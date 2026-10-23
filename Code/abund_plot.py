import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import logging
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import norm
import re
import json
from abundance_contracts import combine_species_statistics, strict_quadrature, species_token, xfe_uncertainty

from config import (
    ATOMIC_Z,
    SOLAR_ABUND_INFO_REL,
    SOLAR_LINES_INFO_REL,
    TC_TABLE_REL,
    resolve_ispec_dir,
)

ISPEC_DIR = resolve_ispec_dir()
TC_PATH = os.path.join(ISPEC_DIR, TC_TABLE_REL)

# ========== For 5_ele_abundance_plot =========
def parse_element_token(s: str):
    """Parse an element label into (base_element, ionization_stage)."""
    parts = str(s).split()
    base = parts[0]
    try:
        ion = int(parts[1]) if len(parts) > 1 else 1
    except ValueError:
        ion = 1
    return base, ion

# Combine species using retained-line counts and propagated uncertainties.
def combine_ions_sem(g):
    """Every valid target line is equal; ion means receive n_i/N weights."""
    return combine_species_statistics(g)


def mkdir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def common_elements_on_Tc(td, rd):
    """Keep elements with condensation temperatures in both target and reference."""
    common = sorted(set(td["Element"]) & set(rd["Element"]))
    if len(common) > 0:
        td = td[td["Element"].isin(common)]
        rd = rd[rd["Element"].isin(common)]
    return td, rd

# Utility functions.
def load_tc(path=TC_PATH):
    """Read and clean the condensation-temperature table. Prefer 50%Tc_K, then Tc_K; return Element and Tc_K."""
    tc_raw = pd.read_csv(path)
    # Identify the condensation-temperature column.
    tc_col = None
    for c in ["50%Tc_K", "Tc50_K", "Tc_K"]:
        if c in tc_raw.columns:
            tc_col = c
            break
    if tc_col is None:
        raise RuntimeError("No condensation-temperature column found; expected 50%Tc_K, Tc50_K, or Tc_K.")

    tc_df = tc_raw[["Element", tc_col]].rename(columns={tc_col: "Tc_K"}).copy()
    # Remove nonnumeric/missing entries and trim whitespace.
    tc_df["Element"] = tc_df["Element"].astype(str).str.strip()
    tc_df["Tc_K"] = pd.to_numeric(tc_df["Tc_K"], errors="coerce")
    tc_df = tc_df.dropna(subset=["Tc_K"])
    return tc_df

def build_ref_for_star(star, a1_path, a2_path):
    """Build reference [X/H] and [X/Fe] tables from Melendez2025 A46 tables. Convert [X/H] = [Fe/H] + [X/Fe] and add their variances assuming independence."""
    a1 = pd.read_csv(a1_path)
    a2 = pd.read_csv(a2_path)

    # Resolve alternative column names.
    def pick(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    col_star_a1 = pick(a1, ["Star","Name","main_id","star","name"])
    col_star_a2 = pick(a2, ["Star","Name","main_id","star","name"])
    col_feh     = pick(a1, ["[Fe/H]","Fe_H","FeH"])
    col_efeh    = pick(a1, ["e_[Fe/H]","[Fe/H]_e","e_Fe_H","Fe_H_e","e_FeH","FeH_e"])
    if not all([col_star_a1, col_star_a2, col_feh]):
        raise RuntimeError("A46 table is missing Star or [Fe/H].")

    # Extract the star's Fe abundance.
    row_a1 = a1.loc[a1[col_star_a1] == star]
    if row_a1.empty:
        raise RuntimeError(f"Star not found in A46_tablea1: {star}")
    feh  = float(row_a1.iloc[0][col_feh])
    efeh = float(row_a1.iloc[0][col_efeh]) if (col_efeh and pd.notna(row_a1.iloc[0][col_efeh])) else np.nan

    # Extract the star's element abundances and uncertainties.
    row_a2 = a2.loc[a2[col_star_a2] == star]
    if row_a2.empty:
        raise RuntimeError(f"Star not found in A46_table2: {star}")

    # Convert to long format: Element, [X/Fe], e_[X/Fe].
    ratio_cols, err_map = [], {}
    for c in row_a2.columns:
        if isinstance(c, str) and re.fullmatch(r"\[[A-Z][a-z]?/Fe\]", c):
            ratio_cols.append(c)
            # Common uncertainty-column names.
            candidates = [f"e_{c}", f"{c}_e"]
            err_map[c] = next((x for x in candidates if x in a2.columns), None)

    recs = []
    for c in ratio_cols:
        val = row_a2.iloc[0][c]
        if pd.isna(val):
            continue
        elem = re.match(r"\[([A-Z][a-z]?)/Fe\]", c).group(1)
        s = row_a2.iloc[0][err_map[c]] if err_map[c] and pd.notna(row_a2.iloc[0][err_map[c]]) else np.nan
        recs.append([elem, float(val), float(s) if pd.notna(s) else np.nan])

    ref_xfe = pd.DataFrame(recs, columns=["Element","[X/Fe]","e_[X/Fe]"])
    ref_xfe["Element"] = ref_xfe["Element"].astype(str).str.strip()

    # Convert to [X/H] and propagate uncertainties.
    ref_xh = ref_xfe.copy()
    ref_xh["[X/H]"]    = ref_xh["[X/Fe]"] + feh
    ref_xh["std_[X/H]"] = strict_quadrature(ref_xh["e_[X/Fe]"], efeh)

    # Add Fe itself.
    ref_xh = pd.concat([ref_xh, pd.DataFrame([["Fe", feh, efeh, 1]], columns=["Element","[X/H]","std_[X/H]","_dummy"])], ignore_index=True)
    ref_xfe = pd.concat([ref_xfe, pd.DataFrame([["Fe", 0.0, 0.0, ]], columns=["Element","[X/Fe]","e_[X/Fe]"])], ignore_index=True)

    return ref_xh[["Element","[X/H]","std_[X/H]"]], ref_xfe[["Element","[X/Fe]","e_[X/Fe]"]]

def _parse_bedell2018_token(colname: str):
    """Parse labels such as [ScII/H], [CI/H], and [CH/H] into an element and a species tag. The tag distinguishes atomic and molecular references."""
    s = str(colname).strip()
    if not (s.startswith("[") and s.endswith("]") and "/H" in s):
        return None, None

    # Extract the species token before /H inside the brackets.
    token = s[1:s.index("/H")]

    # CH is a molecular carbon diagnostic in Bedell2018.
    if token.upper() == "CH":
        return "C", "CH"

    # Parse the one- or two-letter element symbol.
    base = token[0].upper()
    rest = token[1:]
    if len(rest) >= 1 and rest[0].islower():
        base += rest[0]
        rest = rest[1:]

    # The remaining suffix may contain an ionization stage.
    tag = rest if rest else "I"
    return base, tag


def build_ref_for_star_bedell2018_table2(
    star: str,
    table2_path: str,
    feh_override: float | None = None,
    e_feh_override: float | None = None,
    feh_table_path: str | None = None,          # Additional reference field.
    feh_source_hint: str = "Bedell_2018",       # Additional reference field.
    target_species: set[str] | None = None,
    match_target_species: bool = False,
):
    """Build ref_xh and ref_xfe from Bedell2018_table2.csv. The table contains species abundances and errors but no Fe; [X/Fe] requires an external Fe abundance, otherwise it remains unknown."""
    t2 = pd.read_csv(table2_path)

    star_col = "Star" if "Star" in t2.columns else None
    if star_col is None:
        raise RuntimeError(f"[Bedell2018 ref] In {table2_path}: missing Star column")

    # Find the requested star.
    star_key = str(star).strip()
    sel = t2[star_col].astype(str).str.strip() == star_key
    if not sel.any():
        raise RuntimeError(f"[Bedell2018 ref] Star not found in table2: '{star_key}'")

    row = t2.loc[sel].iloc[0]

    # Build a long table for species matching and element aggregation.
    rows = []
    for c in t2.columns:
        if not (isinstance(c, str) and c.startswith("[") and c.endswith("]") and "/H" in c):
            continue

        base, tag = _parse_bedell2018_token(c)
        if base is None:
            continue
        # Table2 is species-level, not per-line. Do not confuse CH with C I.
        token = species_token(f'{base} {tag}')
        if match_target_species and target_species is not None and token not in target_species:
            continue

        xh = pd.to_numeric(row.get(c), errors="coerce")
        err = pd.to_numeric(row.get("e_" + c), errors="coerce")  # Bedell uncertainty columns use e_[X/H].

        if not np.isfinite(xh):
            continue

        rows.append({
            "element": f"{base} {tag}",     # Preserve the element and diagnostic tag for reference matching.
            "[X/H]": float(xh),
            "std_[X/H]": float(err) if np.isfinite(err) else np.nan,
            "n_lines": 1, "error_kind": "reported",  # count is a species placeholder, not a measured line count
        })

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise RuntimeError(f"[Bedell2018 ref] star='{star_key}' has no recognizable [X/H] columns")

    g = combine_ions_for_one_target(df_long)  # -> base_element, [X/H], std_[X/H], n_lines, n_ions

    # Add Fe when available to enable [X/Fe].
    # Resolve Fe/H automatically unless explicitly supplied.
    if feh_override is None and feh_table_path is not None:
        feh_override, e_feh_override = read_feh_from_abundance_with_gaia(
            star=star,
            csv_path=feh_table_path,
            source_hint=feh_source_hint,
        )

    # Append Fe and calculate [X/Fe].
    if feh_override is not None:
        g = pd.concat([g, pd.DataFrame([{
            "base_element": "Fe",
            "[X/H]": float(feh_override),
            "std_[X/H]": float(e_feh_override) if e_feh_override is not None else np.nan,
            "n_lines": 0,
            "n_ions": 1
        }])], ignore_index=True)

    g2 = add_xfe_and_err(g)

    ref_xh = g2.rename(columns={"base_element": "Element"})[["Element", "[X/H]", "std_[X/H]"]]
    ref_xfe = g2.rename(columns={"base_element": "Element"})[["Element", "[X/Fe]", "e_[X/Fe]"]]
    return ref_xh, ref_xfe


def build_ref_for_star_choice(
    star: str,
    ref_name: str,
    bedell_table2_path: str | None = None,
    a46_a1_path: str | None = None,
    a46_a2_path: str | None = None,
    feh_override: float | None = None,
    e_feh_override: float | None = None,
    feh_table_path: str | None = None,
    target_species: set[str] | None = None,
    match_target_species: bool = False,
):
    """Dispatch reference loading by name. A46 requires its parameter and abundance tables; Bedell2018 requires table2 and a Fe reference for [X/Fe]."""
    ref_name = str(ref_name).strip()

    if ref_name.lower() in ["a46", "melendez2025"]:
        # Reuse the A46 reference builder.
        if a46_a1_path is None or a46_a2_path is None:
            raise RuntimeError("ref_name='A46' requires both a46_a1_path and a46_a2_path.")
        return build_ref_for_star(star, a1_path=a46_a1_path, a2_path=a46_a2_path)

    if ref_name.lower() in ["bedell2018", "bedell"]:
        if bedell_table2_path is None:
            raise RuntimeError("ref_name='Bedell2018' requires bedell_table2_path.")
        return build_ref_for_star_bedell2018_table2(
            star=star,
            table2_path=bedell_table2_path,
            feh_override=feh_override,
            e_feh_override=e_feh_override,
            feh_table_path=feh_table_path,     # Forward the option.
            target_species=target_species,
            match_target_species=match_target_species,
            # Optional feh_source_hint defaults to Bedell_2018.
        )

    raise RuntimeError(f"Unknown ref_name: {ref_name}")

def add_tc_to(df_like, tc_df):
    """Attach condensation temperatures to a table with Element or base_element."""
    d = df_like.copy()
    if "Element" not in d.columns and "base_element" in d.columns:
        d = d.rename(columns={"base_element":"Element"})
    # Repeated plotting passes must not create Tc_K_x / Tc_K_y.
    d = d.drop(columns=[c for c in tc_df.columns if c != 'Element' and c in d.columns])
    d = d.merge(tc_df, on="Element", how="left")
    return d

def ensure_element_col(d):
    """Normalize element labels to an Element column."""
    out = d.copy()
    if "Element" in out.columns:
        return out
    if "base_element" in out.columns:
        out = out.rename(columns={"base_element":"Element"})
        return out
    if "element" in out.columns:
        out["Element"] = out["element"].str.split().str[0]
        return out
    raise RuntimeError("DataFrame requires Element, base_element, or element.")



# Plotting functions migrated from step 5.
# Plotting functions.
# ========== fig01 ==========
def plot_xh_by_Z_with_ion(df_in, target, outdir):
    """Plot fig01: species [X/H] ordered by atomic number and ion, with line counts. Requires element, base_element, ion, Z, [X/H], std_[X/H], and n_lines."""
    mkdir(outdir)
    d = df_in.copy()
    order = d.sort_values(["Z","ion","base_element","element"])["element"].tolist()
    d["element"] = pd.Categorical(d["element"], categories=order, ordered=True)
    d = d.sort_values("element")

    plt.figure(figsize=(12,6), dpi=192)
    mask_fe = d["element"].str.startswith("Fe")
    plt.errorbar(d.loc[~mask_fe,"element"], d.loc[~mask_fe,"[X/H]"],
                 yerr=d.loc[~mask_fe,"std_[X/H]"], fmt="o",
                 color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Other elements")
    plt.errorbar(d.loc[ mask_fe,"element"], d.loc[ mask_fe,"[X/H]"],
                 yerr=d.loc[ mask_fe,"std_[X/H]"], fmt="o",
                 color="red",  ecolor="gray", elinewidth=1.5, capsize=4, label="Fe")

    for _, r in d.iterrows():
        plt.text(r["element"], r["[X/H]"]+0.03, f"{int(r['n_lines'])}",
                 ha="center", va="bottom", fontsize=9)

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Element (with ionization stage)", fontsize=14)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] sorted by Z, then ion", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_XH_ordered_by_Z_with_ion.png"), dpi=192)

# ========== fig02 ==========
def plot_xh_by_Z_combined(df_in, target, outdir):
    """Plot fig02: element [X/H] ordered by atomic number. Use the combined retained-line abundance and propagated uncertainty; requires base_element, [X/H], std_[X/H], n_lines, and Z."""
    mkdir(outdir)
    d = df_in.sort_values("Z").copy()
    mask_fe = d["base_element"]=="Fe"

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(d.loc[~mask_fe,"base_element"], d.loc[~mask_fe,"[X/H]"],
                 yerr=d.loc[~mask_fe,"std_[X/H]"], fmt="o", color="blue",
                 ecolor="gray", elinewidth=1.5, capsize=4, markersize=10, label="Other elements")
    plt.errorbar(d.loc[ mask_fe,"base_element"], d.loc[ mask_fe,"[X/H]"],
                 yerr=d.loc[ mask_fe,"std_[X/H]"], fmt="o", color="red",
                 ecolor="gray", elinewidth=1.5, capsize=4, markersize=10, label="Fe")

    for _, r in d.iterrows():
        plt.text(r["base_element"], r["[X/H]"]+0.03, f"{int(r['n_lines'])}",
                 ha="center", va="bottom", fontsize=12)

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] sorted by Z (ions combined)", fontsize=16)
    plt.xticks(d["base_element"].unique(), rotation=45, ha="right", fontsize=13)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_XH_ordered_by_Z.png"), dpi=192)

# ========== fig03 ==========
def plot_xh_by_Tc_ordered(target_df, target, outdir):
    """Plot fig03: [X/H] ordered by condensation temperature without a reference. Requires Element, [X/H], std_[X/H], n_lines, and Tc_K."""
    mkdir(outdir)
    d = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    xticks = d["Element"].tolist()

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(xticks, d["[X/H]"], yerr=d["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4)
    for _, r in d.iterrows():
        n_val = pd.to_numeric(r.get("n_lines", 0), errors="coerce")
        n_show = int(n_val) if np.isfinite(n_val) else 0
        plt.text(r["Element"], r["[X/H]"]+0.03, f"{n_show}",
                 ha="center", va="bottom", fontsize=9)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] ordered by T$_c$", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_XH_ordered_by_Tc.png"), dpi=192)

# ========== fig04 ==========
def plot_xh_vs_Tc_labels(target_df, target, outdir):
    """Plot fig04: [X/H] against numeric condensation temperature with element labels, without a reference. Requires Element, [X/H], std_[X/H], and Tc_K."""
    mkdir(outdir)
    d = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(d["Tc_K"], d["[X/H]"], yerr=d["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4)
    for _, r in d.iterrows():
        plt.text(r["Tc_K"], r["[X/H]"]+0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] vs T$_c$", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04_XH_vs_Tc_labels_Element.png"), dpi=192)

# ========== fig05 ==========
def plot_xh_by_Tc_with_ref(target_df, ref_df, target, ref_name, outdir):
    """Plot fig05: target and reference [X/H] ordered by condensation temperature, with target line counts."""
    mkdir(outdir)
    td = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = rd[rd["Element"].isin(td["Element"])]
    td = td[td["Element"].isin(rd["Element"])]
    xticks = td["Element"].tolist()

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(xticks, td["[X/H]"], yerr=td["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(xticks, rd["[X/H]"], yerr=rd["std_[X/H]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")
    for _, r in td.iterrows():
        plt.text(r["Element"], r["[X/H]"]+0.03, f"{int(r.get('n_lines',0))}",
                 ha="center", va="bottom", fontsize=9)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] ordered by T$_c$ with Ref", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "05_XH_ordered_by_Tc_withRef.png"), dpi=192)

def _annotate_line_counts(data, x=None, y='[X/Fe]'):
    """Annotate actual target lines only; never invent reference line counts."""
    coords=data['Element'].tolist() if x is None else x
    for pos,(_,row) in zip(coords,data.iterrows()):
        n=pd.to_numeric(row.get('n_lines',np.nan),errors='coerce')
        value=pd.to_numeric(row.get(y,np.nan),errors='coerce')
        if np.isfinite(n) and np.isfinite(value):
            plt.text(pos,value+.03,str(int(n)),ha='center',va='bottom',fontsize=9)

# ========== fig06 ==========
def plot_xfe_by_Tc_with_ref(target_xfe_df, ref_xfe_df, target, ref_name, outdir):
    """Plot fig06: target and reference [X/Fe] ordered by condensation temperature, with target line counts."""
    mkdir(outdir)
    td = target_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    td = td[td["Element"].isin(rd["Element"])]
    rd = rd[rd["Element"].isin(td["Element"])]
    xticks = td["Element"].tolist()

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(xticks, td["[X/Fe]"], yerr=td["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(xticks, rd["[X/Fe]"], yerr=rd["e_[X/Fe]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")
    _annotate_line_counts(td)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] ordered by T$_c$ with Ref", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_XFe_ordered_by_Tc_withRef.png"), dpi=192)

# ========== fig06b ==========
def plot_xh_by_Tc_with_ref_and_nohfs(target_hfs_df, target_nohfs_df, ref_df, target, ref_name, outdir):
    """Plot fig06b: HFS, no-HFS, and reference [X/H] ordered by Tc. Show no-HFS values in red only for elements with applied HFS corrections."""
    mkdir(outdir)
    td_hfs = target_hfs_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    rd = ref_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])

    # Main target/reference series use their common elements.
    common_main = sorted(set(td_hfs["Element"]) & set(rd["Element"]))
    td_hfs = td_hfs[td_hfs["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    rd = rd[rd["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    xticks = td_hfs["Element"].tolist()
    x = np.arange(len(xticks))
    pos = {e: i for i, e in enumerate(xticks)}

    # Show finite no-HFS values only where HFS was applied.
    td_no = target_nohfs_df.dropna(subset=["Tc_K"]).copy()
    if "hfs" in td_no.columns:
        td_no = td_no[td_no["hfs"].astype(str).str.lower() == "y"]
    td_no = td_no[td_no["Element"].isin(xticks)].copy()
    td_no = td_no[np.isfinite(pd.to_numeric(td_no.get("[X/H]"), errors="coerce"))]
    td_no = td_no.sort_values(["Tc_K", "Element"])

    plt.figure(figsize=(13, 6), dpi=192)
    plt.errorbar(x, td_hfs["[X/H]"], yerr=td_hfs["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target (HFS)")

    if len(td_no):
        x_no = [pos[e] for e in td_no["Element"].tolist() if e in pos]
        y_no = pd.to_numeric(td_no["[X/H]"], errors="coerce")
        e_no = pd.to_numeric(td_no.get("std_[X/H]", np.nan), errors="coerce")
        plt.errorbar(x_no, y_no, yerr=e_no,
                     fmt="o", mfc="white", mec="red", color="red", ecolor="red",
                     elinewidth=1.2, capsize=3, label="Target (no HFS)")

    rd_aligned = rd.set_index("Element").loc[xticks].reset_index()
    plt.errorbar(x, rd_aligned["[X/H]"], yerr=rd_aligned["std_[X/H]"],
                 fmt="o", mfc="white", mec="green", ecolor="green", elinewidth=1.2, capsize=3,
                 label=f"Ref ({ref_name})")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] ordered by T$_c$ with Ref and no-HFS comparison", fontsize=16)
    plt.xticks(x, xticks, rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06b_XH_ordered_by_Tc_withRef_and_noHFS.png"), dpi=192)

# ========== fig06c ==========
def plot_xfe_by_Tc_with_ref_and_nohfs(target_hfs_xfe_df, target_nohfs_xfe_df, ref_xfe_df, target, ref_name, outdir):
    """Plot fig06c: HFS, no-HFS, and reference [X/Fe] ordered by Tc. Show no-HFS values in red only for elements with applied HFS corrections."""
    mkdir(outdir)
    td_hfs = target_hfs_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    rd = ref_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])

    common_main = sorted(set(td_hfs["Element"]) & set(rd["Element"]))
    td_hfs = td_hfs[td_hfs["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    rd = rd[rd["Element"].isin(common_main)].sort_values(["Tc_K", "Element"])
    xticks = td_hfs["Element"].tolist()
    x = np.arange(len(xticks))
    pos = {e: i for i, e in enumerate(xticks)}

    td_no = target_nohfs_xfe_df.dropna(subset=["Tc_K"]).copy()
    if "hfs" in td_no.columns:
        td_no = td_no[td_no["hfs"].astype(str).str.lower() == "y"]
    td_no = td_no[td_no["Element"].isin(xticks)].copy()
    td_no = td_no[np.isfinite(pd.to_numeric(td_no.get("[X/Fe]"), errors="coerce"))]
    td_no = td_no.sort_values(["Tc_K", "Element"])

    plt.figure(figsize=(13, 6), dpi=192)
    plt.errorbar(x, td_hfs["[X/Fe]"], yerr=td_hfs["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target (HFS)")

    if len(td_no):
        x_no = [pos[e] for e in td_no["Element"].tolist() if e in pos]
        y_no = pd.to_numeric(td_no["[X/Fe]"], errors="coerce")
        e_no = pd.to_numeric(td_no.get("e_[X/Fe]", np.nan), errors="coerce")
        plt.errorbar(x_no, y_no, yerr=e_no,
                     fmt="o", mfc="white", mec="red", color="red", ecolor="red",
                     elinewidth=1.2, capsize=3, label="Target (no HFS)")

    rd_aligned = rd.set_index("Element").loc[xticks].reset_index()
    plt.errorbar(x, rd_aligned["[X/Fe]"], yerr=rd_aligned["e_[X/Fe]"],
                 fmt="o", mfc="white", mec="green", ecolor="green", elinewidth=1.2, capsize=3,
                 label=f"Ref ({ref_name})")

    _annotate_line_counts(td_hfs, x=x)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] ordered by T$_c$ with Ref and no-HFS comparison", fontsize=16)
    plt.xticks(x, xticks, rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06c_XFe_ordered_by_Tc_withRef_and_noHFS.png"), dpi=192)

# ========== fig07 ==========
def plot_xh_vs_Tc_with_ref_labels(target_df, ref_df, target, ref_name, outdir):
    """Plot fig07: target and reference [X/H] against numeric Tc, with element labels."""
    mkdir(outdir)
    td = target_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    td, rd = common_elements_on_Tc(td, rd)

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(td["Tc_K"], td["[X/H]"], yerr=td["std_[X/H]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(rd["Tc_K"], rd["[X/H]"], yerr=rd["std_[X/H]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")

    for _, r in td.iterrows():
        plt.text(r["Tc_K"], r["[X/H]"]+0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    for _, r in rd.iterrows():
        plt.text(r["Tc_K"], r["[X/H]"]-0.03, r["Element"], ha="center", va="top", fontsize=10, color="green")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/H] (dex)", fontsize=14)
    plt.title(f"{target}, [X/H] vs T$_c$ with Ref", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "07_XH_vs_Tc_labels_withRef.png"), dpi=192)

# ========== fig08 ==========
def plot_xfe_vs_Tc_with_ref_labels(target_xfe_df, ref_xfe_df, target, ref_name, outdir):
    """Plot fig08: target and reference [X/Fe] against numeric Tc, with element labels."""
    mkdir(outdir)
    td = target_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    rd = ref_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K","Element"])
    td, rd = common_elements_on_Tc(td, rd)

    plt.figure(figsize=(12,6), dpi=192)
    plt.errorbar(td["Tc_K"], td["[X/Fe]"], yerr=td["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target")
    plt.errorbar(rd["Tc_K"], rd["[X/Fe]"], yerr=rd["e_[X/Fe]"],
                 fmt="o", mfc="white", mec="green", ecolor="green",
                 elinewidth=1.2, capsize=3, label=f"Ref ({ref_name})")

    for _, r in td.iterrows():
        plt.text(r["Tc_K"], r["[X/Fe]"]+0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    for _, r in rd.iterrows():
        plt.text(r["Tc_K"], r["[X/Fe]"]-0.03, r["Element"], ha="center", va="top", fontsize=10, color="green")

    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] vs T$_c$ with Ref", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "08_XFe_vs_Tc_labels_withRef.png"), dpi=192)

# ========== fig06 / 06c / 08 without ref ==========
def plot_xfe_by_Tc_ordered(target_xfe_df, target, outdir):
    """Plot fig06: target [X/Fe] ordered by Tc without a reference. Requires Element, [X/Fe], e_[X/Fe], and Tc_K."""
    mkdir(outdir)
    d = target_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    xticks = d["Element"].tolist()

    plt.figure(figsize=(12, 6), dpi=192)
    plt.errorbar(xticks, d["[X/Fe]"], yerr=d["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4)
    _annotate_line_counts(d)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] ordered by T$_c$", fontsize=16)
    plt.xticks(xticks, rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_XFe_ordered_by_Tc.png"), dpi=192)

def plot_xfe_by_Tc_and_nohfs(target_hfs_xfe_df, target_nohfs_xfe_df, target, outdir):
    """Plot fig06c: HFS and no-HFS [X/Fe] ordered by Tc, without a reference. Show no-HFS values in red only for elements with applied HFS corrections."""
    mkdir(outdir)
    td_hfs = target_hfs_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    xticks = td_hfs["Element"].tolist()
    x = np.arange(len(xticks))
    pos = {e: i for i, e in enumerate(xticks)}

    td_no = target_nohfs_xfe_df.dropna(subset=["Tc_K"]).copy()
    if "hfs" in td_no.columns:
        td_no = td_no[td_no["hfs"].astype(str).str.lower() == "y"]
    td_no = td_no[td_no["Element"].isin(xticks)].copy()
    td_no = td_no[np.isfinite(pd.to_numeric(td_no.get("[X/Fe]"), errors="coerce"))]
    td_no = td_no.sort_values(["Tc_K", "Element"])

    plt.figure(figsize=(13, 6), dpi=192)
    plt.errorbar(x, td_hfs["[X/Fe]"], yerr=td_hfs["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4, label="Target (HFS)")

    if len(td_no):
        x_no = [pos[e] for e in td_no["Element"].tolist() if e in pos]
        y_no = pd.to_numeric(td_no["[X/Fe]"], errors="coerce")
        e_no = pd.to_numeric(td_no.get("e_[X/Fe]", np.nan), errors="coerce")
        plt.errorbar(x_no, y_no, yerr=e_no,
                     fmt="o", mfc="white", mec="red", color="red", ecolor="red",
                     elinewidth=1.2, capsize=3, label="Target (no HFS)")

    _annotate_line_counts(td_hfs, x=x)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] ordered by T$_c$ with no-HFS comparison", fontsize=16)
    plt.xticks(x, xticks, rotation=45, ha="right")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06c_XFe_ordered_by_Tc_and_noHFS.png"), dpi=192)

def plot_xfe_vs_Tc_labels(target_xfe_df, target, outdir):
    """Plot fig08: target [X/Fe] against numeric Tc, with element labels and no reference."""
    mkdir(outdir)
    d = target_xfe_df.dropna(subset=["Tc_K"]).copy().sort_values(["Tc_K", "Element"])
    plt.figure(figsize=(12, 6), dpi=192)
    plt.errorbar(d["Tc_K"], d["[X/Fe]"], yerr=d["e_[X/Fe]"],
                 fmt="o", color="blue", ecolor="gray", elinewidth=1.5, capsize=4)
    for _, r in d.iterrows():
        plt.text(r["Tc_K"], r["[X/Fe]"] + 0.03, r["Element"], ha="center", va="bottom", fontsize=10)
    plt.axhline(0, color="k", ls="--", lw=1)
    plt.xlabel("Condensation temperature T$_c$ (K)", fontsize=14)
    plt.ylabel("[X/Fe] (dex)", fontsize=14)
    plt.title(f"{target}, [X/Fe] vs T$_c$", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "08_XFe_vs_Tc_labels.png"), dpi=192)

# ========== Step 5 pipeline runner ==========
def _step5_agg_hfs_flag(vals):
    vv = {str(v).strip().lower() for v in vals if str(v).strip() != ""}
    if "y" in vv:
        return "y"
    if "f" in vv:
        return "f"
    return "n"


def _step5_ref_disabled(ref_name, ref_star) -> bool:
    """Reference comparison is disabled when REF_NAME or REF_STAR is empty."""

    def _missing(value) -> bool:
        if value is None:
            return True
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        return str(value).strip().lower() in {"", "none", "nan", "<na>"}

    return _missing(ref_name) or _missing(ref_star)


def _step5_build_xfe_from_xh(xh_df):
    fe_row = xh_df.loc[xh_df["Element"] == "Fe"].iloc[0] if (xh_df["Element"] == "Fe").any() else None
    if fe_row is not None:
        xfe = xh_df.copy()
        xfe["[X/Fe]"] = xfe["[X/H]"] - fe_row["[X/H]"]
        xfe["e_[X/Fe]"] = xfe_uncertainty(xfe, fe_row)
        xfe.loc[xfe.Element == 'Fe', 'e_[X/Fe]'] = 0.0
        xfe['uncertainty_kind'] = ('common_parameter_response_plus_independent_statistics'
            if 'sensitivity_teff' in xfe else 'statistical_independence_approximation')

    else:
        xfe = xh_df.assign(**{"[X/Fe]": np.nan, "e_[X/Fe]": np.nan})
    return xfe


def run_step5_ele_abundance_plots(
    *,
    target,
    output_abundances_result_path_linebyline_q2,
    abundance_plot_folder,
    output_folder,
    ref_name,
    ref_star,
    pipeline_options=None,
    ispec_dir=None,
):
    print("[STEP 1] Loading abundance summary and preparing figure output folder.")
    df = pd.read_csv(output_abundances_result_path_linebyline_q2)
    os.makedirs(abundance_plot_folder, exist_ok=True)

    print("[STEP 2] Building element and ion level tables (fig01/fig02).")
    tmp = df["element"].apply(parse_element_token)
    df["base_element"] = tmp.apply(lambda x: x[0])
    df["ion"] = tmp.apply(lambda x: x[1])
    df["Z"] = df["base_element"].map(ATOMIC_Z).fillna(9999)
    df_combined = df.groupby("base_element", as_index=False).apply(combine_ions_sem)
    df_combined["Z"] = df_combined["base_element"].map(ATOMIC_Z)
    plot_xh_by_Z_with_ion(df, target=target, outdir=abundance_plot_folder)
    plot_xh_by_Z_combined(df_combined, target=target, outdir=abundance_plot_folder)

    print("[STEP 3] Preparing Tc-based target tables and HFS flags.")
    tc_df = load_tc(TC_PATH)
    summary_path = output_abundances_result_path_linebyline_q2
    raw_path = summary_path.replace('.csv', '_raw.csv')

    df_summary_src = pd.read_csv(summary_path)
    df_summary_src["element"] = df_summary_src["element"].astype(str)
    if "hfs" not in df_summary_src.columns:
        df_summary_src["hfs"] = "n"

    tmp_sum = df_summary_src["element"].apply(parse_element_token)
    df_summary_src["base_element"] = tmp_sum.apply(lambda x: x[0])
    df_summary_src["ion"] = tmp_sum.apply(lambda x: x[1])
    df_summary_src["[X/H]"] = pd.to_numeric(df_summary_src.get("[X/H]"), errors="coerce")
    df_summary_src["std_[X/H]"] = pd.to_numeric(df_summary_src.get("std_[X/H]"), errors="coerce")
    df_summary_src["n_lines"] = pd.to_numeric(df_summary_src.get("n_lines"), errors="coerce")

    df_combined_hfs = df_summary_src.groupby("base_element", as_index=False).apply(combine_ions_sem)
    df_combined_hfs["Z"] = df_combined_hfs["base_element"].map(ATOMIC_Z)
    target_xh = ensure_element_col(df_combined_hfs).copy()

    flag_sum = df_summary_src[["base_element", "hfs"]].copy()
    flag_sum["hfs"] = flag_sum["hfs"].astype(str)
    flag_sum = flag_sum.groupby("base_element", as_index=False)["hfs"].agg(_step5_agg_hfs_flag)

    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        raw_df["element"] = raw_df["element"].astype(str)
        if "hfs" in raw_df.columns:
            tmp_raw = raw_df["element"].apply(parse_element_token)
            raw_df["base_element"] = tmp_raw.apply(lambda x: x[0])
            h_raw_num = pd.to_numeric(raw_df["hfs"], errors="coerce")
            raw_df["_hraw"] = np.where(np.isfinite(h_raw_num), "y", raw_df["hfs"].astype(str).str.strip().str.lower())
            flag_raw = raw_df.groupby("base_element", as_index=False)["_hraw"].agg(_step5_agg_hfs_flag).rename(columns={"_hraw": "hfs_raw"})
        else:
            flag_raw = pd.DataFrame(columns=["base_element", "hfs_raw"])
    else:
        flag_raw = pd.DataFrame(columns=["base_element", "hfs_raw"])

    hfs_flag_elem = flag_sum.merge(flag_raw, on="base_element", how="outer")
    hfs_flag_elem["hfs"] = hfs_flag_elem.apply(
        lambda r: _step5_agg_hfs_flag([str(r.get("hfs", "n")), str(r.get("hfs_raw", "n"))]), axis=1
    )
    hfs_flag_elem = hfs_flag_elem.rename(columns={"base_element": "Element"})[["Element", "hfs"]]

    target_xh = target_xh.merge(hfs_flag_elem, on="Element", how="left")
    target_xh["hfs"] = target_xh["hfs"].fillna("n")
    target_xh = add_tc_to(target_xh, tc_df)
    target_xfe = _step5_build_xfe_from_xh(target_xh)

    pipeline_options = pipeline_options or {}
    plot_hfs_with_external_ew = bool(pipeline_options.get("plot_hfs_with_external_ews", False))
    use_hfs_plots = (not pipeline_options.get("input_has_external_ews")) or plot_hfs_with_external_ew
    if pipeline_options.get("input_has_external_ews") and plot_hfs_with_external_ew:
        print("[STEP 3] plot_hfs_with_external_ews=True, still drawing HFS/no-HFS compare figures.")

    nohfs_summary_path = os.path.join(output_folder, "HFS", "tables", f"abundances_nohfs_blends_{target}_summary.csv")
    target_xh_nohfs = target_xh.copy()
    target_xh_nohfs.loc[target_xh_nohfs.hfs == "y", "[X/H]"] = np.nan
    target_xh_nohfs.loc[target_xh_nohfs.hfs == "y", "std_[X/H]"] = np.nan
    target_xh_nohfs.loc[target_xh_nohfs.hfs == "y", "n_lines"] = np.nan

    target_xfe_nohfs = target_xfe[["Element", "Z", "hfs"]].copy()
    target_xfe_nohfs["[X/Fe]"] = np.nan
    target_xfe_nohfs["e_[X/Fe]"] = np.nan

    if use_hfs_plots:
        if os.path.exists(nohfs_summary_path):
            nohfs_sum = pd.read_csv(nohfs_summary_path)
            nohfs_sum["element"] = nohfs_sum["element"].astype(str)
            # New summaries expose SEM explicitly. Legacy uncertainty is unknown.
            nohfs_sum['xh_std'] = nohfs_sum.get('xh_sem', np.nan)
            nohfs_sum['xfe_std'] = nohfs_sum.get('xfe_sem', np.nan)
            nohfs_sum = nohfs_sum.rename(columns={
                "element": "Element",
                "xh_mean": "nohfs_xh",
                "xh_std": "nohfs_xh_std",
                "xfe_mean": "nohfs_xfe",
                "xfe_std": "nohfs_xfe_std",
                "n_lines": "nohfs_n_lines",
            })
            target_xh_nohfs = target_xh_nohfs.merge(
                nohfs_sum[["Element", "nohfs_xh", "nohfs_xh_std", "nohfs_n_lines"]],
                on="Element", how="left"
            )
            mask_y = target_xh_nohfs["hfs"].astype(str).str.lower() == "y"
            target_xh_nohfs.loc[mask_y, "[X/H]"] = pd.to_numeric(target_xh_nohfs.loc[mask_y, "nohfs_xh"], errors="coerce")
            target_xh_nohfs.loc[mask_y, "std_[X/H]"] = pd.to_numeric(target_xh_nohfs.loc[mask_y, "nohfs_xh_std"], errors="coerce")
            target_xh_nohfs.loc[mask_y, "n_lines"] = pd.to_numeric(target_xh_nohfs.loc[mask_y, "nohfs_n_lines"], errors="coerce")
            # Paired no-HFS diagnostics have not received HFS parameter
            # sensitivities. Never reuse those sensitivities as no-HFS errors.
            for col in ['stat_err_[X/H]','param_err_[X/H]']+[f'sensitivity_{k}' for k in ['teff','logg','MH','vmic']]:
                if col in target_xh_nohfs:target_xh_nohfs.loc[mask_y,col]=np.nan
            if 'param_err_[X/H]' in target_xh_nohfs:
                target_xh_nohfs.loc[mask_y,'std_[X/H]']=np.nan
            target_xh_nohfs = target_xh_nohfs.drop(columns=["nohfs_xh", "nohfs_xh_std", "nohfs_n_lines"])

            target_xfe_nohfs = target_xfe_nohfs.merge(
                nohfs_sum[["Element", "nohfs_xfe", "nohfs_xfe_std"]],
                on="Element", how="left"
            )
            mask_y2 = target_xfe_nohfs["hfs"].astype(str).str.lower() == "y"
            target_xfe_nohfs.loc[mask_y2, "[X/Fe]"] = pd.to_numeric(target_xfe_nohfs.loc[mask_y2, "nohfs_xfe"], errors="coerce")
            target_xfe_nohfs.loc[mask_y2, "e_[X/Fe]"] = pd.to_numeric(target_xfe_nohfs.loc[mask_y2, "nohfs_xfe_std"], errors="coerce")
            target_xfe_nohfs = target_xfe_nohfs.drop(columns=["nohfs_xfe", "nohfs_xfe_std"])
        else:
            print(f"[warn] no-HFS summary not found for fig06b/06c: {nohfs_summary_path}")
    else:
        print("[STEP 3] Skip no-HFS summary when external EWs are used.")

    target_xfe_nohfs = _step5_build_xfe_from_xh(target_xh_nohfs)
    target_xh_nohfs = add_tc_to(target_xh_nohfs, tc_df)
    target_xfe_nohfs = add_tc_to(target_xfe_nohfs, tc_df)

    print("[STEP 4] Building reference abundance table.")
    if _step5_ref_disabled(ref_name, ref_star):
        print(
            "[STEP 4] No literature reference "
            f"(REF_NAME={ref_name!r}, REF_STAR={ref_star!r}). "
            "Skipping ref table and ref plots."
        )
        print("[STEP 5] Generating abundance plots without a reference.")
        plot_xh_by_Tc_ordered(target_xh, target=target, outdir=abundance_plot_folder)
        plot_xh_vs_Tc_labels(target_xh, target=target, outdir=abundance_plot_folder)
        plot_xfe_by_Tc_ordered(target_xfe, target=target, outdir=abundance_plot_folder)
        plot_xfe_vs_Tc_labels(target_xfe, target=target, outdir=abundance_plot_folder)
        if use_hfs_plots:
            plot_xfe_by_Tc_and_nohfs(
                target_xfe, target_xfe_nohfs, target=target, outdir=abundance_plot_folder
            )
        else:
            print("[STEP 5] Skip 06c (external EWs and no HFS compare).")
        print(f"[STEP 6] Step_element_abundance_plot completed. Output folder: {abundance_plot_folder}")
        return {
            "summary_path": summary_path,
            "plot_dir": abundance_plot_folder,
            "ref_star": None,
            "ref_name": None,
            "used_hfs_compare": bool(use_hfs_plots),
            "nohfs_summary_path": nohfs_summary_path,
        }

    if ispec_dir is None:
        ispec_dir = resolve_ispec_dir()
    if ref_name == "Bedell2018":
        bedell_table2 = os.path.join(ispec_dir, "input/abundances/D_PASTA/BedellHARPS/Bedell2018_table2.csv")
        bedell_feh_table = os.path.join(ispec_dir, "input/abundances/D_PASTA/BedellHARPS/abundance_with_gaia.csv")
        from run_context import REF_MATCH_SPECIES
        match_species = bool(pipeline_options.get('ref_match_species', REF_MATCH_SPECIES))
        used = df_summary_src[np.isfinite(df_summary_src['[X/H]']) & (df_summary_src.n_lines > 0)].copy()
        used['species_token'] = used.element.map(species_token)
        selected = set(used.species_token)
        if match_species:
            columns = pd.read_csv(bedell_table2, nrows=0).columns
            available = set()
            for col in columns:
                base, tag = _parse_bedell2018_token(col)
                if base is not None:
                    available.add(species_token(f'{base} {tag}'))
            accepted = selected & available
            audit = pd.DataFrame([{'target_species': t, 'matched': t in available,
                                   'scope': 'species_abundance_not_per_line'} for t in sorted(selected) if not t.startswith('Fe ')])
            audit.to_csv(os.path.join(abundance_plot_folder, 'reference_species_matching.csv'), index=False)
            aligned = combine_ions_for_one_target(used[used.species_token.isin(accepted) | used.species_token.str.startswith('Fe ')])
            aligned = aligned.rename(columns={'base_element': 'Element'})
            replace=[c for c in aligned.columns if c!='Element']
            target_xh = target_xh.drop(columns=replace,errors='ignore').merge(
                aligned, on='Element', how='left')
            target_xfe = _step5_build_xfe_from_xh(target_xh)
        ref_xh, ref_xfe = build_ref_for_star_choice(
            star=ref_star,
            ref_name="Bedell2018",
            bedell_table2_path=bedell_table2,
            feh_table_path=bedell_feh_table,
            target_species=selected,
            match_target_species=match_species,
        )
    elif ref_name == "Melendez2025":
        pd.DataFrame([{'scope': 'element_only_reference', 'action': 'species_matching_skipped'}]).to_csv(
            os.path.join(abundance_plot_folder, 'reference_species_matching.csv'), index=False)
        a1_path = os.path.join(ispec_dir, "input/abundances/D_PASTA/Melendez2025/A46_tablea1_raw.csv")
        a2_path = os.path.join(ispec_dir, "input/abundances/D_PASTA/Melendez2025/A46_table2_raw.csv")
        ref_xh, ref_xfe = build_ref_for_star_choice(
            star=ref_star,
            ref_name="Melendez2025",
            a46_a1_path=a1_path,
            a46_a2_path=a2_path,
        )
    else:
        raise RuntimeError(f"Unsupported REF_NAME for step 5: {ref_name}")

    ref_xh = add_tc_to(ref_xh, tc_df)
    ref_xfe = add_tc_to(ref_xfe, tc_df)

    print("[STEP 5] Generating all abundance comparison plots.")
    plot_xh_by_Tc_ordered(target_xh, target=target, outdir=abundance_plot_folder)
    plot_xh_vs_Tc_labels(target_xh, target=target, outdir=abundance_plot_folder)
    plot_xh_by_Tc_with_ref(target_xh, ref_xh, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
    plot_xfe_by_Tc_with_ref(target_xfe, ref_xfe, target=target, ref_name=ref_star, outdir=abundance_plot_folder)

    if use_hfs_plots:
        plot_xh_by_Tc_with_ref_and_nohfs(target_xh, target_xh_nohfs, ref_xh, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
        plot_xfe_by_Tc_with_ref_and_nohfs(target_xfe, target_xfe_nohfs, ref_xfe, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
    else:
        print("[STEP 5] Skip fig06b / fig06c (external EWs and no HFS compare).")

    plot_xh_vs_Tc_with_ref_labels(target_xh, ref_xh, target=target, ref_name=ref_star, outdir=abundance_plot_folder)
    plot_xfe_vs_Tc_with_ref_labels(target_xfe, ref_xfe, target=target, ref_name=ref_star, outdir=abundance_plot_folder)

    print(f"[STEP 6] Step_element_abundance_plot completed. Output folder: {abundance_plot_folder}")
    return {
        "summary_path": summary_path,
        "plot_dir": abundance_plot_folder,
        "ref_star": ref_star,
        "ref_name": ref_name,
        "used_hfs_compare": bool(use_hfs_plots),
        "nohfs_summary_path": nohfs_summary_path,
    }


######## ===== For 6_combine_abundances ===== ########
def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def combine_ions_for_one_target(df):
    """Load a target abundance table and combine species. Return base_element, [X/H], std_[X/H], and n_lines using the retained-line aggregation contract."""
    d = df.copy()
    # Parse base element and ionization stage.
    toks = d["element"].astype(str).str.strip()
    d["base_element"] = toks.str.split().str[0]
    # Normalize numeric column names.
    if "[X/H]" not in d.columns:
        raise RuntimeError("Input CSV is missing [X/H].")
    # Convert uncertainty and line-count columns to numeric values.
    d["std_[X/H]"] = pd.to_numeric(d.get("std_[X/H]"), errors="coerce")
    d["n_lines"]    = pd.to_numeric(d.get("n_lines"), errors="coerce")

    # Combine ionization stages using the shared abundance contract.
    g = d.groupby("base_element", as_index=False).apply(combine_ions_sem)
    g = g.reset_index().rename(columns={"level_0":"base_element"})
    # The combined output includes [X/H], std_[X/H], n_lines, and n_ions.
    g = g.drop(columns=['index'],errors='ignore')
    return g

def add_xfe_and_err(g_combined):
    """Calculate [X/Fe] and its uncertainty using the combined Fe abundance."""
    out = g_combined.copy()
    # Locate the Fe row.
    if not (out["base_element"] == "Fe").any():
        out["[X/Fe]"] = np.nan
        out["e_[X/Fe]"] = np.nan
        return out

    fe_row = out.loc[out["base_element"] == "Fe"].iloc[0]
    feh = safe_float(fe_row["[X/H]"])
    e_feh = safe_float(fe_row["std_[X/H]"])
    out["[X/Fe]"]  = out["[X/H]"] - feh
    # Propagate uncertainties; retain unknown errors as unknown.
    out["e_[X/Fe]"] = xfe_uncertainty(out, fe_row)
    out.loc[out.base_element == 'Fe', 'e_[X/Fe]'] = 0.0
    return out

def resolve_target_abundance_csv(root, target):
    """Resolve the abundance summary: prefer abundances_linebyline_<target>.csv, then the legacy abundances_<target>.csv."""
    candidates = [
        os.path.join(root, target, f"abundances_linebyline_{target}.csv"),
        os.path.join(root, target, f"abundances_{target}.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def build_master_abundance_table(ispec_dir, targets=None, out_path=None, tc_path=TC_PATH):
    """Build a multi-target table of [X/H], [X/H]_err, [X/Fe], and [X/Fe]_err per element. If supplied, place Tc_K in the second column."""
    root = os.path.join(ispec_dir, "output")
    if targets is None:
        candidates = []
        for sub in os.listdir(root):
            subdir = os.path.join(root, sub)
            if os.path.isdir(subdir) and resolve_target_abundance_csv(root, sub):
                candidates.append(sub)
        targets = sorted(candidates)

    master = pd.DataFrame({"Element": []})

    for t in targets:
        csv_path = resolve_target_abundance_csv(root, t)
        if csv_path is None:
            print(f"[build_master] skip {t}: no abundance csv found")
            continue
        df = pd.read_csv(csv_path)
        for c in ["element", "[X/H]"]:
            if c not in df.columns:
                raise RuntimeError(f"{csv_path} is missing columns: {c}")

        comb = combine_ions_for_one_target(df)
        comb = add_xfe_and_err(comb)

        comb = comb.rename(columns={"base_element":"Element"})
        comb = comb[["Element", "[X/H]", "std_[X/H]", "[X/Fe]", "e_[X/Fe]"]]
        comb = comb.rename(columns={
            "[X/H]":      f"{t}_[X/H]",
            "std_[X/H]":  f"{t}_[X/H]_err",
            "[X/Fe]":     f"{t}_[X/Fe]",
            "e_[X/Fe]":   f"{t}_[X/Fe]_err",
        })
        master = master.merge(comb, on="Element", how="outer")

    # Optionally sort by atomic number.
    try:
        z_map = {k:int(v) for k,v in ATOMIC_Z.items()}
        master["_Z"] = master["Element"].map(z_map)
        master = master.sort_values(by=["_Z","Element"], kind="stable").drop(columns="_Z")
    except Exception:
        pass

    # Attach condensation temperatures as the second column.
    if tc_path:
        tc_df = load_tc(tc_path)
        master = master.merge(tc_df, on="Element", how="left")
        # Move Tc_K to the second column.
        cols = list(master.columns)
        cols.remove("Tc_K")
        cols = [cols[0], "Tc_K"] + cols[1:]
        master = master[cols]

    if out_path is None:
        out_path = os.path.join(root, "all_targets_abundances.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    master.to_csv(out_path, index=False)
    print(f"Summary completed: {out_path}")
    return master

# ========== For 3_atmos_params_iteration =========
# using q2
# Convert iSpec linemasks to q2 input tables.
# q2 requires wavelength, species, excitation potential, loggf, and stellar EWs.

def load_ispec_params_from_dump(ispec, dump_file):
    """Read fitted parameters from an iSpec atmos_params dump. Return teff, logg, feh (MH), and vt (vmic)."""
    restored = ispec.restore_results(dump_file)
    params = restored[0]
    return {
        "teff": float(params["teff"]),
        "logg": float(params["logg"]),
        "feh":  float(params["MH"]),
        "vt":   float(params["vmic"]),
    }

def _lm_to_q2_line_df(linemasks, star_id):
    """Convert iSpec linemasks to a long table with wavelength, species, ep, gf, and ew. iSpec labels neutral and singly ionized iron as Fe 1 and Fe 2."""
    rows = []
    for lm in linemasks:
        elem = str(lm["element"]).strip()
        if elem not in ["Fe 1", "Fe 2", "Fe I", "Fe II"]:
            continue

        species = 26.0 if elem in ["Fe 1", "Fe I"] else 26.1
        rows.append({
            "wavelength": float(lm["wave_A"]),           # Å
            "species":    float(species),
            "ep":         float(lm["lower_state_eV"]),   # eV
            "gf":         float(lm["loggf"]),            # log(gf)
            "ew":         float(lm["ew"]),               # mÅ
            "id":         str(star_id),
        })

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _merge_fe_ew_overrides(
    df_base: pd.DataFrame,
    ov: Optional[pd.DataFrame],
    wave_tol: float,
    label: str,
) -> pd.DataFrame:
    """Override EWs by matching species, wavelength, ep, and gf against an external long table, keeping target and solar q2 inputs aligned."""
    if ov is None or len(ov) == 0:
        return df_base
    df = df_base.copy()
    ov = ov.copy()
    df["_wk"] = (df["wavelength"].astype(float) / wave_tol).round().astype(np.int64)
    ov["_wk"] = (ov["wavelength"].astype(float) / wave_tol).round().astype(np.int64)
    df["_epk"] = np.round(df["ep"].astype(float), 4)
    df["_gfk"] = np.round(df["gf"].astype(float), 4)
    ov["_epk"] = np.round(ov["ep"].astype(float), 4)
    ov["_gfk"] = np.round(ov["gf"].astype(float), 4)
    keys = ["species", "_wk", "_epk", "_gfk"]
    missing = [k for k in keys if k not in ov.columns]
    if missing:
        raise ValueError(f"Target/Sun Fe override table is missing columns: {missing}")
    sub = ov.drop_duplicates(subset=keys, keep="last")[keys + ["ew"]].rename(columns={"ew": "_ew_ov"})
    out = df.merge(sub, on=keys, how="left")
    n = int(out["_ew_ov"].notna().sum())
    if n:
        out["ew"] = out["_ew_ov"].where(out["_ew_ov"].notna(), out["ew"])
    out = out.drop(columns=["_ew_ov", "_wk", "_epk", "_gfk"], errors="ignore")
    print(f"[export_q2_inputs] FE EW overrides applied ({label}): {n} / {len(df_base)}")
    return out


def export_q2_inputs_wide(
    ispec,
    target_id,
    used_linemasks_fe,              # Final Fe linemasks, including measured EWs.
    solar_lines_csv,                # Master solar EW table, keyed by instrument, resolution, species, and wavelength.
    instrument, resolution,
    out_dir,
    sun_dump_file,                  # Solar atmosphere dump for the selected instrument.
    sun_id="Sun",
    wave_tol=0.001,                 # Angstrom tolerance for matching the same transition.
    target_fe_ew_overrides=None,    # Optional external Fe table: wavelength, species, ep, gf, ew.
    sun_fe_ew_overrides=None,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Target Fe lines in long format.
    df_t = _lm_to_q2_line_df(used_linemasks_fe, target_id)
    print(f"[export_q2_inputs] target Fe lines: {len(df_t)}")
    df_t = _merge_fe_ew_overrides(df_t, target_fe_ew_overrides, wave_tol, "target")

    # 2. Solar Fe lines from the master library.
    sun_master = pd.read_csv(solar_lines_csv)

    # Filter by instrument first.
    sun_inst = sun_master[sun_master["instrument"].astype(str).str.strip() == str(instrument).strip()].copy()
    if len(sun_inst) == 0:
        raise RuntimeError(f"[export_q2_inputs] solar master has no rows for instrument={instrument}")

    # Select the nearest available resolution.
    # Convert numeric and string representations to float.
    avail = pd.to_numeric(sun_inst["resolution"], errors="coerce").values
    if not np.isfinite(avail).any():
        raise RuntimeError(f"[export_q2_inputs] solar master resolution column cannot be parsed for instrument={instrument}")

    req = float(resolution)
    res_use = avail[np.nanargmin(np.abs(avail - req))]

    # Select solar Fe lines at the chosen resolution.
    sun_sel = sun_inst[
        (pd.to_numeric(sun_inst["resolution"], errors="coerce") == float(res_use)) &
        (sun_inst["element"].isin(["Fe I", "Fe II", "Fe 1", "Fe 2"]))
    ].copy()

    if abs(float(res_use) - req) > 1.0:
        print(
            f"[export_q2_inputs] nearest solar R: requested={resolution} -> used={res_use}"
        )
    else:
        print(f"[export_q2_inputs] solar resolution requested={resolution}, used={res_use}")

    # Accept both Fe I and Fe 1 labels.
    def _elem_to_species(x):
        x = str(x).strip()
        return 26.0 if x in ["Fe I", "Fe 1"] else 26.1

    df_s = pd.DataFrame({
        "wavelength": sun_sel["wave_A"].astype(float),
        "species":    sun_sel["element"].map(_elem_to_species).astype(float),
        "ep":         sun_sel["EP"].astype(float),
        "gf":         sun_sel["loggf"].astype(float),
        "ew":         sun_sel["ew_mA"].astype(float),
        "id":         sun_id,
    }).dropna()
    print(f"[export_q2_inputs] sun Fe lines   : {len(df_s)}")
    df_s = _merge_fe_ew_overrides(df_s, sun_fe_ew_overrides, wave_tol, "Sun")

    # 3. Keep common species/wavelength transitions only.
    df_t["wave_key"] = (df_t["wavelength"]/wave_tol).round().astype(int)
    df_s["wave_key"] = (df_s["wavelength"]/wave_tol).round().astype(int)

    key_cols = ["species","wave_key","ep","gf"]
    common = pd.merge(
        df_t[key_cols].drop_duplicates(),
        df_s[key_cols].drop_duplicates(),
        on=key_cols, how="inner"
    )
    print(f"[export_q2_inputs] common Fe lines: {len(common)}")

    df_t2 = df_t.merge(common, on=key_cols, how="inner")
    df_s2 = df_s.merge(common, on=key_cols, how="inner")

    # 4. Build the wide q2 lines table.
    base = common.copy()
    # Use the solar wavelength as the common identifier.
    wave_map = df_s2.groupby(key_cols)["wavelength"].median().reset_index()
    base = base.merge(wave_map, on=key_cols, how="left")

    # Retain each star's measured EW.
    sun_ew = df_s2.groupby(key_cols)["ew"].median().reset_index().rename(columns={"ew": sun_id})
    tar_ew = df_t2.groupby(key_cols)["ew"].median().reset_index().rename(columns={"ew": str(target_id)})

    lines_wide = base.merge(sun_ew, on=key_cols, how="inner").merge(tar_ew, on=key_cols, how="inner")

    # Column order: wavelength, species, ep, gf, then one EW column per star.
    lines_wide = lines_wide[["wavelength","species","ep","gf",sun_id,str(target_id)]].sort_values(
        by=["species","wavelength"]
    )

    lines_csv = os.path.join(out_dir, "lines_q2.csv")
    lines_wide.to_csv(lines_csv, index=False)
    print(f"[export_q2_inputs] lines.csv rows : {len(lines_wide)} -> {lines_csv}")

    # 5. Build stars.csv with solar dump parameters and target initial guesses.
    sun_par = load_ispec_params_from_dump(ispec, sun_dump_file)

    # Target initial parameters are supplied by the caller.
    # Keep initial guesses configurable through function arguments.
    # The calling atmosphere workflow supplies the target initial parameters.

    stars_csv = os.path.join(out_dir, "stars_q2.csv")

    return lines_csv, stars_csv, sun_par

# ========= For 0_Sun_abundance ==========
def _build_sun_atmos_offset_row(
    measured,
    adopted,
    *,
    solar_label,
    instrument,
    resolution,
    atmos_method,
    feh_q2_column=np.nan,
    afe_sun_reference=np.nan,
    note="",
):
    m = {k: float(measured[k]) for k in ("teff", "logg", "MH", "vmic", "alpha")}
    a = {k: float(adopted[k]) for k in ("teff", "logg", "MH", "vmic", "alpha")}
    return {
        "solar_label": str(solar_label),
        "instrument": str(instrument),
        "resolution": int(resolution),
        "atmos_method": str(atmos_method),
        "teff_measured": m["teff"],
        "logg_measured": m["logg"],
        "MH_measured": m["MH"],
        "vmic_measured": m["vmic"],
        "alpha_measured": m["alpha"],
        "feh_q2_column": float(feh_q2_column) if np.isfinite(feh_q2_column) else np.nan,
        "AFe_q2_column": float(feh_q2_column) if np.isfinite(feh_q2_column) else np.nan,
        "AFe_sun_reference": float(afe_sun_reference) if np.isfinite(afe_sun_reference) else np.nan,
        "teff_adopted": a["teff"],
        "logg_adopted": a["logg"],
        "MH_adopted": a["MH"],
        "vmic_adopted": a["vmic"],
        "alpha_adopted": a["alpha"],
        "offset_teff": a["teff"] - m["teff"],
        "offset_logg": a["logg"] - m["logg"],
        "offset_MH": a["MH"] - m["MH"],
        "offset_vmic": a["vmic"] - m["vmic"],
        "offset_alpha": a["alpha"] - m["alpha"],
        "note": str(note),
    }


def _save_sun_atmos_offset_csv(row, per_target_csv, master_csv=None):
    pd.DataFrame([row]).to_csv(per_target_csv, index=False)
    print(f"[0_Sun][Part3] Saved offset table -> {per_target_csv}")
    if master_csv:
        os.makedirs(os.path.dirname(master_csv), exist_ok=True)
        if os.path.isfile(master_csv):
            master = pd.read_csv(master_csv)
            mask = (
                (master["solar_label"].astype(str) == row["solar_label"])
                & (master["instrument"].astype(str) == row["instrument"])
                & (pd.to_numeric(master["resolution"], errors="coerce") == row["resolution"])
            )
            master = master.loc[~mask]
            master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)
        else:
            master = pd.DataFrame([row])
        master.to_csv(master_csv, index=False)
        print(f"[0_Sun][Part3] Updated master offset table -> {master_csv}")


def _resolve_solar_library_paths(*, ispec_dir: str, pipeline_options: dict[str, object]):
    if bool(pipeline_options.get("input_has_external_ews", False)):
        base = os.path.join(ispec_dir, "output", "withEW_Bedell")
        os.makedirs(base, exist_ok=True)
        solar_lines_csv = os.path.join(base, "solar_lines_instru_Bedell2018_Sun.csv")
    else:
        solar_lines_csv = os.path.join(ispec_dir, SOLAR_LINES_INFO_REL)
    solar_abund_csv = os.path.join(ispec_dir, SOLAR_ABUND_INFO_REL)
    return solar_lines_csv, solar_abund_csv


def _to_roman_ion(ele_label: str):
    s = str(ele_label).strip().replace("  ", " ")
    parts = s.split(" ")
    sym = parts[0] if len(parts) else s
    ion_raw = parts[1] if len(parts) > 1 else "1"
    ion_map = {"1": "I", "2": "II", "I": "I", "II": "II", "i": "I", "ii": "II"}
    ion_roman = ion_map.get(str(ion_raw), str(ion_raw))
    ion_num = 1 if ion_roman == "I" else 2 if ion_roman == "II" else ion_roman
    return f"{sym} {ion_roman}", f"{sym} {ion_num}"


def _run_sun_line_abundance_exports(
    *,
    ispec,
    linemasks,
    target: str,
    solar_label: str,
    instrument: str,
    from_resolution: float,
    output_folder: str,
    ispec_dir: str,
    teff: float,
    logg: float,
    mh: float,
    alpha: float,
    vmic: float,
    atmosphere_layers,
    solar_abundances,
    code: str,
    pipeline_options: dict[str, object],
):
    elem_norm = np.array([str(e).strip() for e in linemasks["element"]])
    elements = sorted(set(elem_norm.tolist()))
    solar_lines = []
    for ele in elements:
        sel = elem_norm == ele
        lm_ele = linemasks[sel]
        if len(lm_ele) == 0:
            continue
        spec_abund, _, x_over_h, _ = ispec.determine_abundances(
            atmosphere_layers,
            teff,
            logg,
            mh,
            alpha,
            lm_ele,
            solar_abundances,
            microturbulence_vel=vmic,
            verbose=0,
            code=code,
        )
        spec_abund = np.asarray(spec_abund, dtype=float)
        x_over_h = np.asarray(x_over_h, dtype=float)
        for lm, logeps_m12, xh in zip(lm_ele, spec_abund, x_over_h):
            if not (np.isfinite(logeps_m12) and np.isfinite(xh)):
                continue
            solar_lines.append(
                {
                    "instrument": instrument,
                    "resolution": int(from_resolution),
                    "solar_label": str(solar_label),
                    "element": ele,
                    "wave_A": float(lm["wave_A"]),
                    "EP": float(lm["lower_state_eV"]),
                    "loggf": float(lm["loggf"]),
                    "ew_mA": float(lm["ew"]),
                    "logeps_sun": float(logeps_m12 + 12.0),
                    "logeps_sun_nohfs": float(logeps_m12 + 12.0),
                    "hfs_status": "not_run",
                    "[X/H]_sun_Grevesse": float(xh),
                }
            )
    df_solar_lines = pd.DataFrame(solar_lines)
    print(f"[0_Sun][Part4] Solar per-line rows: {len(df_solar_lines)}")

    solar_lines_csv, solar_abund_csv = _resolve_solar_library_paths(
        ispec_dir=ispec_dir, pipeline_options=pipeline_options
    )
    # Replace only this (instrument, resolution, solar_label) block so other
    # R / labels for the same instrument (e.g. MIKE 65000 vs 45500) are kept.
    def _same_solar_block(df):
        return (
            (df["instrument"].astype(str) == str(instrument))
            & (
                pd.to_numeric(df["resolution"], errors="coerce")
                == int(from_resolution)
            )
            & (df["solar_label"].astype(str) == str(solar_label))
        )

    if os.path.exists(solar_lines_csv):
        master = pd.read_csv(solar_lines_csv)
        if "solar_label" not in master.columns:
            master["solar_label"] = ""
        master = master.loc[~_same_solar_block(master)].copy()
        master = pd.concat([master, df_solar_lines], ignore_index=True)
    else:
        master = df_solar_lines
    master.to_csv(solar_lines_csv, index=False)
    print(f"[0_Sun][Part4] Saved solar_lines_instru -> {solar_lines_csv} rows={len(master)}")

    g = df_solar_lines.groupby("element")
    df_solar_abund = pd.DataFrame(
        {
            "instrument": instrument,
            "resolution": int(from_resolution),
            "solar_label": str(solar_label),
            "element": g.size().index,
            "n_lines": g.size().values,
            "logeps_sun_mean": g["logeps_sun"].mean().values,
            "[X/H]_sun_mean_Grevesse": g["[X/H]_sun_Grevesse"].mean().values,
            "std_[X/H]_sun": g["[X/H]_sun_Grevesse"].std(ddof=1).values,
        }
    )
    if os.path.exists(solar_abund_csv):
        master_a = pd.read_csv(solar_abund_csv)
        if "solar_label" not in master_a.columns:
            master_a["solar_label"] = ""
        master_a = master_a.loc[~_same_solar_block(master_a)].copy()
        master_a = pd.concat([master_a, df_solar_abund], ignore_index=True)
    else:
        master_a = df_solar_abund
    master_a.to_csv(solar_abund_csv, index=False)
    print(f"[0_Sun][Part4] Saved solar_abund_instru -> {solar_abund_csv} rows={len(master_a)}")

    q2_raw_csv = os.path.join(output_folder, f"abundances_linebyline_{target}_raw.csv")
    q2_sum_csv = os.path.join(output_folder, f"abundances_linebyline_{target}.csv")
    if len(df_solar_lines):
        q2_raw = df_solar_lines.copy()
        q2_raw["element_roman"] = q2_raw["element"].map(lambda x: _to_roman_ion(x)[0])
        q2_raw["element_num"] = q2_raw["element"].map(lambda x: _to_roman_ion(x)[1])
        q2_raw["id"] = target
        q2_raw["EP"] = pd.to_numeric(q2_raw.get("EP"), errors="coerce")
        q2_raw["loggf"] = pd.to_numeric(q2_raw.get("loggf"), errors="coerce")
        q2_raw["ew_mA"] = pd.to_numeric(q2_raw.get("ew_mA"), errors="coerce")
        q2_raw["wave_A"] = pd.to_numeric(q2_raw.get("wave_A"), errors="coerce")
        q2_raw["[X/H]_star_Grevesse"] = pd.to_numeric(
            q2_raw.get("[X/H]_sun_Grevesse"), errors="coerce"
        )
        q2_raw["[X/H]_sun_Grevesse"] = 0.0
        q2_raw["d[X/H]"] = q2_raw["[X/H]_star_Grevesse"] - q2_raw["[X/H]_sun_Grevesse"]
        q2_raw["logeps_star"] = pd.to_numeric(q2_raw.get("logeps_sun"), errors="coerce")
        q2_raw["logeps_sun"] = q2_raw["logeps_star"] - q2_raw["d[X/H]"]
        q2_raw["dlogeps"] = q2_raw["logeps_star"] - q2_raw["logeps_sun"]
        q2_raw_out = q2_raw[
            [
                "id",
                "element_roman",
                "wave_A",
                "EP",
                "loggf",
                "ew_mA",
                "logeps_star",
                "logeps_sun",
                "dlogeps",
                "[X/H]_star_Grevesse",
                "[X/H]_sun_Grevesse",
                "d[X/H]",
            ]
        ].rename(columns={"element_roman": "element"})
        q2_raw_out.to_csv(q2_raw_csv, index=False)

        q2_sum = (
            q2_raw.assign(_xh=pd.to_numeric(q2_raw["d[X/H]"], errors="coerce"))
            .groupby("element_num", as_index=False)
            .agg(
                n_lines=("wave_A", "size"),
                **{"[X/H]": ("_xh", "mean"), "std_[X/H]": ("_xh", "std")},
            )
            .rename(columns={"element_num": "element"})
        )
        q2_sum["sem_[X/H]"] = q2_sum["std_[X/H]"] / np.sqrt(q2_sum.n_lines.where(q2_sum.n_lines > 1))
        q2_sum.to_csv(q2_sum_csv, index=False)
        print(f"[0_Sun][Part5] Saved q2 raw -> {q2_raw_csv} rows={len(q2_raw_out)}")
        print(f"[0_Sun][Part5] Saved q2 summary -> {q2_sum_csv} rows={len(q2_sum)}")
    else:
        pd.DataFrame(
            columns=[
                "id",
                "element",
                "wave_A",
                "EP",
                "loggf",
                "ew_mA",
                "logeps_star",
                "logeps_sun",
                "dlogeps",
                "[X/H]_star_Grevesse",
                "[X/H]_sun_Grevesse",
                "d[X/H]",
            ]
        ).to_csv(q2_raw_csv, index=False)
        pd.DataFrame(columns=["element", "n_lines", "[X/H]", "std_[X/H]"]).to_csv(
            q2_sum_csv, index=False
        )
        print("[0_Sun][Part5] No line abundance rows; wrote empty q2 outputs.")

    return {
        "solar_lines_csv": solar_lines_csv,
        "solar_abund_csv": solar_abund_csv,
        "q2_raw_csv": q2_raw_csv,
        "q2_summary_csv": q2_sum_csv,
    }


def _sync_solar_library_after_hfs(
    *,
    export_out: dict[str, object],
    output_folder: str,
    target: str,
    solar_label: str,
    instrument: str,
    from_resolution: float,
    hfs_res: dict[str, object],
) -> dict[str, object]:
    """
    After solar HFS updates output q2 raw/summary, also rewrite matching rows in
    solar_lines_instru.csv / solar_abund_instru.csv so target HFS differentials
    use A_sun,HFS (not the pre-HFS library values written in Part4).
    """
    if not bool(hfs_res.get("ran")):
        return {"synced": False, "reason": "hfs_not_run"}
    if bool(hfs_res.get("dry_run")):
        return {"synced": False, "reason": "hfs_dry_run"}
    if int(hfs_res.get("replaced_n", 0) or 0) <= 0:
        return {"synced": False, "reason": "no_hfs_replacements"}

    q2_raw_csv = str(export_out.get("q2_raw_csv") or "")
    solar_lines_csv = str(export_out.get("solar_lines_csv") or "")
    solar_abund_csv = str(export_out.get("solar_abund_csv") or "")
    if not (q2_raw_csv and os.path.isfile(q2_raw_csv) and solar_lines_csv):
        return {"synced": False, "reason": "missing_q2_raw_or_solar_lines_csv"}

    raw = pd.read_csv(q2_raw_csv)
    if "hfs" not in raw.columns:
        return {"synced": False, "reason": "raw_missing_hfs_column"}

    hfs_flag = raw["hfs"].astype(str).str.strip()
    # Success rows store the previous d[X/H] numeric string; failures are "f".
    ok_mask = (
        hfs_flag.ne("")
        & ~hfs_flag.str.lower().isin(["f", "n", "nan", "none", "<na>"])
    )
    raw_ok = raw.loc[ok_mask].copy()
    if len(raw_ok) == 0:
        return {"synced": False, "reason": "no_ok_hfs_rows_in_raw"}

    raw_ok["element_num"] = raw_ok["element"].map(lambda x: _to_roman_ion(x)[1])
    raw_ok["wave_A"] = pd.to_numeric(raw_ok["wave_A"], errors="coerce")
    raw_ok["logeps_star"] = pd.to_numeric(raw_ok["logeps_star"], errors="coerce")
    raw_ok["d[X/H]"] = pd.to_numeric(raw_ok["d[X/H]"], errors="coerce")
    raw_ok = raw_ok[
        np.isfinite(raw_ok["wave_A"])
        & np.isfinite(raw_ok["logeps_star"])
        & np.isfinite(raw_ok["d[X/H]"])
    ].copy()
    if len(raw_ok) == 0:
        return {"synced": False, "reason": "no_finite_hfs_abundances"}

    if os.path.isfile(solar_lines_csv):
        master = pd.read_csv(solar_lines_csv)
    else:
        return {"synced": False, "reason": "solar_lines_csv_missing"}

    res_i = int(from_resolution)
    inst = str(instrument)
    label = str(solar_label)
    scope = (
        (master["instrument"].astype(str) == inst)
        & (pd.to_numeric(master["resolution"], errors="coerce") == res_i)
        & (master["solar_label"].astype(str) == label)
    )
    if not scope.any():
        # Fall back: instrument+resolution only (older rows without solar_label match)
        scope = (
            (master["instrument"].astype(str) == inst)
            & (pd.to_numeric(master["resolution"], errors="coerce") == res_i)
        )
    if not scope.any():
        return {"synced": False, "reason": "no_matching_solar_library_scope"}

    n_updated = 0
    master_wave = pd.to_numeric(master["wave_A"], errors="coerce")
    for _, r in raw_ok.iterrows():
        ele_num = str(r["element_num"]).strip()
        w = float(r["wave_A"])
        hit = (
            scope
            & (master["element"].astype(str).str.strip() == ele_num)
            & np.isclose(master_wave, w, atol=0.003, rtol=0, equal_nan=False)
        )
        if not hit.any():
            # Also try roman form in library (Ba II vs Ba 2)
            ele_roman = str(r["element"]).strip()
            hit = (
                scope
                & (master["element"].astype(str).str.strip() == ele_roman)
                & np.isclose(master_wave, w, atol=0.003, rtol=0, equal_nan=False)
            )
        if not hit.any():
            continue
        for col, val in [('logeps_sun_nohfs', r.get('logeps_star_nohfs', np.nan)),
                         ('hfs_status', r.get('hfs_status', 'ok')),
                         ('component_signature', r.get('component_signature', ''))]:
            if col not in master.columns:
                master[col] = np.nan if col == 'logeps_sun_nohfs' else ''
            master.loc[hit, col] = val
        master.loc[hit, "logeps_sun"] = float(r["logeps_star"])
        master.loc[hit, "[X/H]_sun_Grevesse"] = float(r["d[X/H]"])
        n_updated += int(hit.sum())

    if n_updated <= 0:
        return {"synced": False, "reason": "no_library_rows_matched", "n_hfs_ok": int(len(raw_ok))}

    master.to_csv(solar_lines_csv, index=False)
    print(
        f"[0_Sun][HFS→library] Updated solar_lines_instru "
        f"({n_updated} row(s)) -> {solar_lines_csv}"
    )

    # Rebuild element means for this instrument/resolution(/label) scope.
    if solar_abund_csv:
        scoped = master.loc[scope].copy()
        if len(scoped):
            g = scoped.groupby("element")
            df_ab = pd.DataFrame(
                {
                    "instrument": inst,
                    "resolution": res_i,
                    "solar_label": label,
                    "element": g.size().index,
                    "n_lines": g.size().values,
                    "logeps_sun_mean": g["logeps_sun"].mean().values,
                    "[X/H]_sun_mean_Grevesse": g["[X/H]_sun_Grevesse"].mean().values,
                    "std_[X/H]_sun": g["[X/H]_sun_Grevesse"].std(ddof=1).values,
                }
            )
            if os.path.isfile(solar_abund_csv):
                master_a = pd.read_csv(solar_abund_csv)
                drop = (
                    (master_a["instrument"].astype(str) == inst)
                    & (pd.to_numeric(master_a["resolution"], errors="coerce") == res_i)
                )
                if "solar_label" in master_a.columns:
                    drop = drop & (master_a["solar_label"].astype(str) == label)
                master_a = master_a.loc[~drop]
                master_a = pd.concat([master_a, df_ab], ignore_index=True)
            else:
                master_a = df_ab
            master_a.to_csv(solar_abund_csv, index=False)
            print(
                f"[0_Sun][HFS→library] Rebuilt solar_abund_instru "
                f"({len(df_ab)} elements) -> {solar_abund_csv}"
            )

    # Optional audit trail next to the Sun HFS tables.
    tables_dir = os.path.join(output_folder, "HFS", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    audit_path = os.path.join(tables_dir, f"solar_library_hfs_sync_{target}.csv")
    raw_ok[
        ["element", "element_num", "wave_A", "logeps_star", "d[X/H]", "hfs"]
    ].rename(
        columns={
            "logeps_star": "logeps_sun_HFS",
            "d[X/H]": "[X/H]_sun_Grevesse_HFS",
            "hfs": "preHFS_d[X/H]",
        }
    ).to_csv(audit_path, index=False)

    return {
        "synced": True,
        "n_updated": int(n_updated),
        "n_hfs_ok": int(len(raw_ok)),
        "solar_lines_csv": solar_lines_csv,
        "solar_abund_csv": solar_abund_csv,
        "audit_csv": audit_path,
    }


def run_sun_part3plus_pipeline(
    *,
    ispec,
    target: str,
    solar_label: str,
    instrument: str,
    from_resolution: float,
    atmos_iter_method: str,
    output_folder: str,
    output_atmos_params_dumpfile_path: str,
    sun_atmos_offset_csv: str,
    sun_atmos_offset_master_csv: str,
    solar_atmos_literature: dict[str, float],
    solar_afe_literature: float,
    pipeline_options: dict[str, object],
    ispec_dir: str,
    spectrum_norm_path: str,
    initial_teff: float,
    initial_logg: float,
    initial_mh: float,
    q2_solution: dict[str, float] | None = None,
    glb_solution: dict[str, object] | None = None,
    linemasks_fe=None,
):
    from atmos_lbl_pipeline import step_log
    from abund_lbl_pipeline import (
        apply_lineqa_for_abundance,
        load_star_and_atmosphere_context,
        plot_all_element_gaussian_fits,
        prepare_abundance_engine,
        run_hfs_blends_and_update_outputs,
        run_nonfe_line_review,
    )

    adopted = dict(solar_atmos_literature)
    atmos_iter_method = str(atmos_iter_method).strip().lower()
    step_log("3", f"Running Sun part3+ pipeline with atmos_iter_method={atmos_iter_method}.")

    if atmos_iter_method == "glb":
        if not glb_solution:
            raise ValueError("glb_solution is required when atmos_iter_method='glb'.")
        params = dict(glb_solution["params"])
        errors = dict(glb_solution["errors"])
        status = dict(glb_solution.get("status", {}))
        x_over_h = glb_solution.get("x_over_h")
        selected_x_over_h = glb_solution.get("selected_x_over_h")
        fitted_lines_params = glb_solution.get("fitted_lines_params")
        used_linemasks = glb_solution.get("used_linemasks")
        spec_abund_fe = glb_solution.get("spec_abund_fe")
        measured = {
            "teff": float(params["teff"]),
            "logg": float(params["logg"]),
            "MH": float(params["MH"]),
            "vmic": float(params["vmic"]),
            "alpha": float(params.get("alpha", 0.0)),
        }
        params_sun = dict(adopted)
        errors_sun = {
            k: float(errors[k]) if pd.notna(errors.get(k)) else np.nan
            for k in ("teff", "logg", "MH", "alpha", "vmic")
        }
        status_sun = {
            "converged": True,
            "method": "literature_solar_anchor_method1",
            "measured_method": "ispec_glb_Fe_EW",
            **status,
        }
        ispec.save_results(
            output_atmos_params_dumpfile_path,
            (
                params_sun,
                errors_sun,
                status_sun,
                x_over_h,
                selected_x_over_h,
                fitted_lines_params,
                used_linemasks,
                spec_abund_fe,
            ),
        )
        row = _build_sun_atmos_offset_row(
            measured,
            adopted,
            solar_label=solar_label,
            instrument=instrument,
            resolution=from_resolution,
            atmos_method="glb",
            note="measured from ispec.model_spectrum_from_ew; adopted=literature solar",
        )
    elif atmos_iter_method == "lbl":
        if not q2_solution:
            raise ValueError("q2_solution is required when atmos_iter_method='lbl'.")
        afe_q2 = float(q2_solution["feh_q2"])
        mh_q2_dex = q2_single_star_feh_column_to_mh(afe_q2, solar_afe_literature)
        measured = {
            "teff": float(q2_solution["teff_q2"]),
            "logg": float(q2_solution["logg_q2"]),
            "MH": float(mh_q2_dex),
            "vmic": float(q2_solution["vt_q2"]),
            "alpha": 0.0,
        }
        params_sun = dict(adopted)
        errors_sun = {"teff": 0.0, "logg": 0.0, "MH": 0.0, "alpha": 0.0, "vmic": 0.0}
        status_sun = {
            "converged": True,
            "method": "literature_solar_anchor_method1",
            "measured_method": "q2_single_star_Fe",
            "q2_teff": float(q2_solution["teff_q2"]),
            "q2_logg": float(q2_solution["logg_q2"]),
            "q2_vmic": float(q2_solution["vt_q2"]),
            "q2_feh_column": float(q2_solution["feh_q2"]),
        }
        if linemasks_fe is None:
            lm_path = os.path.join(
                output_folder, "linemasks", f"{target}_melendez2014_star_fitted_linemasks.txt"
            )
            _lm = ispec.read_line_regions(lm_path)
            _elem = np.array([str(e).strip() for e in _lm["element"]])
            linemasks_fe = _lm[(_elem == "Fe 1") | (_elem == "Fe 2")]

        ispec.save_results(
            output_atmos_params_dumpfile_path,
            (params_sun, errors_sun, status_sun, None, None, None, linemasks_fe, None),
        )
        row = _build_sun_atmos_offset_row(
            measured,
            adopted,
            solar_label=solar_label,
            instrument=instrument,
            resolution=from_resolution,
            atmos_method="lbl",
            feh_q2_column=float(q2_solution["feh_q2"]),
            afe_sun_reference=float(solar_afe_literature),
            note="MH_measured converted from q2 A(Fe) with literature A(Fe)_sun",
        )
    else:
        raise ValueError(f"Unknown atmos_iter_method: {atmos_iter_method!r}")

    _save_sun_atmos_offset_csv(row, sun_atmos_offset_csv, sun_atmos_offset_master_csv)
    step_log("3", f"Saved Sun atmospheric dump and offset table: {output_atmos_params_dumpfile_path}")

    run_find_linemasks = bool(pipeline_options.get("run_find_linemasks", True))
    use_external_ew = bool(pipeline_options.get("input_has_external_ews", False))
    run_part1_spectrum = run_find_linemasks and (not use_external_ew)
    plot_all_elements = bool(pipeline_options.get("sun_plot_all_elements", False))

    ctx4 = load_star_and_atmosphere_context(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        spectrum_norm_path=spectrum_norm_path,
        output_atmos_params_dumpfile_path=output_atmos_params_dumpfile_path,
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_MH=initial_mh,
        ispec_dir=ispec_dir,
        pipeline_options=pipeline_options,
    )
    plot_all_element_gaussian_fits(
        star_spectrum=ctx4["star_spectrum"],
        linemasks=ctx4["linemasks"],
        output_folder=output_folder,
        run_part1_spectrum=run_part1_spectrum,
        plot_all_elements=plot_all_elements,
        progress_every=10,
    )
    run_nonfe_line_review(
        ispec=ispec, star_spectrum=ctx4["star_spectrum"], linemasks=ctx4["linemasks"],
        output_folder=output_folder,
        target=target,
        run_part1_spectrum=run_part1_spectrum,
        plot_all_elements=plot_all_elements,
        pipeline_options=pipeline_options,
    )
    linemasks_used = apply_lineqa_for_abundance(
        ispec=ispec,
        linemasks=ctx4["linemasks"],
        output_folder=output_folder,
        linemask_output_folder=ctx4["linemask_output_folder"],
        target=target,
        pipeline_options=pipeline_options,
    )
    linemasks_used = linemasks_used[linemasks_used["wave_nm"] > 0]
    linemasks_used = linemasks_used[linemasks_used["ew"] > 0]

    engine = prepare_abundance_engine(
        ispec=ispec,
        ispec_dir=ispec_dir,
        teff=ctx4["teff"],
        logg=ctx4["logg"],
        mh=ctx4["mh"],
        alpha=ctx4["alpha"],
        code="moog",
    )
    export_out = _run_sun_line_abundance_exports(
        ispec=ispec,
        linemasks=linemasks_used,
        target=target,
        solar_label=solar_label,
        instrument=instrument,
        from_resolution=from_resolution,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        teff=ctx4["teff"],
        logg=ctx4["logg"],
        mh=ctx4["mh"],
        alpha=ctx4["alpha"],
        vmic=ctx4["vmic"],
        atmosphere_layers=engine["atmosphere_layers"],
        solar_abundances=engine["solar_abundances"],
        code=engine["code"],
        pipeline_options=pipeline_options,
    )

    hfs_res = run_hfs_blends_and_update_outputs(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        atmosphere_layers=engine["atmosphere_layers"],
        teff=ctx4["teff"],
        logg=ctx4["logg"],
        mh=ctx4["mh"],
        output_abundances_result_path_linebyline_q2=export_out["q2_summary_csv"],
        pipeline_options=pipeline_options,
        vmic=ctx4['vmic'],
        is_solar_reference=True,
        solar_label=solar_label,
        run_hfs_blends=bool(pipeline_options.get("sun_run_hfs_blends", True)),
        dry_run=bool(pipeline_options.get("sun_hfs_dry_run", False)),
    )
    # Part4 wrote solar_lines_instru before HFS; sync A_sun,HFS back so target
    # HFS differentials are A_star,HFS - A_sun,HFS (not A_sun,noHFS).
    library_sync = _sync_solar_library_after_hfs(
        export_out=export_out,
        output_folder=output_folder,
        target=target,
        solar_label=solar_label,
        instrument=instrument,
        from_resolution=from_resolution,
        hfs_res={
            **dict(hfs_res or {}),
            "dry_run": bool(pipeline_options.get("sun_hfs_dry_run", False)),
        },
    )
    if library_sync.get("synced"):
        step_log(
            "7",
            f"Synced HFS solar abundances into library "
            f"(n_updated={library_sync.get('n_updated')}).",
        )
    else:
        step_log(
            "7",
            f"Skip solar-library HFS sync: {library_sync.get('reason', 'unknown')}",
        )
    step_log("6", "Sun part3+ pipeline completed.")
    return {
        "export": export_out,
        "hfs": hfs_res,
        "library_sync": library_sync,
        "dump_file": output_atmos_params_dumpfile_path,
    }


def run_sun_atmos_iteration(
    *,
    ispec,
    linemasks,
    atmos_iter_method: str,
    output_folder: str,
    solar_label: str,
    initial_teff: float,
    initial_logg: float,
    initial_mh: float,
    code: str = "moog",
    model: str | None = None,
    solar_abundances_file: str | None = None,
):
    """
    Execute Sun atmospheric-parameter iteration (glb/lbl) and return structured
    outputs for downstream run_sun_part3plus_pipeline().
    """
    from atmos_lbl_pipeline import step_log

    atmos_iter_method = str(atmos_iter_method).strip().lower()
    if model is None:
        model = os.path.join(resolve_ispec_dir(), "input", "atmospheres", "MARCS.GES") + os.sep
    if solar_abundances_file is None:
        if "ATLAS" in model:
            solar_abundances_file = os.path.join(
                resolve_ispec_dir(), "input", "abundances", "Grevesse.1998", "stdatom.dat"
            )
        else:
            solar_abundances_file = os.path.join(
                resolve_ispec_dir(), "input", "abundances", "Grevesse.2007", "stdatom.dat"
            )

    if atmos_iter_method == "glb":
        step_log("2", "Running Sun global atmospheric fit (ispec.model_spectrum_from_ew).")
        modeled_layers_pack = ispec.load_modeled_layers_pack(model)
        solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
        initial_alpha = 0.0
        initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_mh)
        if not ispec.valid_atmosphere_target(
            modeled_layers_pack,
            {"teff": initial_teff, "logg": initial_logg, "MH": initial_mh, "alpha": initial_alpha},
        ):
            step_log("2", "Warning: initial parameters are outside model grid.")
        results = ispec.model_spectrum_from_ew(
            linemasks,
            modeled_layers_pack,
            solar_abundances,
            initial_teff,
            initial_logg,
            initial_mh,
            initial_alpha,
            initial_vmic,
            free_params=["teff", "logg", "vmic"],
            adjust_model_metalicity=True,
            max_iterations=15,
            enhance_abundances=True,
            outliers_detection="robust",
            outliers_weight_limit=0.90,
            tmp_dir=None,
            code=code,
        )
        params, errors, status, x_over_h, selected_x_over_h, fitted_lines_params, used_linemasks = results
        step_log(
            "2",
            f"GLB solution: Teff={float(params['teff']):.2f}, logg={float(params['logg']):.3f}, "
            f"[M/H]={float(params['MH']):.3f}, vmic={float(params['vmic']):.3f}",
        )
        return {
            "q2_solution": None,
            "linemasks_fe": None,
            "glb_solution": {
                "params": params,
                "errors": errors,
                "status": status,
                "x_over_h": x_over_h,
                "selected_x_over_h": selected_x_over_h,
                "fitted_lines_params": fitted_lines_params,
                "used_linemasks": used_linemasks,
                "spec_abund_fe": np.asarray(x_over_h, dtype=float),
            },
        }

    if atmos_iter_method == "lbl":
        step_log("2", "Running Sun LBL atmospheric fit (q2 specpars).")
        import q2

        elem_col = np.array([str(e).strip() for e in linemasks["element"]])
        linemasks_fe = linemasks[(elem_col == "Fe 1") | (elem_col == "Fe 2")]
        q2_dir = os.path.join(output_folder, "q2_work")
        os.makedirs(q2_dir, exist_ok=True)
        initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_mh)
        stars_csv, lines_csv = export_q2_inputs_single_wide(
            linemasks_fe,
            star_id=solar_label,
            teff=initial_teff,
            logg=initial_logg,
            feh=initial_mh,
            vt=float(initial_vmic),
            out_dir=q2_dir,
            fe_only=True,
        )

        data = q2.Data(stars_csv, lines_csv)
        sp = q2.specpars.SolvePars(grid="marcs")
        sp.errors = True
        solution_csv = os.path.join(q2_dir, "solution.csv")
        q2.specpars.solve_all(data, sp, solution_csv)
        sol = pd.read_csv(solution_csv)
        row = sol.loc[sol["id"] == str(solar_label)].iloc[0]
        teff_q2, logg_q2, feh_q2, vt_q2 = map(
            float, [row["teff"], row["logg"], row["feh"], row["vt"]]
        )
        step_log(
            "2",
            f"LBL solution: Teff={teff_q2:.2f}, logg={logg_q2:.3f}, feh_col={feh_q2:.3f}, vt={vt_q2:.3f}",
        )
        return {
            "linemasks_fe": linemasks_fe,
            "glb_solution": None,
            "q2_solution": {
                "teff_q2": teff_q2,
                "logg_q2": logg_q2,
                "feh_q2": feh_q2,
                "vt_q2": vt_q2,
            },
        }

    raise ValueError(f"Unknown atmos_iter_method: {atmos_iter_method!r}")


def _to_bool(x):
    """Robust bool conversion for 'discarded' or similar columns."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")

def export_q2_inputs_single_wide(
    linemasks,
    star_id,
    teff, logg, feh, vt,
    out_dir,
    fe_only=True,
    wave_tol=0.001,         # Å; used to group same line
    discard_col="discarded", # if exists in linemasks
):
    """
    Export q2 inputs for a SINGLE star (Sun-only use case).
    Writes:
      - stars_q2.csv : columns [id, teff, logg, feh, vt]
      - lines_q2.csv : wide format: [wavelength, species, ep, gf, <star_id>]
                       where <star_id> column stores EW (mA).

    Parameters
    ----------
    linemasks : numpy structured array / table
        iSpec linemasks that contains at least:
        element, wave_A, lower_state_eV, loggf, ew
    star_id : str
        The star label used as column name in lines_q2.csv
    teff, logg, feh, vt : float
        Initial parameters for q2
    out_dir : str
        Directory to write q2 inputs
    fe_only : bool
        If True, keep only Fe I/Fe II lines (recommended for specpars)
    wave_tol : float
        Wavelength tolerance (Å) for grouping duplicates
    discard_col : str
        Column name for discarded flag, if present

    Returns
    -------
    (stars_csv_path, lines_csv_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) build long table from linemasks ----------
    rows = []
    names = getattr(linemasks, "dtype", None).names if hasattr(linemasks, "dtype") else None
    if names is None:
        raise TypeError("linemasks must be a numpy structured array with dtype.names")

    required = ["element", "wave_A", "lower_state_eV", "loggf", "ew"]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"linemasks missing required columns: {missing}")

    has_discard = (discard_col in names)

    for lm in linemasks:
        if has_discard and _to_bool(lm[discard_col]):
            continue

        elem = str(lm["element"]).strip()

        # keep only Fe lines for atmospheric parameter solving
        if fe_only:
            if elem not in ("Fe 1", "Fe 2", "Fe I", "Fe II"):
                continue
            species = 26.0 if elem in ("Fe 1", "Fe I") else 26.1
        else:
            # For non-Fe-only usage, you must provide/compute species properly.
            # Here we keep it strict to avoid silent wrong species.
            raise ValueError("fe_only=False is not supported in this helper (species would be ambiguous).")

        waveA = pd.to_numeric(lm["wave_A"], errors="coerce")
        ep    = pd.to_numeric(lm["lower_state_eV"], errors="coerce")
        gf    = pd.to_numeric(lm["loggf"], errors="coerce")
        ew    = pd.to_numeric(lm["ew"], errors="coerce")  # iSpec ew is typically in mÅ

        if not (np.isfinite(waveA) and np.isfinite(ep) and np.isfinite(gf) and np.isfinite(ew)):
            continue
        if ew <= 0:
            continue

        rows.append({
            "wavelength": float(waveA),
            "species": float(species),
            "ep": float(ep),
            "gf": float(gf),
            "ew": float(ew),
        })

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        raise RuntimeError("No valid Fe lines found in linemasks for q2 export.")

    # ---------- 2) group duplicates (same species + close wavelength + same ep/gf) ----------
    df["wave_key"] = (df["wavelength"] / wave_tol).round().astype(int)
    key_cols = ["species", "wave_key", "ep", "gf"]

    g = df.groupby(key_cols, as_index=False).agg(
        wavelength=("wavelength", "median"),
        ew=("ew", "median"),
    )

    # ---------- 3) make wide lines table ----------
    lines_wide = g[["wavelength", "species", "ep", "gf"]].copy()
    lines_wide[str(star_id)] = g["ew"].astype(float)
    lines_wide = lines_wide.sort_values(by=["species", "wavelength"], kind="stable")

    lines_csv = os.path.join(out_dir, "lines_q2.csv")
    lines_wide.to_csv(lines_csv, index=False)

    # ---------- 4) stars.csv ----------
    stars_df = pd.DataFrame([{
        "id": str(star_id),
        "teff": float(teff),
        "logg": float(logg),
        "feh":  float(feh),
        "vt":   float(vt),
    }])
    stars_csv = os.path.join(out_dir, "stars_q2.csv")
    stars_df.to_csv(stars_csv, index=False)

    print(f"[export_q2_inputs_single_wide] lines: {len(lines_wide)} -> {lines_csv}")
    print(f"[export_q2_inputs_single_wide] stars: {len(stars_df)} -> {stars_csv}")
    return stars_csv, lines_csv

def _pick_col(df, candidates):
    """Return the first available column from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def save_solar_afe_reference_from_q2_solution(
    solution_csv,
    solar_id,
    instrument,
    resolution,
    out_json,
    extra_meta=None,
):
    """Extract the final solar solution from q2 and save it as JSON. In absolute single-star mode, q2 commonly stores A(Fe) in its feh column."""
    sol = pd.read_csv(solution_csv)

    # Accept alternative identifier columns.
    id_col = _pick_col(sol, ["id", "star", "name", "Star", "Name"])
    if id_col is None:
        raise ValueError(f"Cannot find star id column in {solution_csv}")

    sub = sol.loc[sol[id_col].astype(str) == str(solar_id)]
    if sub.empty:
        raise ValueError(f"Cannot find solar_id={solar_id} in {solution_csv}")

    # Use the last iteration when multiple rows exist for a star.
    row = sub.iloc[-1]

    teff_col = _pick_col(sol, ["teff", "Teff"])
    logg_col = _pick_col(sol, ["logg", "Logg"])
    vt_col   = _pick_col(sol, ["vt", "vmic", "Vt", "Vmic"])
    feh_col  = _pick_col(sol, ["feh", "[Fe/H]", "FeH", "AFe", "afe", "A(Fe)"])

    if feh_col is None:
        raise ValueError("Cannot find feh/AFe column in q2 solution.csv")

    rec = {
        "solar_id": str(solar_id),
        "instrument": str(instrument),
        "resolution": int(resolution),
        # Save the measured solar A(Fe), not a differential metallicity.
        "AFe_sun": float(row[feh_col]),
        "teff": float(row[teff_col]) if teff_col else np.nan,
        "logg": float(row[logg_col]) if logg_col else np.nan,
        "vt":   float(row[vt_col])   if vt_col else np.nan,
    }

    if extra_meta:
        rec["meta"] = dict(extra_meta)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    print(f"[save_solar_afe_reference] Saved -> {out_json}")
    return rec


def load_solar_afe_reference(out_json):
    """Read JSON written by save_solar_afe_reference_from_q2_solution."""
    with open(out_json, "r", encoding="utf-8") as f:
        rec = json.load(f)
    return rec


def afe_to_feh(AFe_star, AFe_sun):
    """Convert absolute A(Fe) to [Fe/H] by subtracting the solar A(Fe)."""
    return float(AFe_star) - float(AFe_sun)


# log10(N_Fe/N_H)+12 at solar metallicity on the adopted abundance scale.
SOLAR_AFE_AT_FEH_ZERO = {
    "Grevesse1998": 7.50,
    "Grevesse2007": 7.45,
    "Asplund2009": 7.50,
}


def q2_single_star_feh_column_to_mh(feh_column, afe_sun_reference):
    """Convert q2's absolute single-star feh (A(Fe)) to [Fe/H] for compatibility with the iSpec/MOOG model metallicity."""
    return afe_to_feh(feh_column, afe_sun_reference)


def load_solar_afe_reference(in_json, instrument=None, resolution=None):
    """Read Sun_AFe_reference.json and optionally check instrument/resolution consistency. Return a dictionary containing at least AFe_sun."""
    with open(in_json, "r", encoding="utf-8") as f:
        rec = json.load(f)

    if instrument is not None and "instrument" in rec:
        if str(rec["instrument"]) != str(instrument):
            raise ValueError(f"Solar ref instrument mismatch: {rec['instrument']} vs {instrument}")

    if resolution is not None and "resolution" in rec:
        if int(rec["resolution"]) != int(resolution):
            raise ValueError(f"Solar ref resolution mismatch: {rec['resolution']} vs {resolution}")

    if "AFe_sun" not in rec:
        raise KeyError(f"'AFe_sun' not found in {in_json}")

    return rec


def read_feh_from_abundance_with_gaia(
    star: str,
    csv_path: str,
    source_hint: str = "Bedell_2018",
):
    """Read stellar [Fe/H] and uncertainty from a long abundance table. Expected columns: star_name, element_code, abundance, uncertainty, source, element_str. Return (feh, e_feh)."""
    df = pd.read_csv(csv_path)

    # 1. Select the star.
    star_key = str(star).strip()
    df = df[df["star_name"].astype(str).str.strip() == star_key]
    if df.empty:
        raise RuntimeError(f"[read_feh] In {csv_path}: star not found: '{star_key}'")

    # 2. Prefer the requested reference source when multiple sources exist.
    if source_hint and "source" in df.columns:
        df_src = df[df["source"].astype(str).str.contains(source_hint, na=False)]
        if not df_src.empty:
            df = df_src

    # 3. Select Fe using element_str or atomic number 26.
    fe = df[
        (df["element_str"].astype(str).str.strip().str.upper() == "FE")
        | (pd.to_numeric(df["element_code"], errors="coerce") == 26.0)
    ]
    if fe.empty:
        raise RuntimeError(f"[read_feh] star='{star_key}' in {csv_path} has no Fe row (source_hint='{source_hint}'）")

    # 4. Use the first matching row.
    r = fe.iloc[0]
    feh = float(pd.to_numeric(r["abundance"], errors="coerce"))
    e_feh = float(pd.to_numeric(r["uncertainty"], errors="coerce")) if "uncertainty" in fe.columns else np.nan
    return feh, e_feh
