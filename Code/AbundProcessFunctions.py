import os
import sys
import numpy as np
import pandas as pd
import logging
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import norm
from userlist import *
import re
import json

# ========== For 5_ele_abundance_plot =========
def parse_element_token(s: str):
    """Parse element symbol and ionization stage, return (base_element, ionization_stage)."""
    parts = str(s).split()
    base = parts[0]
    try:
        ion = int(parts[1]) if len(parts) > 1 else 1
    except ValueError:
        ion = 1
    return base, ion

# ---------- Combine ionization stages: use Standard Error of the Mean (SEM) as the uncertainty ----------
def combine_ions_sem(g):
    """
    g: Rows corresponding to different ionization stages of the same base element.
    Returns:
      mean_xh : Mean [X/H] averaged over ionization stages
      sem_xh  : Standard error of [X/H] across ionization stages (std/sqrt(N)),
               where N is the number of ionization stages used.
               If N==1, fall back to the existing std_[X/H] of that stage
               (alternatively, you could change this to np.nan).
      n_lines : Sum of n_lines over ionization stages
      n_ions  : Number of ionization stages included in the merge
    """
    vals = g["[X/H]"].astype(float).to_numpy()
    k = np.isfinite(vals).sum()
    mean_xh = np.nanmean(vals)

    if k > 1:
        sem_xh = np.nanstd(vals, ddof=1) / np.sqrt(k)
    else:
        # Fallback for a single ionization stage (keep that stage's uncertainty; could also be np.nan)
        sem_xh = float(g["std_[X/H]"].iloc[0]) if len(g) > 0 else np.nan

    n_lines = g["n_lines"].sum()
    return pd.Series({"[X/H]": mean_xh, "std_[X/H]": sem_xh, "n_lines": n_lines, "n_ions": k})

def mkdir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def common_elements_on_Tc(td, rd):
    """Keep only elements that have Tc in both target/ref to avoid misalignment."""
    common = sorted(set(td["Element"]) & set(rd["Element"]))
    if len(common) > 0:
        td = td[td["Element"].isin(common)]
        rd = rd[rd["Element"].isin(common)]
    return td, rd

# ========== Utility functions ==========
def load_tc(path=TC_PATH):
    """Read/clean the Tc table; keep only Element and Tc (prefer 50%Tc_K; fallback to Tc_K). Return tc_df: Element, Tc_K."""
    tc_raw = pd.read_csv(path)
    # Identify Tc column
    tc_col = None
    for c in ["50%Tc_K", "Tc50_K", "Tc_K"]:
        if c in tc_raw.columns:
            tc_col = c
            break
    if tc_col is None:
        raise RuntimeError("Tc column not found in the Tc table (expected 50%Tc_K / Tc50_K / Tc_K)")

    tc_df = tc_raw[["Element", tc_col]].rename(columns={tc_col: "Tc_K"}).copy()
    # Clean: drop non-numeric/missing, strip spaces
    tc_df["Element"] = tc_df["Element"].astype(str).str.strip()
    tc_df["Tc_K"] = pd.to_numeric(tc_df["Tc_K"], errors="coerce")
    tc_df = tc_df.dropna(subset=["Tc_K"])
    return tc_df

def build_ref_for_star(star, a1_path, a2_path):
    """
    Build reference abundances for a given star from Meléndez2025 A46 tables (two outputs):
      - ref_xh : Element, [X/H], std_[X/H]
      - ref_xfe: Element, [X/Fe], e_[X/Fe]
    Uncertainty propagation:
      - [X/H] = [Fe/H] + [X/Fe],  sigma^2 = sigma_FeH^2 + sigma_XFe^2 (assuming independence)
    """
    a1 = pd.read_csv(a1_path)
    a2 = pd.read_csv(a2_path)

    # Fallback selector for column names
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
        raise RuntimeError("A46 table is missing Star or [Fe/H] columns")

    # Retrieve Fe/H for this star
    row_a1 = a1.loc[a1[col_star_a1] == star]
    if row_a1.empty:
        raise RuntimeError(f"Cannot find {star} in A46_tablea1")
    feh  = float(row_a1.iloc[0][col_feh])
    efeh = float(row_a1.iloc[0][col_efeh]) if (col_efeh and pd.notna(row_a1.iloc[0][col_efeh])) else np.nan

    # Extract all [X/Fe] values and uncertainties for this star
    row_a2 = a2.loc[a2[col_star_a2] == star]
    if row_a2.empty:
        raise RuntimeError(f"Cannot find {star} in A46_table2")

    # Convert to long format: Element, [X/Fe], e_[X/Fe]
    ratio_cols, err_map = [], {}
    for c in row_a2.columns:
        if isinstance(c, str) and re.fullmatch(r"\[[A-Z][a-z]?/Fe\]", c):
            ratio_cols.append(c)
            # Common uncertainty column naming conventions
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

    # Convert to [X/H] and propagate uncertainty
    ref_xh = ref_xfe.copy()
    ref_xh["[X/H]"]     = ref_xh["[X/Fe]"] + feh
    ref_xh["std_[X/H]"] = np.sqrt(
        np.where(np.isfinite(ref_xh["e_[X/Fe]"]), ref_xh["e_[X/Fe]"]**2, 0.0) +
        (efeh**2 if np.isfinite(efeh) else 0.0)
    )

    # Add Fe itself
    ref_xh = pd.concat(
        [ref_xh, pd.DataFrame([["Fe", feh, efeh, 1]], columns=["Element","[X/H]","std_[X/H]","_dummy"])],
        ignore_index=True
    )
    ref_xfe = pd.concat(
        [ref_xfe, pd.DataFrame([["Fe", 0.0, efeh, ]], columns=["Element","[X/Fe]","e_[X/Fe]"])],
        ignore_index=True
    )

    return ref_xh[["Element","[X/H]","std_[X/H]"]], ref_xfe[["Element","[X/Fe]","e_[X/Fe]"]]

def _parse_bedell2018_token(colname: str):
    """
    Input: column name like '[ScII/H]', '[CI/H]', '[CH/H]'
    Output: (base_element, tag), where tag is only used to preserve source labeling
            and does not affect the final base_element merging.
    """
    s = str(colname).strip()
    if not (s.startswith("[") and s.endswith("]") and "/H" in s):
        return None, None

    # Extract token inside brackets, before '/H': e.g., 'ScII', 'CI', 'CH'
    token = s[1:s.index("/H")]

    # Special case: CH -> map to C (Bedell often uses CH features for C)
    if token.upper() == "CH":
        return "C", "CH"

    # Parse element symbol (1-2 chars: first uppercase, optional second lowercase)
    base = token[0].upper()
    rest = token[1:]
    if len(rest) >= 1 and rest[0].islower():
        base += rest[0]
        rest = rest[1:]

    # rest may be I/II/III... or empty
    tag = rest if rest else "I"
    return base, tag


def build_ref_for_star_bedell2018_table2(
    star: str,
    table2_path: str,
    feh_override: float | None = None,
    e_feh_override: float | None = None,
    feh_table_path: str | None = None,          # <- new
    feh_source_hint: str = "Bedell_2018",       # <- new
):
    """
    Build reference abundances from Bedell2018_table2.csv.
    - ref_xh : Element, [X/H], std_[X/H]
    - ref_xfe: Element, [X/Fe], e_[X/Fe] (requires feh_override; otherwise all NaN)

    Typical table2 columns:
      Star, [CI/H], e_[CI/H], [CH/H], e_[CH/H], ... (usually does not include Fe)
    """
    t2 = pd.read_csv(table2_path)

    star_col = "Star" if "Star" in t2.columns else None
    if star_col is None:
        raise RuntimeError(f"[Bedell2018 ref] Cannot find 'Star' column in {table2_path}")

    # Locate the row for this star
    star_key = str(star).strip()
    sel = t2[star_col].astype(str).str.strip() == star_key
    if not sel.any():
        raise RuntimeError(f"[Bedell2018 ref] Cannot find star='{star_key}' in table2")

    row = t2.loc[sel].iloc[0]

    # Build a long-form table to reuse combine_ions_for_one_target()
    rows = []
    for c in t2.columns:
        if not (isinstance(c, str) and c.startswith("[") and c.endswith("]") and "/H" in c):
            continue

        base, tag = _parse_bedell2018_token(c)
        if base is None:
            continue

        xh  = pd.to_numeric(row.get(c), errors="coerce")
        err = pd.to_numeric(row.get("e_" + c), errors="coerce")  # Bedell uncertainty columns: 'e_[X/H]'

        if not np.isfinite(xh):
            continue

        rows.append({
            "element": f"{base} {tag}",     # only to enable base_element merging; tag does not matter
            "[X/H]": float(xh),
            "std_[X/H]": float(err) if np.isfinite(err) else np.nan,
            "n_lines": 1,                   # Bedell table has no per-line counts; set to 1 as a placeholder
        })

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise RuntimeError(f"[Bedell2018 ref] star='{star_key}' produced no parsed [X/H] columns")

    g = combine_ions_for_one_target(df_long)  # -> base_element, [X/H], std_[X/H], n_lines, n_ions

    # If you want ref_xfe to be valid: you must add an Fe row
    # ----------------- Auto-fetch Fe/H if not manually overridden -----------------
    if feh_override is None and feh_table_path is not None:
        feh_override, e_feh_override = read_feh_from_abundance_with_gaia(
            star=star,
            csv_path=feh_table_path,
            source_hint=feh_source_hint,
        )

    # ----------------- Append Fe row, then compute [X/Fe] -----------------
    if feh_override is not None:
        g = pd.concat([g, pd.DataFrame([{
            "base_element": "Fe",
            "[X/H]": float(feh_override),
            "std_[X/H]": float(e_feh_override) if e_feh_override is not None else np.nan,
            "n_lines": 0,
            "n_ions": 1
        }])], ignore_index=True)

    g2 = add_xfe_and_err(g)

    ref_xh  = g2.rename(columns={"base_element": "Element"})[["Element", "[X/H]", "std_[X/H]"]]
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
):
    """
    Unified entry point: choose reference builder by ref_name.
    - ref_name='A46' (or 'Melendez2025'): uses build_ref_for_star (requires a1/a2 paths)
    - ref_name='Bedell2018': uses Bedell table2 (requires bedell_table2_path; X/Fe also needs Fe/H)
    """
    ref_name = str(ref_name).strip()

    if ref_name.lower() in ["a46", "melendez2025"]:
        # Reuse existing function: build_ref_for_star(star, a1_path, a2_path)
        if a46_a1_path is None or a46_a2_path is None:
            raise RuntimeError("ref_name='A46' requires both a46_a1_path and a46_a2_path")
        return build_ref_for_star(star, a1_path=a46_a1_path, a2_path=a46_a2_path)

    if ref_name.lower() in ["bedell2018", "bedell"]:
        if bedell_table2_path is None:
            raise RuntimeError("ref_name='Bedell2018' requires bedell_table2_path (i.e., Bedell2018_table2.csv)")
        return build_ref_for_star_bedell2018_table2(
            star=star,
            table2_path=bedell_table2_path,
            feh_override=feh_override,
            e_feh_override=e_feh_override,
            feh_table_path=feh_table_path,     # <- pass-through
            # feh_source_hint=... (optional; defaults to Bedell_2018)
        )

    raise RuntimeError(f"Unknown ref_name: {ref_name}")

def add_tc_to(df_like, tc_df):
    """Merge Tc into the table; df_like must contain 'Element' or 'base_element'."""
    d = df_like.copy()
    if "Element" not in d.columns and "base_element" in d.columns:
        d = d.rename(columns={"base_element":"Element"})
    d = d.merge(tc_df, on="Element", how="left")
    return d

def ensure_element_col(d):
    """Standardize to an 'Element' column from either Element/base_element/element naming."""
    out = d.copy()
    if "Element" in out.columns:
        return out
    if "base_element" in out.columns:
        out = out.rename(columns={"base_element":"Element"})
        return out
    if "element" in out.columns:
        out["Element"] = out["element"].str.split().str[0]
        return out
    raise RuntimeError("DataFrame has none of: Element / base_element / element columns")

######## ===== For 6_combine_abundances ===== ########
def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def combine_ions_for_one_target(df):
    """
    Input: abundances_<target>.csv for a given target (long format, includes ionization stages)
    Output: merged DataFrame with columns:
      base_element, [X/H], std_[X/H], n_lines
    """
    d = df.copy()
    # Base element name & ionization stage
    toks = d["element"].astype(str).str.strip()
    d["base_element"] = toks.str.split().str[0]
    # Validate numeric column names
    if "[X/H]" not in d.columns:
        raise RuntimeError("input csv is missing the [X/H] column")
    # Be tolerant: convert std and n_lines to numeric
    d["std_[X/H]"] = pd.to_numeric(d.get("std_[X/H]"), errors="coerce")
    d["n_lines"]    = pd.to_numeric(d.get("n_lines"), errors="coerce").fillna(0).astype(int)

    # Merge ionization stages by base element (using your SEM rule)
    g = d.groupby("base_element", as_index=False).apply(combine_ions_sem)
    g = g.reset_index().rename(columns={"level_0":"base_element"})
    # combine_ions_sem returns columns: "[X/H]","std_[X/H]","n_lines","n_ions"
    g = g[["base_element","[X/H]","std_[X/H]","n_lines","n_ions"]]
    return g

def add_xfe_and_err(g_combined):
    """
    Given the merged table, derive [X/Fe] and e_[X/Fe] from Fe's [X/H] and std_[X/H].
    """
    out = g_combined.copy()
    # Find Fe row
    if not (out["base_element"] == "Fe").any():
        out["[X/Fe]"] = np.nan
        out["e_[X/Fe]"] = np.nan
        return out

    fe_row = out.loc[out["base_element"] == "Fe"].iloc[0]
    feh = safe_float(fe_row["[X/H]"])
    e_feh = safe_float(fe_row["std_[X/H]"])
    out["[X/Fe]"]  = out["[X/H]"] - feh
    # Error propagation (if a component is missing, use whatever is available)
    out["e_[X/Fe]"] = np.sqrt(
        np.where(np.isfinite(out["std_[X/H]"]), out["std_[X/H]"]**2, 0.0) +
        (e_feh**2 if np.isfinite(e_feh) else 0.0)
    )
    return out

def build_master_abundance_table(ispec_dir, targets=None, out_path=None, tc_path=TC_PATH):
    """
    Build a master table:
      rows = elements; columns = per-target [X/H], [X/H]_err, [X/Fe], [X/Fe]_err.
    If tc_path is provided, merge Tc_K and place it as the 2nd column.
    """
    root = os.path.join(ispec_dir, "output")
    if targets is None:
        candidates = []
        for sub in os.listdir(root):
            subdir = os.path.join(root, sub)
            if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, f"abundances_{sub}.csv")):
                candidates.append(sub)
        targets = sorted(candidates)

    master = pd.DataFrame({"Element": []})

    for t in targets:
        csv_path = os.path.join(root, t, f"abundances_{t}.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for c in ["element", "[X/H]"]:
            if c not in df.columns:
                raise RuntimeError(f"{csv_path} is missing column: {c}")

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

    # (Optional) sort by atomic number
    try:
        z_map = {k:int(v) for k,v in ATOMIC_Z.items()}
        master["_Z"] = master["Element"].map(z_map)
        master = master.sort_values(by=["_Z","Element"], kind="stable").drop(columns="_Z")
    except Exception:
        pass

    # ==== New: merge Tc and place it as the 2nd column ====
    if tc_path:
        tc_df = load_tc(tc_path)
        master = master.merge(tc_df, on="Element", how="left")
        # Move Tc_K to the second column
        cols = list(master.columns)
        cols.remove("Tc_K")
        cols = [cols[0], "Tc_K"] + cols[1:]
        master = master[cols]

    if out_path is None:
        out_path = os.path.join(root, "all_targets_abundances.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    master.to_csv(out_path, index=False)
    print(f"Master table written: {out_path}")
    return master

# ========== For 3_atmos_params_iteration =========
# using q2
# Convert iSpec linemasks to the q2-required lines.csv format
# Key point: q2 lines.csv requires: id, wave, species, EP, loggf, EW

def load_ispec_params_from_dump(ispec, dump_file):
    """
    Read iterated atmospheric parameters from iSpec atmos_params_*.dump
    (the file saved via ispec.save_results).
    Returns: teff, logg, feh(MH), vt(vmic)
    """
    restored = ispec.restore_results(dump_file)
    params = restored[0]
    return {
        "teff": float(params["teff"]),
        "logg": float(params["logg"]),
        "feh":  float(params["MH"]),
        "vt":   float(params["vmic"]),
    }

def _lm_to_q2_line_df(linemasks, star_id):
    """
    Convert iSpec linemasks (with ew/wave_A/EP/loggf, etc.) into a long-form table:
    wavelength,species,ep,gf,ew
    Note: iSpec element strings are typically 'Fe 1'/'Fe 2' (not 'Fe I'/'Fe II').
    """
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

def export_q2_inputs_wide(
    ispec,
    target_id,
    used_linemasks_fe,              # Final Fe linemasks used in step 3_ (contains EW)
    solar_lines_csv,                # Master solar EW table (instrument/resolution/element/wave_A/EP/loggf/ew_mA)
    instrument, resolution,
    out_dir,
    sun_dump_file,                  # ★NEW: instrument-specific Sun iSpec dump (used to read Teff/logg/MH/vmic)
    sun_id="Sun",
    wave_tol=0.001,                 # Å: used to align the same spectral line
):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) Target Fe lines (long table) ----------
    df_t = _lm_to_q2_line_df(used_linemasks_fe, target_id)
    print(f"[export_q2_inputs] target Fe lines: {len(df_t)}")

    # ---------- 2) Sun Fe lines (from your master solar EW table) ----------
    sun_master = pd.read_csv(solar_lines_csv)

    # ---------- NEW: filter by instrument first ----------
    sun_inst = sun_master[sun_master["instrument"].astype(str).str.strip() == str(instrument).strip()].copy()
    if len(sun_inst) == 0:
        raise RuntimeError(f"[export_q2_inputs] solar master has no rows for instrument={instrument}")

    # ---------- NEW: automatically select the closest available resolution ----------
    # Note: cast to float to tolerate int/float/string representations
    avail = pd.to_numeric(sun_inst["resolution"], errors="coerce").values
    if not np.isfinite(avail).any():
        raise RuntimeError(f"[export_q2_inputs] solar master resolution column cannot be parsed for instrument={instrument}")

    req = float(resolution)
    res_use = avail[np.nanargmin(np.abs(avail - req))]

    # ---------- NEW: select Sun Fe lines at res_use ----------
    sun_sel = sun_inst[
        (pd.to_numeric(sun_inst["resolution"], errors="coerce") == float(res_use)) &
        (sun_inst["element"].isin(["Fe I", "Fe II", "Fe 1", "Fe 2"]))
    ].copy()

    print(f"[export_q2_inputs] solar resolution requested={resolution}, used={res_use}")

    # Support both 'Fe I' and 'Fe 1' notations
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

    # ---------- 3) Keep only common lines (align by species + wavelength) ----------
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

    # ---------- 4) Build q2 wide-format lines table ----------
    base = common.copy()
    # Use Sun wavelength as the output wavelength (often more stable)
    wave_map = df_s2.groupby(key_cols)["wavelength"].median().reset_index()
    base = base.merge(wave_map, on=key_cols, how="left")

    # EW values for Sun/target
    sun_ew = df_s2.groupby(key_cols)["ew"].median().reset_index().rename(columns={"ew": sun_id})
    tar_ew = df_t2.groupby(key_cols)["ew"].median().reset_index().rename(columns={"ew": str(target_id)})

    lines_wide = base.merge(sun_ew, on=key_cols, how="inner").merge(tar_ew, on=key_cols, how="inner")

    # Column order required by q2: wavelength,species,ep,gf, then EW columns for each star
    lines_wide = lines_wide[["wavelength","species","ep","gf",sun_id,str(target_id)]].sort_values(
        by=["species","wavelength"]
    )

    lines_csv = os.path.join(out_dir, "lines_q2.csv")
    lines_wide.to_csv(lines_csv, index=False)
    print(f"[export_q2_inputs] lines.csv rows : {len(lines_wide)} -> {lines_csv}")

    # ---------- 5) stars.csv (Sun parameters read from dump; target uses initial values) ----------
    sun_par = load_ispec_params_from_dump(ispec, sun_dump_file)

    # For the target initial values: keep using your current initial_teff/logg/MH/vmic
    # For flexibility, it is recommended to build stars_df in your 3_ script and save it there.
    # (So here we only return the path placeholder and Sun params.)

    stars_csv = os.path.join(out_dir, "stars_q2.csv")

    return lines_csv, stars_csv, sun_par

# ========= For 0_Sun_abundance ==========
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
    wave_tol=0.001,          # Å; used to group the same line
    discard_col="discarded", # if present in linemasks
):
    """
    Export q2 inputs for a SINGLE star (Sun-only use case).
    Writes:
      - stars_q2.csv : columns [id, teff, logg, feh, vt]
      - lines_q2.csv : wide format: [wavelength, species, ep, gf, <star_id>]
                       where the <star_id> column stores EW (mÅ).

    Parameters
    ----------
    linemasks : numpy structured array / table
        iSpec linemasks that contain at least:
        element, wave_A, lower_state_eV, loggf, ew
    star_id : str
        The star label used as the column name in lines_q2.csv
    teff, logg, feh, vt : float
        Initial parameters for q2
    out_dir : str
        Directory to write q2 inputs
    fe_only : bool
        If True, keep only Fe I/Fe II lines (recommended for specpars)
    wave_tol : float
        Wavelength tolerance (Å) for grouping duplicates
    discard_col : str
        Column name for the discarded-flag, if present

    Returns
    -------
    (stars_csv_path, lines_csv_path)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) build a long table from linemasks ----------
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

        # Keep only Fe lines for atmospheric-parameter solving
        if fe_only:
            if elem not in ("Fe 1", "Fe 2", "Fe I", "Fe II"):
                continue
            species = 26.0 if elem in ("Fe 1", "Fe I") else 26.1
        else:
            # For non-Fe-only usage, you must provide/compute 'species' properly.
            # Here we keep it strict to avoid silently using incorrect species.
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

    # ---------- 3) make a wide-format lines table ----------
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
    """Pick the first existing column name in df from a list of candidates."""
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
    """
    Extract the Sun's final solution from q2 solution.csv and save as JSON.
    Note: In single-star mode (no reference), q2 often writes A(Fe) into the 'feh' column.
    """
    sol = pd.read_csv(solution_csv)

    # Flexible handling of the id column name
    id_col = _pick_col(sol, ["id", "star", "name", "Star", "Name"])
    if id_col is None:
        raise ValueError(f"Cannot find star id column in {solution_csv}")

    sub = sol.loc[sol[id_col].astype(str) == str(solar_id)]
    if sub.empty:
        raise ValueError(f"Cannot find solar_id={solar_id} in {solution_csv}")

    # If multiple rows exist (e.g., multiple outputs), take the last one
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
        # Note: this stores the Sun-derived A(Fe) (in single-star mode it typically lands in the 'feh' column)
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
    """Load the JSON written by save_solar_afe_reference_from_q2_solution."""
    with open(out_json, "r", encoding="utf-8") as f:
        rec = json.load(f)
    return rec


def afe_to_feh(AFe_star, AFe_sun):
    """Convert absolute abundance A(Fe) to [Fe/H]."""
    return float(AFe_star) - float(AFe_sun)


def load_solar_afe_reference(in_json, instrument=None, resolution=None):
    """
    Load Sun_AFe_reference.json saved by 0_Sun and optionally validate instrument/resolution.
    Returns a dict that includes at least AFe_sun.
    """
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
    """
    Read [Fe/H] and its uncertainty for a star from a long-format table abundance_with_gaia.csv.
    Expected columns: star_name, element_code, abundance, uncertainty, source, element_str

    Returns:
        feh (float), e_feh (float)
    """
    df = pd.read_csv(csv_path)

    # 1) select star
    star_key = str(star).strip()
    df = df[df["star_name"].astype(str).str.strip() == star_key]
    if df.empty:
        raise RuntimeError(f"[read_feh] Cannot find star='{star_key}' in {csv_path}")

    # 2) prefer the specified source (if multiple sources exist for the same star)
    if source_hint and "source" in df.columns:
        df_src = df[df["source"].astype(str).str.contains(source_hint, na=False)]
        if not df_src.empty:
            df = df_src

    # 3) find Fe: element_str == 'Fe' OR element_code == 26
    fe = df[
        (df["element_str"].astype(str).str.strip().str.upper() == "FE")
        | (pd.to_numeric(df["element_code"], errors="coerce") == 26.0)
    ]
    if fe.empty:
        raise RuntimeError(f"[read_feh] star='{star_key}' has no Fe row in {csv_path} (source_hint='{source_hint}')")

    # 4) if multiple rows exist: take the first; you can change this to an average if desired
    r = fe.iloc[0]
    feh = float(pd.to_numeric(r["abundance"], errors="coerce"))
    e_feh = float(pd.to_numeric(r["uncertainty"], errors="coerce")) if "uncertainty" in fe.columns else np.nan
    return feh, e_feh