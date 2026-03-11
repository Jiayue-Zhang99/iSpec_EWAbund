import os
import re
import logging
import numpy as np
import pandas as pd
from userlist import *

# =============================================================================
# Generic linelist utilities (iSpec-friendly)
#
# Goal:
#   1) Read ANY reference linelist (TSV/CSV/space-separated) with minimal assumptions.
#   2) Convert it into an iSpec-style atomic linelist (TSV).
#   3) Optionally: "project" (update) the values (e.g., loggf, EP) onto a BASE linelist
#      (e.g., VALD/GES atomic_lines.tsv) by matching (element, ion, wavelength, EP) with tolerances.
#
# This is a generalized version (NOT tied to the Meléndez 2014 linelist).
# =============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# 0) Helpers: robust parsing and I/O
# =============================================================================

def detect_separator(path: str) -> str:
    """
    Detect delimiter from the first non-empty, non-comment line.
    Supports: tab, comma, semicolon, whitespace.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "\t" in s:
                return "\t"
            if "," in s:
                return ","
            if ";" in s:
                return ";"
            # fallback: whitespace
            return r"\s+"
    return r"\s+"


def read_table_any(path: str, dtype=str) -> pd.DataFrame:
    """
    Read a table with auto-detected separator. Keeps empty strings (no default NA).
    """
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, dtype=dtype, keep_default_na=False, comment="#", engine="python")
    return df


def to_float(x):
    """Safe float conversion."""
    try:
        v = float(x)
        return v
    except Exception:
        return np.nan


def fmt_float(x, nd=6) -> str:
    """Format float to fixed decimals; return empty string if NaN."""
    if not np.isfinite(x):
        return ""
    return f"{float(x):.{nd}f}"


# =============================================================================
# 1) Element / ion parsing
# =============================================================================

_ROMAN_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}

def roman_to_int(s: str):
    s = str(s).strip().upper()
    return _ROMAN_MAP.get(s, None)

def normalize_element_ion_token(token: str):
    """
    Normalize element+ion token into (symbol, ion_int).
    Supports examples:
      "Fe 1", "Fe 2"
      "Fe I", "Fe II"
      "FeI", "FeII"
      "Ti2", "TiII"
      "C" (ion defaults to 1)
    """
    s = str(token).strip()
    if not s:
        return ("", np.nan)

    # Case 1: "Fe 1" / "Fe I" style
    parts = s.split()
    if len(parts) >= 2:
        sym = parts[0].strip()
        ion_raw = parts[1].strip()
        # integer ion?
        if ion_raw.isdigit():
            return (sym, int(ion_raw))
        # roman ion?
        ri = roman_to_int(ion_raw)
        if ri is not None:
            return (sym, ri)
        return (sym, np.nan)

    # Case 2: compact "FeII" / "FeI" / "Ti2"
    m = re.match(r"^([A-Z][a-z]?)([0-9]+|I|II|III|IV|V|VI)?$", s)
    if m:
        sym = m.group(1)
        tail = m.group(2)
        if tail is None or tail == "":
            return (sym, 1)
        if tail.isdigit():
            return (sym, int(tail))
        ri = roman_to_int(tail)
        if ri is not None:
            return (sym, ri)
        return (sym, np.nan)

    # Fallback: treat the first 1–2 letters as element symbol
    m2 = re.match(r"^([A-Z][a-z]?).*$", s)
    if m2:
        return (m2.group(1), 1)

    return ("", np.nan)


def build_ispec_element_str(sym: str, ion: int) -> str:
    """
    iSpec commonly uses 'Fe 1' / 'Fe 2' style in 'element' column.
    """
    sym = str(sym).strip()
    if not sym:
        return ""
    if ion is None or not np.isfinite(float(ion)):
        return sym
    return f"{sym} {int(ion)}"


# =============================================================================
# 2) Column mapping: accept many linelist formats
# =============================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column names into a canonical set if possible.

    Canonical targets we try to produce:
      element_token    (e.g. 'Fe 1', 'FeI', 'Fe II', 'Fe')
      ion              (optional; can be derived from element_token)
      wave_A           (wavelength in Angstrom)
      wave_nm          (wavelength in nm) (optional; can be derived)
      lower_state_eV   (excitation potential / EP in eV)
      loggf            (log(gf))
      ew               (equivalent width in mA) (optional)
    """
    d = df.copy()
    cols = {c: c.strip() for c in d.columns}
    d.rename(columns=cols, inplace=True)

    # Lowercase helper map
    low = {c.lower(): c for c in d.columns}

    def pick(*names):
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    # element token candidates
    c_elem = pick("element", "species", "elem", "Element", "Species")
    c_ion  = pick("ion", "Ion", "ionization", "stage", "charge")

    # wavelength candidates
    c_waveA = pick("wave_A", "wavelength_A", "wavelength", "lambda_A", "lambda", "wl_A", "lam_A")
    c_wavenm = pick("wave_nm", "wavelength_nm", "wl_nm", "lambda_nm")

    # EP candidates
    c_ep = pick("lower_state_eV", "EP", "ep", "excitation_potential", "chi", "elow", "e_low")

    # loggf candidates
    c_loggf = pick("loggf", "log_gf", "log(gf)", "gf", "loggf_value")

    # EW candidates (optional)
    c_ew = pick("ew", "EW", "ew_mA", "EW_mA", "ew_ma")

    # Create canonical columns (keep originals too)
    if c_elem:
        d["element_token"] = d[c_elem]
    else:
        d["element_token"] = ""

    if c_ion:
        d["ion"] = d[c_ion]
    elif "ion" not in d.columns:
        d["ion"] = ""

    if c_waveA:
        d["wave_A"] = d[c_waveA]
    elif "wave_A" not in d.columns:
        d["wave_A"] = ""

    if c_wavenm:
        d["wave_nm"] = d[c_wavenm]
    elif "wave_nm" not in d.columns:
        d["wave_nm"] = ""

    if c_ep:
        d["lower_state_eV"] = d[c_ep]
    elif "lower_state_eV" not in d.columns:
        d["lower_state_eV"] = ""

    if c_loggf:
        d["loggf"] = d[c_loggf]
    elif "loggf" not in d.columns:
        d["loggf"] = ""

    if c_ew:
        d["ew"] = d[c_ew]
    elif "ew" not in d.columns:
        d["ew"] = ""

    return d


def build_canonical_ispec_linelist(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Build an iSpec-friendly atomic linelist with key columns:
      element, ion, wave_A, wave_nm, lower_state_eV, loggf
    plus any extra columns that already exist (kept as-is).

    Notes:
      - If ion is missing, we parse it from element_token.
      - If wave_nm is missing, we derive it from wave_A.
    """
    d = normalize_columns(df_in)

    # Parse element symbol and ion
    sym_ion = d["element_token"].map(normalize_element_ion_token)
    d["_sym"] = [a for a, _ in sym_ion]
    d["_ion_from_token"] = [b for _, b in sym_ion]

    # Parse ion column if available
    ion_num = d["ion"].map(lambda x: int(x) if str(x).strip().isdigit() else np.nan)
    d["_ion"] = pd.Series(ion_num).fillna(pd.Series(d["_ion_from_token"])).astype("Int64")

    # Wavelength and EP/loggf numeric
    d["_wave_A"] = d["wave_A"].map(to_float)
    d["_wave_nm"] = d["wave_nm"].map(to_float)
    d["_ep"] = d["lower_state_eV"].map(to_float)
    d["_loggf"] = d["loggf"].map(to_float)
    d["_ew"] = d["ew"].map(to_float)

    # Derive missing wave_nm from wave_A
    m_missing_nm = ~np.isfinite(d["_wave_nm"]) & np.isfinite(d["_wave_A"])
    d.loc[m_missing_nm, "_wave_nm"] = d.loc[m_missing_nm, "_wave_A"] * 0.1

    # Build iSpec 'element' string like "Fe 1"
    d["element"] = [build_ispec_element_str(s, i if pd.notna(i) else np.nan) for s, i in zip(d["_sym"], d["_ion"])]

    # Ensure required columns exist
    out = d.copy()
    out["ion"] = out["_ion"].astype("Int64")
    out["wave_A"] = out["_wave_A"]
    out["wave_nm"] = out["_wave_nm"]
    out["lower_state_eV"] = out["_ep"]
    out["loggf"] = out["_loggf"]
    out["ew"] = out["_ew"]

    # Drop obviously invalid rows
    out = out.dropna(subset=["wave_A", "lower_state_eV"])
    out = out[out["element"].astype(str).str.len() > 0]

    return out


# =============================================================================
# 3) Matching / updating a BASE linelist using a REFERENCE linelist
# =============================================================================

def match_and_update_base_linelist(
    ref_df: pd.DataFrame,
    base_df: pd.DataFrame,
    tol_wave_A: float = 0.005,
    tol_ep_eV: float = 0.02,
    update_cols=("loggf", "lower_state_eV"),
):
    """
    Match REF lines onto BASE lines within (element symbol, ion) buckets,
    then update selected columns in BASE (e.g., loggf and EP).

    Matching keys:
      - same (symbol, ion)
      - |wave_A_base - wave_A_ref| <= tol_wave_A
      - |EP_base - EP_ref| <= tol_ep_eV

    If multiple candidates exist, the nearest normalized distance is used.
    Each BASE row is updated at most once.

    Returns:
      updated_base_df, report_df
    """
    ref = build_canonical_ispec_linelist(ref_df).copy()
    base = build_canonical_ispec_linelist(base_df).copy()

    # Keep original base columns (for writing back later)
    base["_rowid"] = np.arange(len(base), dtype=int)

    # For bucket matching, use parsed sym/ion
    base_sym_ion = base["element"].map(normalize_element_ion_token)
    base["_sym"] = [a for a, _ in base_sym_ion]
    base["_ion"] = [b for _, b in base_sym_ion]
    base["_wave_A"] = pd.to_numeric(base["wave_A"], errors="coerce")
    base["_ep"] = pd.to_numeric(base["lower_state_eV"], errors="coerce")

    ref_sym_ion = ref["element"].map(normalize_element_ion_token)
    ref["_sym"] = [a for a, _ in ref_sym_ion]
    ref["_ion"] = [b for _, b in ref_sym_ion]
    ref["_wave_A"] = pd.to_numeric(ref["wave_A"], errors="coerce")
    ref["_ep"] = pd.to_numeric(ref["lower_state_eV"], errors="coerce")

    base = base.dropna(subset=["_sym", "_ion", "_wave_A", "_ep"]).copy()
    ref = ref.dropna(subset=["_sym", "_ion", "_wave_A", "_ep"]).copy()

    base_g = base.groupby(["_sym", "_ion"])
    ref_g = ref.groupby(["_sym", "_ion"])
    keys = sorted(set(base_g.groups.keys()) & set(ref_g.groups.keys()))

    matched_base_rowids = set()
    report = []

    for (sym, ion) in keys:
        bs = base_g.get_group((sym, ion))
        rs = ref_g.get_group((sym, ion))

        for _, r in rs.iterrows():
            rw, rep = float(r["_wave_A"]), float(r["_ep"])

            cand = bs[(np.abs(bs["_wave_A"] - rw) <= tol_wave_A) &
                      (np.abs(bs["_ep"] - rep) <= tol_ep_eV)]

            if cand.empty:
                continue

            # Choose nearest candidate in normalized distance
            dw = (cand["_wave_A"] - rw) / tol_wave_A
            de = (cand["_ep"] - rep) / tol_ep_eV
            score = np.sqrt(dw**2 + de**2)
            picked = cand.loc[score.idxmin()]
            rid = int(picked["_rowid"])

            # Each base row updated at most once
            if rid in matched_base_rowids:
                continue

            matched_base_rowids.add(rid)

            report.append({
                "element": f"{sym} {int(ion)}",
                "ref_wave_A": rw,
                "ref_EP_eV": rep,
                "ref_loggf": float(r.get("loggf", np.nan)),
                "base_rowid": rid,
                "base_wave_A": float(picked["_wave_A"]),
                "base_EP_eV": float(picked["_ep"]),
                "base_loggf_before": float(picked.get("loggf", np.nan)),
                "d_wave_A": float(picked["_wave_A"] - rw),
                "d_EP_eV": float(picked["_ep"] - rep),
            })

    report_df = pd.DataFrame(report)

    # Apply updates
    updated = base_df.copy()
    # We must locate rows in the ORIGINAL base_df; use the matched rowids from canonical base
    # To do that robustly, we rely on base["_rowid"] being aligned with the base_df after canonicalization.
    # If base_df has extra columns/rows removed during canonicalization, this alignment could break.
    # Therefore: we update on the canonical 'base' and return it as an updated base linelist.
    base_updated = base.copy()

    if not report_df.empty:
        # Build quick lookup (rowid -> ref values)
        by_rid = {int(r["base_rowid"]): r for _, r in report_df.iterrows()}

        for rid, rr in by_rid.items():
            if "loggf" in update_cols and "ref_loggf" in rr and np.isfinite(rr["ref_loggf"]):
                base_updated.loc[base_updated["_rowid"] == rid, "loggf"] = rr["ref_loggf"]
            if "lower_state_eV" in update_cols and np.isfinite(rr["ref_EP_eV"]):
                base_updated.loc[base_updated["_rowid"] == rid, "lower_state_eV"] = rr["ref_EP_eV"]

    return base_updated, report_df


# =============================================================================
# 4) Writing iSpec TSV
# =============================================================================

def write_ispec_atomic_linelist_tsv(df: pd.DataFrame, out_path: str):
    """
    Write an iSpec-readable atomic linelist TSV.
    Keeps all columns, but ensures key columns are formatted cleanly.
    """
    out = df.copy()

    # Ensure canonical numeric columns are properly formatted as strings (fixed decimals),
    # which makes the TSV stable across platforms (avoids scientific notation).
    for col in ["wave_A", "wave_nm", "lower_state_eV", "loggf"]:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy()
            out[col] = [fmt_float(v, nd=6) for v in vals]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Saved iSpec linelist TSV -> {out_path} (rows={len(out)})")


# =============================================================================
# 5) Example usage (edit paths only; code logic stays general)
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Path notes (EDIT THESE FOR YOUR MACHINE / PROJECT):
    #
    # ref_linelist_path:
    #   Your "reference" linelist whose loggf/EP you trust (any format: TSV/CSV/space).
    #
    # base_linelist_path:
    #   A "base" linelist with full iSpec columns (commonly VALD/GES atomic_lines.tsv).
    #   We match reference lines onto this base and update selected fields.
    #
    # out_linelist_path:
    #   The output TSV that you will feed into iSpec (e.g., ispec.read_atomic_linelist()).
    #
    # out_report_path:
    #   A matching report for auditing which lines were updated.
    # -------------------------------------------------------------------------

    # --- Use paths from userlist.py (recommended) ---
    # Reference linelist (your trusted list; in your current setup this is Melendez -> iSpec TSV)
    ref_linelist_path = output_linelist_path

    # Base linelist with full iSpec columns (VALD/GES atomic_lines.tsv style)
    base_linelist_path = vald_atomic_lines_path

    # Output updated linelist (put it under the same linelist folder for this pipeline)
    out_linelist_path = os.path.join(output_linelist_folder, "atomic_lines_updated.tsv")

    # Output match report (CSV) for auditing
    out_report_path = os.path.join(output_linelist_folder, "match_report.csv")

    # 1) Read any reference/base tables
    ref_df = read_table_any(ref_linelist_path, dtype=str)
    base_df = read_table_any(base_linelist_path, dtype=str)

    # 2) Update base with reference values (tolerances can be tuned)
    updated_base, report_df = match_and_update_base_linelist(
        ref_df,
        base_df,
        tol_wave_A=0.005,   # Angstrom tolerance for wavelength matching
        tol_ep_eV=0.02,     # eV tolerance for EP matching
        update_cols=("loggf", "lower_state_eV"),
    )

    # 3) Save updated linelist (iSpec-ready)
    write_ispec_atomic_linelist_tsv(updated_base, out_linelist_path)

    # 4) Save report (optional but recommended)
    os.makedirs(os.path.dirname(out_report_path), exist_ok=True)
    report_df.to_csv(out_report_path, index=False)
    logger.info(f"Saved match report -> {out_report_path} (rows={len(report_df)})")