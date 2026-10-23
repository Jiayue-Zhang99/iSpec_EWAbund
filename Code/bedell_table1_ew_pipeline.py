"""Inject Bedell2018 Table1 EWs into iSpec linemasks for q2/MOOG analysis. By default Fe retains pipeline template EWs; non-Fe rows with positive external EWs replace matching rows or append missing transitions using Table1 atomic data."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# Convert MOOG species codes to iSpec labels such as Fe 1.
from ispec.lines import __get_element as _moog_species_to_element_label


def _normalize_table1_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "log(gf)" in df.columns:
        df = df.rename(columns={"log(gf)": "loggf"})
    if "loggf" not in df.columns:
        raise KeyError("Bedell2018_table1.csv requires log(gf) or loggf.")
    return df


def load_bedell_table1_raw(path: str) -> pd.DataFrame:
    df = _normalize_table1_columns(pd.read_csv(path))
    req = {"Wavelength", "Species", "EP", "loggf"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Bedell2018_table1 is missing columns: {missing}")
    if "Src" in df.columns:
        src_empty = df["Src"].isna() | (df["Src"].astype(str).str.strip() == "")
        df = df.assign(_prefer=src_empty.astype(int))
        df = df.sort_values("_prefer", ascending=False).drop(columns=["_prefer"])
    df = df.drop_duplicates(subset=["Wavelength", "Species"], keep="first")
    df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors="coerce")
    df["Species"] = pd.to_numeric(df["Species"], errors="coerce")
    return df.dropna(subset=["Wavelength", "Species"])


def resolve_linemask_template(
    target: str,
    ispec_dir: str,
    explicit: Optional[str] = None,
) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    candidates = [
        os.path.join(
            ispec_dir,
            "output",
            "MegantestSample",
            target,
            "linemasks",
            f"{target}_melendez2014_star_fitted_linemasks.txt",
        ),
        os.path.join(
            ispec_dir,
            "output",
            target,
            "linemasks",
            f"{target}_melendez2014_star_fitted_linemasks.txt",
        ),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No fitted-linemask template found. First run standard 3a Part I for this target,\n"
        "or set PIPELINE_OPTIONS['bedell_ew_linemask_template'].\n"
        "Tried:\n  " + "\n  ".join(candidates)
    )


def _species_float_from_lm(lm) -> float:
    s = str(lm["spectrum_moog_species"]).strip()
    return float(s)


def _match_key(wave_a: float, species: float, wave_decimals: int, sp_decimals: int) -> tuple[float, float]:
    return (round(float(wave_a), wave_decimals), round(float(species), sp_decimals))


def _is_fe_moog_species(sp: float) -> bool:
    return bool(
        np.isclose(sp, 26.0, rtol=0, atol=1e-4) | np.isclose(sp, 26.1, rtol=0, atol=1e-4)
    )


def _is_fe_element(elem: str) -> bool:
    t = str(elem).strip()
    return t in ("Fe 1", "Fe 2", "Fe I", "Fe II")


def _table1_has_fe_rows(tbl: pd.DataFrame) -> bool:
    sp = pd.to_numeric(tbl["Species"], errors="coerce")
    m = np.isfinite(sp.to_numpy(dtype=float))
    if not m.any():
        return False
    s = sp.to_numpy(dtype=float)[m]
    return bool(np.any(np.isclose(s, 26.0, rtol=0, atol=1e-4) | np.isclose(s, 26.1, rtol=0, atol=1e-4)))


def _load_chemical_elements_and_molecules(ispec, ispec_dir: str) -> tuple[np.ndarray, np.ndarray]:
    root = ispec_dir.rstrip(os.sep)
    ce_path = os.path.join(root, "input", "abundances", "chemical_elements_symbols.dat")
    mol_path = os.path.join(root, "input", "abundances", "molecular_symbols.dat")
    return ispec.read_chemical_elements(ce_path), ispec.read_molecular_symbols(mol_path)


def _template_key_set(star_linemasks, wave_decimals: int, species_decimals: int) -> set[tuple[float, float]]:
    keys: set[tuple[float, float]] = set()
    for i in range(len(star_linemasks)):
        try:
            sp = _species_float_from_lm(star_linemasks[i])
        except Exception:
            continue
        w = float(star_linemasks[i]["wave_A"])
        keys.add(_match_key(w, sp, wave_decimals, species_decimals))
    return keys


def _pick_donor_index(star_linemasks, elem_label: str) -> int:
    """Prefer a donor row with the same element label; otherwise use row zero."""
    want = str(elem_label).strip()
    for i in range(len(star_linemasks)):
        if str(star_linemasks[i]["element"]).strip() == want:
            return i
    return 0


def _append_row_from_table1(
    star_linemasks: np.ndarray,
    donor_idx: int,
    wave_a: float,
    species: float,
    ep: float,
    loggf: float,
    ew: float,
    elem_label: str,
    wave_decimals: int,
    species_decimals: int,
) -> np.ndarray:
    """Copy the donor dtype and fill Table1 atomic fields, EW, and a basic fitting window."""
    dtype = star_linemasks.dtype
    nr = np.empty(1, dtype=dtype)
    donor = star_linemasks[donor_idx]
    for name in dtype.names:
        nr[name][0] = donor[name]

    wnm = float(wave_a) / 10.0
    half_nm = 0.015
    # Fixed-width numpy element strings may truncate labels.
    nr["element"][0] = elem_label
    nr["wave_A"][0] = float(wave_a)
    nr["wave_nm"][0] = wnm
    nr["loggf"][0] = float(loggf)
    nr["lower_state_eV"][0] = float(ep)
    nr["spectrum_moog_species"][0] = f"{species:.1f}"
    nr["ew"][0] = float(ew)
    nr["ewr"][0] = 0.0
    nr["wave_peak"][0] = wnm
    nr["wave_base"][0] = wnm - half_nm
    nr["wave_top"][0] = wnm + half_nm
    nr["note"][0] = elem_label
    # Mark as supported by MOOG, consistent with the template.
    for _k in ("spectrum_support", "turbospectrum_support", "moog_support", "width_support", "synthe_support", "sme_support"):
        if _k in dtype.names:
            nr[_k][0] = "T"
    return nr


def build_fitted_linemasks_from_bedell_table1(
    ispec,
    table1_path: str,
    column_key: str,
    template_path: Optional[str],
    target: str,
    ispec_dir: str,
    out_fitted_path: str,
    wave_match_decimals: int = 2,
    species_decimals: int = 1,
    fallback_pipeline_fe_ew_if_no_table1_fe: bool = True,
    use_pipeline_fe_only: bool = True,
    merge_missing_table1_lines: bool = True,
) -> np.ndarray:
    """Read a linemask template and replace or append external EWs. use_pipeline_fe_only preserves template Fe lines; merge_missing_table1_lines appends missing non-Fe transitions. fallback_pipeline_fe_ew_if_no_table1_fe applies only when pipeline-only Fe mode is disabled."""
    tpl = resolve_linemask_template(target, ispec_dir, template_path)
    tbl = load_bedell_table1_raw(table1_path)
    if column_key not in tbl.columns:
        raise KeyError(
            f"Table1 has no column {column_key!r}. Expected HIPxxxx or Sun; "
            f"set inlist.PIPELINE_OPTIONS_OVERRIDE['bedell_ew_column'] if needed."
        )

    tbl["_ew"] = pd.to_numeric(tbl[column_key], errors="coerce")
    tbl_u = tbl.dropna(subset=["_ew"]).copy()
    if tbl_u.empty:
        raise ValueError(f"Column {column_key} in {table1_path} has no valid EWs")

    chem, mol = _load_chemical_elements_and_molecules(ispec, ispec_dir)

    if use_pipeline_fe_only:
        skip_fe_overwrite = True
        print(
            "[bedell_table1_ew] Pipeline-only Fe: retain template Fe I/Fe II EWs and lines; "
            "replace or append non-Fe EWs from Table1."
        )
    else:
        has_fe_in_table = _table1_has_fe_rows(tbl_u)
        skip_fe_overwrite = bool(fallback_pipeline_fe_ew_if_no_table1_fe and (not has_fe_in_table))
        if skip_fe_overwrite:
            print(
                "[bedell_table1_ew] Table1 has no Fe rows: retain pipeline template Fe EWs; "
                "replace other element EWs from Table1."
            )

    ew_map: dict[tuple[float, float], float] = {}
    for _, r in tbl_u.iterrows():
        sp = float(r["Species"])
        if use_pipeline_fe_only and _is_fe_moog_species(sp):
            continue
        if not use_pipeline_fe_only and skip_fe_overwrite and _is_fe_moog_species(sp):
            continue
        k = _match_key(float(r["Wavelength"]), sp, wave_match_decimals, species_decimals)
        ew_map[k] = float(r["_ew"])

    star_linemasks = ispec.read_line_regions(tpl)
    star_linemasks = star_linemasks.copy()

    n_match = 0
    n_skip_zero = 0
    n_skip_fe_pipeline = 0
    for i in range(len(star_linemasks)):
        elem = str(star_linemasks[i]["element"]).strip()
        if skip_fe_overwrite and _is_fe_element(elem):
            n_skip_fe_pipeline += 1
            continue
        try:
            sp = _species_float_from_lm(star_linemasks[i])
        except Exception:
            continue
        w = float(star_linemasks[i]["wave_A"])
        k = _match_key(w, sp, wave_match_decimals, species_decimals)
        if k in ew_map:
            ew_val = ew_map[k]
            if ew_val > 0:
                star_linemasks["ew"][i] = ew_val
                n_match += 1
            else:
                n_skip_zero += 1

    n_tpl_before = len(star_linemasks)
    n_append = 0
    n_append_skip = 0
    if merge_missing_table1_lines:
        present = _template_key_set(star_linemasks, wave_match_decimals, species_decimals)
        blocks: list[np.ndarray] = []
        for _, r in tbl_u.iterrows():
            sp = float(r["Species"])
            # Do not append Table1 Fe rows in pipeline-only Fe mode.
            if _is_fe_moog_species(sp):
                continue
            ew0 = float(r["_ew"])
            if not (ew0 > 0 and np.isfinite(ew0)):
                continue
            w_a = float(r["Wavelength"])
            k = _match_key(w_a, sp, wave_match_decimals, species_decimals)
            if k in present:
                continue
            species_str = f"{sp:.1f}"
            el = _moog_species_to_element_label(chem, mol, species_str)
            if el == "Discard":
                n_append_skip += 1
                continue
            donor_idx = _pick_donor_index(star_linemasks, el)
            ep = float(pd.to_numeric(r.get("EP"), errors="coerce") or 0.0)
            lg = float(pd.to_numeric(r.get("loggf"), errors="coerce") or 0.0)
            row = _append_row_from_table1(
                star_linemasks,
                donor_idx,
                w_a,
                sp,
                ep,
                lg,
                ew0,
                el,
                wave_match_decimals,
                species_decimals,
            )
            blocks.append(row)
            present.add(k)
            n_append += 1
        if blocks:
            star_linemasks = np.concatenate([star_linemasks] + blocks, axis=0)
            sort_ix = np.argsort(star_linemasks["wave_A"].astype(float))
            star_linemasks = star_linemasks[sort_ix]

    os.makedirs(os.path.dirname(out_fitted_path), exist_ok=True)
    ispec.write_line_regions(star_linemasks, out_fitted_path, extended=True)

    _elem = np.array([str(e).strip() for e in star_linemasks["element"]])
    iron_mask = (_elem == "Fe 1") | (_elem == "Fe 2")
    iron_star_linemasks = star_linemasks[iron_mask]
    fe_path = os.path.join(os.path.dirname(out_fitted_path), f"{target}_melendez2014_star_fe_linemasks.txt")
    ispec.write_line_regions(iron_star_linemasks, fe_path)

    zeroed = ispec.reset_fitted_data_fields(star_linemasks)
    zpath = os.path.join(
        os.path.dirname(out_fitted_path),
        f"{target}_melendez2014_star_zeroed_fitted_linemasks.txt",
    )
    ispec.write_line_regions(zeroed, zpath, extended=True)

    n_final = len(star_linemasks)
    print(
        f"[bedell_table1_ew] template={tpl}\n"
        f"[bedell_table1_ew] wrote fitted -> {out_fitted_path}\n"
        f"[bedell_table1_ew] EW overwrite hits: {n_match} (on original {n_tpl_before} template rows); "
        f"appended from Table1: {n_append}; final line count: {n_final}\n"
        f"[bedell_table1_ew] table EW<=0 skipped in overwrite: {n_skip_zero}; "
        f"Fe rows skipped in overwrite pass: {n_skip_fe_pipeline}; "
        f"append skipped (Discard species): {n_append_skip}; "
        f"unique table keys used for non-Fe overwrite map: {len(ew_map)}"
    )
    return star_linemasks


def fe_ew_override_frames_for_q2(
    table1_path: str,
    target_col: str,
    sun_col: str = "Sun",
    fallback_pipeline_fe_ew_if_no_table1_fe: bool = True,
    use_pipeline_fe_only: bool = True,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Extract Fe I/Fe II external EWs as wavelength, species, ep, gf, ew. Return (None, None) in pipeline-only Fe mode so q2 uses the pipeline linemasks."""
    if use_pipeline_fe_only:
        print(
            "[bedell_table1_ew] q2 pipeline-only Fe mode: ignore Table1 Fe EWs; "
            "use Fe from pipeline linemasks."
        )
        return None, None

    df = load_bedell_table1_raw(table1_path)
    for c in (target_col, sun_col):
        if c not in df.columns:
            raise KeyError(f"Table1 is missing column {c!r}")

    sp = pd.to_numeric(df["Species"], errors="coerce")
    fe_mask = np.isclose(sp.to_numpy(dtype=float), 26.0, rtol=0, atol=1e-4) | np.isclose(
        sp.to_numpy(dtype=float), 26.1, rtol=0, atol=1e-4
    )
    fe = df.loc[fe_mask].copy()
    if fe.empty:
        if fallback_pipeline_fe_ew_if_no_table1_fe:
            print(
                f"[bedell_table1_ew] q2：{table1_path} has no Fe rows; do not override Fe EWs "
                f"(target uses linemask Fe; Sun uses the solar library)."
            )
            return None, None
        raise ValueError(f"{table1_path} has no Fe transitions with Species 26.0 or 26.1")

    def _part(sub: pd.DataFrame, col: str) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "wavelength": pd.to_numeric(sub["Wavelength"], errors="coerce"),
                "species": sub["Species"].astype(float),
                "ep": pd.to_numeric(sub["EP"], errors="coerce"),
                "gf": pd.to_numeric(sub["loggf"], errors="coerce"),
                "ew": pd.to_numeric(sub[col], errors="coerce"),
            }
        )
        out = out.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["wavelength", "species", "ep", "gf", "ew"]
        )
        return out

    return _part(fe, target_col), _part(fe, sun_col)
