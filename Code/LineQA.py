import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_DELETED_COLS = [
    "element",
    "wave_A",
    "wave_nm",
    "loggf",
    "lower_state_eV",
    "lower_state_cm1",
    "lower_j",
    "upper_state_eV",
    "upper_state_cm1",
    "upper_j",
    "upper_g",
    "lande_lower",
    "lande_upper",
    "spectrum_transition_type",
    "turbospectrum_rad",
    "rad",
    "stark",
    "waals",
    "waals_single_gamma_format",
    "turbospectrum_fdamp",
    "spectrum_fudge_factor",
    "theoretical_depth",
    "theoretical_ew",
    "lower_orbital_type",
    "upper_orbital_type",
    "molecule",
    "spectrum_synthe_isotope",
    "ion",
    "spectrum_moog_species",
    "turbospectrum_species",
    "width_species",
    "reference_code",
    "spectrum_support",
    "turbospectrum_support",
    "moog_support",
    "width_support",
    "synthe_support",
    "sme_support",
]

DEFAULT_MODIFIED_COLS = ["element", "wave_A", "wave_nm", "loggf", "ew_mA_new"]


def ensure_tsv_with_header(path: str, columns: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, sep="\t", index=False)


def _read_tsv(path: str) -> pd.DataFrame:
    if (path is None) or (not os.path.exists(path)):
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_deleted_table(path: str) -> pd.DataFrame:
    df = _read_tsv(path)
    if df.empty:
        return pd.DataFrame(columns=["element", "wave_nm"])

    out = df.copy()
    if "element" not in out.columns or "wave_nm" not in out.columns:
        # Accept partially populated manual tables where possible.
        cols = [c for c in ["element", "wave_nm"] if c in out.columns]
        out = out[cols]

    out["element"] = out["element"].astype(str).str.strip()
    out["wave_nm"] = pd.to_numeric(out["wave_nm"], errors="coerce")
    out = out.dropna(subset=["wave_nm"])
    return out[["element", "wave_nm"]]


def load_modified_table(path: str) -> pd.DataFrame:
    df = _read_tsv(path)
    if df.empty:
        return pd.DataFrame(columns=["element", "wave_nm", "ew_mA_new"])

    out = df.copy()
    if "element" in out.columns:
        out["element"] = out["element"].astype(str).str.strip()
    if "wave_nm" in out.columns:
        out["wave_nm"] = pd.to_numeric(out["wave_nm"], errors="coerce")
    if "ew_mA_new" in out.columns:
        out["ew_mA_new"] = pd.to_numeric(out["ew_mA_new"], errors="coerce")

    keep = [c for c in ["element", "wave_nm", "ew_mA_new"] if c in out.columns]
    out = out[keep].dropna(subset=["wave_nm"])
    if "ew_mA_new" in out.columns:
        out = out.dropna(subset=["ew_mA_new"])
    return out


def _round(x, decimals: int):
    return np.round(np.asarray(x, dtype=float), decimals)


@dataclass(frozen=True)
class LineEditPaths:
    deleted_tsv: str
    modified_tsv: str

    @staticmethod
    def in_target_output(output_folder: str) -> "LineEditPaths":
        return LineEditPaths(
            deleted_tsv=os.path.join(output_folder, "linemasks", "Lines_deleted.tsv"),
            modified_tsv=os.path.join(output_folder, "linemasks", "Lines_EW_modified.tsv"),
        )

    def ensure_exists(self) -> None:
        ensure_tsv_with_header(self.deleted_tsv, DEFAULT_DELETED_COLS)
        ensure_tsv_with_header(self.modified_tsv, DEFAULT_MODIFIED_COLS)


def filter_structured_array_by_deleted(
    arr,
    deleted_df: pd.DataFrame,
    *,
    element_col: str = "element",
    wave_nm_col: str = "wave_nm",
    decimals: int = 4,
    only_elements: Optional[Iterable[str]] = None,
):
    """Filter an iSpec structured array using deleted rows keyed by element and rounded wave_nm."""
    if deleted_df is None or deleted_df.empty or arr is None:
        return arr

    names = getattr(arr, "dtype", None).names if hasattr(arr, "dtype") else None
    if not names or (element_col not in names) or (wave_nm_col not in names):
        return arr

    d = deleted_df.copy()
    d["element"] = d["element"].astype(str).str.strip()
    d["wave_nm"] = pd.to_numeric(d["wave_nm"], errors="coerce")
    d = d.dropna(subset=["wave_nm"])
    if d.empty:
        return arr

    if only_elements is not None:
        only = {str(x).strip() for x in only_elements}
        d = d[d["element"].isin(only)]
        if d.empty:
            return arr

    elem_arr = pd.Series(arr[element_col]).astype(str).str.strip().to_numpy()
    wave_arr = _round(arr[wave_nm_col], decimals)

    dkey = pd.DataFrame(
        {
            "element": d["element"].astype(str).str.strip().to_numpy(),
            "wave_nm": _round(d["wave_nm"], decimals),
        }
    ).drop_duplicates()

    # Use a set for efficient lookup.
    key_set = set(zip(dkey["element"].tolist(), dkey["wave_nm"].tolist()))
    keep = np.array([(e, w) not in key_set for e, w in zip(elem_arr, wave_arr)], dtype=bool)
    return arr[keep]


def apply_modified_ews_to_linemasks(
    linemasks,
    modified_df: pd.DataFrame,
    *,
    decimals: int = 4,
    element_col: str = "element",
    wave_nm_col: str = "wave_nm",
    ew_col: str = "ew",
):
    """Apply manual EWs in mAngstrom to a copy of the linemasks array."""
    if modified_df is None or modified_df.empty or linemasks is None:
        return linemasks

    names = getattr(linemasks, "dtype", None).names if hasattr(linemasks, "dtype") else None
    if not names or (element_col not in names) or (wave_nm_col not in names) or (ew_col not in names):
        return linemasks

    m = modified_df.copy()
    if "element" in m.columns:
        m["element"] = m["element"].astype(str).str.strip()
    if "wave_nm" in m.columns:
        m["wave_nm"] = pd.to_numeric(m["wave_nm"], errors="coerce")
    if "ew_mA_new" in m.columns:
        m["ew_mA_new"] = pd.to_numeric(m["ew_mA_new"], errors="coerce")
    m = m.dropna(subset=["wave_nm", "ew_mA_new"])
    if m.empty:
        return linemasks

    out = linemasks.copy()
    elem_arr = pd.Series(out[element_col]).astype(str).str.strip().to_numpy()
    wave_arr = _round(out[wave_nm_col], decimals)

    # Keep the last manual entry for each element/wavelength pair.
    m["wave_nm_r"] = _round(m["wave_nm"], decimals)
    m = m.drop_duplicates(subset=["element", "wave_nm_r"], keep="last")
    ew_map = {(r["element"], r["wave_nm_r"]): float(r["ew_mA_new"]) for _, r in m.iterrows()}

    for i, (e, w) in enumerate(zip(elem_arr, wave_arr)):
        key = (e, float(w))
        if key in ew_map:
            out[ew_col][i] = ew_map[key]
    return out


def prepare_line_edit_tables(output_folder: str) -> LineEditPaths:
    """Ensure both manual-edit tables exist under output/<target>/linemasks and return their paths."""
    paths = LineEditPaths.in_target_output(output_folder)
    paths.ensure_exists()
    return paths

