import os
import re
import shlex
import shutil
import subprocess
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hfs_contracts import (no_hfs_logeps, attach_solar_baselines, component_signature,
                           apply_pair_quality, profile_quality)


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, object]) -> None:
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    import logging
    logging.getLogger(__name__).debug("HFS diagnostic: %s", payload)


# Elements requested for HFS processing
ELEMENT_CONFIG: Dict[str, Dict[str, object]] = {
    "V": {"species": 23.0, "ion": "1"},
    "Mn": {"species": 25.0, "ion": "1"},
    "Co": {"species": 27.0, "ion": "1"},
    "Cu": {"species": 29.0, "ion": "1"},
    "Ba": {"species": 56.1, "ion": "2"},
}

# Grevesse 2007-like solar photospheric abundances A(X).
# Used to convert absolute logeps to differential [X/H] = A(X)_star - A(X)_sun.
SOLAR_LOGEPS: Dict[str, float] = {
    "V": 3.93,
    "Mn": 5.39,
    "Co": 4.92,
    "Cu": 4.21,
    "Ba": 2.17,
}


@dataclass
class HFSPaths:
    base: Path
    linelists_raw: Path
    linelists_blends: Path
    runs: Path
    tables: Path
    figs: Path


def _ensure_dirs(root: Path) -> HFSPaths:
    paths = HFSPaths(
        base=root,
        linelists_raw=root / "linelists_raw",
        linelists_blends=root / "linelists_blends",
        runs=root / "runs",
        tables=root / "tables",
        figs=root / "figs",
    )
    for p in (paths.base, paths.linelists_raw, paths.linelists_blends, paths.runs, paths.tables, paths.figs):
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _normalize_element_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace(" II", " 2").replace(" I", " 1")
    m = re.match(r"^([A-Za-z]+)\s*([12])?$", s)
    if not m:
        return s
    base = m.group(1)
    ion = m.group(2)
    if ion:
        return f"{base} {ion}"
    return base


def _infer_base_linelist(output_folder: str, target: str) -> Path:
    """Prefer final linelist_for_<target>.tsv over copied pre-delete list."""
    base = Path(output_folder)
    final_linelist = base / f"linelist_for_{target}.tsv"
    if final_linelist.exists():
        return final_linelist
    fallback = base / "linemasks" / f"linelist_for_{target}_copied.tsv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No linelist found for {target}. Tried: {final_linelist} and {fallback}"
    )


def _load_base_linelist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "element" not in df.columns or "wave_A" not in df.columns:
        raise ValueError(f"Linelist missing required columns: {path}")
    df = df.copy()
    df["element_norm"] = df["element"].map(_normalize_element_name)
    df["wave_A"] = pd.to_numeric(df["wave_A"], errors="coerce")
    df["lower_state_eV"] = pd.to_numeric(df.get("lower_state_eV", np.nan), errors="coerce")
    df["loggf"] = pd.to_numeric(df.get("loggf", np.nan), errors="coerce")
    df["spectrum_moog_species"] = pd.to_numeric(df.get("spectrum_moog_species", np.nan), errors="coerce")
    return df


def collect_hfs_target_lines(
    output_folder: str,
    target: str,
    elements: Sequence[str] = ("V", "Mn", "Co", "Cu", "Ba"),
    *,
    use_external_input_ew: bool = False,
) -> pd.DataFrame:
    """Build the HFS target line table. Read anchor EWs from q2_work/lines_q2.csv, then optionally from the raw abundance table. In external EW mode, preserve finite q2 EWs; otherwise matching raw EWs may replace them."""
    linelist_path = _infer_base_linelist(output_folder, target)
    df = _load_base_linelist(linelist_path)

    rows: List[Dict[str, object]] = []
    for ele in elements:
        cfg = ELEMENT_CONFIG[ele]
        species = float(cfg["species"])
        ion = str(cfg["ion"])
        ele_norm = f"{ele} {ion}"
        sub = df[df["element_norm"] == ele_norm].copy()
        if sub.empty:
            # fallback by species if element label differs
            sub = df[np.isclose(df["spectrum_moog_species"], species, equal_nan=False)].copy()
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            rows.append(
                {
                    "target": target,
                    "element": ele,
                    "element_norm": ele_norm,
                    "ion": ion,
                    "species": species,
                    "wave_A": float(r["wave_A"]),
                    "ep_eV": float(r["lower_state_eV"]) if np.isfinite(r["lower_state_eV"]) else np.nan,
                    "loggf": float(r["loggf"]) if np.isfinite(r["loggf"]) else np.nan,
                    "waals_single_gamma_format": float(r.get("waals_single_gamma_format", 0.0))
                    if np.isfinite(pd.to_numeric(r.get("waals_single_gamma_format", 0.0), errors="coerce"))
                    else 0.0,
                    "theoretical_ew": float(r.get("theoretical_ew", np.nan))
                    if np.isfinite(pd.to_numeric(r.get("theoretical_ew", np.nan), errors="coerce"))
                    else np.nan,
                    "source_linelist": str(linelist_path),
                }
            )
    out = pd.DataFrame(rows).sort_values(["element", "wave_A"]).reset_index(drop=True)
    out["observed_ew"] = np.nan
    out["logeps_sun_ref"] = np.nan
    out["logeps_star_ref"] = np.nan
    out["d_xh_ref"] = np.nan
    out["q2_matched"] = False

    # Prefer strict one-to-one line mapping from q2_work/lines_q2.csv.
    q2_lines = Path(output_folder) / "q2_work" / "lines_q2.csv"
    if q2_lines.exists():
        try:
            q2 = pd.read_csv(q2_lines)
            q2["wavelength"] = pd.to_numeric(q2.get("wavelength"), errors="coerce")
            q2["species"] = pd.to_numeric(q2.get("species"), errors="coerce")
            q2[target] = pd.to_numeric(q2.get(target), errors="coerce")
            q2 = q2[np.isfinite(q2["wavelength"]) & np.isfinite(q2["species"]) & np.isfinite(q2[target])]
            used_q2_idx: set = set()
            for i, r in out.iterrows():
                m = q2[
                    (np.isclose(q2["wavelength"], float(r["wave_A"]), atol=0.003, rtol=0))
                    & (np.isclose(q2["species"], float(r["species"]), atol=0.06))
                ]
                if len(m):
                    mm = m.copy()
                    mm["__d"] = (mm["wavelength"] - float(r["wave_A"])).abs() + 10.0 * (
                        mm["species"] - float(r["species"])
                    ).abs()
                    mm = mm.sort_values("__d")
                    pick = None
                    for idx in mm.index.tolist():
                        if int(idx) not in used_q2_idx:
                            pick = idx
                            break
                    if pick is None:
                        continue
                    used_q2_idx.add(int(pick))
                    out.at[i, "observed_ew"] = float(q2.loc[pick, target])
                    out.at[i, "q2_matched"] = True
        except Exception:
            pass

    # Add no-HFS line-by-line solar zero point for consistent [X/H] definition.
    raw_q2 = Path(output_folder) / f"abundances_linebyline_{target}_raw.csv"
    if raw_q2.exists():
        try:
            rr = pd.read_csv(raw_q2)
            rr["element"] = rr.get("element", "").astype(str).str.strip()
            rr["wave_A"] = pd.to_numeric(rr.get("wave_A"), errors="coerce")
            rr["ew_mA"] = pd.to_numeric(rr.get("ew_mA"), errors="coerce")
            rr["logeps_star"] = pd.to_numeric(rr.get("logeps_star"), errors="coerce")
            rr["logeps_sun"] = pd.to_numeric(rr.get("logeps_sun"), errors="coerce")
            rr["d[X/H]"] = pd.to_numeric(rr.get("d[X/H]"), errors="coerce")
            rr = rr[np.isfinite(rr["wave_A"])]
            used_rr_idx: set = set()
            for i, r in out.iterrows():
                ion_label = "II" if str(r.get("ion", "1")) == "2" else "I"
                expected_element = f"{str(r['element']).strip()} {ion_label}"
                m = rr[
                    np.isclose(rr["wave_A"], float(r["wave_A"]), atol=0.003, rtol=0)
                    & (rr["element"] == expected_element)
                ]
                if len(m):
                    mm = m.copy()
                    mm["__d"] = (mm["wave_A"] - float(r["wave_A"])).abs()
                    mm = mm.sort_values("__d")
                    pick = None
                    for idx in mm.index.tolist():
                        if int(idx) not in used_rr_idx:
                            pick = idx
                            break
                    if pick is None:
                        continue
                    used_rr_idx.add(int(pick))
                    ew_raw = float(rr.loc[pick, "ew_mA"]) if np.isfinite(
                        pd.to_numeric(rr.loc[pick, "ew_mA"], errors="coerce")
                    ) else np.nan
                    if np.isfinite(ew_raw):
                        cur_ew = pd.to_numeric(out.at[i, "observed_ew"], errors="coerce")
                        if not (use_external_input_ew and np.isfinite(cur_ew)):
                            out.at[i, "observed_ew"] = ew_raw
                    out.at[i, "q2_matched"] = True
                    if np.isfinite(pd.to_numeric(rr.loc[pick, "logeps_star"], errors="coerce")):
                        out.at[i, "logeps_star_ref"] = no_hfs_logeps(rr.loc[pick])
                    if np.isfinite(pd.to_numeric(rr.loc[pick, "logeps_sun"], errors="coerce")):
                        out.at[i, "logeps_sun_ref"] = float(rr.loc[pick, "logeps_sun"])
                    if np.isfinite(pd.to_numeric(rr.loc[pick, "d[X/H]"], errors="coerce")):
                        out.at[i, "d_xh_ref"] = float(rr.loc[pick, "d[X/H]"])
        except Exception:
            pass

    # Prefer lines_q2_hfs.csv in external EW mode; lines_q2.csv may contain only Fe.
    if use_external_input_ew:
        hfs_lines = Path(output_folder) / "q2_work" / "lines_q2_hfs.csv"
        if hfs_lines.is_file():
            try:
                q2h = pd.read_csv(hfs_lines)
                q2h["wavelength"] = pd.to_numeric(q2h.get("wavelength"), errors="coerce")
                q2h["species"] = pd.to_numeric(q2h.get("species"), errors="coerce")
                q2h[target] = pd.to_numeric(q2h.get(target), errors="coerce")
                q2h = q2h[
                    np.isfinite(q2h["wavelength"])
                    & np.isfinite(q2h["species"])
                    & np.isfinite(q2h[target])
                ]
                used_hfs_idx: set = set()
                for i, r in out.iterrows():
                    m = q2h[
                        (np.isclose(q2h["wavelength"], float(r["wave_A"]), atol=0.003, rtol=0))
                        & (np.isclose(q2h["species"], float(r["species"]), atol=0.06))
                    ]
                    if len(m):
                        mm = m.copy()
                        mm["__d"] = (mm["wavelength"] - float(r["wave_A"])).abs() + 10.0 * (
                            mm["species"] - float(r["species"])
                        ).abs()
                        mm = mm.sort_values("__d")
                        pick = None
                        for idx in mm.index.tolist():
                            if int(idx) not in used_hfs_idx:
                                pick = idx
                                break
                        if pick is None:
                            continue
                        used_hfs_idx.add(int(pick))
                        out.at[i, "observed_ew"] = float(q2h.loc[pick, target])
                        out.at[i, "q2_matched"] = True
            except Exception:
                pass

    # Enforce one-to-one baseline correctness: only keep lines with q2_work anchor EW.
    if out["q2_matched"].any():
        out = out[out["q2_matched"]].copy().reset_index(drop=True)
    return out


def write_lines_q2_for_hfs_from_linelist_raw(
    output_folder: str,
    target: str,
    elements: Sequence[str] = ("V", "Mn", "Co", "Cu", "Ba"),
) -> str:
    """Export HFS EWs to q2_work/lines_q2_hfs.csv. Use the target linelist and prefer raw abundance ew_mA values, matching the external-EW abundance workflow."""
    linelist_path = _infer_base_linelist(output_folder, target)
    df = _load_base_linelist(linelist_path)
    raw_path = Path(output_folder) / f"abundances_linebyline_{target}_raw.csv"
    ew_by_key: Dict[tuple[float, str], float] = {}
    if raw_path.is_file():
        rr = pd.read_csv(raw_path)
        rr["element"] = rr.get("element", "").astype(str).str.strip()
        rr["wave_A"] = pd.to_numeric(rr.get("wave_A"), errors="coerce")
        rr["ew_mA"] = pd.to_numeric(rr.get("ew_mA"), errors="coerce")
        for _, row in rr.iterrows():
            if np.isfinite(row["wave_A"]) and np.isfinite(row["ew_mA"]):
                ew_by_key[(round(float(row["wave_A"]), 3), str(row["element"]))] = float(row["ew_mA"])

    rows: List[Dict[str, object]] = []
    for ele in elements:
        cfg = ELEMENT_CONFIG[ele]
        species = float(cfg["species"])
        ion = str(cfg["ion"])
        ele_norm = f"{ele} {ion}"
        ion_roman = "II" if ion == "2" else "I"
        sub = df[df["element_norm"] == ele_norm].copy()
        if sub.empty:
            sub = df[np.isclose(df["spectrum_moog_species"], species, equal_nan=False)].copy()
        for _, r in sub.iterrows():
            w = float(r["wave_A"])
            el_label = f"{ele} {ion_roman}"
            ew = ew_by_key.get((round(w, 3), el_label), np.nan)
            if not np.isfinite(ew):
                ew = ew_by_key.get((round(w, 3), ele_norm.replace(" 1", " I").replace(" 2", " II")), np.nan)
            if not np.isfinite(ew):
                continue
            rows.append(
                {
                    "wavelength": w,
                    "species": species,
                    "ep": float(r["lower_state_eV"]) if np.isfinite(r["lower_state_eV"]) else np.nan,
                    "gf": float(r["loggf"]) if np.isfinite(r["loggf"]) else np.nan,
                    target: float(ew),
                }
            )
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError(
            f"No HFS lines with EW found for {target}. Check {linelist_path} and {raw_path}."
        )
    out_dir = Path(output_folder) / "q2_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lines_q2_hfs.csv"
    out_df = out_df.sort_values(["species", "wavelength"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)
    print(f"[HFS] wrote {len(out_df)} HFS EW rows -> {out_path}")
    return str(out_path)


def _find_linemake_executable() -> Optional[str]:
    env = os.environ.get("ISPEC_LINEMAKE_BIN") or os.environ.get("LINEMAKE_BIN")
    if env:
        env_path = Path(env).expanduser()
        if env_path.exists():
            _agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H6",
                location="HFSBlends.py:_find_linemake_executable",
                message="linemake env candidate details",
                data={
                    "env_value": env,
                    "resolved": str(env_path),
                    "is_file": env_path.is_file(),
                    "is_executable": os.access(str(env_path), os.X_OK),
                },
            )
        else:
            _agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H6",
                location="HFSBlends.py:_find_linemake_executable",
                message="linemake env path does not exist",
                data={"env_value": env, "resolved": str(env_path)},
            )
    if env and Path(env).expanduser().exists():
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H1",
            location="HFSBlends.py:_find_linemake_executable",
            message="linemake resolved from env",
            data={"env_value": env, "resolved": str(Path(env).expanduser())},
        )
        return str(Path(env).expanduser())
    which_hits: Dict[str, Optional[str]] = {}
    for c in ["linemake", "linemake.go", "linemake.py"]:
        p = shutil.which(c)
        which_hits[c] = p
        if p:
            _agent_debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="HFSBlends.py:_find_linemake_executable",
                message="linemake resolved from which",
                data={"candidate": c, "resolved": p},
            )
            return p
    _agent_debug_log(
        run_id="pre-fix",
        hypothesis_id="H1",
        location="HFSBlends.py:_find_linemake_executable",
        message="linemake not found",
        data={
            "env_ISPEC_LINEMAKE_BIN": os.environ.get("ISPEC_LINEMAKE_BIN"),
            "env_LINEMAKE_BIN": os.environ.get("LINEMAKE_BIN"),
            "which_hits": which_hits,
        },
    )
    return None


def _build_linemake_argv(
    linemake_bin: str, species: float, wave: float
) -> List[str]:
    """
    Build argv for optional non-interactive linemake wrapper.

    If ISPEC_LINEMAKE_ARGV is set, it is appended after ``linemake_bin``, with
    ``str.format`` placeholders ``{species}``, ``{wave}``, ``{w}`` (``{w}`` = wave to 3 decimals).
    Default: ``[bin, str(species), f'{wave:.3f}']`` — vmplacco linemake is usually interactive.
    """
    tpl = os.environ.get("ISPEC_LINEMAKE_ARGV")
    if tpl:
        extra = tpl.format(species=species, wave=wave, w=f"{wave:.3f}")
        return [linemake_bin, *shlex.split(extra)]
    return [linemake_bin, str(species), f"{wave:.3f}"]


def _build_linemake_stdin_for_native(wave: float) -> str:
    """
    Build scripted stdin for native interactive vmplacco linemake.
    Keeps wavelength limits within the same 1000 A bin to avoid known crash.
    """
    base = int(wave // 1000) * 1000
    w_lo = max(wave - 0.2, base + 0.001)
    w_hi = min(wave + 0.2, base + 999.999)
    return f"{w_lo:.3f}\n{w_hi:.3f}\nn\nn\n"


def _parse_moog_style_numeric_lines(text: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            w = float(parts[0])
            sp = float(parts[1])
            ep = float(parts[2])
            lg = float(parts[3])
        except ValueError:
            continue
        rows.append({"wave_A": float(w), "species": float(sp), "ep_eV": float(ep), "loggf": float(lg)})
    return pd.DataFrame(rows)


def _extract_linemake_components_from_run(
    run_dir: Path,
    species: float,
    wave_A: float,
    expected_ep_eV: Optional[float] = None,
    expected_loggf: Optional[float] = None,
    ep_tol_eV: float = 0.20,
    loggf_sum_tol_dex: Optional[float] = None,
    tol_A: float = 0.5,
) -> pd.DataFrame:
    """
    Read native linemake outputs (outsort/outlines) and extract HFS components.
    The nearest line to anchor wavelength is treated as anchor-like and excluded
    from components to avoid double-counting with the EW anchor line.
    """
    frames: List[pd.DataFrame] = []
    for p in [run_dir / "outsort", run_dir / "outlines"]:
        if p.exists():
            df = _parse_moog_style_numeric_lines(p.read_text(errors="ignore"))
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df = df[np.isclose(df["species"], float(species), atol=0.06)]
    if df.empty:
        return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])
    df = df[np.abs(df["wave_A"] - float(wave_A)) <= tol_A].copy()
    if df.empty:
        return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])

    # Keep only transitions consistent with anchor EP when available.
    if expected_ep_eV is not None and np.isfinite(expected_ep_eV):
        df = df[np.abs(df["ep_eV"] - float(expected_ep_eV)) <= float(ep_tol_eV)].copy()
        if df.empty:
            return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])

    comp = df.copy()
    if comp.empty:
        return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])

    # Optional pre-check only; final physical QC + renormalization happens downstream.
    if (
        loggf_sum_tol_dex is not None
        and expected_loggf is not None
        and np.isfinite(expected_loggf)
    ):
        comp_loggf_sum = float(np.log10(np.sum(np.power(10.0, comp["loggf"].astype(float)))))
        if abs(comp_loggf_sum - float(expected_loggf)) > float(loggf_sum_tol_dex):
            return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])

    comp["is_component"] = True
    return comp.sort_values("wave_A").reset_index(drop=True)


def _enforce_component_physics(
    comp: pd.DataFrame,
    anchor_ep_eV: float,
    anchor_loggf: float,
    ep_tol_eV: float = 0.15,
    max_abs_delta_loggf: float = 1.0,
) -> Tuple[pd.DataFrame, str]:
    """
    Enforce basic physical consistency for extracted HFS components:
    1) EP consistency with anchor transition.
    2) Reasonable total gf consistency with parent line.
    3) Renormalize component loggf so sum(gf_components) == gf_parent.
    """
    if comp.empty:
        return comp, "empty"

    c = comp.copy()
    if np.isfinite(anchor_ep_eV):
        c = c[np.abs(pd.to_numeric(c["ep_eV"], errors="coerce") - float(anchor_ep_eV)) <= ep_tol_eV].copy()
    if c.empty:
        return c, "drop_all_ep_mismatch"

    gf_sum = np.sum(10.0 ** pd.to_numeric(c["loggf"], errors="coerce"))
    if (not np.isfinite(gf_sum)) or gf_sum <= 0:
        return pd.DataFrame(columns=comp.columns), "invalid_component_gf_sum"
    if not np.isfinite(anchor_loggf):
        return c, "keep_no_anchor_loggf"

    sum_loggf = float(np.log10(gf_sum))
    delta = sum_loggf - float(anchor_loggf)
    if abs(delta) > max_abs_delta_loggf:
        return pd.DataFrame(columns=comp.columns), f"drop_unphysical_gf_sum_delta={delta:.3f}"

    # Renormalize components to match parent line total gf exactly.
    c["loggf"] = pd.to_numeric(c["loggf"], errors="coerce") - delta
    c, merge_note = _merge_duplicate_components(c)
    return c, f"renorm_delta={delta:.3f};{merge_note}"


def _merge_duplicate_components(
    comp: pd.DataFrame,
    wave_round_decimals: int = 3,
    species_round_decimals: int = 2,
    ep_round_decimals: int = 3,
) -> Tuple[pd.DataFrame, str]:
    """
    Merge numerically duplicate HFS components by summing gf in linear space.
    This avoids over-counting when line lists contain repeated identical transitions.
    """
    if comp.empty:
        return comp, "merge_dup_none"
    c = comp.copy()
    for col in ["wave_A", "species", "ep_eV", "loggf"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c[
        np.isfinite(c["wave_A"])
        & np.isfinite(c["species"])
        & np.isfinite(c["ep_eV"])
        & np.isfinite(c["loggf"])
    ].copy()
    if c.empty:
        return c, "merge_dup_none"

    c["_wkey"] = c["wave_A"].round(wave_round_decimals)
    c["_skey"] = c["species"].round(species_round_decimals)
    c["_ekey"] = c["ep_eV"].round(ep_round_decimals)
    n_before = int(len(c))

    c["_gf_lin"] = np.power(10.0, pd.to_numeric(c["loggf"], errors="coerce"))
    merged = (
        c.groupby(["_wkey", "_skey", "_ekey"], as_index=False)
        .agg(
            wave_A=("wave_A", "mean"),
            species=("species", "mean"),
            ep_eV=("ep_eV", "mean"),
            gf_sum=("_gf_lin", "sum"),
        )
    )
    merged["loggf"] = np.where(
        pd.to_numeric(merged["gf_sum"], errors="coerce") > 0,
        np.log10(pd.to_numeric(merged["gf_sum"], errors="coerce")),
        np.nan,
    )
    merged["is_component"] = True
    merged = merged.drop(columns=["gf_sum"], errors="ignore")
    merged = merged[np.isfinite(pd.to_numeric(merged["loggf"], errors="coerce"))].copy()
    merged = merged.sort_values("wave_A").reset_index(drop=True)
    n_after = int(len(merged))
    return merged, f"merge_dup_removed={max(n_before - n_after, 0)}"


def write_template_hfs_raw_snapshot(
    out_raw_dir: Path,
    template_dir: Path,
    element: str,
    species: float,
    wave_A: float,
    ep_eV: float,
    loggf: float,
) -> Path:
    """
    Always write a reproducible HFS snapshot from Iris template files so linelists_raw/
    is populated even without vmplacco linemake (which is interactive by default).
    """
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{element}_{wave_A:.3f}"
    out_path = out_raw_dir / f"{tag}.template_hfs_snapshot.dat"
    comp = _read_hfs_template_components(template_dir, species, wave_A, tol_A=0.5)
    ep0 = float(ep_eV) if np.isfinite(ep_eV) else 0.0
    lg0 = float(loggf) if np.isfinite(loggf) else -99.0
    lines: List[str] = [
        f"# HFS raw snapshot (Iris template + EW linelist anchor) tag={tag}",
        f"# species={species} wave_A={wave_A}",
        f"# template_dir={template_dir}",
        "#",
        "# Anchor (positive wavelength, same convention as build_blends_linelists):",
        f"{wave_A:10.3f} {species:9.1f} {ep0:9.2f} {lg0:9.2f}",
    ]
    n_comp = int(comp["is_component"].sum()) if not comp.empty else 0
    if n_comp > 0:
        lines.append("# HFS components (negative wavelength, from template is_component):")
        for _, c in comp[comp["is_component"]].iterrows():
            lines.append(
                f"{-abs(float(c['wave_A'])):10.3f} {float(c['species']):9.1f} "
                f"{float(c['ep_eV']):9.2f} {float(c['loggf']):9.2f}"
            )
    else:
        lines.append("# No HFS components matched in template within tolerance (see blends_manifest).")
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def generate_hfs_components_with_linemake(
    target_lines: pd.DataFrame,
    out_raw_dir: Path,
    hfs_template_dir: Path,
    linemake_bin: Optional[str] = None,
    dry_run: bool = False,
    linemake_timeout_sec: int = 12,
) -> pd.DataFrame:
    """
    Record HFS raw artifacts under linelists_raw/.

    - Always writes {tag}.template_hfs_snapshot.dat from Iris templates (non-empty folder).
    - Optional vmplacco ``linemake`` is normally **interactive**; set ISPEC_LINEMAKE_ARGV
      if your install supports non-interactive argv, or rely on template snapshots only.
    """
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    linemake_bin = linemake_bin or _find_linemake_executable()

    run_rows: List[Dict[str, object]] = []
    for _, r in target_lines.iterrows():
        element = r["element"]
        wave = float(r["wave_A"])
        species = float(r["species"])
        ep_ev = float(r["ep_eV"]) if "ep_eV" in r.index else np.nan
        lg = float(r["loggf"]) if "loggf" in r.index else np.nan
        tag = f"{element}_{wave:.3f}"
        out_file = out_raw_dir / f"{tag}.linemake.dat"
        comp_file = out_raw_dir / f"{tag}.linemake_components.csv"
        run_dir = out_raw_dir / f"{tag}.linemake_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        snap = write_template_hfs_raw_snapshot(
            out_raw_dir, hfs_template_dir, element, species, wave, ep_ev, lg
        )

        if linemake_bin is None:
            note_path = out_raw_dir / f"{tag}.linemake_note.txt"
            note_path.write_text(
                "No linemake binary found (PATH, ISPEC_LINEMAKE_BIN, LINEMAKE_BIN). "
                "Template snapshot is in .template_hfs_snapshot.dat.\n"
            )
            run_rows.append(
                {
                    "element": element,
                    "wave_A": wave,
                    "species": species,
                    "out_file": str(out_file),
                    "template_snapshot": str(snap),
                    "status": "skip_no_linemake",
                    "cmd": "",
                    "stderr": "linemake executable not found",
                    "components_file": str(comp_file),
                    "n_components": 0,
                }
            )
            continue

        cmd_list = _build_linemake_argv(linemake_bin, species, wave)
        cmd_str = " ".join(cmd_list)
        use_native_stdin = (
            os.environ.get("ISPEC_LINEMAKE_ARGV") in (None, "")
            and Path(linemake_bin).name in {"linemake.go", "linemake"}
        )
        scripted_stdin = _build_linemake_stdin_for_native(wave) if use_native_stdin else None
        _agent_debug_log(
            run_id="post-fix",
            hypothesis_id="H8",
            location="HFSBlends.py:generate_hfs_components_with_linemake",
            message="linemake execution mode selected",
            data={
                "element": element,
                "wave_A": wave,
                "species": species,
                "linemake_bin": linemake_bin,
                "use_native_stdin": use_native_stdin,
                "has_custom_argv": bool(os.environ.get("ISPEC_LINEMAKE_ARGV")),
            },
        )

        if dry_run:
            (out_raw_dir / f"{tag}.linemake_dry_run.txt").write_text(
                f"dry_run=True: would run:\n  {cmd_str}\n"
                "vmplacco/linemake is interactive unless you use a wrapper; "
                "set ISPEC_LINEMAKE_ARGV for non-interactive argv.\n"
            )
            run_rows.append(
                {
                    "element": element,
                    "wave_A": wave,
                    "species": species,
                    "out_file": str(out_file),
                    "template_snapshot": str(snap),
                    "status": "dry_run",
                    "cmd": cmd_str,
                    "stderr": "",
                    "components_file": str(comp_file),
                    "n_components": 0,
                }
            )
            continue

        try:
            proc = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                check=False,
                timeout=linemake_timeout_sec,
                input=scripted_stdin,
                cwd=str(run_dir),
            )
            out_file.write_text(proc.stdout if proc.stdout else "")
            err_path = out_raw_dir / f"{tag}.linemake_stderr.txt"
            err_path.write_text(proc.stderr if proc.stderr else "")
            # Preserve native outputs for traceability.
            for nm in ["outlines", "outsort"]:
                src = run_dir / nm
                if src.exists():
                    shutil.copyfile(src, out_raw_dir / f"{tag}.{nm}")
            comp = _extract_linemake_components_from_run(
                run_dir,
                species=species,
                wave_A=wave,
                expected_ep_eV=ep_ev if np.isfinite(ep_ev) else None,
                expected_loggf=lg if np.isfinite(lg) else None,
                ep_tol_eV=0.20,
                loggf_sum_tol_dex=None,
                tol_A=0.5,
            )
            if comp.empty:
                comp_file.write_text("wave_A,species,ep_eV,loggf,is_component\n")
            else:
                comp.to_csv(comp_file, index=False)
            status = "ok" if proc.returncode == 0 else f"fail_rc_{proc.returncode}"
            if proc.returncode != 0 and not (proc.stdout or "").strip():
                status = f"{status}_maybe_interactive"
            run_rows.append(
                {
                    "element": element,
                    "wave_A": wave,
                    "species": species,
                    "out_file": str(out_file),
                    "template_snapshot": str(snap),
                    "status": status,
                    "cmd": cmd_str,
                    "stderr": (proc.stderr or "").strip()[:2000],
                    "components_file": str(comp_file),
                    "n_components": int(len(comp)) if 'comp' in locals() else 0,
                }
            )
        except subprocess.TimeoutExpired:
            (out_raw_dir / f"{tag}.linemake_stderr.txt").write_text(
                f"timeout after {linemake_timeout_sec}s (likely interactive linemake).\n"
            )
            run_rows.append(
                {
                    "element": element,
                    "wave_A": wave,
                    "species": species,
                    "out_file": str(out_file),
                    "template_snapshot": str(snap),
                    "status": "skip_timeout_interactive",
                    "cmd": cmd_str,
                    "stderr": "subprocess timeout",
                    "components_file": str(comp_file),
                    "n_components": 0,
                }
            )
        except Exception as e:
            run_rows.append(
                {
                    "element": element,
                    "wave_A": wave,
                    "species": species,
                    "out_file": str(out_file),
                    "template_snapshot": str(snap),
                    "status": "exception",
                    "cmd": cmd_str,
                    "stderr": str(e),
                    "components_file": str(comp_file),
                    "n_components": 0,
                }
            )
    return pd.DataFrame(run_rows)


def _read_hfs_template_components(template_dir: Path, species: float, wave_A: float, tol_A: float = 0.3) -> pd.DataFrame:
    """
    Fallback parser using existing Iris template files when linemake raw output is unavailable.
    """
    species_file_map = {
        23.0: "lines_Iris_23.0.dat",
        25.0: "lines_Iris_25.0.dat",
        27.0: "lines_Iris_27.0.dat",
        29.0: "lines_Iris_29.0.dat",
        56.1: "lines_Iris_56.1.dat",
    }
    f = template_dir / species_file_map.get(species, "")
    if not f.exists():
        return pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])

    rows = []
    for line in f.read_text().splitlines():
        s = line.strip()
        if (not s) or s.lower().startswith("hfs lines"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            w = float(parts[0])
            sp = float(parts[1])
            ep = float(parts[2])
            lg = float(parts[3])
        except ValueError:
            continue
        rows.append({"wave_A": abs(w), "species": sp, "ep_eV": ep, "loggf": lg, "is_component": w < 0})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[np.isclose(df["species"], species)]
    if df.empty:
        return df
    df = df[np.abs(df["wave_A"] - wave_A) <= tol_A]
    return df.sort_values("wave_A").reset_index(drop=True)


def build_blends_linelists(
    target_lines: pd.DataFrame,
    base_linelist_df: pd.DataFrame,
    hfs_template_dir: Path,
    out_blends_dir: Path,
    linemake_runs_df: Optional[pd.DataFrame] = None,
    use_hfs_components: bool = True,
    line_file_prefix: str = "lines",
    window_A: float = 2.0,
    blend_margin_A: float = 0.15,
) -> pd.DataFrame:
    """
    Build MOOG-blends input line lists (anchor + all non-target lines as negative-wavelength blends).
    """
    out_blends_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, object]] = []

    for _, r in target_lines.iterrows():
        element = r["element"]
        species = float(r["species"])
        wave0 = float(r["wave_A"])
        ep0 = float(r["ep_eV"]) if np.isfinite(r["ep_eV"]) else 0.0
        loggf0 = float(r["loggf"]) if np.isfinite(r["loggf"]) else -99.0
        waals0 = float(r.get("waals_single_gamma_format", 0.0)) if np.isfinite(pd.to_numeric(r.get("waals_single_gamma_format", 0.0), errors="coerce")) else 0.0
        ew0 = float(r.get("observed_ew", np.nan)) if np.isfinite(pd.to_numeric(r.get("observed_ew", np.nan), errors="coerce")) else np.nan
        if not np.isfinite(ew0):
            ew0 = float(r.get("theoretical_ew", np.nan)) if np.isfinite(pd.to_numeric(r.get("theoretical_ew", np.nan), errors="coerce")) else 0.0
        tag = f"{element}_{wave0:.3f}"
        out_file = out_blends_dir / f"{line_file_prefix}_{tag}.dat"

        comp = pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])
        comp_qc_note = "no_hfs_mode"
        component_source = "no_hfs"
        if use_hfs_components:
            component_source = "none"
            if linemake_runs_df is not None and ("components_file" in linemake_runs_df.columns):
                mrun = linemake_runs_df[
                    (linemake_runs_df["element"].astype(str) == str(element))
                    & np.isclose(pd.to_numeric(linemake_runs_df["wave_A"], errors="coerce"), wave0, atol=0.003, rtol=0)
                    & np.isclose(pd.to_numeric(linemake_runs_df["species"], errors="coerce"), species, atol=0.06)
                ]
                if len(mrun):
                    cpath = Path(str(mrun.iloc[0].get("components_file", "")))
                    if cpath.exists():
                        try:
                            cdf = pd.read_csv(cpath)
                            for col in ["wave_A", "species", "ep_eV", "loggf"]:
                                cdf[col] = pd.to_numeric(cdf.get(col), errors="coerce")
                            cdf = cdf[
                                np.isfinite(cdf["wave_A"])
                                & np.isfinite(cdf["species"])
                                & np.isfinite(cdf["ep_eV"])
                                & np.isfinite(cdf["loggf"])
                            ].copy()
                            if len(cdf):
                                cdf["is_component"] = True
                                comp = cdf[["wave_A", "species", "ep_eV", "loggf", "is_component"]]
                                comp, comp_qc_note = _enforce_component_physics(
                                    comp,
                                    anchor_ep_eV=ep0,
                                    anchor_loggf=loggf0,
                                )
                                if len(comp) > 0:
                                    component_source = "linemake"
                        except Exception:
                            comp = pd.DataFrame(columns=["wave_A", "species", "ep_eV", "loggf", "is_component"])
                            comp_qc_note = "component_parse_exception"
            # Fallback: if linemake components are insufficient, use Iris template components.
            if len(comp) < 2:
                tdf = _read_hfs_template_components(hfs_template_dir, species, wave0, tol_A=0.5)
                if not tdf.empty:
                    tdf = tdf[tdf["is_component"] == True].copy()
                if not tdf.empty:
                    tdf, tnote = _enforce_component_physics(
                        tdf,
                        anchor_ep_eV=ep0,
                        anchor_loggf=loggf0,
                    )
                    if len(tdf) >= 2:
                        comp = tdf[["wave_A", "species", "ep_eV", "loggf", "is_component"]]
                        component_source = "Iris"
                        comp_qc_note = f"{comp_qc_note};fallback_Iris|{tnote}"
                    else:
                        comp_qc_note = f"{comp_qc_note};Iris_insufficient"
                else:
                    comp_qc_note = f"{comp_qc_note};Iris_not_found"
        n_comp_total = int(len(comp))
        # Scientific requirement: a meaningful HFS split needs >=2 components.
        has_valid_hfs = (not use_hfs_components) or (n_comp_total >= 2)

        # Default anchor uses the original parent line parameters.
        anchor_wave = wave0
        anchor_ep = ep0
        anchor_loggf = loggf0

        # For HFS mode, promote one real component as anchor and keep the rest as blends.
        # This avoids parent+components double counting in blends.
        comp_for_blends = comp.copy()
        if use_hfs_components and n_comp_total >= 2:
            anchor_idx = (pd.to_numeric(comp_for_blends["wave_A"], errors="coerce") - wave0).abs().idxmin()
            anchor_row = comp_for_blends.loc[anchor_idx]
            anchor_wave = float(anchor_row["wave_A"])
            anchor_ep = float(anchor_row["ep_eV"])
            anchor_loggf = float(anchor_row["loggf"])
            comp_for_blends = comp_for_blends.drop(index=anchor_idx).reset_index(drop=True)
        n_comp_blends = int(len(comp_for_blends))

        # Background lines: keep only physically overlapping neighbors around the target profile.
        comp_support = [anchor_wave, wave0]
        if len(comp_for_blends):
            comp_support.extend(pd.to_numeric(comp_for_blends["wave_A"], errors="coerce").dropna().tolist())
        local_min = float(np.nanmin(comp_support)) - float(blend_margin_A)
        local_max = float(np.nanmax(comp_support)) + float(blend_margin_A)
        # Keep compatibility with broad run window_A; local support is the real inclusion criterion.
        global_min = wave0 - float(window_A)
        global_max = wave0 + float(window_A)
        bg = base_linelist_df[
            (base_linelist_df["wave_A"] >= max(local_min, global_min))
            & (base_linelist_df["wave_A"] <= min(local_max, global_max))
            & np.isfinite(base_linelist_df["spectrum_moog_species"])
            & np.isfinite(base_linelist_df["lower_state_eV"])
            & np.isfinite(base_linelist_df["loggf"])
        ].copy()
        bg = (
            bg.sort_values("wave_A")
            .drop_duplicates(subset=["wave_A", "spectrum_moog_species", "lower_state_eV", "loggf"], keep="first")
            .copy()
        )

        lines_txt: List[str] = ["wavelength species lower_state_eV loggf damping d0 equivalent_width comment"]
        # anchor line (positive wavelength)
        lines_txt.append(
            f"{anchor_wave:10.3f}{species:10.1f}{anchor_ep:10.3f}{anchor_loggf:10.3f}{waals0:10.2f}{0.0:10.2f}{ew0:10.2f} {'':>10s}"
        )
        # HFS components (negative wavelength)
        if n_comp_blends > 0:
            for _, c in comp_for_blends.iterrows():
                lines_txt.append(
                    f"{-abs(float(c['wave_A'])):10.3f}{float(c['species']):10.1f}{float(c['ep_eV']):10.3f}{float(c['loggf']):10.3f}{0.0:10.2f}{0.0:10.2f}{0.0:10.2f} {'':>10s}"
                )
        # Background lines must be negative wavelength so blends does NOT fit EW on them.
        for _, b in bg.iterrows():
            bw = float(b["wave_A"])
            bsp = float(b["spectrum_moog_species"])
            if np.isclose(bw, wave0, atol=0.003, rtol=0) and np.isclose(bsp, species, atol=0.01):
                continue
            waals_b = float(b.get("waals_single_gamma_format", 0.0)) if np.isfinite(pd.to_numeric(b.get("waals_single_gamma_format", 0.0), errors="coerce")) else 0.0
            lines_txt.append(
                f"{-abs(bw):10.3f}{bsp:10.1f}{float(b['lower_state_eV']):10.3f}{float(b['loggf']):10.3f}{waals_b:10.2f}{0.0:10.2f}{0.0:10.2f} {'':>10s}"
            )

        out_file.write_text("\n".join(lines_txt) + "\n")
        manifest_rows.append(
            {
                "element": element,
                "wave_A": wave0,
                "species": species,
                "observed_ew": ew0,
                "logeps_sun_ref": float(r.get("logeps_sun_ref", np.nan))
                if np.isfinite(pd.to_numeric(r.get("logeps_sun_ref", np.nan), errors="coerce"))
                else np.nan,
                "d_xh_ref": float(r.get("d_xh_ref", np.nan))
                if np.isfinite(pd.to_numeric(r.get("d_xh_ref", np.nan), errors="coerce"))
                else np.nan,
                "line_file": str(out_file),
                "n_hfs_components": n_comp_total,
                "n_hfs_components_blends": n_comp_blends,
                "n_background_lines": int(len(bg)),
                "status": (
                    "ok"
                    if has_valid_hfs
                    else "invalid_no_hfs_component"
                ),
                "mode": "hfs" if use_hfs_components else "no_hfs",
                "component_qc": comp_qc_note,
                "component_source": component_source,
            }
        )
    return pd.DataFrame(manifest_rows)


def write_batch_par_from_template(
    template_path: Path,
    out_path: Path,
    lines_in: str,
    model_in: str,
    standard_out: str,
    summary_out: str,
    window_A: float = 2.0,
    step_A: float = 0.01,
    blen_species: Optional[float] = None,
) -> None:
    text = template_path.read_text().splitlines()
    new_lines: List[str] = []
    found_blen = False
    for ln in text:
        s = ln.strip()
        if s.startswith("standard_out"):
            new_lines.append(f"standard_out '{standard_out}'")
        elif s.startswith("summary_out"):
            new_lines.append(f"summary_out '{summary_out}'")
        elif s.startswith("lines_in"):
            new_lines.append(f"lines_in '{lines_in}'")
        elif s.startswith("model_in"):
            new_lines.append(f"model_in '{model_in}'")
        elif s.lower().startswith("blenlimits"):
            found_blen = True
            new_lines.append("blenlimits")
        elif found_blen and re.match(r"^\s*[-+0-9.]+\s+[-+0-9.]+\s+[-+0-9.]+\s*$", s):
            # blends template third field controls the solved species (e.g. 56 for Ba).
            # Use current target species to avoid hard-coded Ba behavior.
            sp = int(round(float(blen_species))) if blen_species is not None else int(float(s.split()[2]))
            new_lines.append(f"{window_A:.1f} {step_A:.2f} {sp}")
            found_blen = False
        else:
            new_lines.append(ln)
    out_path.write_text("\n".join(new_lines) + "\n")


def write_batch_par_abfind(
    out_path: Path,
    lines_in: str,
    model_in: str,
    standard_out: str = "moog.std",
    summary_out: str = "moog.sum",
) -> None:
    """
    Write a minimal MOOG abfind batch file.
    """
    lines = [
        "abfind",
        f"standard_out '{standard_out}'",
        f"summary_out '{summary_out}'",
        f"lines_in '{lines_in}'",
        f"model_in '{model_in}'",
        "atmosphere 1",
        "molecules 1",
        "lines 1",
        "damping 1",
        "flux/int 0",
        "freeform 0",
        "units 0",
        "plot 0",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def _find_moog_executable(preferred: Optional[str] = None) -> Optional[str]:
    if preferred and Path(preferred).exists():
        return preferred
    env = os.environ.get("MOOGSILENT_PATH")
    if env and Path(env).exists():
        return env
    for c in ["MOOGSILENT", "MOOG", "moogsilent"]:
        p = shutil.which(c)
        if p:
            return p
    return None


def run_moog_blends_batch(
    run_dir: Path,
    moog_bin: Optional[str] = None,
    timeout_sec: int = 120,
    dry_run: bool = False,
) -> Tuple[str, str, int]:
    """
    Execute MOOG blends with batch.par in run_dir.
    Returns (stdout, stderr, returncode).
    """
    moog = _find_moog_executable(moog_bin)
    if moog is None:
        return "", "MOOG executable not found", 127
    if dry_run:
        return "", f"dry_run: {moog}", 0

    # MOOG blends can ask several interactive prompts (e.g. molecular equilibrium redo).
    # Feed deterministic "no" answers to keep batch mode non-interactive.
    moog_stdin = os.environ.get("ISPEC_MOOG_STDIN", "n\nn\nn\nn\nn\n")

    prev = Path.cwd()
    try:
        os.chdir(run_dir)
        proc = subprocess.run(
            [moog],
            input=moog_stdin,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        _agent_debug_log(
            run_id="post-fix",
            hypothesis_id="H9",
            location="HFSBlends.py:run_moog_blends_batch",
            message="moog batch finished",
            data={
                "run_dir": str(run_dir),
                "returncode": int(proc.returncode),
                "stderr_has_eof": "End of file" in (proc.stderr or ""),
                "stderr_has_getasci": "Getasci.f" in (proc.stderr or ""),
            },
        )
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        os.chdir(prev)


def _strip_ansi(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _parse_fort99_groups(fort99_file: Path) -> List[Tuple[float, float]]:
    """
    Parse MOOG fort.99 diagnostic groups: "USING THE LINE GROUP IN THE RANGE: a b".
    """
    if not fort99_file.exists():
        return []
    txt = fort99_file.read_text(errors="ignore")
    groups: List[Tuple[float, float]] = []
    for m in re.finditer(
        r"USING THE LINE GROUP IN THE RANGE:\s*([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)",
        txt,
        flags=re.IGNORECASE,
    ):
        try:
            a = float(m.group(1))
            b = float(m.group(2))
            groups.append((min(a, b), max(a, b)))
        except Exception:
            continue
    return groups


def _assess_moog_run_quality(
    run_dir: Path,
    stdout: str,
    stderr: str,
    expected_wave_A: Optional[float] = None,
) -> Dict[str, object]:
    """
    Assess whether a blends run is scientifically usable for line-by-line abundance.
    """
    stdout_clean = _strip_ansi(stdout or "")
    stderr_clean = _strip_ansi(stderr or "")
    std_text = ""
    std_file = run_dir / "moog.std"
    if std_file.exists():
        std_text = std_file.read_text(errors="ignore")
    merged = "\n".join([stdout_clean, stderr_clean, std_text]).upper()
    non_converged = ("MAX OF 30 ITERATIONS" in merged) or ("I QUIT!" in merged and "ITERATION" in merged)

    groups = _parse_fort99_groups(run_dir / "fort.99")
    n_groups = len(groups)
    target_group_matches = 0
    if expected_wave_A is not None and np.isfinite(expected_wave_A):
        w0 = float(expected_wave_A)
        target_group_matches = int(sum(1 for a, b in groups if (a - 0.06) <= w0 <= (b + 0.06)))
    single_group_strict = (n_groups == 1) and (target_group_matches == 1)
    return {
        "non_converged": bool(non_converged),
        "n_groups": int(n_groups),
        "target_group_matches": int(target_group_matches),
        "single_group_strict": bool(single_group_strict),
    }


def parse_moog_blends_summary(
    summary_file: Path,
    expected_element: Optional[str] = None,
    expected_wave_A: Optional[float] = None,
    expected_species: Optional[float] = None,
) -> Dict[str, object]:
    """
    Robust parser for MOOG blends summary output.

    Strategy:
    1) If "Abundance Results for Species ..." block exists, use the block that matches
       expected element and pick anchor-like row near expected wave/species.
    2) Else parse explicit "element X: abundance = ..." lines with element check.
    3) Else fallback to generic abundance tokens, but only if expected element does not
       conflict with species blocks found in the file.
    """
    if not summary_file.exists():
        return {"status": "missing_summary", "logeps_hfs": np.nan, "note": "summary_out missing"}

    txt = summary_file.read_text(errors="ignore")
    lines = txt.splitlines()
    expected_elem_norm = str(expected_element).strip().lower() if expected_element else None

    # 1) Species-block parsing (same table shape as abfind output on many MOOG builds).
    species_pat = re.compile(r"Abundance Results for Species\s+([A-Za-z]+)\s+([IVX]+)", re.IGNORECASE)
    blocks: Dict[str, List[List[float]]] = {}
    current_elem: Optional[str] = None
    for ln in lines:
        sm = species_pat.search(ln)
        if sm:
            current_elem = sm.group(1).strip().lower()
            blocks.setdefault(current_elem, [])
            continue
        if current_elem is None:
            continue
        parts = ln.split()
        if len(parts) < 8:
            continue
        try:
            vals = list(map(float, parts[:8]))
        except ValueError:
            continue
        blocks[current_elem].append(vals)

    if blocks:
        if expected_elem_norm and expected_elem_norm not in blocks:
            avail = ",".join(sorted(blocks.keys()))
            return {
                "status": "unexpected_element_in_summary",
                "logeps_hfs": np.nan,
                "note": f"expected element={expected_element}, available={avail}",
            }
        pick_elem = expected_elem_norm if expected_elem_norm in blocks else sorted(blocks.keys())[0]
        rows = blocks.get(pick_elem, [])
        if rows:
            valid_rows = [
                v
                for v in rows
                if (
                    np.isfinite(v[6])
                    and abs(float(v[6])) < 99.0
                    and (not np.isfinite(v[7]) or abs(float(v[7])) < 99.0)
                )
            ]
            if not valid_rows:
                return {
                    "status": "unresolved",
                    "logeps_hfs": np.nan,
                    "note": "no_valid_abundance_rows_in_species_block",
                }
            if expected_wave_A is not None and expected_species is not None:
                best = min(
                    valid_rows,
                    key=lambda v: abs(v[0] - float(expected_wave_A))
                    + 10.0 * abs(v[1] - float(expected_species)),
                )
                if abs(best[0] - float(expected_wave_A)) > 0.06 or abs(best[1] - float(expected_species)) > 0.12:
                    return {
                        "status": "no_anchor_line_match",
                        "logeps_hfs": np.nan,
                        "note": (
                            f"best wave={best[0]:.3f}, species={best[1]:.2f}; "
                            f"expected wave={float(expected_wave_A):.3f}, species={float(expected_species):.2f}"
                        ),
                    }
                return {"status": "ok", "logeps_hfs": float(best[6]), "note": "parsed_species_block_anchor"}
            return {"status": "ok", "logeps_hfs": float(valid_rows[0][6]), "note": "parsed_species_block_first_row"}

    # 2) Explicit element abundance lines are diagnostics only for blends mode.
    em = re.search(
        r"element\s+([A-Za-z]+)\s*:\s*abundance\s*=\s*([+-]?\d+\.\d+)",
        txt,
        flags=re.IGNORECASE,
    )
    if em:
        elem = em.group(1).strip()
        val = float(em.group(2))
        return {
            "status": "unresolved",
            "logeps_hfs": np.nan,
            "note": f"parsed_element_line_not_allowed element={elem} abundance={val:.3f}",
        }

    # 3) Generic hints are also unresolved for strict anchor matching.
    patterns = [
        r"\baverage abundance\s*=\s*([+-]?\d+\.\d+)",
        r"\babundance\s*=\s*([+-]?\d+\.\d+)",
        r"\blog\s*eps\s*=\s*([+-]?\d+\.\d+)",
        r"\bA\(\w+\)\s*=\s*([+-]?\d+\.\d+)",
        r"\bmean\s*abundance\s*([+-]?\d+\.\d+)",
    ]
    for p in patterns:
        m = re.search(p, txt, flags=re.IGNORECASE)
        if m:
            return {
                "status": "unresolved",
                "logeps_hfs": np.nan,
                "note": f"generic_abundance_token_not_allowed value={float(m.group(1)):.3f}",
            }
    return {"status": "unresolved", "logeps_hfs": np.nan, "note": "no_species_anchor_match"}


def parse_moog_abfind_summary(
    summary_file: Path,
    expected_element: str,
    expected_wave_A: float,
    expected_species: float,
) -> Dict[str, object]:
    """
    Parse MOOG abfind summary and extract abundance for the target anchor line.
    """
    if not summary_file.exists():
        return {"status": "missing_summary", "logeps_hfs": np.nan, "note": "summary_out missing"}
    lines = summary_file.read_text(errors="ignore").splitlines()

    in_species_block = False
    numeric_rows: List[List[float]] = []
    species_pat = re.compile(r"Abundance Results for Species\s+([A-Za-z]+)\s+([IVX]+)", re.IGNORECASE)
    for ln in lines:
        sm = species_pat.search(ln)
        if sm:
            in_species_block = sm.group(1).strip().lower() == expected_element.lower()
            continue
        if not in_species_block:
            continue
        parts = ln.split()
        if len(parts) < 8:
            continue
        try:
            vals = list(map(float, parts[:8]))
        except ValueError:
            continue
        numeric_rows.append(vals)

    if not numeric_rows:
        return {
            "status": "no_target_species_block",
            "logeps_hfs": np.nan,
            "note": f"no abfind species block for {expected_element}",
        }

    # Format: wave, species, EP, logGF, EWin, logRWin, abund, delavg
    best = min(
        numeric_rows,
        key=lambda v: abs(v[0] - expected_wave_A) + 10.0 * abs(v[1] - expected_species),
    )
    if abs(best[0] - expected_wave_A) > 0.02 or abs(best[1] - expected_species) > 0.06:
        return {
            "status": "no_anchor_line_match",
            "logeps_hfs": np.nan,
            "note": (
                f"best line wave={best[0]:.3f}, species={best[1]:.2f}, "
                f"expected wave={expected_wave_A:.3f}, species={expected_species:.2f}"
            ),
        }
    return {"status": "ok", "logeps_hfs": float(best[6]), "note": "parsed_abfind_anchor"}


def _compute_fe_reference_from_ew_csv(ew_csv: Path) -> Tuple[float, float]:
    """Both returned entries are the common equal-line Fe reference.

    Keep the historical tuple API, but do not give neutral/ionized elements
    different Fe denominators from the final element plots.
    """
    if not ew_csv.exists():
        return np.nan, np.nan
    df = pd.read_csv(ew_csv)
    if "element" not in df.columns or "[X/H]" not in df.columns:
        return np.nan, np.nan
    from abundance_contracts import combine_species_statistics, species_token
    lines=df[df.element.map(species_token).str.startswith('Fe ')].copy()
    fe=float(combine_species_statistics(lines)['[X/H]']) if len(lines) else np.nan
    return fe, fe


def _ensure_moog_model_footer(model_file: Path, mh: float, microturbulence: float) -> None:
    """
    Ensure MOOG model file has the footer blocks required by MOOGSILENT:
    vmicro line, NATOMS, and NMOL list.

    iSpec's write_atmosphere(..., code='moog') writes the atmospheric layers but
    may omit these blocks; then MOOG fails with Inmodel.f EOF.
    """
    if not model_file.exists():
        return
    if not np.isfinite(mh) or not np.isfinite(microturbulence) or microturbulence < 0:
        raise ValueError("Invalid adopted HFS metallicity or microturbulence")
    lines = model_file.read_text(errors="ignore").splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith("NATOMS")]
    if indices:
        i = indices[0]
        lines[i-1] = f"  {float(microturbulence):.2f}"
        count = int(lines[i].split()[1])
        if count != 0 and abs(float(lines[i].split()[-1])-mh) > .011:
            raise ValueError("Cannot rescale an explicit chemical mixture by rewriting its MH header")
        lines[i] = f"NATOMS=   {count} {float(mh):.2f}"
    else:
        lines += [f"  {float(microturbulence):.2f}", f"NATOMS=   0 {float(mh):.2f}"]
    if not any(line.startswith("NMOL") for line in lines):
        lines += ["NMOL      28",
                  "  101.0   106.0   107.0   108.0   112.0  126.0",
                  "  606.0   607.0   608.0", "  707.0   708.0",
                  "  808.0   812.0   822.0   823.0   840.0",
                  "  10108.0 10820.0 60808.0",
                  "  6.1     7.1     8.1   12.1  20.1  22.1  23.1  26.1  40.1"]
    model_file.write_text("\n".join(lines)+"\n")


def _build_nohfs_from_q2(
    target_lines: pd.DataFrame,
    target: str,
    output_folder: str,
    fe1_ref: float,
    fe2_ref: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, r in target_lines.iterrows():
        element = str(r["element"])
        ion = str(r.get("ion", "1"))
        wave_A = float(r["wave_A"])
        species = float(r["species"])
        xh = pd.to_numeric(r.get("d_xh_nohfs_paired", np.nan), errors="coerce")
        logeps_star = pd.to_numeric(r.get("logeps_star_ref", np.nan), errors="coerce")
        fe_ref = fe2_ref if ion == "2" else fe1_ref
        xfe = (float(xh) - float(fe_ref)) if (np.isfinite(xh) and np.isfinite(fe_ref)) else np.nan
        status = "ok" if np.isfinite(xh) else "unresolved"
        note = "paired_star_noHFS_minus_sun_noHFS" if status == "ok" else "missing_q2_linebyline_match"
        rows.append(
            {
                "target": target,
                "mode": "no_hfs",
                "element": element,
                "ion": ion,
                "wave_A": wave_A,
                "species": species,
                "n_hfs_components": 0,
                "logeps_noHFS": float(logeps_star) if np.isfinite(logeps_star) else np.nan,
                "[X/H]_noHFS": float(xh) if np.isfinite(xh) else np.nan,
                "[X/Fe]_noHFS": float(xfe) if np.isfinite(xfe) else np.nan,
                "status": status,
                "note": note,
                "source_linelist": str(_infer_base_linelist(output_folder, target)),
                "line_file": "",
                "run_dir": "",
            }
        )
    return pd.DataFrame(rows).sort_values(["element", "wave_A"]).reset_index(drop=True)


def run_hfs_blends_for_target(
    target: str,
    output_folder: str,
    ispec_dir: str,
    model_in: str,
    ew_result_csv: Optional[str] = None,
    moog_bin: Optional[str] = None,
    linemake_bin: Optional[str] = None,
    window_A: float = 2.0,
    dry_run: bool = False,
    blends_batch_template: Optional[str] = None,
    require_linemake: bool = True,
    enable_implausible_hfs_shift_gate: Optional[bool] = None,
    *,
    use_external_input_ew: bool = False,
    vmic: Optional[float] = None,
    model_mh: Optional[float] = None,
    solar_label: Optional[str] = None,
    is_solar_reference: bool = False,
) -> pd.DataFrame:
    """Run HFS postprocessing and write HFS/tables/abundances_hfs_blends_<target>.csv. External EW mode preserves q2 anchor EWs. enable_implausible_hfs_shift_gate is compatibility-only; paired-baseline and provenance checks replace amplitude cuts."""
    if dry_run:
        print(
            f"[HFS] dry_run=True for {target}: MOOG will not execute, "
            "so logeps/[X/H]/[X/Fe] columns will be NaN by design."
        )

    model_path = Path(model_in).expanduser()
    if not dry_run and (not model_path.exists()):
        raise FileNotFoundError(
            "HFS model atmosphere file not found: "
            f"{model_path}. Please provide a valid MOOG model file via model_in."
        )

    out_root = Path(output_folder) / "HFS"
    paths = _ensure_dirs(out_root)

    template_dir = Path(ispec_dir) / "input" / "linelists" / "HFS"
    batch_template_path = Path(blends_batch_template).expanduser() if blends_batch_template else (template_dir / "batch.par")
    if not batch_template_path.exists():
        raise FileNotFoundError(
            "MOOG blends batch template not found: "
            f"{batch_template_path}. Please provide a valid blends template via blends_batch_template."
        )
    resolved_linemake_bin = linemake_bin or _find_linemake_executable()
    _agent_debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="HFSBlends.py:run_hfs_blends_for_target",
        message="linemake resolution before requirement check",
        data={
            "target": target,
            "linemake_bin_arg": linemake_bin,
            "resolved_linemake_bin": resolved_linemake_bin,
            "require_linemake": require_linemake,
        },
    )
    if require_linemake and (resolved_linemake_bin is None):
        _agent_debug_log(
            run_id="pre-fix",
            hypothesis_id="H4",
            location="HFSBlends.py:run_hfs_blends_for_target",
            message="raising missing linemake error",
            data={"target": target, "reason": "require_linemake_and_none"},
        )
        raise FileNotFoundError(
            "linemake executable not found (PATH, ISPEC_LINEMAKE_BIN, LINEMAKE_BIN). "
            "Please set ISPEC_LINEMAKE_BIN to your linemake binary path."
        )

    # 1) Target line selection from final linelist
    target_lines = collect_hfs_target_lines(
        output_folder=output_folder,
        target=target,
        use_external_input_ew=use_external_input_ew,
    )
    if target_lines.empty:
        raise RuntimeError(f"No HFS target lines found for {target}.")

    if vmic is None or model_mh is None:
        raise ValueError("HFS requires explicit vmic and model_mh from the adopted atmosphere")
    target_lines = attach_solar_baselines(
        target_lines, ispec_dir, solar_label, is_solar_reference
    )
    # The old amplitude gate is deliberately retired. Quality checks use the
    # paired baseline, components and MOOG convergence; amplitudes are audited.
    # persist mapping audit
    target_lines.to_csv(paths.tables / f"hfs_target_lines_{target}.csv", index=False)

    # 2) linemake generation (record only if unavailable)
    linemake_runs = generate_hfs_components_with_linemake(
        target_lines=target_lines,
        out_raw_dir=paths.linelists_raw,
        hfs_template_dir=template_dir,
        linemake_bin=resolved_linemake_bin,
        dry_run=dry_run,
    )
    linemake_runs.to_csv(paths.tables / f"linemake_runs_{target}.csv", index=False)

    # 3) Build blends line lists (HFS and no-HFS, same blends setup)
    base_linelist_df = _load_base_linelist(_infer_base_linelist(output_folder, target))
    manifest_hfs = build_blends_linelists(
        target_lines=target_lines,
        base_linelist_df=base_linelist_df,
        hfs_template_dir=template_dir,
        out_blends_dir=paths.linelists_blends,
        linemake_runs_df=linemake_runs,
        use_hfs_components=True,
        line_file_prefix="lines_hfs",
        window_A=window_A,
    )
    manifest_hfs.to_csv(paths.tables / f"blends_manifest_{target}.csv", index=False)

    # 4) Run blends per line
    ew_csv = Path(ew_result_csv) if ew_result_csv else Path(output_folder) / f"abundances_linebyline_{target}.csv"
    fe1_ref, fe2_ref = _compute_fe_reference_from_ew_csv(ew_csv)
    mh_ref = float(model_mh)

    def _lookup_target_line_row(element: str, wave_A: float, species: float) -> Optional[pd.Series]:
        tt = target_lines[
            (target_lines["element"].astype(str) == str(element))
            & np.isclose(pd.to_numeric(target_lines["wave_A"], errors="coerce"), float(wave_A), atol=0.003, rtol=0)
            & np.isclose(pd.to_numeric(target_lines["species"], errors="coerce"), float(species), atol=0.06)
        ]
        if len(tt):
            return tt.iloc[0]
        return None

    def _run_manifest(manifest_df: pd.DataFrame, mode: str) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        # Per-element line count in current HFS run.
        # Used to relax the plausibility gate for single-line elements.
        line_count_by_element = (
            manifest_df["element"].astype(str).value_counts().to_dict()
            if "element" in manifest_df.columns
            else {}
        )
        for _, m in manifest_df.iterrows():
            element = str(m["element"])
            wave_A = float(m["wave_A"])
            species = float(m["species"])
            ion = ELEMENT_CONFIG[element]["ion"]
            line_file = Path(str(m["line_file"]))
            tag = f"{element}_{wave_A:.3f}"
            run_dir = paths.runs / f"{mode}_{tag}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Strict science rule: HFS rows without components are invalid and excluded.
            if (mode == "hfs") and (int(pd.to_numeric(m.get("n_hfs_components"), errors="coerce")) < 2):
                rows.append(
                    {
                        "target": target,
                        "mode": mode,
                        "element": element,
                        "ion": ion,
                        "wave_A": wave_A,
                        "species": species,
                        "n_hfs_components": int(pd.to_numeric(m.get("n_hfs_components"), errors="coerce") or 0),
                        "logeps_HFS": np.nan,
                        "[X/H]_HFS": np.nan,
                        "[X/Fe]_HFS": np.nan,
                        "status": "invalid_no_hfs_component",
                        "note": "n_hfs_components<2",
                        "source_linelist": str(_infer_base_linelist(output_folder, target)),
                        "line_file": str(line_file),
                        "run_dir": str(run_dir),
                    }
                )
                continue

            line_in_run = run_dir / "lines.in"
            shutil.copyfile(line_file, line_in_run)
            model_in_run = run_dir / "model.in"
            if model_path.exists():
                shutil.copyfile(model_path, model_in_run)
                if not dry_run:
                    _ensure_moog_model_footer(model_in_run, mh=mh_ref, microturbulence=float(vmic))
            else:
                model_in_run = model_path

            batch_path = run_dir / "batch.par"
            std_name = "moog.std"
            sum_name = "moog.sum"
            write_batch_par_from_template(
                template_path=batch_template_path,
                out_path=batch_path,
                lines_in=line_in_run.name,
                model_in=model_in_run.name if isinstance(model_in_run, Path) else str(model_in_run),
                standard_out=std_name,
                summary_out=sum_name,
                window_A=window_A,
                blen_species=species,
            )

            stdout, stderr, rc = run_moog_blends_batch(
                run_dir=run_dir,
                moog_bin=moog_bin,
                timeout_sec=180,
                dry_run=dry_run,
            )
            (run_dir / "run.stdout.txt").write_text(stdout or "")
            (run_dir / "run.stderr.txt").write_text(stderr or "")
            run_qc = _assess_moog_run_quality(
                run_dir=run_dir,
                stdout=stdout,
                stderr=stderr,
                expected_wave_A=wave_A,
            )

            if dry_run:
                parse = {"status": "dry_run", "logeps_hfs": np.nan, "note": "dry_run"}
            else:
                parse = parse_moog_blends_summary(
                    run_dir / sum_name,
                    expected_element=element,
                    expected_wave_A=wave_A,
                    expected_species=species,
                )
            logeps = parse["logeps_hfs"]

            # The parser checks MOOG's abundance field and species/anchor;
            # a large abundance difference is not evidence of a scale error.
            fe_ref = fe2_ref if ion == "2" else fe1_ref

            if dry_run:
                status = "dry_run"
                note = parse.get("note", "")
            else:
                # parsed_element_line / generic tokens are forced unresolved by parser.
                if parse.get("status") == "ok" and rc == 0:
                    status = "ok"
                elif parse.get("status") == "ok" and rc != 0:
                    status = f"run_rc_{rc}|ok"
                else:
                    status = "unresolved"
                note = parse.get("note", "")
            if rc != 0 and stderr:
                note = f"{note}; {stderr.strip()[:200]}"
            if run_qc.get("non_converged", False):
                status = "unresolved"
                note = f"{note}; moog_non_converged_max_iterations".strip("; ")

            # Single-line fallback: when an element has only one target line in this run,
            # allow element-level abundance token only for strict single-group converged runs.
            if (
                mode == "hfs"
                and status == "unresolved"
                and rc == 0
                and (f"parsed_element_line_not_allowed element={element} " in str(note))
                and (not run_qc.get("non_converged", False))
                and run_qc.get("single_group_strict", False)
            ):
                mm = re.search(r"abundance=([+-]?\d+\.\d+)", str(note))
                if mm:
                    fallback_logeps = float(mm.group(1))
                    if np.isfinite(fallback_logeps) and abs(fallback_logeps) < 99.0:
                        logeps = fallback_logeps
                        status = "ok"
                        note = (
                            f"{note}; fallback_single_line_element"
                            f"|n_groups={int(run_qc.get('n_groups', 0))}"
                        )

            # Amplitude differences cannot certify or invalidate HFS physics.
            # In particular, never compare star(noHFS)-Sun(HFS) to both-HFS.
            tref = _lookup_target_line_row(element, wave_A, species)
            pair = tref.to_dict() if tref is not None else {}
            baseline = pd.to_numeric(pair.get("d_xh_nohfs_paired", np.nan), errors="coerce")
            solar_hfs = pd.to_numeric(pair.get("logeps_sun_hfs", np.nan), errors="coerce")
            if is_solar_reference or np.isfinite(solar_hfs):
                xh_hfs = float(logeps)-float(solar_hfs) if np.isfinite(logeps) else np.nan
                xfe_hfs = xh_hfs-fe_ref if np.isfinite(fe_ref) else np.nan
            else:
                xh_hfs = np.nan
                xfe_hfs = np.nan
            try:
                signature = component_signature(line_in_run, int(m.get("n_hfs_components", 0)))
            except (OSError, ValueError):
                signature = ""

            rows.append(
                {
                    "target": target,
                    "mode": mode,
                    "element": element,
                    "ion": ion,
                    "wave_A": wave_A,
                    "species": species,
                    "n_hfs_components": int(pd.to_numeric(m.get("n_hfs_components"), errors="coerce") or 0),
                    "component_signature": signature,
                    **profile_quality(output_folder, target, f'{element} {ion}', wave_A,
                                      pair.get('observed_ew', np.nan)),
                    "solar_component_signature": pair.get("solar_component_signature", ""),
                    "solar_pair_status": pair.get("solar_pair_status", "missing"),
                    "solar_label_used": pair.get("solar_label_used", ""),
                    "logeps_sun_nohfs": pair.get("logeps_sun_nohfs", np.nan),
                    "logeps_sun_hfs": solar_hfs,
                    "[X/H]_noHFS_paired": baseline,
                    "differential_hfs_correction": xh_hfs-baseline,
                    "moog_non_converged": bool(run_qc.get("non_converged", False)),
                    "logeps_HFS": logeps,
                    "[X/H]_HFS": xh_hfs,
                    "[X/Fe]_HFS": xfe_hfs,
                    "hfs_source_used": (
                        "Iris_retry"
                        if "retry_with_Iris_ok" in str(note)
                        else str(m.get("component_source", "unknown"))
                    ),
                    "status": status,
                    "note": note,
                    "source_linelist": str(_infer_base_linelist(output_folder, target)),
                    "line_file": str(line_file),
                    "run_dir": str(run_dir),
                }
            )
        return pd.DataFrame(rows).sort_values(["element", "wave_A"]).reset_index(drop=True)

    def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        xh_valid = pd.to_numeric(df["[X/H]_HFS"], errors="coerce").where(df["status"] == "ok")
        xfe_valid = pd.to_numeric(df["[X/Fe]_HFS"], errors="coerce").where(df["status"] == "ok")
        df2 = df.copy()
        df2["_xh_valid"] = xh_valid
        df2["_xfe_valid"] = xfe_valid
        grp = df2.groupby(["target", "element", "ion"], as_index=False).agg(
            n_lines_total=("wave_A", "size"),
            n_lines=("status", lambda s: int(np.sum(pd.Series(s).astype(str) == "ok"))),
            n_ok=("status", lambda s: int(np.sum(pd.Series(s).astype(str) == "ok"))),
            xh_mean=("_xh_valid", "mean"),
            xh_std=("_xh_valid", "std"),
            xfe_mean=("_xfe_valid", "mean"),
            xfe_std=("_xfe_valid", "std"),
        )
        grp["xh_sem"] = grp.xh_std / np.sqrt(grp.n_ok.where(grp.n_ok > 1))
        grp["xfe_sem"] = grp.xfe_std / np.sqrt(grp.n_ok.where(grp.n_ok > 1))
        return grp

    decisions = {}
    review_file = Path(output_folder) / 'HFS' / 'review_decisions.csv'
    if review_file.exists():
        review = pd.read_csv(review_file)
        for _, row in review.iterrows():
            decision = str(row['decision']).strip().lower()
            reason = row.get('reason', '')
            if decision not in {'accept', 'reject'} or pd.isna(reason) or not str(reason).strip():
                raise ValueError('HFS review decisions require accept/reject and a reason')
            key = (str(row['element']).strip(), round(float(row['wave_A']), 3))
            if key in decisions:
                raise ValueError(f'Duplicate HFS review decision: {key}')
            decisions[key] = decision
    out_hfs = apply_pair_quality(_run_manifest(manifest_hfs, mode="hfs"), is_solar_reference, decisions)
    out_hfs_csv = paths.tables / f"abundances_hfs_blends_{target}.csv"
    out_hfs.to_csv(out_hfs_csv, index=False)
    sum_hfs = _build_summary(out_hfs)
    if not sum_hfs.empty:
        sum_hfs.to_csv(paths.tables / f"abundances_hfs_blends_{target}_summary.csv", index=False)

    out_nohfs = _build_nohfs_from_q2(
        target_lines=target_lines,
        target=target,
        output_folder=output_folder,
        fe1_ref=fe1_ref,
        fe2_ref=fe2_ref,
    )
    # Match the accepted both-HFS set exactly; no independent baseline selection.
    accepted = {(str(r.element), round(float(r.wave_A), 3)) for _, r in out_hfs[out_hfs.status == "ok"].iterrows()}
    out_nohfs = out_nohfs[[ (str(r.element), round(float(r.wave_A), 3)) in accepted for _, r in out_nohfs.iterrows() ]].copy()
    out_nohfs_csv = paths.tables / f"abundances_nohfs_{target}.csv"
    out_nohfs.to_csv(out_nohfs_csv, index=False)
    # Backward-compatible alias path, content still comes from q2 line-by-line.
    out_nohfs.to_csv(paths.tables / f"abundances_nohfs_blends_{target}.csv", index=False)
    # Build no-HFS summary from no-HFS naming.
    if not out_nohfs.empty:
        xh_valid = pd.to_numeric(out_nohfs["[X/H]_noHFS"], errors="coerce").where(out_nohfs["status"] == "ok")
        xfe_valid = pd.to_numeric(out_nohfs["[X/Fe]_noHFS"], errors="coerce").where(out_nohfs["status"] == "ok")
        dno = out_nohfs.copy()
        dno["_xh_valid"] = xh_valid
        dno["_xfe_valid"] = xfe_valid
        sum_nohfs = dno.groupby(["target", "element", "ion"], as_index=False).agg(
            n_lines_total=("wave_A", "size"),
            n_lines=("status", lambda s: int(np.sum(pd.Series(s).astype(str) == "ok"))),
            n_ok=("status", lambda s: int(np.sum(pd.Series(s).astype(str) == "ok"))),
            xh_mean=("_xh_valid", "mean"),
            xh_std=("_xh_valid", "std"),
            xfe_mean=("_xfe_valid", "mean"),
            xfe_std=("_xfe_valid", "std"),
        )
    else:
        sum_nohfs = pd.DataFrame(columns=['target','element','ion','n_lines_total','n_lines','n_ok','xh_mean','xh_std','xfe_mean','xfe_std'])
    if not sum_nohfs.empty:
        sum_nohfs["xh_sem"] = sum_nohfs.xh_std / np.sqrt(sum_nohfs.n_ok.where(sum_nohfs.n_ok > 1))
        sum_nohfs["xfe_sem"] = sum_nohfs.xfe_std / np.sqrt(sum_nohfs.n_ok.where(sum_nohfs.n_ok > 1))
        sum_nohfs.to_csv(paths.tables / f"abundances_nohfs_{target}_summary.csv", index=False)
        sum_nohfs.to_csv(paths.tables / f"abundances_nohfs_blends_{target}_summary.csv", index=False)
    else:
        sum_nohfs.to_csv(paths.tables / f"abundances_nohfs_{target}_summary.csv", index=False)
        sum_nohfs.to_csv(paths.tables / f"abundances_nohfs_blends_{target}_summary.csv", index=False)

    return out_hfs


def run_hfs_blends_for_targets(
    targets: Sequence[Dict[str, str]],
    ispec_dir: str,
    model_in: str,
    ew_csv_name_tpl: str = "abundances_linebyline_{target}.csv",
    dry_run: bool = False,
    blends_batch_template: Optional[str] = None,
    require_linemake: bool = True,
    *,
    use_external_input_ew: bool = False,
) -> pd.DataFrame:
    """
    Batch wrapper for multiple targets.
    targets item format:
      {"target": "HIP10303", "subdir": "MegantestSample",
       "vmic": 1.0, "model_mh": 0.1, "model_in": "/path/to/model"}
    Atmospheric parameters must be supplied per target; no common vt fallback.
    """
    rows = []
    for t in targets:
        name = t["target"]
        sub = t.get("subdir")
        if sub:
            out_folder = os.path.join(ispec_dir, "output", sub, name)
        else:
            out_folder = os.path.join(ispec_dir, "output", name)
        ew_csv = os.path.join(out_folder, ew_csv_name_tpl.format(target=name))
        try:
            df = run_hfs_blends_for_target(
                target=name,
                output_folder=out_folder,
                ispec_dir=ispec_dir,
                model_in=t.get('model_in', model_in),
                vmic=float(t['vmic']),
                model_mh=float(t['model_mh']),
                solar_label=t.get('solar_label'),
                is_solar_reference=bool(t.get('is_solar_reference', False)),
                ew_result_csv=ew_csv,
                dry_run=dry_run,
                blends_batch_template=blends_batch_template,
                require_linemake=require_linemake,
                use_external_input_ew=use_external_input_ew,
            )
            ok = int((df["status"] == "ok").sum()) if "status" in df.columns else 0
            rows.append({"target": name, "n_rows": len(df), "n_ok": ok, "status": "ok"})
        except Exception as e:
            rows.append({"target": name, "n_rows": 0, "n_ok": 0, "status": f"error: {e}"})
    return pd.DataFrame(rows)


def build_ew_vs_hfs_diagnostics(
    target: str,
    output_folder: str,
    hfs_csv: Optional[str] = None,
    ew_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Build EW-vs-HFS comparison table and quick diagnostic figure.
    """
    out_folder = Path(output_folder) / "HFS"
    tables = out_folder / "tables"
    figs = out_folder / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    hfs_path = Path(hfs_csv) if hfs_csv else tables / f"abundances_hfs_blends_{target}_summary.csv"
    ew_path = Path(ew_csv) if ew_csv else tables / f"abundances_nohfs_{target}_summary.csv"

    if not hfs_path.exists() or not ew_path.exists():
        return pd.DataFrame(), None

    h = pd.read_csv(hfs_path)
    e = pd.read_csv(ew_path)
    if "xh_mean" in e.columns:
        e = e.rename(columns={"xh_mean": "[X/H]", "xfe_mean": "[X/Fe]"})
    if h.empty or e.empty:
        return pd.DataFrame(), None

    # normalize labels
    e = e.copy()
    if "[X/Fe]" not in e.columns:
        # Reconstruct [X/Fe] from [X/H] and Fe references if absent
        from abundance_contracts import combine_species_statistics, species_token
        iron=e[e.element.map(species_token).str.startswith('Fe ')]
        fe=float(combine_species_statistics(iron)['[X/H]']) if len(iron) else np.nan

        def _ew_xfe(row):
            xh = pd.to_numeric(row.get("[X/H]"), errors="coerce")
            if not np.isfinite(xh):
                return np.nan
            return xh - fe if np.isfinite(fe) else np.nan

        e["[X/Fe]"] = e.apply(_ew_xfe, axis=1)

    e["element_base"] = e["element"].astype(str).str.replace(" I", "", regex=False).str.replace(" II", "", regex=False).str.split().str[0]
    h["element_base"] = h["element"].astype(str).str.split().str[0]

    h2 = h[["element_base", "xh_mean", "xfe_mean", "n_lines"]].copy().rename(columns={"n_lines": "n_lines_HFS"})
    e2 = e[["element_base", "element", "[X/H]", "[X/Fe]", "n_lines"]].copy().rename(columns={"n_lines": "n_lines_EW"})
    m = pd.merge(h2, e2, on="element_base", how="left")
    m = m.rename(
        columns={
            "xh_mean": "[X/H]_HFS_mean",
            "xfe_mean": "[X/Fe]_HFS_mean",
            "[X/H]": "[X/H]_EW",
            "[X/Fe]": "[X/Fe]_EW",
            "n_lines_HFS": "n_lines_HFS",
            "n_lines_EW": "n_lines_EW",
        }
    )
    if "[X/H]_EW" in m.columns:
        m["d[X/H]_HFS-EW"] = pd.to_numeric(m["[X/H]_HFS_mean"], errors="coerce") - pd.to_numeric(m["[X/H]_EW"], errors="coerce")
    if "[X/Fe]_EW" in m.columns:
        m["d[X/Fe]_HFS-EW"] = pd.to_numeric(m["[X/Fe]_HFS_mean"], errors="coerce") - pd.to_numeric(m["[X/Fe]_EW"], errors="coerce")

    out_cmp = tables / f"ew_vs_hfs_{target}.csv"
    m.to_csv(out_cmp, index=False)

    fig_path: Optional[Path] = None
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(figs / ".mplconfig"))

        plot_elems = list(ELEMENT_CONFIG.keys())
        def _col_one(elem_base: str, col: str) -> float:
            hit = m.loc[m["element_base"].astype(str) == elem_base, col]
            if hit.empty:
                return np.nan
            return float(pd.to_numeric(hit.iloc[0], errors="coerce"))

        xh_ew = [_col_one(el, "[X/H]_EW") for el in plot_elems]
        xh_hfs = [_col_one(el, "[X/H]_HFS_mean") for el in plot_elems]
        xfe_ew = [_col_one(el, "[X/Fe]_EW") for el in plot_elems]
        xfe_hfs = [_col_one(el, "[X/Fe]_HFS_mean") for el in plot_elems]

        any_ew = any(np.isfinite(v) for v in xh_ew)
        any_hfs = any(np.isfinite(v) for v in xh_hfs)
        have_delta = "d[X/H]_HFS-EW" in m.columns and m["d[X/H]_HFS-EW"].notna().any()

        if any_ew or any_hfs:
            x = np.arange(len(plot_elems))
            w = 0.35
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
            ax0, ax1 = axes
            ax0.bar(x - w / 2, xh_ew, width=w, label="EW [X/H]", alpha=0.85)
            ax0.bar(x + w / 2, xh_hfs, width=w, label="HFS [X/H]", alpha=0.85)
            ax0.axhline(0, color="k", ls="--", lw=0.8)
            ax0.set_xticks(x)
            ax0.set_xticklabels(plot_elems)
            ax0.set_ylabel("[X/H]")
            ax0.grid(axis="y", alpha=0.3)
            ax0.legend(fontsize=8)

            ax1.bar(x - w / 2, xfe_ew, width=w, label="EW [X/Fe]", alpha=0.85)
            ax1.bar(x + w / 2, xfe_hfs, width=w, label="HFS [X/Fe]", alpha=0.85)
            ax1.axhline(0, color="k", ls="--", lw=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(plot_elems)
            ax1.set_ylabel("[X/Fe]")
            ax1.grid(axis="y", alpha=0.3)
            ax1.legend(fontsize=8)

            notes: List[str] = []
            if not any_hfs:
                notes.append("HFS abundances missing (dry run or MOOG parse failed); HFS bars may be empty")
            if have_delta:
                notes.append("See ew_vs_hfs_delta_*.png for Δ(HFS−EW)")
            fig.suptitle(f"{target}: EW vs HFS (per-element means)", y=1.02)
            if notes:
                fig.text(0.5, 0.02, " · ".join(notes), ha="center", fontsize=9, color="0.35")

            plt.tight_layout(rect=(0, 0.08, 1, 0.96))
            fig_path = figs / f"ew_vs_hfs_{target}.png"
            plt.savefig(fig_path, dpi=160)
            plt.close(fig)

            if have_delta:
                dd = m[pd.notna(m["d[X/H]_HFS-EW"])].copy()
                if len(dd) > 0:
                    plt.figure(figsize=(10, 4), dpi=160)
                    xi = np.arange(len(dd))
                    plt.axhline(0, color="k", ls="--", lw=1)
                    plt.plot(xi, dd["d[X/H]_HFS-EW"], "o-", label="d[X/H] HFS-EW")
                    if "d[X/Fe]_HFS-EW" in dd.columns and dd["d[X/Fe]_HFS-EW"].notna().any():
                        plt.plot(xi, dd["d[X/Fe]_HFS-EW"], "s-", label="d[X/Fe] HFS-EW")
                    plt.xticks(xi, dd["element_base"], rotation=45, ha="right")
                    plt.ylabel("Difference (dex)")
                    plt.title(f"{target}: Δ(HFS − EW)")
                    plt.grid(alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(figs / f"ew_vs_hfs_delta_{target}.png", dpi=160)
                    plt.close()
    except Exception:
        fig_path = None

    return m, (str(fig_path) if fig_path else None)

