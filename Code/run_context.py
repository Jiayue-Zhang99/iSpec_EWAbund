from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Any

import pandas as pd

from config import (
    ATOMIC_Z,
    INPUT_SPECTRA_INFO_REL,
    LOG_LEVEL,
    SOLAR_ABUND_INFO_REL,
    SOLAR_LINES_INFO_REL,
    TC_TABLE_REL,
    merge_pipeline_options,
    resolve_ispec_dir,
)


EMPTY_SOLAR_COLUMNS = [
    "instrument",
    "resolution",
    "solar_label",
    "element",
    "wave_A",
    "EP",
    "loggf",
    "ew_mA",
    "logeps_sun",
    "[X/H]_sun_Grevesse",
]


def configure_root_logger(level: str = LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(level.upper()))
    return logger


def bootstrap_ispec(ispec_dir: str):
    if os.path.abspath(ispec_dir) not in sys.path:
        sys.path.insert(0, os.path.abspath(ispec_dir))
    import ispec  # noqa

    return ispec


def _default_external_ew_path(ispec_dir: str, pipeline_options: dict) -> str:
    return pipeline_options.get("bedell2018_table1_path") or os.path.join(
        ispec_dir, "output", "Bedell_method_stats", "Bedell2018_table1.csv"
    )


def _build_input_output_paths(
    *,
    ispec_dir: str,
    target: str,
    note: str,
    row_target: pd.Series,
    pipeline_options: dict,
) -> dict[str, Any]:
    unit_is_Angstrom = False
    if note == "megantest":
        input_spectra_path = (
            ispec_dir
            + "input/spectra/HARPS_combined_solar_twin_spectra/"
            + target
            + "_n.fits"
        )
        output_folder = ispec_dir + "output/MegantestSample/" + target + "/"
        spectrum_norm_path = output_folder + target + "_n.fits"
    else:
        input_spectra_path = (
            ispec_dir
            + "input/spectra/"
            + target
            + "/ADP.2014-11-05T09_35_00.720_378.2-691.3nm.fits"
        )
        unit_is_Angstrom = True
        output_folder = ispec_dir + "output/" + target + "/"
        spectrum_norm_path = (
            output_folder
            + "spectra_"
            + target
            + "_Bedell_HARPS_"
            + str(row_target["target_name2"])
            + "_n.fits"
        )

    if pipeline_options.get("input_has_external_ews"):
        output_folder = os.path.join(ispec_dir, "output", "withEW_Bedell", target) + os.sep
        if note == "megantest":
            spectrum_norm_path = output_folder + target + "_n.fits"
        else:
            spectrum_norm_path = (
                output_folder
                + "spectra_"
                + target
                + "_Bedell_HARPS_"
                + str(row_target["target_name2"])
                + "_n.fits"
            )
        if not pipeline_options.get("external_ew_table_path"):
            pipeline_options["external_ew_table_path"] = _default_external_ew_path(
                ispec_dir, pipeline_options
            )

    return {
        "input_spectra_path": input_spectra_path,
        "output_folder": output_folder,
        "spectrum_norm_path": spectrum_norm_path,
        "unit_is_Angstrom": unit_is_Angstrom,
    }


def _read_default_run_context_override() -> dict[str, Any]:
    """
    Read run-context override from inlist.py if available.
    This keeps notebook calls unchanged while allowing users to tune paths/flags in inlist.
    """
    try:
        from inlist import RUN_CONTEXT_OVERRIDE  # type: ignore

        if isinstance(RUN_CONTEXT_OVERRIDE, dict):
            return dict(RUN_CONTEXT_OVERRIDE)
    except Exception:
        pass
    return {}


def _load_solar_context(
    *,
    ispec_dir: str,
    instrument_name: str,
    from_resolution: int | float,
    pipeline_options: dict,
) -> dict[str, Any]:
    solar_abund_info_csv = os.path.join(ispec_dir, SOLAR_ABUND_INFO_REL)
    df_solar_info = pd.read_csv(solar_abund_info_csv)
    mask_exact = (df_solar_info["instrument"] == instrument_name) & (
        df_solar_info["resolution"] == from_resolution
    )
    if not df_solar_info[mask_exact].empty:
        row_sun = df_solar_info[mask_exact].iloc[0]
    else:
        row_sun = df_solar_info[df_solar_info["instrument"] == instrument_name].iloc[0]

    solar_label = row_sun["solar_label"]
    solar_resolution = row_sun["resolution"]

    if pipeline_options.get("input_has_external_ews"):
        solar_lines_csv = os.path.join(
            ispec_dir,
            "output",
            "withEW_Bedell",
            "solar_lines_instru_Bedell2018_Sun.csv",
        )
    else:
        solar_lines_csv = os.path.join(ispec_dir, SOLAR_LINES_INFO_REL)

    if os.path.isfile(solar_lines_csv):
        solar_lines_master = pd.read_csv(solar_lines_csv)
    elif pipeline_options.get("input_has_external_ews"):
        solar_lines_master = pd.DataFrame(columns=EMPTY_SOLAR_COLUMNS)
    else:
        solar_lines_master = pd.read_csv(solar_lines_csv)

    solar_lines_instr = solar_lines_master[
        (solar_lines_master["instrument"] == instrument_name)
        & (solar_lines_master["resolution"] == from_resolution)
    ].copy()

    return {
        "solar_abund_info_csv": solar_abund_info_csv,
        "df_solar_info": df_solar_info,
        "row_sun": row_sun,
        "solar_label": solar_label,
        "solar_resolution": solar_resolution,
        "solar_lines_csv": solar_lines_csv,
        "solar_lines_master": solar_lines_master,
        "solar_lines_instr": solar_lines_instr,
    }


def load_run_context(
    target: str,
    pipeline_options_override: dict | None = None,
    run_context_override: dict | None = None,
) -> dict[str, Any]:
    ispec_dir = resolve_ispec_dir()
    pipeline_options = merge_pipeline_options(pipeline_options_override)
    user_ctx_override = _read_default_run_context_override()
    if run_context_override:
        user_ctx_override.update(run_context_override)

    input_spectra_info = os.path.join(ispec_dir, INPUT_SPECTRA_INFO_REL)
    df_input_info = pd.read_excel(input_spectra_info)
    row_target = df_input_info.loc[df_input_info["target_name"] == target].iloc[0]
    note = user_ctx_override.get("note") or row_target["note"]

    basic_paths = _build_input_output_paths(
        ispec_dir=ispec_dir,
        target=target,
        note=note,
        row_target=row_target,
        pipeline_options=pipeline_options,
    )
    for k in ("input_spectra_path", "spectrum_norm_path", "unit_is_Angstrom", "output_folder"):
        v = user_ctx_override.get(k)
        if v is not None:
            basic_paths[k] = v

    instrument_name = row_target["instrument"]
    from_resolution = row_target["R"]
    solar_ctx = _load_solar_context(
        ispec_dir=ispec_dir,
        instrument_name=instrument_name,
        from_resolution=from_resolution,
        pipeline_options=pipeline_options,
    )

    output_folder = basic_paths["output_folder"]
    solar_resolution = solar_ctx["solar_resolution"]

    ctx: dict[str, Any] = {
        "ispec_dir": ispec_dir,
        "LOG_LEVEL": LOG_LEVEL,
        "logger": configure_root_logger(LOG_LEVEL),
        "target": target,
        "PIPELINE_OPTIONS": pipeline_options,
        "input_spectra_info": input_spectra_info,
        "df_input_info": df_input_info,
        "row_target": row_target,
        "note": note,
        "instrument_name": instrument_name,
        "from_resolution": from_resolution,
        **basic_paths,
        **solar_ctx,
        "Obs_time_raw": row_target["Start of Obs"],
        "target_ra_raw": row_target["RA"],
        "target_dec_raw": row_target["Dec"],
        "to_resolution": solar_resolution,
        "LBV_correction": False,
        "LRV_correction": False,
        "normalization_whole_template": user_ctx_override.get(
            "normalization_whole_template", True
        ),
        "normalization_segmts_template": user_ctx_override.get(
            "normalization_segmts_template", True
        ),
        "normalization_segmts_polynomy": user_ctx_override.get(
            "normalization_segmts_polynomy", True
        ),
        "output_linelist_folder": ispec_dir + "input/linelists/linelist_melendez2014/",
        "vald_atomic_lines_path": ispec_dir
        + "input/linelists/transitions/VALD.300_1100nm/atomic_lines copy.tsv",
        "initial_teff": row_target["Teff"],
        "initial_logg": row_target["logg"],
        "initial_MH": row_target["[Fe/H]"],
        "fit_width": 0.2,
        "ATOMIC_Z": ATOMIC_Z,
        "TC_PATH": os.path.join(ispec_dir, TC_TABLE_REL),
        "REF_STAR": row_target["target_name2"],
        "REF_NAME": "Bedell2018",
    }

    ctx["input_linelist_path"] = (
        ctx["output_linelist_folder"] + "linelist_melendez2014.txt"
    )
    ctx["converted_linelist_path"] = (
        ctx["output_linelist_folder"] + "linelist_melendez2014_ispec.tsv"
    )
    ctx["output_linelist_path"] = ctx["converted_linelist_path"]
    ctx["linelist_target_path"] = output_folder + "linelist_for_" + target + ".tsv"
    _dump_candidates = [
        output_folder + f"atmos_params_{target}_q2.dump",
        output_folder + f"atmos_params_{target}.dump",
        output_folder + f"atmos_params_{target}_h1.dump",
    ]
    _existing_dump = next((p for p in _dump_candidates if os.path.isfile(p)), None)
    ctx["output_atmos_params_dumpfile_path"] = _existing_dump or _dump_candidates[1]
    ctx["output_abundances_result_path_linebyline_q2"] = (
        output_folder + f"abundances_linebyline_q2_{target}.csv"
    )
    ctx["output_abundances_result_path_linebyline"] = (
        output_folder + f"abundances_linebyline_{target}.csv"
    )
    ctx["output_abundances_result_path_Grevesse"] = (
        output_folder + f"abundances_Grevesse_{target}.csv"
    )
    ctx["output_abundances_result_path"] = output_folder + f"abundances_{target}.csv"
    ctx["abundance_plot_folder"] = output_folder + "abundance_plots/"

    return ctx


def prepare_outputs(ctx: dict[str, Any]) -> None:
    output_folder = ctx["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    input_spectra_path = ctx["input_spectra_path"]
    spectrum_norm_path = ctx["spectrum_norm_path"]
    target = ctx["target"]
    note = ctx["note"]
    pipeline_options = ctx["PIPELINE_OPTIONS"]
    ispec_dir = ctx["ispec_dir"]

    if note == "megantest" and os.path.isfile(input_spectra_path):
        dst = os.path.join(output_folder, os.path.basename(input_spectra_path))
        if not os.path.isfile(dst):
            shutil.copy2(input_spectra_path, dst)

    if pipeline_options.get("input_has_external_ews"):
        if os.path.isfile(input_spectra_path):
            shutil.copy2(input_spectra_path, spectrum_norm_path)

        dst_lm = os.path.join(output_folder, "linemasks")
        os.makedirs(dst_lm, exist_ok=True)
        src_lm_dirs = [
            os.path.join(ispec_dir, "output", "MegantestSample", target, "linemasks"),
            os.path.join(ispec_dir, "output", target, "linemasks"),
        ]
        for name in ("Lines_EW_modified.tsv", "Lines_deleted.tsv"):
            src = None
            for d in src_lm_dirs:
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    src = p
                    break
            outp = os.path.join(dst_lm, name)
            if src is not None:
                shutil.copy2(src, outp)
            elif not os.path.isfile(outp):
                if name == "Lines_EW_modified.tsv":
                    pd.DataFrame(
                        columns=["element", "wave_A", "wave_nm", "loggf", "ew_mA_new"]
                    ).to_csv(outp, sep="\t", index=False)
                else:
                    pd.DataFrame(columns=["element", "wave_nm"]).to_csv(
                        outp, sep="\t", index=False
                    )


def apply_context_globals(
    namespace: dict[str, Any],
    *,
    target: str,
    pipeline_options_override: dict | None = None,
    run_context_override: dict | None = None,
    prepare_io: bool = True,
) -> dict[str, Any]:
    ctx = load_run_context(target, pipeline_options_override, run_context_override)
    if prepare_io:
        prepare_outputs(ctx)
    ctx["ispec"] = bootstrap_ispec(ctx["ispec_dir"])
    namespace.update(ctx)
    return ctx
