from __future__ import annotations

import argparse
import os
from typing import Any

from inlist import PIPELINE_OPTIONS_OVERRIDE, target as default_target
from run_context import apply_context_globals
from atmos_lbl_pipeline import (
    apply_deleted_lines_to_atomic_linelist,
    compute_fe_abundance,
    export_q2_dump_csv,
    fast_apply_lineqa_to_linemasks,
    filter_and_save_linemasks,
    load_or_prepare_inputs,
    plot_fe_abundance_trends,
    plot_gaussian_fits_by_element,
    prepare_manual_line_edit_tables,
    run_find_linemasks_part1,
    run_q2_fit_and_export_inputs,
    save_q2_dump,
    step_log,
)


def run_pipeline(target_name: str) -> dict[str, Any]:
    apply_context_globals(
        globals(),
        target=target_name,
        pipeline_options_override=PIPELINE_OPTIONS_OVERRIDE,
        prepare_io=True,
    )

    run_find_linemasks = PIPELINE_OPTIONS.get("run_find_linemasks", True)
    use_external_ew = PIPELINE_OPTIONS.get("input_has_external_ews", False)
    run_part1_spectrum = run_find_linemasks and (not use_external_ew)

    step_log("0", f"Start 3a CLI pipeline for target={target}")
    step_log(
        "0",
        f"Flags: run_find_linemasks={run_find_linemasks}, use_external_ew={use_external_ew}",
    )

    _inputs = load_or_prepare_inputs(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        output_linelist_path=output_linelist_path,
        linelist_target_path=linelist_target_path,
        spectrum_norm_path=spectrum_norm_path,
        ispec_dir=ispec_dir,
        row_target=row_target,
        PIPELINE_OPTIONS=PIPELINE_OPTIONS,
        RUN_PART1_SPECTRUM=run_part1_spectrum,
        run_find_linemasks=run_find_linemasks,
    )
    star_spectrum = _inputs["star_spectrum"]
    atomic_linelist = _inputs["atomic_linelist"]
    star_linemasks = _inputs["star_linemasks"]
    recover_star_linemasks = _inputs["recover_star_linemasks"]
    linemasks = _inputs["linemasks"]
    _bedell_q2_tar_ew = _inputs["bedell_q2_tar_ew"]
    _bedell_q2_sun_ew = _inputs["bedell_q2_sun_ew"]

    if run_part1_spectrum:
        star_linemasks = run_find_linemasks_part1(
            ispec=ispec,
            star_spectrum=star_spectrum,
            atomic_linelist=atomic_linelist,
            from_resolution=from_resolution,
            to_resolution=to_resolution,
            min_depth=0.05,
            max_depth=1.00,
        )
        _saved = filter_and_save_linemasks(
            ispec=ispec,
            target=target,
            output_folder=output_folder,
            star_linemasks=star_linemasks,
            ew_min=10,
            ew_max=100,
        )
        star_linemasks = _saved["star_linemasks"]
        recover_star_linemasks = _saved["recover_star_linemasks"]
        linemask_output_folder = _saved["linemask_output_folder"]

        plot_gaussian_fits_by_element(
            star_spectrum=star_spectrum,
            linemasks=star_linemasks,
            output_dir=os.path.join(output_folder, "figs_Fe_GaussianFits"),
            element_filter=["Fe 1", "Fe 2"],
            prefix="FeFit",
            w_range=0.25,
            progress_every=10,
        )

        prepare_manual_line_edit_tables(
            output_folder=output_folder,
            target=target,
            output_linelist_path=output_linelist_path,
            PIPELINE_OPTIONS=PIPELINE_OPTIONS,
        )

        apply_deleted_lines_to_atomic_linelist(
            ispec=ispec,
            output_folder=output_folder,
            linelist_target_path=linelist_target_path,
        )

        _fast_lq = fast_apply_lineqa_to_linemasks(
            ispec=ispec,
            target=target,
            output_folder=output_folder,
            star_linemasks=star_linemasks,
            ew_min=10,
            ew_max=100,
        )
        star_linemasks = _fast_lq["star_linemasks"]
        recover_star_linemasks = _fast_lq["recover_star_linemasks"]
        step_log("7", f"Fast LineQA kept lines: {len(star_linemasks)}")
    else:
        linemask_output_folder = os.path.join(output_folder, "linemasks")
        step_log("2-7", "Part I skipped; using existing fitted linemasks.")

    lm_path = os.path.join(linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.txt")
    linemasks = ispec.read_line_regions(lm_path)
    linemasks = linemasks[linemasks["wave_nm"] > 0]
    linemasks = linemasks[linemasks["ew"] > 0]

    _fe_abund = compute_fe_abundance(
        ispec=ispec,
        linemasks=linemasks,
        ispec_dir=ispec_dir,
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_MH=initial_MH,
        initial_feh_err=globals().get("initial_feh_err"),
    )
    linemasks = _fe_abund["linemasks"]
    x_over_h = _fe_abund["x_over_h"]

    plot_fe_abundance_trends(
        linemasks=linemasks,
        x_over_h=x_over_h,
        use_external_ew=use_external_ew,
        show_plots=PIPELINE_OPTIONS.get("show_fe_trend_plots", True),
    )

    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    _q2 = run_q2_fit_and_export_inputs(
        ispec=ispec,
        linemasks=linemasks,
        target=target,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        solar_label=solar_label,
        solar_lines_csv=solar_lines_csv,
        instrument_name=instrument_name,
        from_resolution=from_resolution,
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_MH=initial_MH,
        initial_vmic=initial_vmic,
        PIPELINE_OPTIONS=PIPELINE_OPTIONS,
        bedell_q2_tar_ew=_bedell_q2_tar_ew,
        bedell_q2_sun_ew=_bedell_q2_sun_ew,
    )

    _dump = save_q2_dump(
        ispec=ispec,
        output_atmos_params_dumpfile_path=output_atmos_params_dumpfile_path,
        solution_row=_q2["row"],
        linemasks_fe=_q2["linemasks_fe"],
        target=target,
        solution_csv=_q2["solution_csv"],
    )
    step_log(
        "10",
        f"Reference parameters: Teff = {initial_teff}, logg = {initial_logg}, [Fe/H] = {initial_MH}",
    )
    csv_atmos_q2 = export_q2_dump_csv(
        ispec=ispec,
        target=target,
        output_atmos_params_dumpfile_path=output_atmos_params_dumpfile_path,
    )
    step_log("12", f"Step_atmospheric_paramters completed. q2 dump={_dump['dump_file_q2']}")
    step_log("12", f"Step_atmospheric_paramters completed. q2 csv={csv_atmos_q2}")

    return {"q2": _q2, "dump": _dump, "csv": csv_atmos_q2}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 3a atmospheric-parameter LBL pipeline in one command."
    )
    parser.add_argument(
        "--target",
        default=default_target,
        help=f"Target star name (default from inlist.py: {default_target})",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_pipeline(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
