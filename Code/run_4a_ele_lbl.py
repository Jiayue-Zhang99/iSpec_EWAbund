from __future__ import annotations

import argparse
from typing import Any

from inlist import PIPELINE_OPTIONS_OVERRIDE, target as default_target
from run_context import apply_context_globals
from atmos_lbl_pipeline import step_log
from abund_lbl_pipeline import (
    apply_lineqa_for_abundance,
    compute_lbl_differential_and_export,
    load_star_and_atmosphere_context,
    plot_all_element_gaussian_fits,
    prepare_abundance_engine,
    run_hfs_blends_and_update_outputs,
    run_nonfe_line_review,
)


def run_pipeline(target_name: str, *, run_hfs: bool = True, hfs_dry_run: bool = False) -> dict[str, Any]:
    apply_context_globals(
        globals(),
        target=target_name,
        pipeline_options_override=PIPELINE_OPTIONS_OVERRIDE,
        prepare_io=True,
    )

    run_find_linemasks = PIPELINE_OPTIONS.get("run_find_linemasks", True)
    use_external_ew = PIPELINE_OPTIONS.get("input_has_external_ews", False)
    run_part1_spectrum = run_find_linemasks and (not use_external_ew)
    plot_all_elements = True

    step_log("0", f"Start 4a CLI pipeline for target={target}")
    step_log(
        "0",
        f"Flags: run_find_linemasks={run_find_linemasks}, use_external_ew={use_external_ew}, run_hfs={run_hfs}",
    )

    ctx4a = load_star_and_atmosphere_context(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        spectrum_norm_path=spectrum_norm_path,
        output_atmos_params_dumpfile_path=output_atmos_params_dumpfile_path,
        initial_teff=initial_teff,
        initial_logg=initial_logg,
        initial_MH=initial_MH,
    )
    star_spectrum = ctx4a["star_spectrum"]
    linemask_output_folder = ctx4a["linemask_output_folder"]
    linemasks = ctx4a["linemasks"]
    teff = ctx4a["teff"]
    logg = ctx4a["logg"]
    mh = ctx4a["mh"]
    alpha = ctx4a["alpha"]
    vmic = ctx4a["vmic"]

    plot_res = plot_all_element_gaussian_fits(
        star_spectrum=star_spectrum,
        linemasks=linemasks,
        output_folder=output_folder,
        run_part1_spectrum=run_part1_spectrum,
        plot_all_elements=plot_all_elements,
        progress_every=10,
    )
    step_log("2", f"Element Gaussian diagnostics ready: dir={plot_res['base_output_dir']}")

    run_nonfe_line_review(
        output_folder=output_folder,
        target=target,
        run_part1_spectrum=run_part1_spectrum,
        plot_all_elements=plot_all_elements,
        pipeline_options=PIPELINE_OPTIONS,
    )

    linemasks = apply_lineqa_for_abundance(
        ispec=ispec,
        linemasks=linemasks,
        output_folder=output_folder,
        linemask_output_folder=linemask_output_folder,
        target=target,
        pipeline_options=PIPELINE_OPTIONS,
    )

    linemasks = linemasks[linemasks["wave_nm"] > 0]
    linemasks = linemasks[linemasks["ew"] > 0]
    engine = prepare_abundance_engine(
        ispec=ispec,
        ispec_dir=ispec_dir,
        teff=teff,
        logg=logg,
        mh=mh,
        alpha=alpha,
        code="moog",
    )

    lbl = compute_lbl_differential_and_export(
        ispec=ispec,
        linemasks=linemasks,
        target=target,
        ispec_dir=ispec_dir,
        instrument_name=instrument_name,
        from_resolution=from_resolution,
        solar_lines_csv=globals().get("solar_lines_csv"),
        atmosphere_layers=engine["atmosphere_layers"],
        solar_abundances=engine["solar_abundances"],
        teff=teff,
        logg=logg,
        mh=mh,
        alpha=alpha,
        vmic=vmic,
        code=engine["code"],
        output_abundances_result_path_linebyline_q2=output_abundances_result_path_linebyline_q2,
    )
    step_log("6", f"4a output summary path: {lbl['summary_path']}")

    hfs = run_hfs_blends_and_update_outputs(
        ispec=ispec,
        target=target,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        atmosphere_layers=engine["atmosphere_layers"],
        teff=teff,
        logg=logg,
        mh=mh,
        output_abundances_result_path_linebyline_q2=output_abundances_result_path_linebyline_q2,
        pipeline_options=PIPELINE_OPTIONS,
        run_hfs_blends=run_hfs,
        dry_run=hfs_dry_run,
    )
    if hfs.get("ran"):
        step_log("7", f"HFS detailed report saved: {hfs.get('report_path')}")

    step_log("8", "Step_element_abundance completed.")
    return {"lbl": lbl, "hfs": hfs}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 4a elemental abundance LBL pipeline in one command."
    )
    parser.add_argument(
        "--target",
        default=default_target,
        help=f"Target star name (default from inlist.py: {default_target})",
    )
    parser.add_argument(
        "--skip-hfs",
        action="store_true",
        help="Skip HFS blends stage.",
    )
    parser.add_argument(
        "--hfs-dry-run",
        action="store_true",
        help="Run HFS stage in dry-run mode.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_pipeline(args.target, run_hfs=(not args.skip_hfs), hfs_dry_run=args.hfs_dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
