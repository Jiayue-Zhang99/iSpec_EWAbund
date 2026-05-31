from __future__ import annotations

import argparse
from typing import Any

from inlist import PIPELINE_OPTIONS_OVERRIDE, target as default_target
from run_context import apply_context_globals
from atmos_lbl_pipeline import step_log
from abund_plot import run_step5_ele_abundance_plots


def run_pipeline(target_name: str) -> dict[str, Any]:
    apply_context_globals(
        globals(),
        target=target_name,
        pipeline_options_override=PIPELINE_OPTIONS_OVERRIDE,
        prepare_io=True,
    )

    step_log("0", f"Start 5 CLI pipeline for target={target}")
    step_log("1", "Running abundance plot pipeline from summary tables.")
    result = run_step5_ele_abundance_plots(
        target=target,
        output_abundances_result_path_linebyline_q2=output_abundances_result_path_linebyline_q2,
        abundance_plot_folder=abundance_plot_folder,
        output_folder=output_folder,
        ref_name=REF_NAME,
        ref_star=REF_STAR,
        pipeline_options=PIPELINE_OPTIONS,
        ispec_dir=ispec_dir,
    )
    step_log("2", f"Step_element_abundance_plot completed. folder={result['plot_dir']}")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run step 5 elemental abundance plotting in one command."
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
