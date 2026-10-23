"""Solar external-EW Part I: combine pipeline template Fe with Bedell2018 Table1 Sun EWs. Requires a fitted-linemask template from a prior standard solar Part I run or the current notebook session."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Mapping, Optional

import numpy as np

from bedell_table1_ew_pipeline import build_fitted_linemasks_from_bedell_table1, resolve_linemask_template


def _default_table1_path(ispec_dir: str, pipeline_options: Optional[Mapping[str, Any]]) -> str:
    if pipeline_options:
        p = pipeline_options.get("external_ew_table_path")
        if p and os.path.isfile(str(p)):
            return str(p)
        p2 = pipeline_options.get("bedell2018_table1_path")
        if p2 and os.path.isfile(str(p2)):
            return str(p2)
    return os.path.join(ispec_dir, "output", "Bedell_method_stats", "Bedell2018_table1.csv")


def _sun_template_path(
    ispec_dir: str,
    target: str,
    pipeline_options: Optional[Mapping[str, Any]],
    output_folder: Optional[str],
) -> str:
    if pipeline_options:
        for k in ("bedell_ew_sun_linemask_template", "bedell_ew_linemask_template"):
            p = pipeline_options.get(k)
            if p and os.path.isfile(str(p)):
                return str(p)
    if output_folder:
        loc = os.path.join(
            str(output_folder).rstrip(os.sep),
            "linemasks",
            f"{target}_melendez2014_star_fitted_linemasks.txt",
        )
        if os.path.isfile(loc):
            return loc
    p0 = os.path.join(
        ispec_dir,
        "output",
        "Sun",
        target,
        "linemasks",
        f"{target}_melendez2014_star_fitted_linemasks.txt",
    )
    if os.path.isfile(p0):
        return p0
    return resolve_linemask_template(target, ispec_dir, None)


def run_sun_bedell_table1_external_part1(
    ispec,
    *,
    ispec_dir: str,
    target: str,
    output_folder: str,
    spectrum_norm_path: str,
    output_linelist_path: str,
    linelist_target_path: str,
    star_spectrum: np.ndarray,
    pipeline_options: Optional[Mapping[str, Any]] = None,
    copy_linelist_from_base: bool = True,
) -> tuple:
    """Build and save solar linemasks using Table1 Sun EWs, retaining template Fe. Return star_linemasks, iron_star_linemasks, atomic_linelist, recover_star_linemasks, and zeroed_star_linemasks."""
    if copy_linelist_from_base:
        shutil.copyfile(output_linelist_path, linelist_target_path)
        print(f"Copied {output_linelist_path} -> {linelist_target_path}")
    atomic_linelist_file = linelist_target_path
    print(atomic_linelist_file)

    logging.info("Table1 external EW (Sun): preparing atomic linelist range...")
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file)
    atomic_linelist = ispec.read_atomic_linelist(
        atomic_linelist_file,
        wave_base=float(np.min(star_spectrum["waveobs"])),
        wave_top=float(np.max(star_spectrum["waveobs"])),
    )
    print(
        f"Successfully read {len(atomic_linelist)} lines from the linelist in range "
        f"{np.min(star_spectrum['waveobs'])} to {np.max(star_spectrum['waveobs'])}."
    )
    print(
        "Linelist wavelength range:",
        np.min(atomic_linelist["wave_nm"]),
        "to",
        np.max(atomic_linelist["wave_nm"]),
        "nm.",
    )

    po = pipeline_options or {}
    table1_path = _default_table1_path(ispec_dir, po)
    template_path = _sun_template_path(ispec_dir, target, po, output_folder)

    linemask_output_folder = os.path.join(output_folder, "linemasks")
    os.makedirs(linemask_output_folder, exist_ok=True)
    out_fitted = os.path.join(
        linemask_output_folder,
        f"{target}_melendez2014_star_fitted_linemasks.txt",
    )

    star_linemasks = build_fitted_linemasks_from_bedell_table1(
        ispec,
        table1_path=table1_path,
        column_key="Sun",
        template_path=template_path,
        target=target,
        ispec_dir=ispec_dir,
        out_fitted_path=out_fitted,
        fallback_pipeline_fe_ew_if_no_table1_fe=po.get(
            "bedell_fallback_pipeline_fe_ew_if_no_table1_fe", True
        ),
        use_pipeline_fe_only=po.get("bedell_use_pipeline_fe_only", True),
        merge_missing_table1_lines=po.get("bedell_merge_missing_table1_lines", True),
    )

    iron = (star_linemasks["element"] == "Fe 1") | (star_linemasks["element"] == "Fe 2")
    iron_star_linemasks = star_linemasks[iron]

    ispec.write_line_regions(
        star_linemasks,
        os.path.join(linemask_output_folder, f"{target}_melendez2014_star_linemasks.txt"),
    )
    ispec.write_line_regions(
        iron_star_linemasks,
        os.path.join(linemask_output_folder, f"{target}_melendez2014_star_fe_linemasks.txt"),
    )
    recover_star_linemasks = ispec.read_line_regions(out_fitted)
    zeroed_star_linemasks = ispec.reset_fitted_data_fields(star_linemasks)
    ispec.write_line_regions(
        zeroed_star_linemasks,
        os.path.join(
            linemask_output_folder,
            f"{target}_melendez2014_star_zeroed_fitted_linemasks.txt",
        ),
        extended=True,
    )
    ispec.write_atomic_linelist(
        star_linemasks,
        os.path.join(linemask_output_folder, f"{target}_melendez2014_star_atomic_linelist.txt"),
    )

    return star_linemasks, iron_star_linemasks, atomic_linelist, recover_star_linemasks, zeroed_star_linemasks
