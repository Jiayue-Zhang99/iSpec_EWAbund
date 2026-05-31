from __future__ import annotations

import copy
import os
from pathlib import Path


def resolve_ispec_dir() -> str:
    """
    Resolve iSpec root directory.

    Priority:
    1) ISPEC_DIR environment variable
    2) parent directory of Code/
    """
    env = os.environ.get("ISPEC_DIR")
    if env:
        p = Path(env).expanduser().resolve()
    else:
        p = Path(__file__).resolve().parent.parent
    s = str(p)
    return s if s.endswith(os.sep) else s + os.sep


LOG_LEVEL = "info"

INPUT_SPECTRA_INFO_REL = os.path.join("input", "spectra", "input_spectra_info.xlsx")
SOLAR_ABUND_INFO_REL = os.path.join("input", "solar_abund_instru.csv")
SOLAR_LINES_INFO_REL = os.path.join("input", "solar_lines_instru.csv")
TC_TABLE_REL = os.path.join("input", "Tc_elements_Lodders2003_table8.csv")

SUN_ATMOS_LITERATURE_DEFAULT = {
    "teff": 5772.0,
    "logg": 4.44,
    "MH": 0.0,
    "alpha": 0.0,
    "vmic": 1.0,
}
SUN_AFE_REFERENCE_DEFAULT = 7.45  # Grevesse 2007 solar A(Fe)

DEFAULT_PIPELINE_OPTIONS = {
    # ----- input forms -----
    # True: input is already normalized, False: input is not normalized
    "input_is_already_normalized": True,
    # ----- external EW -----
    # True: input has external EW, False: input does not have external EW
    "input_has_external_ews": False,
    # external EW table path
    "external_ew_table_path": None,
    # bedell2018 table1 path
    "bedell2018_table1_path": None,
    # bedell2018 table1 EW column name
    "bedell_ew_column": None,
    # bedell2018 table1 linemask template path
    "bedell_ew_linemask_template": None,
    # True: EW only from pipeline, False: EW from table1
    "bedell_use_pipeline_fe_only": True,
    # True: merge missing lines from table1 to template, False: not merge
    "bedell_merge_missing_table1_lines": True,
    # True: fallback to pipeline EW if no table1 EW, False: not fallback
    "bedell_fallback_pipeline_fe_ew_if_no_table1_fe": True, # True: fallback to pipeline EW if no table1 EW, False: not fallback
    # ----- abundance mode -----
    # "average_differential" means to use the average differential mode
    # "line_by_line" means to use the line by line mode
    "abundance_differential_mode": "line_by_line",  
    # ----- find linemasks -----
    # True: run find linemasks, False: not run find linemasks
    "run_find_linemasks": True,
    # ----- apply lineqa in abundance step -----
    # True: apply lineqa in abundance step, False: not apply lineqa in abundance step
    "apply_lineqa_in_abundance_step": True,
    # ----- persist filtered linemasks for abund -----
    # True: persist filtered linemasks for abund, False: not persist filtered linemasks for abund
    "persist_filtered_linemasks_for_abund": True,
    "line_review_enabled": True,
    # ----- interactive EW quality control page -----
    # "inline" means to use the inline mode in jupyter notebook
    # "browser" means to use the browser mode to open the quality control page in the browser
    "line_review_ui_mode": "browser", 
    # fixed reason options (good means to keep, not write into deleted table)
    "line_review_reason_options": [
        "good",
        "continuum_low",
        "continuum_high",
        "strong_line",
        "crowded_region",
        "low_EW_value",
        "other",
    ],
    # ----- plotting behavior -----
    # True: show Fe trend figures interactively (plt.show)
    # False: do not pop up Fe trend figures
    "show_fe_trend_plots": True,
    # ----- plot HFS with external EW -----
    # False (default): when input_has_external_ews=True, fig06 only uses q2 result without HFS, no 06b/06c
    # True: same as input_has_external_ews=False (read HFS/no-HFS summary, plot 06b/06c)
    # Note: this option is only used when input_has_external_ews=True
    "plot_hfs_with_external_ews": True,
    # ----- 0_Sun_abundance part3+ controls -----
    # "lbl" means to use the line-by-line atmosphere iteration method
    # "glb" means to use the global atmosphere iteration method
    "sun_atmos_iter_method": "lbl",
    # True: plot all elements at abundance step
    "sun_plot_all_elements": True,
    # True: run HFS blends and update outputs
    "sun_run_hfs_blends": True,
    # True: dry run HFS blends and update outputs
    "sun_hfs_dry_run": False,
}

ATOMIC_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, 
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, 
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, 
    "Sr": 38, "Y": 39, "Zr": 40, "Ba": 56, "La": 57, "Ce": 58, "Nd": 60, "Eu": 63,
}


def default_pipeline_options() -> dict:
    return copy.deepcopy(DEFAULT_PIPELINE_OPTIONS)


def merge_pipeline_options(override: dict | None = None) -> dict:
    out = default_pipeline_options()
    if override:
        out.update(override)
    return out
