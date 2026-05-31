from __future__ import annotations

# Run-level input list (user-adjustable settings).
# This file is intended to hold the knobs users frequently tune.

target = "HIP1954"

# Only put values you want to override from config.DEFAULT_PIPELINE_OPTIONS.
PIPELINE_OPTIONS_OVERRIDE: dict = {
    # Disable interactive Fe-trend plot popups in 3a CLI/notebook helper calls.
    "show_fe_trend_plots": False,
    "input_has_external_ews": False,
    "sun_atmos_iter_method": "lbl",
    "sun_plot_all_elements": True,
    "sun_run_hfs_blends": True,
    "sun_hfs_dry_run": False,
}

# Optional context overrides used by run_context.load_run_context().
# Keep any item as None to use the default auto-derived value.
RUN_CONTEXT_OVERRIDE: dict = {
    # Force note branch if needed ("megantest" or other value). None -> from input_spectra_info.xlsx
    "note": None,

    # Paths users may frequently tweak
    # (especially when note != "megantest" and custom local files are used).
    "input_spectra_path": None,
    "spectrum_norm_path": None,
    "unit_is_Angstrom": None,
    "output_folder": None,

    # Step1 normalization knobs
    "normalization_whole_template": True,
    "normalization_segmts_template": True,
    "normalization_segmts_polynomy": True,
}
