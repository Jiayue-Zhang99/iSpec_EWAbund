from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def step_log(step: str, message: str, *, progress: str | None = None) -> None:
    if progress:
        print(f"[STEP {step}][{progress}] {message}")
    else:
        print(f"[STEP {step}] {message}")


@contextmanager
def _quiet_matplotlib_fonts():
    """
    Temporarily silence matplotlib font-manager warnings.
    This is useful for environments without Ubuntu serif fonts.
    """
    font_logger = logging.getLogger("matplotlib.font_manager")
    old_level = font_logger.level
    old_family = plt.rcParams.get("font.family")
    old_serif = plt.rcParams.get("font.serif")
    font_logger.setLevel(logging.ERROR)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "DejaVu Sans", "Times New Roman"]
    try:
        yield
    finally:
        font_logger.setLevel(old_level)
        if old_family is not None:
            plt.rcParams["font.family"] = old_family
        if old_serif is not None:
            plt.rcParams["font.serif"] = old_serif


def load_or_prepare_inputs(
    *,
    ispec,
    target: str,
    output_folder: str,
    output_linelist_path: str,
    linelist_target_path: str,
    spectrum_norm_path: str,
    ispec_dir: str,
    row_target: pd.Series,
    PIPELINE_OPTIONS: dict[str, Any],
    RUN_PART1_SPECTRUM: bool,
    run_find_linemasks: bool,
):
    bedell_q2_tar_ew = None
    bedell_q2_sun_ew = None

    if PIPELINE_OPTIONS.get("input_has_external_ews"):
        from bedell_table1_ew_pipeline import (
            build_fitted_linemasks_from_bedell_table1,
            fe_ew_override_frames_for_q2,
        )

        step_log("1", "External EW mode enabled (Bedell Table1).")
        linemask_output_folder = os.path.join(output_folder, "linemasks")
        os.makedirs(linemask_output_folder, exist_ok=True)
        shutil.copyfile(output_linelist_path, linelist_target_path)
        atomic_linelist_file = linelist_target_path

        table1_path = PIPELINE_OPTIONS.get("external_ew_table_path") or os.path.join(
            ispec_dir, "output", "Bedell_method_stats", "Bedell2018_table1.csv"
        )
        ew_col = PIPELINE_OPTIONS.get("bedell_ew_column") or str(row_target["target_name2"])
        tpl = PIPELINE_OPTIONS.get("bedell_ew_linemask_template")
        fitted_out = os.path.join(
            linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.txt"
        )

        fe_only = PIPELINE_OPTIONS.get("bedell_use_pipeline_fe_only", True)
        merge_t1 = PIPELINE_OPTIONS.get("bedell_merge_missing_table1_lines", True)
        fb_fe = PIPELINE_OPTIONS.get("bedell_fallback_pipeline_fe_ew_if_no_table1_fe", True)
        star_linemasks = build_fitted_linemasks_from_bedell_table1(
            ispec=ispec,
            table1_path=table1_path,
            column_key=ew_col,
            template_path=tpl,
            target=target,
            ispec_dir=ispec_dir,
            out_fitted_path=fitted_out,
            fallback_pipeline_fe_ew_if_no_table1_fe=fb_fe,
            use_pipeline_fe_only=fe_only,
            merge_missing_table1_lines=merge_t1,
        )
        bedell_q2_tar_ew, bedell_q2_sun_ew = fe_ew_override_frames_for_q2(
            table1_path,
            ew_col,
            "Sun",
            fallback_pipeline_fe_ew_if_no_table1_fe=fb_fe,
            use_pipeline_fe_only=fe_only,
        )

        recover_star_linemasks = star_linemasks
        elem_col = np.array([str(e).strip() for e in star_linemasks["element"]])
        iron_star_linemasks = star_linemasks[(elem_col == "Fe 1") | (elem_col == "Fe 2")]
        star_spectrum = ispec.read_spectrum(spectrum_norm_path)
        atomic_linelist = ispec.read_atomic_linelist(
            atomic_linelist_file,
            wave_base=np.min(star_spectrum["waveobs"]),
            wave_top=np.max(star_spectrum["waveobs"]),
        )
        linemasks = star_linemasks
        linemasks = linemasks[linemasks["wave_nm"] > 0]
        linemasks = linemasks[linemasks["ew"] > 0]
        step_log(
            "1",
            f"Prepared external-EW linemasks: raw={len(star_linemasks)}, Fe={len(iron_star_linemasks)}, filtered={len(linemasks)}",
        )
        return {
            "star_spectrum": star_spectrum,
            "atomic_linelist": atomic_linelist,
            "star_linemasks": star_linemasks,
            "iron_star_linemasks": iron_star_linemasks,
            "recover_star_linemasks": recover_star_linemasks,
            "linemasks": linemasks,
            "atomic_linelist_file": atomic_linelist_file,
            "bedell_q2_tar_ew": bedell_q2_tar_ew,
            "bedell_q2_sun_ew": bedell_q2_sun_ew,
        }

    if RUN_PART1_SPECTRUM:
        step_log("1", "Standard mode: loading spectrum and linelist.")
        star_spectrum = ispec.read_spectrum(spectrum_norm_path)
        shutil.copyfile(output_linelist_path, linelist_target_path)
        atomic_linelist_file = linelist_target_path
        atomic_linelist = ispec.read_atomic_linelist(
            atomic_linelist_file,
            wave_base=np.min(star_spectrum["waveobs"]),
            wave_top=np.max(star_spectrum["waveobs"]),
        )
        step_log("1", f"Atomic lines in range: {len(atomic_linelist)}")
        return {
            "star_spectrum": star_spectrum,
            "atomic_linelist": atomic_linelist,
            "atomic_linelist_file": atomic_linelist_file,
            "star_linemasks": None,
            "iron_star_linemasks": None,
            "recover_star_linemasks": None,
            "linemasks": None,
            "bedell_q2_tar_ew": None,
            "bedell_q2_sun_ew": None,
        }

    linemask_output_folder = os.path.join(output_folder, "linemasks")
    fitted_lm = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.txt"
    )
    if not os.path.isfile(fitted_lm):
        raise FileNotFoundError(
            f"run_find_linemasks=False but missing fitted linemasks:\n  {fitted_lm}"
        )
    step_log("1", "Part I skipped; loading existing fitted linemasks.")
    shutil.copyfile(output_linelist_path, linelist_target_path)
    atomic_linelist_file = linelist_target_path
    star_linemasks = ispec.read_line_regions(fitted_lm)
    recover_star_linemasks = star_linemasks
    elem_col = np.array([str(e).strip() for e in star_linemasks["element"]])
    iron_star_linemasks = star_linemasks[(elem_col == "Fe 1") | (elem_col == "Fe 2")]
    star_spectrum = ispec.read_spectrum(spectrum_norm_path)
    atomic_linelist = ispec.read_atomic_linelist(
        atomic_linelist_file,
        wave_base=np.min(star_spectrum["waveobs"]),
        wave_top=np.max(star_spectrum["waveobs"]),
    )
    linemasks = star_linemasks
    linemasks = linemasks[linemasks["wave_nm"] > 0]
    linemasks = linemasks[linemasks["ew"] > 0]
    return {
        "star_spectrum": star_spectrum,
        "atomic_linelist": atomic_linelist,
        "star_linemasks": star_linemasks,
        "iron_star_linemasks": iron_star_linemasks,
        "recover_star_linemasks": recover_star_linemasks,
        "linemasks": linemasks,
        "atomic_linelist_file": atomic_linelist_file,
        "bedell_q2_tar_ew": None,
        "bedell_q2_sun_ew": None,
    }


def run_find_linemasks_part1(
    *,
    ispec,
    star_spectrum,
    atomic_linelist,
    from_resolution: float,
    to_resolution: float,
    min_depth: float = 0.05,
    max_depth: float = 1.0,
):
    step_log("2", "Running find_linemasks on spectrum.", progress="1/3")
    step_log("2", "Preparing smoothed spectrum (if needed).", progress="1/3")
    if from_resolution <= to_resolution:
        smoothed_star_spectrum = star_spectrum
    else:
        smoothed_star_spectrum = ispec.convolve_spectrum(star_spectrum, to_resolution)
    step_log("2", "Fitting continuum model.", progress="2/3")
    star_continuum_model = ispec.fit_continuum(
        star_spectrum, fixed_value=1.0, model="Fixed value"
    )
    step_log("2", "Executing ispec.find_linemasks (this may take a while).", progress="3/3")
    star_linemasks = ispec.find_linemasks(
        star_spectrum,
        star_continuum_model,
        atomic_linelist=atomic_linelist,
        max_atomic_wave_diff=0.005,
        telluric_linelist=None,
        vel_telluric=0.0,
        minimum_depth=min_depth,
        maximum_depth=max_depth,
        smoothed_spectrum=smoothed_star_spectrum,
        check_derivatives=False,
        discard_gaussian=False,
        discard_voigt=True,
        closest_match=False,
    )
    step_log("2", f"find_linemasks completed: {len(star_linemasks)} raw lines.")
    return star_linemasks


def filter_and_save_linemasks(
    *,
    ispec,
    target: str,
    output_folder: str,
    star_linemasks,
    ew_min: float = 10.0,
    ew_max: float = 100.0,
):
    step_log("3", "Filtering and saving fitted linemasks.")
    star_linemasks = star_linemasks[star_linemasks["wave_nm"] != 0]
    star_linemasks = star_linemasks[star_linemasks["ew"] != 0]
    star_linemasks = star_linemasks[
        (star_linemasks["ew"] >= ew_min) & (star_linemasks["ew"] <= ew_max)
    ]
    iron = (star_linemasks["element"] == "Fe 1") | (star_linemasks["element"] == "Fe 2")
    iron_star_linemasks = star_linemasks[iron]

    linemask_output_folder = os.path.join(output_folder, "linemasks")
    os.makedirs(linemask_output_folder, exist_ok=True)
    star_linemasks_txt = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_linemasks.txt"
    )
    fe_linemasks_txt = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_fe_linemasks.txt"
    )
    fitted_linemasks_txt = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.txt"
    )
    zeroed_linemasks_txt = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_zeroed_fitted_linemasks.txt"
    )
    atomic_linemasks_txt = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_atomic_linelist.txt"
    )

    ispec.write_line_regions(star_linemasks, star_linemasks_txt)
    ispec.write_line_regions(iron_star_linemasks, fe_linemasks_txt)
    ispec.write_line_regions(star_linemasks, fitted_linemasks_txt, extended=True)
    recover_star_linemasks = ispec.read_line_regions(fitted_linemasks_txt)
    zeroed_star_linemasks = ispec.reset_fitted_data_fields(star_linemasks)
    ispec.write_line_regions(zeroed_star_linemasks, zeroed_linemasks_txt, extended=True)
    ispec.write_atomic_linelist(star_linemasks, atomic_linemasks_txt)
    step_log(
        "3",
        f"Saved linemasks: kept={len(star_linemasks)} Fe={len(iron_star_linemasks)}",
    )
    return {
        "star_linemasks": star_linemasks,
        "iron_star_linemasks": iron_star_linemasks,
        "recover_star_linemasks": recover_star_linemasks,
        "linemask_output_folder": linemask_output_folder,
        "fitted_linemasks_txt": fitted_linemasks_txt,
    }


def plot_gaussian_fits_by_element(
    *,
    star_spectrum,
    linemasks,
    output_dir: str,
    element_filter: list[str] | None = None,
    prefix: str = "LineFit",
    w_range: float = 0.25,
    progress_every: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "deleted"), exist_ok=True)
    element_filter = element_filter or ["Fe 1", "Fe 2"]
    mask = np.isin(np.array([str(e).strip() for e in linemasks["element"]]), element_filter)
    lines = linemasks[mask]

    def gaussian(x, mu, sig, A, baseline):
        return baseline + A * np.exp(-((x - mu) ** 2) / (2 * sig**2))

    total = len(lines)
    saved = 0
    skipped = 0
    step_log("4", f"Start gaussian plotting for {total} lines.", progress=f"0/{total}")
    with _quiet_matplotlib_fonts():
        for idx, line in enumerate(lines, start=1):
            mu = line["mu"]
            sig = line["sig"]
            A = line["A"]
            baseline = line["baseline"]
            if sig == 0 or mu == 0:
                skipped += 1
                if idx % progress_every == 0 or idx == total:
                    step_log(
                        "4",
                        f"plotting in progress: saved={saved}, skipped={skipped}",
                        progress=f"{idx}/{total}",
                    )
                continue

            xmask = (star_spectrum["waveobs"] >= mu - w_range) & (
                star_spectrum["waveobs"] <= mu + w_range
            )
            wave = star_spectrum["waveobs"][xmask]
            flux = star_spectrum["flux"][xmask]
            fit_x = np.linspace(mu - w_range, mu + w_range, 300)
            fit_y = gaussian(fit_x, mu, sig, A, baseline)
            plt.figure(figsize=(8, 5), dpi=128)
            plt.plot(wave, flux, label="Observed Spectrum", color="blue", lw=0.7)
            plt.plot(fit_x, fit_y, "--", label="Gaussian Fit", color="red")
            plt.axvline(mu, color="orange", linestyle=":", label=f"$\\mu$ = {mu:.3f} nm")
            plt.title(f"Gaussian Fit: {line['element']} {line['wave_A']:.2f} A", fontsize=16)
            plt.xlabel("Wavelength (nm)", fontsize=16)
            plt.ylabel("Normalized Flux", fontsize=16)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.ticklabel_format(style="plain", axis="x")
            text = (
                f"log(gf) = {line['loggf']:.2f}\n"
                f"EP = {line['lower_state_eV']:.2f} eV\n"
                f"EW = {line['ew']:.1f} mA"
            )
            plt.text(
                0.75,
                0.05,
                text,
                transform=plt.gca().transAxes,
                fontsize=13,
                bbox=dict(facecolor="white", alpha=0.8),
            )
            plt.legend(loc="lower left", fontsize=13)
            plt.tight_layout()
            out_path = os.path.join(
                output_dir, f"{prefix}_{idx}_{line['element']}_{line['wave_A']:.2f}.png"
            )
            plt.savefig(out_path)
            plt.close()
            saved += 1
            if idx % progress_every == 0 or idx == total:
                step_log(
                    "4",
                    f"plotting in progress: saved={saved}, skipped={skipped}",
                    progress=f"{idx}/{total}",
                )
    step_log("4", f"Gaussian plotting completed. saved={saved}, skipped={skipped}")
    return {"total": total, "saved": saved, "skipped": skipped}


def prepare_manual_line_edit_tables(
    *,
    output_folder: str,
    target: str,
    output_linelist_path: str,
    PIPELINE_OPTIONS: dict[str, Any],
):
    step_log("5", "Preparing manual line-edit tables.")
    linemasks_dir = os.path.join(output_folder, "linemasks")
    os.makedirs(linemasks_dir, exist_ok=True)
    copied_linelist = os.path.join(linemasks_dir, f"linelist_for_{target}_copied.tsv")
    shutil.copyfile(output_linelist_path, copied_linelist)
    deleted_file_path = os.path.join(linemasks_dir, "Lines_deleted.tsv")
    modified_file_path = os.path.join(linemasks_dir, "Lines_EW_modified.tsv")

    cols_deleted = [
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
    cols_modified = ["element", "wave_A", "wave_nm", "loggf", "ew_mA_new"]

    for path, cols in (
        (deleted_file_path, cols_deleted),
        (modified_file_path, cols_modified),
    ):
        if not os.path.exists(path):
            pd.DataFrame(columns=cols).to_csv(path, sep="\t", index=False)

    if PIPELINE_OPTIONS.get("line_review_enabled", True):
        from line_review_ui import run_line_review

        run_line_review(
            output_folder=output_folder,
            target=target,
            figure_dir=os.path.join(output_folder, "figs_Fe_GaussianFits"),
            deleted_tsv_path=deleted_file_path,
            linelist_reference_path=copied_linelist,
            ui_mode=PIPELINE_OPTIONS.get("line_review_ui_mode", "inline"),
            reason_options=PIPELINE_OPTIONS.get("line_review_reason_options"),
            enabled=True,
        )
    return {
        "deleted_file_path": deleted_file_path,
        "modified_file_path": modified_file_path,
        "copied_linelist": copied_linelist,
    }


def apply_deleted_lines_to_atomic_linelist(
    *,
    ispec,
    output_folder: str,
    linelist_target_path: str,
):
    step_log("6", "Applying deleted-line table to atomic linelist.")
    atomic_linelist = ispec.read_atomic_linelist(linelist_target_path)
    from LineQA import prepare_line_edit_tables, load_deleted_table, filter_structured_array_by_deleted

    paths = prepare_line_edit_tables(output_folder)
    df_deleted = load_deleted_table(paths.deleted_tsv)
    filtered_linelist = filter_structured_array_by_deleted(
        atomic_linelist,
        df_deleted,
        element_col="element",
        wave_nm_col="wave_nm",
        decimals=4,
        only_elements=None,
    )
    ispec.write_atomic_linelist(filtered_linelist, linelist_target_path)
    # use step_log to print the number of deleted lines
    step_log("6", f"Deleted {len(df_deleted)} lines from atomic linelist.")
    step_log("6", f"Filtered atomic linelist: {len(filtered_linelist)} lines.")
    return filtered_linelist


def fast_apply_lineqa_to_linemasks(
    *,
    ispec,
    target: str,
    output_folder: str,
    star_linemasks=None,
    ew_min: float = 10.0,
    ew_max: float = 100.0,
):
    from LineQA import prepare_line_edit_tables, load_deleted_table, filter_structured_array_by_deleted

    step_log("7", "Applying LineQA deletions directly to fitted linemasks.")
    linemask_output_folder = os.path.join(output_folder, "linemasks")
    os.makedirs(linemask_output_folder, exist_ok=True)
    fitted_lm_path = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.txt"
    )

    if os.path.isfile(fitted_lm_path):
        star_linemasks = ispec.read_line_regions(fitted_lm_path)
    elif star_linemasks is None:
        raise FileNotFoundError(f"Missing fitted linemasks: {fitted_lm_path}")

    paths = prepare_line_edit_tables(output_folder)
    deleted = load_deleted_table(paths.deleted_tsv)
    star_linemasks = filter_structured_array_by_deleted(
        star_linemasks,
        deleted,
        element_col="element",
        wave_nm_col="wave_nm",
        decimals=4,
        only_elements=None,
    )
    star_linemasks = star_linemasks[star_linemasks["wave_nm"] != 0]
    star_linemasks = star_linemasks[star_linemasks["ew"] != 0]
    star_linemasks = star_linemasks[
        (star_linemasks["ew"] >= ew_min) & (star_linemasks["ew"] <= ew_max)
    ]
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
    ispec.write_line_regions(star_linemasks, fitted_lm_path, extended=True)
    recover_star_linemasks = ispec.read_line_regions(fitted_lm_path)
    zeroed_star_linemasks = ispec.reset_fitted_data_fields(star_linemasks)
    ispec.write_line_regions(
        zeroed_star_linemasks,
        os.path.join(
            linemask_output_folder, f"{target}_melendez2014_star_zeroed_fitted_linemasks.txt"
        ),
        extended=True,
    )
    ispec.write_atomic_linelist(
        star_linemasks,
        os.path.join(linemask_output_folder, f"{target}_melendez2014_star_atomic_linelist.txt"),
    )
    return {
        "star_linemasks": star_linemasks,
        "iron_star_linemasks": iron_star_linemasks,
        "recover_star_linemasks": recover_star_linemasks,
    }


def set_line_ew_in_linemask(
    linemask_path: str,
    element: str,
    wave_value: float,
    ew_mA: float,
    wave_unit: str = "A",
    tol: float = 0.0,
    note_tag: str = "EW_man=%.1f mA",
):
    if not os.path.exists(linemask_path):
        raise FileNotFoundError(linemask_path)
    df = pd.read_csv(linemask_path, sep="\t")
    if "element" not in df.columns:
        raise KeyError("linemask missing 'element' column")
    elem_col = df["element"].astype(str).str.strip()
    has_A = "wave_A" in df.columns
    has_nm = "wave_nm" in df.columns
    if not (has_A or has_nm):
        raise KeyError("linemask missing 'wave_A' or 'wave_nm'")
    if has_nm:
        wave_nm_series = pd.to_numeric(df["wave_nm"], errors="coerce")
    else:
        wave_nm_series = pd.to_numeric(df["wave_A"], errors="coerce") / 10.0
    wave_target_nm = wave_value / 10.0 if wave_unit.upper() == "A" else wave_value
    tol_nm = (tol / 10.0) if (wave_unit.upper() == "A") else tol
    m_elem = elem_col == element
    if tol_nm == 0:
        m_wave = np.round(wave_nm_series, 3) == np.round(wave_target_nm, 3)
    else:
        m_wave = np.abs(wave_nm_series - wave_target_nm) <= tol_nm
    idx = np.where(m_elem & m_wave)[0]
    if len(idx) == 0:
        raise ValueError(
            f"Line not found: element='{element}', lambda~{wave_target_nm:.3f} nm (tol={tol_nm:.4g})"
        )
    if len(idx) > 1:
        j = idx[np.argmin(np.abs(wave_nm_series.iloc[idx].values - wave_target_nm))]
        idx = [j]
    i = idx[0]
    wrote_any = False
    if "ew" in df.columns:
        df.at[i, "ew"] = float(ew_mA)
        wrote_any = True
    if "ew_mA" in df.columns:
        df.at[i, "ew_mA"] = float(ew_mA)
        wrote_any = True
    if not wrote_any:
        df["ew"] = pd.to_numeric(df.get("ew", np.nan), errors="coerce")
        df.at[i, "ew"] = float(ew_mA)
    if "note" not in df.columns:
        df["note"] = ""
    current_note = "" if pd.isna(df.at[i, "note"]) else str(df.at[i, "note"])
    tag = f"[{(note_tag % ew_mA) if '%' in note_tag else note_tag}]"
    if tag not in current_note:
        df.at[i, "note"] = (current_note + (" " if current_note.strip() else "") + tag).strip()
    df.to_csv(linemask_path, sep="\t", index=False)
    return i


def compute_fe_abundance(
    *,
    ispec,
    linemasks,
    ispec_dir: str,
    initial_teff: float,
    initial_logg: float,
    initial_MH: float,
    initial_feh_err: float | None = None,
):
    step_log("8", "Computing Fe abundance with MOOG.")
    linemasks = linemasks[linemasks["wave_nm"] > 0]
    linemasks = linemasks[linemasks["ew"] > 0]
    code = "moog"
    teff = initial_teff
    logg = initial_logg
    MH = initial_MH
    alpha = 0.00
    microturbulence_vel = 1.0
    model = ispec_dir + "/input/atmospheres/MARCS.GES/"
    if "ATLAS" in model:
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    else:
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    atmosphere_layers = ispec.interpolate_atmosphere_layers(
        modeled_layers_pack, {"teff": teff, "logg": logg, "MH": MH, "alpha": alpha}, code=code
    )
    spec_abund, normal_abund, x_over_h, x_over_fe = ispec.determine_abundances(
        atmosphere_layers,
        teff,
        logg,
        MH,
        alpha,
        linemasks,
        solar_abundances,
        microturbulence_vel=microturbulence_vel,
        verbose=1,
        code=code,
    )
    bad = np.isnan(x_over_h)
    fe1 = linemasks["element"] == "Fe 1"
    fe2 = linemasks["element"] == "Fe 2"
    fe1_abund = x_over_h[np.logical_and(fe1, ~bad)]
    fe2_abund = x_over_h[np.logical_and(fe2, ~bad)]
    step_log(
        "8",
        f"[Fe/H] ref={MH:.4f} +/- {initial_feh_err if initial_feh_err is not None else float('nan')}",
    )
    step_log(
        "8",
        f"Fe I lines={len(fe1_abund)} mean={np.mean(fe1_abund):.4f} std={np.std(fe1_abund):.4f}",
    )
    step_log(
        "8",
        f"Fe II lines={len(fe2_abund)} mean={np.mean(fe2_abund):.4f} std={np.std(fe2_abund):.4f}",
    )
    spec_abund_fe = spec_abund[np.logical_or(fe1, fe2)]
    return {
        "linemasks": linemasks,
        "spec_abund": spec_abund,
        "normal_abund": normal_abund,
        "x_over_h": x_over_h,
        "x_over_fe": x_over_fe,
        "fe1_abund": fe1_abund,
        "fe2_abund": fe2_abund,
        "spec_abund_fe": spec_abund_fe,
        "model": model,
        "code": code,
        "teff": teff,
        "logg": logg,
        "MH": MH,
        "alpha": alpha,
        "microturbulence_vel": microturbulence_vel,
        "solar_abundances_file": solar_abundances_file,
    }


def plot_abundance_trend(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    label: str,
    outfile: str | None = None,
    show_plot: bool = True,
):
    from scipy.stats import linregress

    with _quiet_matplotlib_fonts():
        plt.figure(figsize=(6, 4), dpi=120)
        plt.scatter(x, y, c="blue", label=label, s=40)
        slope, intercept, _, _, _ = linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        plt.plot(x_fit, slope * x_fit + intercept, "r--", label=f"Slope = {slope:.3f}")
        plt.xlabel(xlabel, fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile)
        elif show_plot:
            plt.show()
        plt.close()


def plot_fe_abundance_trends(
    *,
    linemasks,
    x_over_h,
    use_external_ew: bool,
    show_plots: bool = True,
):
    if use_external_ew:
        step_log("9", "Skipping Fe trend plots in external-EW mode.")
        return
    fe1_mask = (linemasks["element"] == "Fe 1") & (~np.isnan(x_over_h))
    fe2_mask = (linemasks["element"] == "Fe 2") & (~np.isnan(x_over_h))
    fe1_abund = x_over_h[fe1_mask]
    fe2_abund = x_over_h[fe2_mask]
    jobs = [
        (
            linemasks["lower_state_eV"][fe1_mask],
            fe1_abund,
            "Excitation Potential (eV)",
            "[Fe I/H]",
            "[Fe I/H] vs. Excitation Potential",
            "Fe I",
        ),
        (
            np.log10(linemasks["ew"][fe1_mask] / (linemasks["wave_nm"][fe1_mask] * 10)),
            fe1_abund,
            "log(EW/lambda)",
            "[Fe I/H]",
            "[Fe I/H] vs. log(REW)",
            "Fe I",
        ),
        (
            linemasks["lower_state_eV"][fe2_mask],
            fe2_abund,
            "Excitation Potential (eV)",
            "[Fe II/H]",
            "[Fe II/H] vs. Excitation Potential",
            "Fe II",
        ),
        (
            np.log10(linemasks["ew"][fe2_mask] / (linemasks["wave_nm"][fe2_mask] * 10)),
            fe2_abund,
            "log(EW/lambda)",
            "[Fe II/H]",
            "[Fe II/H] vs. log(REW)",
            "Fe II",
        ),
    ]
    total = len(jobs)
    step_log("9", "Start Fe trend plotting.", progress=f"0/{total}")
    for i, job in enumerate(jobs, start=1):
        plot_abundance_trend(*job, show_plot=show_plots)
        step_log("9", f"Rendered plot: {job[4]}", progress=f"{i}/{total}")
    step_log("9", "Fe trend plotting completed.")


def run_q2_fit_and_export_inputs(
    *,
    ispec,
    linemasks,
    target: str,
    output_folder: str,
    ispec_dir: str,
    solar_label: str,
    solar_lines_csv: str,
    instrument_name: str,
    from_resolution: float,
    initial_teff: float,
    initial_logg: float,
    initial_MH: float,
    initial_vmic: float,
    PIPELINE_OPTIONS: dict[str, Any],
    bedell_q2_tar_ew=None,
    bedell_q2_sun_ew=None,
):
    from abund_plot import export_q2_inputs_wide
    import q2

    step_log("10", "Preparing q2 inputs and running atmospheric fit.")
    elem_col = np.array([str(e).strip() for e in linemasks["element"]])
    linemasks_fe = linemasks[(elem_col == "Fe 1") | (elem_col == "Fe 2")]
    sun_dump_file = os.path.join(
        ispec_dir, "output", "Sun", solar_label, f"atmos_params_{solar_label}.dump"
    )
    q2_dir = os.path.join(output_folder, "q2_work")
    os.makedirs(q2_dir, exist_ok=True)

    q2_kwargs = {}
    if PIPELINE_OPTIONS.get("input_has_external_ews") and bedell_q2_tar_ew is not None:
        q2_kwargs["target_fe_ew_overrides"] = bedell_q2_tar_ew
        q2_kwargs["sun_fe_ew_overrides"] = bedell_q2_sun_ew

    lines_csv, stars_csv, sun_par = export_q2_inputs_wide(
        ispec=ispec,
        target_id=str(target),
        used_linemasks_fe=linemasks_fe,
        solar_lines_csv=solar_lines_csv,
        instrument=instrument_name,
        resolution=from_resolution,
        out_dir=q2_dir,
        sun_dump_file=sun_dump_file,
        sun_id="Sun",
        wave_tol=0.001,
        **q2_kwargs,
    )
    stars_df = pd.DataFrame(
        [
            {
                "id": "Sun",
                "teff": sun_par["teff"],
                "logg": sun_par["logg"],
                "feh": sun_par["feh"],
                "vt": sun_par["vt"],
            },
            {
                "id": str(target),
                "teff": float(initial_teff),
                "logg": float(initial_logg),
                "feh": float(initial_MH),
                "vt": float(initial_vmic),
            },
        ]
    )
    stars_df.to_csv(stars_csv, index=False)

    cwd0 = os.getcwd()
    try:
        os.chdir(q2_dir)
        with _quiet_matplotlib_fonts():
            data = q2.Data("stars_q2.csv", "lines_q2.csv")
            sp = q2.specpars.SolvePars(grid="marcs")
            sp.errors = True
            solution_csv = os.path.join(q2_dir, f"solution_q2_{target}.csv")
            q2.specpars.solve_all(data, sp, solution_csv, reference_star="Sun")
    finally:
        os.chdir(cwd0)

    sol = pd.read_csv(solution_csv)
    row = sol.loc[sol["id"] == str(target)].iloc[0]
    teff_q2 = float(row["teff"])
    logg_q2 = float(row["logg"])
    feh_q2 = float(row["feh"])
    vt_q2 = float(row["vt"])
    step_log(
        "10",
        f"q2 solution: Teff={teff_q2:.2f} logg={logg_q2:.4f} [Fe/H]={feh_q2:.4f} vt={vt_q2:.4f}",
    )
    return {
        "linemasks_fe": linemasks_fe,
        "q2_dir": q2_dir,
        "solution_csv": solution_csv,
        "row": row,
        "teff_q2": teff_q2,
        "logg_q2": logg_q2,
        "feh_q2": feh_q2,
        "vt_q2": vt_q2,
    }


def save_q2_dump(
    *,
    ispec,
    output_atmos_params_dumpfile_path: str,
    solution_row: pd.Series,
    linemasks_fe,
    target: str,
    solution_csv: str,
):
    step_log("11", "Saving q2 payload to iSpec-style dump.")
    err_teff_q2 = (
        float(solution_row["err_teff"])
        if "err_teff" in solution_row and pd.notna(solution_row["err_teff"])
        else np.nan
    )
    err_logg_q2 = (
        float(solution_row["err_logg"])
        if "err_logg" in solution_row and pd.notna(solution_row["err_logg"])
        else np.nan
    )
    err_feh_candidates = []
    for k in ["err_feh", "err_feh_"]:
        if k in solution_row and pd.notna(solution_row[k]):
            err_feh_candidates.append(float(solution_row[k]))
    err_feh_q2 = err_feh_candidates[0] if err_feh_candidates else np.nan
    err_vt_q2 = (
        float(solution_row["err_vt"])
        if "err_vt" in solution_row and pd.notna(solution_row["err_vt"])
        else np.nan
    )
    params_q2 = {
        "teff": int(round(float(solution_row["teff"]))),
        "logg": float(solution_row["logg"]),
        "MH": float(solution_row["feh"]),
        "alpha": 0.0,
        "vmic": float(solution_row["vt"]),
    }
    errors_q2 = {
        "teff": float(err_teff_q2),
        "logg": float(err_logg_q2),
        "MH": float(err_feh_q2),
        "alpha": np.nan,
        "vmic": float(err_vt_q2),
    }
    status_q2 = {
        "method": "q2_line_by_line_differential",
        "reference_star": "Sun",
        "grid": "marcs",
        "q2_solution_csv": os.path.basename(solution_csv),
    }
    dump_payload = (
        params_q2,
        errors_q2,
        status_q2,
        None,
        None,
        None,
        linemasks_fe,
        None,
    )
    dump_file_q2 = output_atmos_params_dumpfile_path.replace(".dump", "_q2.dump")
    ispec.mkdir_p(os.path.dirname(dump_file_q2))
    ispec.save_results(dump_file_q2, dump_payload)
    step_log("11", f"Saved q2 dump: {dump_file_q2}")
    return {"dump_file_q2": dump_file_q2, "params_q2": params_q2, "errors_q2": errors_q2}


def export_q2_dump_csv(*, ispec, target: str, output_atmos_params_dumpfile_path: str):
    dump_file_q2 = output_atmos_params_dumpfile_path.replace(".dump", "_q2.dump")
    csv_out = dump_file_q2.replace(".dump", ".csv")
    if not os.path.isfile(dump_file_q2):
        raise FileNotFoundError(f"Missing q2 dump: {dump_file_q2}")
    payload = ispec.restore_results(dump_file_q2)
    params = payload[0]
    errors = payload[1] if len(payload) > 1 else {}
    status = payload[2] if len(payload) > 2 else None

    def _fv(d, k):
        if not isinstance(d, dict):
            return float("nan")
        v = d.get(k, float("nan"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    row = {
        "starname": str(target),
        "dump_path": dump_file_q2,
        "teff": _fv(params, "teff"),
        "logg": _fv(params, "logg"),
        "MH": _fv(params, "MH"),
        "vmic": _fv(params, "vmic"),
        "alpha": _fv(params, "alpha"),
        "teff_err": _fv(errors, "teff"),
        "logg_err": _fv(errors, "logg"),
        "MH_err": _fv(errors, "MH"),
        "vmic_err": _fv(errors, "vmic"),
        "alpha_err": _fv(errors, "alpha"),
        "status_json": (
            json.dumps(status, ensure_ascii=False, default=str)
            if isinstance(status, dict)
            else str(status)
        ),
    }
    pd.DataFrame([row]).to_csv(csv_out, index=False)
    step_log("12", f"Wrote q2 atmos CSV: {csv_out}")
    return csv_out
