from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atmos_lbl_pipeline import step_log


def load_star_and_atmosphere_context(
    *,
    ispec,
    target: str,
    output_folder: str,
    spectrum_norm_path: str,
    output_atmos_params_dumpfile_path: str,
    initial_teff: float,
    initial_logg: float,
    initial_MH: float,
) -> dict[str, Any]:
    step_log("1", "Loading spectrum, fitted linemasks, and atmospheric dump.")
    star_spectrum = ispec.read_spectrum(spectrum_norm_path)
    linemask_output_folder = os.path.join(output_folder, "linemasks")
    linemasks_path = os.path.join(
        linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.txt"
    )
    linemasks = ispec.read_line_regions(linemasks_path)
    step_log("1", f"Loaded fitted linemasks: {len(linemasks)}")

    payload = ispec.restore_results(output_atmos_params_dumpfile_path)
    params = payload[0]
    errors = payload[1]
    status_q2 = payload[2] if len(payload) > 2 else {}

    teff = float(params["teff"])
    logg = float(params["logg"])
    mh = float(params["MH"])
    vmic = float(params["vmic"])
    alpha = float(params["alpha"])
    tefferr = float(errors.get("teff", np.nan))
    loggerr = float(errors.get("logg", np.nan))
    mherr = float(errors.get("MH", np.nan))
    vmicerr = float(errors.get("vmic", np.nan))

    step_log(
        "1",
        f"Reference parameters: Teff={initial_teff}, logg={initial_logg}, [Fe/H]={initial_MH}",
    )
    step_log(
        "1",
        f"Fitted parameters: Teff={teff:.2f}±{tefferr:.2f}, "
        f"logg={logg:.2f}±{loggerr:.2f}, [M/H]={mh:.2f}±{mherr:.2f}, vmic={vmic:.2f}±{vmicerr:.2f}",
    )

    return {
        "star_spectrum": star_spectrum,
        "linemask_output_folder": linemask_output_folder,
        "linemasks": linemasks,
        "status_q2": status_q2,
        "teff": teff,
        "logg": logg,
        "mh": mh,
        "alpha": alpha,
        "vmic": vmic,
    }


def plot_all_element_gaussian_fits(
    *,
    star_spectrum,
    linemasks,
    output_folder: str,
    run_part1_spectrum: bool,
    plot_all_elements: bool = True,
    progress_every: int = 10,
) -> dict[str, Any]:
    if not plot_all_elements:
        step_log("2", "Skip element Gaussian plots: plot_all_elements=False.")
        return {"base_output_dir": os.path.join(output_folder, "figs_ele_GaussianFits"), "plotted": 0}
    if not run_part1_spectrum:
        step_log("2", "Skip element Gaussian plots: run_part1_spectrum=False.")
        return {"base_output_dir": os.path.join(output_folder, "figs_ele_GaussianFits"), "plotted": 0}

    step_log("2", "Plotting Gaussian diagnostics for all non-Fe elements.")
    base_output_dir = os.path.join(output_folder, "figs_ele_GaussianFits")
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "deleted"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "modified"), exist_ok=True)

    # Keep behavior compatible with line_review_ui non-Fe pattern.
    def gaussian(x, mu, sig, A, baseline):
        return baseline + A * np.exp(-((x - mu) ** 2) / (2 * sig**2))

    all_elements = sorted(np.unique(np.array([str(e).strip() for e in linemasks["element"]])))
    all_elements = [e for e in all_elements if "Fe" not in e]
    total_elements = len(all_elements)
    plotted = 0
    step_log("2", "Element plotting started.", progress=f"0/{total_elements}")

    for ei, ele in enumerate(all_elements, start=1):
        ele_lines = linemasks[np.array([str(e).strip() == ele for e in linemasks["element"]])]
        if len(ele_lines) == 0:
            step_log("2", f"Skip empty element group: {ele}", progress=f"{ei}/{total_elements}")
            continue

        ele_output_dir = os.path.join(base_output_dir, ele.replace(" ", "_"))
        os.makedirs(ele_output_dir, exist_ok=True)
        valid_lines = 0
        for idx, line in enumerate(ele_lines, start=1):
            mu = line["mu"]
            sig = line["sig"]
            A = line["A"]
            baseline = line["baseline"]
            if sig == 0 or mu == 0:
                continue

            mask = (star_spectrum["waveobs"] >= mu - 0.25) & (star_spectrum["waveobs"] <= mu + 0.25)
            wave = star_spectrum["waveobs"][mask]
            flux = star_spectrum["flux"][mask]
            fit_x = np.linspace(mu - 0.25, mu + 0.25, 300)
            fit_y = gaussian(fit_x, mu, sig, A, baseline)

            plt.figure(figsize=(8, 5), dpi=128)
            plt.plot(wave, flux, label="Observed Spectrum", color="blue", lw=0.7)
            plt.plot(fit_x, fit_y, "--", label="Gaussian Fit", color="red")
            plt.axvline(mu, color="orange", linestyle=":", label=f"mu = {mu:.3f} nm")
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

            # Keep filename compatible with line_review_ui _ELE_PATTERN.
            out_path = os.path.join(ele_output_dir, f"{ele}_{idx}_{line['wave_A']:.2f}.png")
            plt.savefig(out_path)
            plt.close()
            plotted += 1
            valid_lines += 1
            if idx % progress_every == 0 or idx == len(ele_lines):
                step_log(
                    "2",
                    f"Element {ele}: rendered={valid_lines}/{len(ele_lines)}",
                    progress=f"{ei}/{total_elements}",
                )

    step_log("2", f"Element Gaussian plotting completed. total_rendered={plotted}")
    return {"base_output_dir": base_output_dir, "plotted": plotted}


def run_nonfe_line_review(
    *,
    output_folder: str,
    target: str,
    run_part1_spectrum: bool,
    plot_all_elements: bool,
    pipeline_options: dict[str, Any],
) -> None:
    if not run_part1_spectrum or not plot_all_elements or not pipeline_options.get("line_review_enabled", True):
        step_log("3", "Skip interactive line review.")
        return
    from line_review_ui import run_line_review

    step_log("3", "Starting interactive line review for non-Fe lines.")
    run_line_review(
        output_folder=output_folder,
        target=target,
        figure_dir=os.path.join(output_folder, "figs_ele_GaussianFits"),
        deleted_tsv_path=os.path.join(output_folder, "linemasks", "Lines_deleted.tsv"),
        linelist_reference_path=os.path.join(
            output_folder, "linemasks", f"linelist_for_{target}_copied.tsv"
        ),
        ui_mode=pipeline_options.get("line_review_ui_mode", "inline"),
        reason_options=pipeline_options.get("line_review_reason_options"),
        enabled=True,
    )


def apply_lineqa_for_abundance(
    *,
    ispec,
    linemasks,
    output_folder: str,
    linemask_output_folder: str,
    target: str,
    pipeline_options: dict[str, Any],
):
    if not pipeline_options.get("apply_lineqa_in_abundance_step", True):
        step_log("4", "Skip LineQA application in abundance step.")
        return linemasks
    from LineQA import (
        apply_modified_ews_to_linemasks,
        filter_structured_array_by_deleted,
        load_deleted_table,
        load_modified_table,
        prepare_line_edit_tables,
    )

    step_log("4", "Applying LineQA (deleted lines + EW modifications).")
    paths = prepare_line_edit_tables(output_folder)
    deleted = load_deleted_table(paths.deleted_tsv)
    modified = load_modified_table(paths.modified_tsv)
    out = filter_structured_array_by_deleted(linemasks, deleted, decimals=4)
    out = apply_modified_ews_to_linemasks(out, modified, decimals=4)
    step_log("4", f"LineQA applied: kept={len(out)}, deleted={len(deleted)}, modified={len(modified)}")

    if pipeline_options.get("persist_filtered_linemasks_for_abund", True):
        out_path = os.path.join(
            linemask_output_folder, f"{target}_melendez2014_star_fitted_linemasks.filtered_for_abund.txt"
        )
        ispec.write_line_regions(out, out_path, extended=True)
        step_log("4", f"Saved filtered linemasks for abundance: {out_path}")
    return out


def prepare_abundance_engine(
    *,
    ispec,
    ispec_dir: str,
    teff: float,
    logg: float,
    mh: float,
    alpha: float,
    code: str = "moog",
) -> dict[str, Any]:
    step_log("5", "Preparing atmosphere model and solar abundance reference.")
    model = os.path.join(ispec_dir, "input", "atmospheres", "MARCS.GES") + os.sep
    if "ATLAS" in model:
        solar_abundances_file = os.path.join(
            ispec_dir, "input", "abundances", "Grevesse.1998", "stdatom.dat"
        )
    else:
        solar_abundances_file = os.path.join(
            ispec_dir, "input", "abundances", "Grevesse.2007", "stdatom.dat"
        )

    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)
    if not ispec.valid_atmosphere_target(
        modeled_layers_pack, {"teff": teff, "logg": logg, "MH": mh, "alpha": alpha}
    ):
        step_log("5", "Warning: atmospheric parameters are outside the model grid.")
    atmosphere_layers = ispec.interpolate_atmosphere_layers(
        modeled_layers_pack, {"teff": teff, "logg": logg, "MH": mh, "alpha": alpha}, code=code
    )
    return {
        "code": code,
        "model": model,
        "solar_abundances_file": solar_abundances_file,
        "solar_abundances": solar_abundances,
        "atmosphere_layers": atmosphere_layers,
    }


def _norm_elem(e):
    s = str(e).strip()
    return s.replace(" 1", " I").replace(" 2", " II")


def compute_lbl_differential_and_export(
    *,
    ispec,
    linemasks,
    target: str,
    ispec_dir: str,
    instrument_name: str,
    from_resolution: float,
    solar_lines_csv: str | None,
    atmosphere_layers,
    solar_abundances,
    teff: float,
    logg: float,
    mh: float,
    alpha: float,
    vmic: float,
    code: str,
    output_abundances_result_path_linebyline_q2: str,
) -> dict[str, Any]:
    step_log("6", "Computing line-by-line abundances and differential [X/H].")
    solar_lines_csv = solar_lines_csv or os.path.join(ispec_dir, "input", "solar_lines_instru.csv")
    if not os.path.isfile(solar_lines_csv):
        raise FileNotFoundError(f"Missing solar line library: {solar_lines_csv}")
    sun_master = pd.read_csv(solar_lines_csv)

    sun_inst = sun_master[
        sun_master["instrument"].astype(str).str.strip() == str(instrument_name).strip()
    ].copy()
    if sun_inst.empty:
        raise RuntimeError(f"No solar_lines rows for instrument={instrument_name}")
    avail = pd.to_numeric(sun_inst["resolution"], errors="coerce").values
    if not np.isfinite(avail).any():
        raise RuntimeError(f"Invalid resolution in solar_lines for instrument={instrument_name}")
    req = float(from_resolution)
    res_use = float(avail[np.nanargmin(np.abs(avail - req))])
    sun_sel = sun_inst[np.isclose(pd.to_numeric(sun_inst["resolution"], errors="coerce"), res_use, atol=1e-3)]
    if sun_sel.empty:
        sun_sel = sun_inst.copy()
        step_log("6", f"Warning: no exact solar resolution match at R={res_use:g}, using all rows.")
    else:
        step_log("6", f"Using solar line library at R={res_use:g}, rows={len(sun_sel)}")
    sun_sel["element_norm"] = sun_sel["element"].map(_norm_elem)

    elem_norm_all = np.array([_norm_elem(e) for e in linemasks["element"]])
    if "discarded" in linemasks.dtype.names:
        disc = linemasks["discarded"]
        if disc.dtype == np.bool_:
            mask_use = ~disc
        elif np.issubdtype(disc.dtype, np.number):
            mask_use = disc == 0
        else:
            disc_str = np.array([str(x).strip().lower() for x in disc])
            mask_use = ~np.isin(disc_str, ["true", "1", "yes", "y", "t"])
    else:
        mask_use = np.ones(len(linemasks), dtype=bool)
    linemasks_use = linemasks[mask_use]
    elem_norm_use = elem_norm_all[mask_use]

    elements = np.unique(elem_norm_use)
    star_lines: list[dict[str, Any]] = []
    for i, ele in enumerate(elements, start=1):
        sel = elem_norm_use == ele
        lm_ele = linemasks_use[sel]
        if len(lm_ele) == 0:
            continue
        spec_abund, _, x_over_h, _ = ispec.determine_abundances(
            atmosphere_layers,
            teff,
            logg,
            mh,
            alpha,
            lm_ele,
            solar_abundances,
            microturbulence_vel=vmic,
            verbose=0,
            code=code,
        )
        spec_abund = np.asarray(spec_abund, dtype=float)
        x_over_h = np.asarray(x_over_h, dtype=float)
        for lm, logeps_m12, xh in zip(lm_ele, spec_abund, x_over_h):
            if not (np.isfinite(logeps_m12) and np.isfinite(xh)):
                continue
            star_lines.append(
                {
                    "id": str(target),
                    "element": ele,
                    "wave_A": float(lm["wave_A"]),
                    "EP": float(lm["lower_state_eV"]),
                    "loggf": float(lm["loggf"]),
                    "ew_mA": float(lm["ew"]),
                    "logeps_star": float(logeps_m12 + 12.0),
                    "[X/H]_star_Grevesse": float(xh),
                }
            )
        if i % 5 == 0 or i == len(elements):
            step_log("6", f"Element abundance progress: {i}/{len(elements)}")

    df_star_lines = pd.DataFrame(star_lines)
    step_log("6", f"Computed star per-line rows: {len(df_star_lines)}")

    wave_tol = 0.001
    wave_key = lambda w: np.round(np.asarray(w, dtype=float) / wave_tol).astype(np.int64) * wave_tol

    sun_m = sun_sel[["element_norm", "wave_A", "logeps_sun", "[X/H]_sun_Grevesse"]].copy()
    sun_m = sun_m.rename(columns={"element_norm": "element"})
    sun_m["wave_key"] = wave_key(sun_m["wave_A"])
    sun_m = sun_m.drop_duplicates(subset=["element", "wave_key"], keep="first")

    df_s = df_star_lines.copy()
    df_s["wave_key"] = wave_key(df_s["wave_A"])
    df_merge = df_s.merge(
        sun_m[["element", "wave_key", "logeps_sun", "[X/H]_sun_Grevesse"]],
        on=["element", "wave_key"],
        how="inner",
    )
    df_merge["dlogeps"] = df_merge["logeps_star"] - df_merge["logeps_sun"]
    df_merge["d[X/H]"] = df_merge["[X/H]_star_Grevesse"] - df_merge["[X/H]_sun_Grevesse"]

    df_merge_out = df_merge[
        [
            "id",
            "element",
            "wave_A",
            "EP",
            "loggf",
            "ew_mA",
            "logeps_star",
            "logeps_sun",
            "dlogeps",
            "[X/H]_star_Grevesse",
            "[X/H]_sun_Grevesse",
            "d[X/H]",
        ]
    ].copy()

    raw_path = output_abundances_result_path_linebyline_q2.replace(".csv", "_raw.csv")
    df_merge_out.to_csv(raw_path, index=False)
    step_log("6", f"Saved raw per-line abundance file: {raw_path}")

    roman2int = {"I": 1, "II": 2, "III": 3, "IV": 4}

    def to_num_ion_token(s):
        parts = str(s).strip().split()
        if len(parts) == 1:
            return f"{parts[0]} 1"
        base, ion = parts[0], parts[1].upper()
        if ion in roman2int:
            return f"{base} {roman2int[ion]}"
        try:
            return f"{base} {int(float(ion))}"
        except Exception:
            return f"{base} 1"

    df_for_plot = df_merge_out.copy()
    df_for_plot["element"] = df_for_plot["element"].apply(to_num_ion_token)
    g = df_for_plot.groupby("element")["d[X/H]"]
    df_summary = g.agg(n_lines="count", **{"[X/H]": "mean"}).reset_index()
    df_summary["std_[X/H]"] = g.std(ddof=1).reset_index(drop=True)

    order = [x for x in ["Fe 1", "Fe 2"] if x in df_summary["element"].values]
    others = [x for x in df_summary["element"].tolist() if x not in order]
    df_summary["__ord"] = pd.Categorical(
        df_summary["element"], categories=order + sorted(others), ordered=True
    )
    df_summary = df_summary.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    df_summary.to_csv(output_abundances_result_path_linebyline_q2, index=False)
    step_log("6", f"Saved summary abundance file for Step 5 plotting: {output_abundances_result_path_linebyline_q2}")
    return {"raw_path": raw_path, "summary_path": output_abundances_result_path_linebyline_q2, "summary_df": df_summary}


def run_hfs_blends_and_update_outputs(
    *,
    ispec,
    target: str,
    output_folder: str,
    ispec_dir: str,
    atmosphere_layers,
    teff: float,
    logg: float,
    mh: float,
    output_abundances_result_path_linebyline_q2: str,
    pipeline_options: dict[str, Any],
    run_hfs_blends: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not run_hfs_blends:
        step_log("7", "Skip HFS blends by configuration.")
        return {"ran": False}
    from HFSBlends import build_ew_vs_hfs_diagnostics, run_hfs_blends_for_target

    step_log("7", "Running HFS blends post-processing.")
    hfs_model_in = os.path.join(output_folder, "HFS", "runs", f"model_{target}.in")
    os.makedirs(os.path.dirname(hfs_model_in), exist_ok=True)
    ispec.write_atmosphere(
        atmosphere_layers, teff, logg, mh, atmosphere_filename=hfs_model_in, code="moog"
    )
    if (not os.path.exists(hfs_model_in)) or os.path.getsize(hfs_model_in) == 0:
        raise RuntimeError(f"Failed to export MOOG model atmosphere: {hfs_model_in}")

    linemake_bin_cfg = os.environ.get("ISPEC_LINEMAKE_BIN") or os.environ.get("LINEMAKE_BIN")
    if not linemake_bin_cfg:
        local_linemake = os.path.join(ispec_dir, "tools", "linemake", "linemake.go")
        if os.path.exists(local_linemake):
            linemake_bin_cfg = local_linemake
            os.environ["ISPEC_LINEMAKE_BIN"] = linemake_bin_cfg

    use_external_ew = bool(pipeline_options.get("input_has_external_ews", False))
    hfs_df = run_hfs_blends_for_target(
        target=target,
        output_folder=output_folder,
        ispec_dir=ispec_dir,
        model_in=hfs_model_in,
        ew_result_csv=output_abundances_result_path_linebyline_q2,
        moog_bin=None,
        linemake_bin=linemake_bin_cfg,
        window_A=2.0,
        dry_run=dry_run,
        require_linemake=True,
        enable_implausible_hfs_shift_gate=True,
        use_external_input_ew=use_external_ew,
    )
    hfs_sum_out = os.path.join(output_folder, "HFS", "tables", f"abundances_hfs_blends_{target}_summary.csv")
    hfs_sum = pd.read_csv(hfs_sum_out) if os.path.exists(hfs_sum_out) else pd.DataFrame()

    raw_path = output_abundances_result_path_linebyline_q2.replace(".csv", "_raw.csv")
    sum_path = output_abundances_result_path_linebyline_q2
    hfs_elems = {"Cu", "Mn", "Co", "Ba", "V"}

    def canon_ele_ion(ele_text):
        s = str(ele_text).strip().replace("  ", " ")
        parts = s.split(" ")
        sym = parts[0] if parts else s
        ion_raw = parts[1] if len(parts) > 1 else "1"
        ion_u = str(ion_raw).upper()
        if ion_u in ["I", "1"]:
            ion = "1"
        elif ion_u in ["II", "2"]:
            ion = "2"
        else:
            ion = str(ion_raw)
        return sym, ion

    replaced_n = 0
    failed_hfs_n = 0
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        raw_df["wave_A"] = pd.to_numeric(raw_df.get("wave_A"), errors="coerce")
        hfs_ok = hfs_df.copy()
        hfs_ok = hfs_ok[
            (hfs_ok.get("mode", "hfs").astype(str) == "hfs")
            & (hfs_ok["status"].astype(str) == "ok")
        ].copy()
        hfs_ok["wave_A"] = pd.to_numeric(hfs_ok.get("wave_A"), errors="coerce")
        hfs_ok["logeps_HFS"] = pd.to_numeric(hfs_ok.get("logeps_HFS"), errors="coerce")
        hfs_ok["[X/H]_HFS"] = pd.to_numeric(hfs_ok.get("[X/H]_HFS"), errors="coerce")
        hfs_ok = hfs_ok[
            np.isfinite(hfs_ok["wave_A"])
            & np.isfinite(hfs_ok["logeps_HFS"])
            & np.isfinite(hfs_ok["[X/H]_HFS"])
        ].copy()
        raw_df["hfs"] = ""

        for i, r in raw_df.iterrows():
            sym, ion = canon_ele_ion(r.get("element", ""))
            if sym not in hfs_elems:
                raw_df.at[i, "hfs"] = ""
                continue
            w = pd.to_numeric(r.get("wave_A", np.nan), errors="coerce")
            old_dxh = pd.to_numeric(r.get("d[X/H]", np.nan), errors="coerce")
            cand = hfs_ok[
                (hfs_ok["element"].astype(str).str.strip() == sym)
                & (hfs_ok["ion"].astype(str).str.strip() == ion)
            ].copy()
            if np.isfinite(w) and len(cand):
                cand["dw"] = (cand["wave_A"] - float(w)).abs()
                cand = cand.sort_values("dw")
                if float(cand.iloc[0]["dw"]) <= 0.003:
                    best = cand.iloc[0]
                    raw_df.at[i, "hfs"] = old_dxh if np.isfinite(old_dxh) else ""
                    raw_df.at[i, "logeps_star"] = float(best["logeps_HFS"])
                    raw_df.at[i, "d[X/H]"] = float(best["[X/H]_HFS"])
                    if "[X/H]_sun_Grevesse" in raw_df.columns:
                        xh_sun = pd.to_numeric(raw_df.at[i, "[X/H]_sun_Grevesse"], errors="coerce")
                        raw_df.at[i, "[X/H]_star_Grevesse"] = (
                            float(best["[X/H]_HFS"]) + float(xh_sun)
                            if np.isfinite(xh_sun)
                            else float(best["[X/H]_HFS"])
                        )
                    if "logeps_sun" in raw_df.columns:
                        lsun = pd.to_numeric(raw_df.at[i, "logeps_sun"], errors="coerce")
                        if np.isfinite(lsun):
                            raw_df.at[i, "dlogeps"] = float(best["logeps_HFS"]) - float(lsun)
                    replaced_n += 1
                    continue
            raw_df.at[i, "hfs"] = "f"
            failed_hfs_n += 1
        raw_df.to_csv(raw_path, index=False)

    if os.path.exists(sum_path):
        sum_df = pd.read_csv(sum_path)
        sum_df["hfs"] = "n"

        def base_sym(e):
            return canon_ele_ion(e)[0]

        if len(hfs_sum):
            hs = hfs_sum.copy()
            hs["n_ok"] = pd.to_numeric(hs.get("n_ok"), errors="coerce")
            hs["xh_mean"] = pd.to_numeric(hs.get("xh_mean"), errors="coerce")
            hs["xh_std"] = pd.to_numeric(hs.get("xh_std"), errors="coerce")
            for i, r in sum_df.iterrows():
                sym = base_sym(r.get("element", ""))
                if sym not in hfs_elems:
                    continue
                m = hs[hs["element"].astype(str).str.strip() == sym].copy()
                if m.empty:
                    sum_df.at[i, "hfs"] = "f"
                    continue
                mr = m.iloc[0]
                n_ok = pd.to_numeric(mr.get("n_ok", np.nan), errors="coerce")
                if np.isfinite(n_ok) and int(n_ok) > 0 and np.isfinite(
                    pd.to_numeric(mr.get("xh_mean", np.nan), errors="coerce")
                ):
                    sum_df.at[i, "[X/H]"] = float(mr["xh_mean"])
                    if "std_[X/H]" in sum_df.columns and np.isfinite(
                        pd.to_numeric(mr.get("xh_std", np.nan), errors="coerce")
                    ):
                        sum_df.at[i, "std_[X/H]"] = float(mr["xh_std"])
                    sum_df.at[i, "hfs"] = "y"
                else:
                    sum_df.at[i, "hfs"] = "f"
        else:
            for i, r in sum_df.iterrows():
                if base_sym(r.get("element", "")) in hfs_elems:
                    sum_df.at[i, "hfs"] = "f"
        sum_df.to_csv(sum_path, index=False)

    cmp_df, cmp_fig = build_ew_vs_hfs_diagnostics(target=target, output_folder=output_folder)

    # Write a compact human-readable HFS run report.
    tables_dir = os.path.join(output_folder, "HFS", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    report_path = os.path.join(tables_dir, f"hfs_run_report_{target}.txt")

    rep = hfs_df.copy()
    for col in ("wave_A", "species", "n_hfs_components", "[X/H]_HFS", "[X/Fe]_HFS"):
        if col in rep.columns:
            rep[col] = pd.to_numeric(rep[col], errors="coerce")

    lines: list[str] = []
    lines.append(f"HFS Run Report for {target}")
    lines.append("=" * 72)
    lines.append(f"model_in: {hfs_model_in}")
    lines.append(f"hfs_rows: {len(rep)}")
    if len(rep):
        ok_rows = int((rep["status"].astype(str) == "ok").sum())
        lines.append(f"hfs_ok_rows: {ok_rows}/{len(rep)}")
    lines.append("")
    if len(rep):
        for elem in ["Cu", "Mn", "Co", "Ba", "V"]:
            d = rep[rep["element"].astype(str) == elem].copy().sort_values("wave_A")
            if d.empty:
                continue
            ok_n = int((d["status"].astype(str) == "ok").sum())
            lines.append(f"[{elem}] n_ok={ok_n}/{len(d)}")
            for _, rr in d.iterrows():
                w = pd.to_numeric(rr.get("wave_A", np.nan), errors="coerce")
                xh = pd.to_numeric(rr.get("[X/H]_HFS", np.nan), errors="coerce")
                st = str(rr.get("status", ""))
                note = str(rr.get("note", ""))
                lines.append(
                    f"  - line {w:.3f}A | status={st} | [X/H]_HFS={xh:.3f} | note={note}"
                    if np.isfinite(w) and np.isfinite(xh)
                    else f"  - line {w:.3f}A | status={st} | note={note}"
                    if np.isfinite(w)
                    else f"  - line status={st} | note={note}"
                )
            lines.append("")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    step_log(
        "7",
        f"HFS completed: rows={len(hfs_df)}, replaced={replaced_n}, failed={failed_hfs_n}, ew_vs_hfs_rows={len(cmp_df)}",
    )
    step_log("7", f"HFS detailed report path: {report_path}")
    if cmp_fig:
        step_log("7", f"EW-vs-HFS figure: {cmp_fig}")

    return {
        "ran": True,
        "hfs_rows": len(hfs_df),
        "replaced_n": replaced_n,
        "failed_hfs_n": failed_hfs_n,
        "cmp_rows": len(cmp_df),
        "report_path": report_path,
    }
