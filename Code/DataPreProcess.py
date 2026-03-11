# DataPreProcess.py
# ------------------------------------------------------------
# NOTE (per your requirements):
# 1) Keep absolute iSpec path (ispec_dir) exactly as in your original notebook.
# 2) Keep the original 1_data_processing functions' structure as unchanged as possible,
#    including commented-out mask_file candidates.
# ------------------------------------------------------------

import os
import sys
import logging
import re
import numpy as np
import matplotlib.pyplot as plt

# These are used by the original notebook-style parsing functions
import pandas as pd

# astropy is used for robust RA/Dec parsing (safe fallback; you were already using it in the notebook context)
try:
    from astropy.coordinates import Angle
    import astropy.units as u
except Exception:
    Angle = None
    u = None

# ------------------------------------------------------------
# iSpec import (keep absolute path)
# ------------------------------------------------------------
ispec_dir = "/Users/jiayue/iSpec/"
print("ispec_dir =", ispec_dir)
sys.path.insert(0, os.path.abspath(ispec_dir))
import ispec  # noqa

logger = logging.getLogger(__name__)

from userlist import target

# ------------------------------------------------------------
# 1) read/write spectrum (keep your original function)
# ------------------------------------------------------------
def read_write_spectrum(filepath, starname, output_dir, unit_is_Angstrom=False):
    #--- Reading spectra -----------------------------------------------------------
    logging.info("Reading spectra")
    star_spectrum = ispec.read_spectrum(filepath)
    if unit_is_Angstrom:
        star_spectrum['waveobs'] = star_spectrum['waveobs'] / 10.0 # convert from Angstrom to nm
    ##--- Save spectrum ------------------------------------------------------------
    logging.info("Saving spectrum...")
    os.makedirs(output_dir, exist_ok=True)
    ispec.write_spectrum(star_spectrum, output_dir+"spectra_" + starname + ".fits")
    print("Spectrum saved to " + output_dir+"spectra_" + starname + ".fits")
    return star_spectrum


# ------------------------------------------------------------
# 2) metadata parsing (keep notebook-style I/O)
# ------------------------------------------------------------
def parse_obs_time(val):
    """
    Supprt format like '2024/9/8 8:37:17'、'2013/5/31 10:17'
    return [Y, M, D, h, m, s]
    """
    # keep original behavior (pandas)
    dt = pd.to_datetime(val, dayfirst=False, infer_datetime_format=True)
    return [int(dt.year), int(dt.month), int(dt.day), int(dt.hour), int(dt.minute), int(dt.second)]


def parse_ra_to_hms(val):
    """
    RA support:
        - 'HH:MM:SS.SS' (hour angle)
        - numeric (degree) (will be converted to hour angle)
    return [H, M, S]
    """
    s = str(val).strip()

    # robust (astropy) if available
    if Angle is not None and u is not None:
        try:
            if ":" in s:
                ang = Angle(s, unit=u.hourangle)
            else:
                ang = Angle(float(s), unit=u.deg).to(u.hourangle)
            hms = ang.hms
            return [int(hms.h), int(hms.m), float(hms.s)]
        except Exception:
            pass

    # fallback manual parsing
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return [h, m, sec]
    else:
        deg = float(s)
        total_hours = deg / 15.0
        h = int(total_hours)
        m = int((total_hours - h) * 60.0)
        sec = (total_hours - h - m/60.0) * 3600.0
        return [h, m, sec]


def parse_dec_to_dms(val):
    """
    Dec support:
        - '±DD:MM:SS.S' (degree angle)
        - numeric (degree)
    return [deg, arcmin, arcsec]  (sign on deg only)
    """
    s = str(val).strip()

    # robust (astropy) if available
    if Angle is not None and u is not None:
        try:
            ang = Angle(s, unit=u.deg) if ":" in s else Angle(float(s), unit=u.deg)
            dms = ang.signed_dms
            return [int(dms.d), int(abs(dms.m)), float(abs(dms.s))]
        except Exception:
            pass

    # fallback manual parsing
    if ":" in s:
        sign = -1 if s.startswith("-") else 1
        ss = s[1:] if s[0] in "+-" else s
        parts = [p.strip() for p in ss.split(":")]
        d = int(parts[0]) * sign
        m = int(parts[1]) if len(parts) > 1 else 0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return [d, abs(m), abs(sec)]
    else:
        deg = float(s)
        sign = -1 if deg < 0 else 1
        adeg = abs(deg)
        d = int(adeg) * sign
        m_float = (adeg - int(adeg)) * 60.0
        m = int(m_float)
        sec = (m_float - m) * 60.0
        return [d, m, sec]


# ------------------------------------------------------------
# 3) Telluric shift via telluric mask (notebook-compatible)
# ------------------------------------------------------------
def determine_tellurics_shift_with_mask(star_spectrum,
                                        telluric_mask_file=None,
                                        minimum_depth=0.0,
                                        lower_velocity_limit=-100,
                                        upper_velocity_limit=100,
                                        velocity_step=0.5,
                                        mask_depth=0.01,
                                        fourier=False,
                                        only_one_peak=True):
    """
    Estimate telluric-based velocity shift and apply correction.

    Return:
        dv (km/s), dv_err (km/s), corrected_spectrum
    """
    # default telluric mask path (use absolute ispec_dir as you did before)
    if telluric_mask_file is None:
        telluric_mask_file = ispec_dir + "input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"

    telluric_linelist = ispec.read_telluric_linelist(telluric_mask_file, minimum_depth=minimum_depth)

    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, telluric_linelist,
                                                  lower_velocity_limit=lower_velocity_limit,
                                                  upper_velocity_limit=upper_velocity_limit,
                                                  velocity_step=velocity_step,
                                                  mask_depth=mask_depth,
                                                  fourier=fourier,
                                                  only_one_peak=only_one_peak)

    components = len(models)
    print("Number of components found: ", components)

    dv = np.round(models[0].mu(), 2)       # km/s
    dv_err = np.round(models[0].emu(), 2)  # km/s
    print("Telluric velocity shift: ", dv, "±", dv_err, "km/s")

    # Apply telluric-based correction
    corrected_spectrum = ispec.correct_velocity(star_spectrum, dv)
    return dv, dv_err, corrected_spectrum


# ------------------------------------------------------------
# 4) Barycentric velocity correction (fix time-order robustness, keep call style)
# ------------------------------------------------------------
def calculate_barycentric_velocity(star_spectrum, Obstime, RA, Dec):
    """
    Barycentric velocity correction using iSpec.

    Inputs (notebook style):
        Obstime: [Y, M, D, h, m, s]   (from parse_obs_time)
        RA:      [H, M, S]           (from parse_ra_to_hms)
        Dec:     [deg, arcmin, arcsec] (from parse_dec_to_dms)

    Return:
        barycentric_vel (km/s), corrected_spectrum
    """
    # --- Robustly accept old/wrong ordering if ever passed ---
    # If the first element looks like a day (<= 31) and third looks like a year (>= 1900),
    # swap to (Y, M, D, h, m, s).
    if len(Obstime) >= 3:
        if (Obstime[0] <= 31) and (Obstime[2] >= 1900):
            # looks like [D, M, Y, ...]
            day, month, year, hours, minutes, seconds = Obstime
            Obstime_use = [year, month, day, hours, minutes, seconds]
        else:
            Obstime_use = Obstime
    else:
        Obstime_use = Obstime

    year, month, day, hours, minutes, seconds = Obstime_use
    RA_h, RA_m, RA_s = RA
    Dec_d, Dec_m, Dec_s = Dec

    barycentric_vel = ispec.calculate_barycentric_velocity_correction(
        (int(year), int(month), int(day), int(hours), int(minutes), float(seconds)),
        (int(RA_h), int(RA_m), float(RA_s), int(Dec_d), int(Dec_m), float(Dec_s))
    )

    print("Barycentric velocity correction:", barycentric_vel, "km/s")

    # Apply barycentric correction
    corrected_spectrum = ispec.correct_velocity(star_spectrum, barycentric_vel)
    return barycentric_vel, corrected_spectrum


# ------------------------------------------------------------
# 5) Radial velocity with CCF mask (keep commented mask_file candidates)
# ------------------------------------------------------------
def determine_radial_velocity_with_mask(targetspectrum, mask_file=None):
    """
    Determine RV using a cross-correlation mask.

    NOTE:
    - In your original notebook, mask_file was usually defined outside the function.
    - To minimize notebook changes, mask_file is optional:
        * if mask_file is None, it will try to use a global variable named `mask_file`.
    """
    # try to keep original behavior: allow external/global `mask_file`
    if mask_file is None:
        try:
            mask_file = ispec_dir + "input/linelists/CCF/Narval.Sun.370_1048nm/mask.lst"#globals()["mask_file"]
        except Exception:
            raise ValueError("mask_file is None and no global `mask_file` is defined in notebook.")

    # (keep your commented candidates, unchanged)
    #mask_file = ispec_dir + "input/linelists/CCF/Atlas.Arcturus.372_926nm/mask.lst""
    #mask_file = ispec_dir + "input/linelists/CCF/Atlas.Sun.372_926nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.A0.350_1095nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.F0.360_698nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.G2.375_679nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.K0.378_679nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.K5.378_680nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/HARPS_SOPHIE.M5.400_687nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/Synthetic.Sun.350_1100nm/mask.lst"
    #mask_file = ispec_dir + "input/linelists/CCF/VALD.Sun.300_1100nm/mask.lst"

    ccf_mask = ispec.read_cross_correlation_mask(mask_file)

    models, ccf = ispec.cross_correlate_with_mask(targetspectrum, ccf_mask, \
                            lower_velocity_limit=-200, upper_velocity_limit=200, \
                            velocity_step=1.0, mask_depth=0.01, \
                            fourier=False)

    # Number of models represent the number of components
    components = len(models)
    print("Number of components found: ", components)
    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    print("Radial velocity: ", rv, "±", rv_err, "km/s")
    return rv, rv_err


# ------------------------------------------------------------
# 6) Small helper to apply any velocity correction (optional, non-breaking)
# ------------------------------------------------------------
def apply_velocity_correction(star_spectrum, dv_kms):
    """
    Apply iSpec velocity correction (dv in km/s).
    Non-breaking helper for cleaner notebook code.
    """
    return ispec.correct_velocity(star_spectrum, float(dv_kms))

# ------------------------------------------------------------
# 7) Whole-spectrum normalization with template (keep original structure)
# ------------------------------------------------------------
def normalize_whole_spectrum_with_template(inputfile, from_resolution):
    """
    Normalize a whole spectrum using a template-driven continuum fit (iSpec model="Template").

    Parameters
    ----------
    inputfile : dict-like
        iSpec spectrum object (e.g., returned by ispec.read_spectrum).
    from_resolution : float
        Resolution of the input spectrum (R). iSpec uses this to match the template/continuum scale.

    Returns
    -------
    normalized_star_spectrum : dict-like
        Normalized spectrum.
    star_continuum_model : callable
        Continuum model returned by iSpec (callable on wavelength array).
    """
    star_spectrum = inputfile

    # Read the template spectrum (Solar synthetic template shipped with iSpec)
    synth_spectrum = ispec.read_spectrum(
        ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz"
    )

    # --- Continuum fit ----------------------------------------------------------
    model = "Template"
    # Automatic: 1 spline every ~5 nm (internally used to apply a Gaussian-like smoothing)
    nknots = None
    median_wave_range = 5.0

    # Regions to ignore in continuum fitting (strong absorption lines)
    strong_lines = ispec.read_line_regions(
        ispec_dir + "/input/regions/strong_lines/absorption_lines.txt"
    )
    # Alternative choices (kept as in your notebook; commented out intentionally)
    # strong_lines = ispec.read_line_regions(ispec_dir + "/input/regions/relevant/relevant_line_masks.txt")
    # strong_lines = None

    star_continuum_model = ispec.fit_continuum(
        star_spectrum,
        from_resolution=from_resolution,
        ignore=strong_lines,
        nknots=nknots,
        median_wave_range=median_wave_range,
        model=model,
        template=synth_spectrum
    )

    # --- Continuum normalization ------------------------------------------------
    logging.info("Continuum normalization (Template)...")
    normalized_star_spectrum = ispec.normalize_spectrum(
        star_spectrum, star_continuum_model, consider_continuum_errors=False
    )

    # If the spectrum is already normalized and you want to force a flat continuum:
    # star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")

    return normalized_star_spectrum, star_continuum_model


def plot_spectrum_with_normalization(
    raw_spectrum, normalized_spectrum, continuum_model,
    starname=None, zoominrange1=[585, 595], zoominrange2=[588.7, 590]
):
    """
    Plot the raw spectrum with the fitted continuum model and the normalized spectrum,
    including two zoom-in panels.

    Parameters
    ----------
    raw_spectrum : dict-like
        Raw spectrum (before normalization; may also be before velocity corrections depending on your workflow).
    normalized_spectrum : dict-like
        Normalized spectrum.
    continuum_model : callable
        Continuum model returned by iSpec.fit_continuum (callable on wavelength array).
    starname : str or None
        Star name used in plot titles. If None, uses "target".
    zoominrange1 : list [wmin, wmax]
        First zoom-in wavelength range in nm.
    zoominrange2 : list [wmin, wmax]
        Second zoom-in wavelength range in nm.
    """
    import matplotlib.pyplot as plt

    if starname is None:
        starname = "target"

    wave = normalized_spectrum["waveobs"]
    flux = normalized_spectrum["flux"]
    continuum_flux = continuum_model(wave)

    # Figure 1: raw spectrum + continuum model
    plt.figure(figsize=(12, 6), dpi=192)
    plt.plot(raw_spectrum["waveobs"], raw_spectrum["flux"], label="Spectrum", linewidth=0.5)
    plt.plot(wave, continuum_flux, label="Continuum", color="red", linewidth=2)
    plt.legend()
    plt.title(f"{starname} with continuum, before normalization")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux")

    # Figure 2: stacked panels with zoom-ins
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), dpi=192)

    # subplot 1: raw + continuum
    axs[0].plot(raw_spectrum["waveobs"], raw_spectrum["flux"], label="Spectrum", linewidth=0.5)
    axs[0].plot(wave, continuum_flux, label="Continuum", color="red")
    axs[0].legend()
    axs[0].set_xlim(raw_spectrum["waveobs"][0], raw_spectrum["waveobs"][-1])
    axs[0].set_title(f"{starname} with continuum, before normalization")

    # subplot 2: full normalized spectrum
    axs[1].plot(wave, flux, linewidth=0.5)
    axs[1].set_title(f"{starname}, after normalization")
    axs[1].axhline(y=1, color="orange", linewidth=1)
    axs[1].text(wave[0], 0.9, "y=1", color="orange", ha="left", va="bottom")
    axs[1].set_xlim(wave[0], wave[-1])
    axs[1].set_ylim(0, 1.1)
    axs[1].axvspan(zoominrange1[0], zoominrange1[1], color="red", alpha=0.2)

    # subplot 3: zoominrange1
    axs[2].plot(wave, flux, linewidth=0.8)
    axs[2].axhline(y=1, color="orange", linewidth=1)
    axs[2].text(zoominrange1[0], 0.9, "y=1", color="orange", ha="left", va="bottom")
    axs[2].set_xlim(zoominrange1[0], zoominrange1[1])
    axs[2].set_ylim(0, 1.1)
    axs[2].axvspan(zoominrange2[0], zoominrange2[1], color="red", alpha=0.2)

    # subplot 4: zoominrange2
    axs[3].plot(wave, flux, linewidth=0.8)
    axs[3].axhline(y=1, color="orange", linewidth=1)
    axs[3].text(zoominrange2[0], 0.9, "y=1", color="orange", ha="left", va="bottom")
    axs[3].set_xlim(zoominrange2[0], zoominrange2[1])
    axs[3].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 8) Continuum normalization using only continuum regions (keep original structure)
# ------------------------------------------------------------
def normalize_spectrum_using_continuum_regions(star_spectrum, from_resolution):
    """
    Consider only continuum regions for the fit, strategy 'median+max'
    """

    #--- Continuum fit -------------------------------------------------------------
    model = "Splines" # "Polynomy"
    degree = 2
    nknots = int((star_spectrum['waveobs'][-1] - star_spectrum['waveobs'][0]) / 2.0)   # Automatic: 1 spline every 5 nm

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0

    continuum_regions = ispec.read_continuum_regions(ispec_dir + "/input/regions/fe_lines_continuum.txt")
    star_continuum_model = ispec.fit_continuum(star_spectrum, from_resolution=from_resolution, \
                            continuum_regions=continuum_regions, nknots=nknots, degree=degree, \
                            median_wave_range=median_wave_range, \
                            max_wave_range=max_wave_range, \
                            model=model, order=order, \
                            automatic_strong_line_detection=True, \
                            strong_line_probability=0.5, \
                            use_errors_for_fitting=True)

    #--- Continuum normalization ---------------------------------------------------
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum, star_continuum_model, consider_continuum_errors=False)
    # Use a fixed value because the spectrum is already normalized
    #star_continuum_model = ispec.fit_continuum(star_spectrum, fixed_value=1.0, model="Fixed value")
    return normalized_star_spectrum, star_continuum_model