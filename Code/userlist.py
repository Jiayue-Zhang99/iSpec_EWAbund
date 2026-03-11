import os
import sys
import logging
import pandas as pd

################################################################################
#--- iSpec directory -------------------------------------------------------------
# Absolute path to the local iSpec installation directory.
# All input/output paths in this pipeline are constructed relative to this path.
#ispec_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
ispec_dir = "/Users/jiayue/iSpec/"
print("ispec_dir =", ispec_dir)

# Add iSpec directory to Python path
sys.path.insert(0, os.path.abspath(ispec_dir))
import ispec

#--- Change LOG level ----------------------------------------------------------
# Available levels: "debug", "info", "warning", "error"
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger()  # root logger, shared by all modules
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################

################################################################################
### --- User inputs --- ###
# Target star name (must match entries in input_spectra_info.xlsx)
target = "example"
print("target:", target)

# Path to the raw observed spectrum (FITS file)
# This should be the wavelength-calibrated, extracted 1D spectrum
input_spectra_path = (
    ispec_dir + "input/spectra/" + target +
    "/example_target_file_name.fits"
)

# Whether the wavelength unit in the input spectrum is Angstrom (True) or nm (False)
unit_is_Angstrom = True

# Output directory for all results related to this target
output_folder = ispec_dir + "output/" + target + "/"

# Path to the normalized spectrum used in subsequent analysis steps
# This file is typically generated in 1_data_processing
spectrum_norm_path = (
    output_folder + "spectra_" + target + "_normed_splines_segmts.fits"
)

### --- Read target metadata from input_spectra_info.xlsx --- ###
# This Excel file stores observing and stellar parameters for all targets
input_spectra_info = "../input/spectra/input_spectra_info.xlsx"
df_input_info = pd.read_excel(input_spectra_info)

# Select the row corresponding to the current target
# If multiple rows exist, the first one is used
row_target = df_input_info.loc[
    df_input_info["target_name"] == target
].iloc[0]

################################################################################
# From 0_Sun_abundance
# Instrument name and resolving power used for this target
instrument_name = row_target["instrument"]
from_resolution = row_target["R"]

### --- Read solar abundance reference for this instrument --- ###
# This table stores instrument-dependent solar abundance references
solar_abund_info_csv = os.path.join(
    ispec_dir, "input", "solar_abund_instru.csv"
)
df_solar_info = pd.read_csv(solar_abund_info_csv)

# Try to find an exact match in instrument + resolution
mask_exact = (
    (df_solar_info["instrument"] == instrument_name) &
    (df_solar_info["resolution"] == from_resolution)
)

if not df_solar_info[mask_exact].empty:
    row_sun = df_solar_info[mask_exact].iloc[0]
    print(
        "Sun's info found for instrument %s at resolution %d"
        % (instrument_name, from_resolution)
    )
else:
    # If no exact resolution match is found,
    # fall back to the first solar entry for the same instrument
    row_sun = df_solar_info[
        df_solar_info["instrument"] == instrument_name
    ].iloc[0]
    print(
        "Exact match not found. Using Sun's info for instrument %s at resolution %d"
        % (instrument_name, row_sun["resolution"])
    )

# Solar reference label and resolution actually used
solar_label      = row_sun["solar_label"]
solar_resolution = row_sun["resolution"]

# --- Read solar per-line abundance reference ---
# This file stores line-by-line solar abundances for different instruments
solar_lines_csv = os.path.join(
    ispec_dir, "input", "solar_lines_instru.csv"
)
solar_lines_master = pd.read_csv(solar_lines_csv)

# Select solar reference lines for the current instrument and resolution
solar_lines_instr = solar_lines_master[
    (solar_lines_master["instrument"] == instrument_name) &
    (solar_lines_master["resolution"] == from_resolution)
].copy()

################################################################################
# For 1_data_processing
# Observation time (used for barycentric correction if enabled)
Obs_time_raw = row_target["Start of Obs"]

# Target coordinates (used for barycentric correction)
target_ra_raw  = row_target["RA"]
target_dec_raw = row_target["Dec"]

# Target spectrum will be degraded to the solar reference resolution
to_resolution = solar_resolution

# Flags controlling velocity corrections
LBV_correction = False  # whether to apply barycentric velocity correction
LRV_correction = False  # whether to apply radial velocity correction

# Flags controlling different normalization strategies
normalization_whole_template  = True   # normalize using whole-spectrum template
normalization_segmts_template = True   # normalize in segments using template
normalization_segmts_polynomy = True   # normalize in segments using polynomial fitting

################################################################################
# For 2_linelist_modify
# Directory containing the reference linelist (user-provided / literature / curated)
ref_linelist_folder = (
    ispec_dir + "input/linelists/linelist_reference/"
)

# Original reference linelist file (raw format: txt/csv/tsv, etc.)
ref_linelist_raw_path = (
    ref_linelist_folder + "linelist_reference.txt"
)

# Converted reference linelist in iSpec TSV format
ref_linelist_ispec_path = (
    ref_linelist_folder + "linelist_reference_ispec.tsv"
)

# Final reference linelist used for abundance analysis (iSpec TSV)
output_linelist_path = (
    ref_linelist_folder + "linelist_reference_ispec.tsv"
)

# VALD atomic line database used for cross-matching
vald_atomic_lines_path = (
    ispec_dir +
    "input/linelists/transitions/VALD.300_1100nm/atomic_lines copy.tsv"
)

################################################################################
# For 3_atmos_params_iteration
# Initial guesses for stellar atmospheric parameters
initial_teff = row_target["Teff"]
initial_logg = row_target["logg"]
initial_MH   = row_target["[Fe/H]"]

# Linelist path copied into the target output directory
linelist_target_path = (
    output_folder + "linelist_for_" + target + ".tsv"
)

# Width (in nm) of the fitting window around each spectral line
fit_width = 0.2

# Dump file storing fitted atmospheric parameters and EW results
output_atmos_params_dumpfile_path = (
    output_folder + "atmos_params_%s.dump" % target
)

################################################################################
# For 4_ele_abundance
# Output abundance tables in different formats
output_abundances_result_path_linebyline_q2 = (
    output_folder + "abundances_linebyline_q2_" + target + ".csv"
)
output_abundances_result_path_linebyline = (
    output_folder + "abundances_linebyline_" + target + ".csv"
)
output_abundances_result_path_Grevesse = (
    output_folder + "abundances_Grevesse_" + target + ".csv"
)
output_abundances_result_path = (
    output_folder + "abundances_" + target + ".csv"
)

################################################################################
# For 5_ele_abundance_plot
# Atomic number table used for plotting abundance trends
ATOMIC_Z = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Sr":38,"Y":39,"Zr":40,
    "Ba":56,"La":57,"Ce":58,"Nd":60,"Eu":63
}

# Output directory for abundance plots
abundance_plot_folder = output_folder + "abundance_plots/"

# Condensation temperature table (Lodders 2003, Table 8)
TC_PATH = "../input/Tc_elements_Lodders2003_table8.csv"

# Reference star used for differential comparison
REF_STAR = row_target['target_name2']

# Reference abundance pattern source
# Options: "Melendez2025" or "Bedell2018"
REF_NAME = "Melendez2025"

################################################################################
# For 6_combine_abundances