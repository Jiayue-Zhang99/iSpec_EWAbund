# Data and output contracts

## Spectrum inputs

Internally wavelengths are in nm, normalized flux is dimensionless, and `err` is flux standard deviation (sigma). EWs are in mAngstrom; atomic `wave_A` is in Angstrom and `wave_nm` in nm. Inverse variance is `ivar=1/sigma^2`; it must not be stored directly as SIGMA.

`spectrum_io.py` recognizes supported FITS and explicit text headers. Legacy `*_stitched_auto.txt` files are interpreted under the alexmods wavelength/flux/ivar convention. Unrecognized headerless text requires explicit error semantics; do not silently guess. `DataPreProcess.read_write_spectrum` performs the configured wavelength conversion. Example FITS use AWAV/FLUX/SIGMA columns with nm wavelength data and must not be divided by ten again.

## Target metadata

Working file: `input/spectra/input_spectra_info.xlsx`. Required context fields include `target_name`, `target_name2`, `note`, `instrument`, `R`, `Start of Obs`, `RA`, `Dec`, `Teff`, `logg`, `[Fe/H]`, and `[Fe/H]_err`; retain the additional example columns for preprocessing and provenance. `target_name2` is the reference-table identifier, e.g. HIP42333. Use signed sexagesimal Dec; negative sub-degree declinations must retain the minus sign.

Do not put multiple instruments under the same target identifier: the current loader takes the first matching target row. The example is restricted to a single MIKE row. Its initial parameters are initialization metadata, not advertised final results.

## Solar libraries

`input/solar_abund_instru.csv`: instrument, resolution, solar_label, element, n_lines, logeps_sun_mean, [X/H]_sun_mean_Grevesse, std_[X/H]_sun.

`input/solar_lines_instru.csv`: instrument, resolution, solar_label, element, wave_A, EP, loggf, ew_mA, logeps_sun, [X/H]_sun_Grevesse; HFS-aware tables also carry logeps_sun_nohfs, hfs_status and component_signature.

The solar workflow produces masks, atmosphere dumps and HFS components in addition to these CSVs. Keep those products together. Copying just a solar CSV is insufficient for complete uncertainty/HFS provenance replay. Make a distinct solar_label for each resolution. Current matching primarily selects instrument and nearest resolution, so avoid multiple conflicting labels for the same instrument/resolution.

## Literature references (optional, not bundled)

- Bedell2018: `input/abundances/D_PASTA/BedellHARPS/Bedell2018_table2.csv`, with Star and species columns such as [CI/H], e_[CI/H], [CH/H], [ScII/H]. Fe/H is resolved from `abundance_with_gaia.csv` in the same directory; its long-form columns include star_name, element_code, abundance, uncertainty, source, element_str.
- Melendez2025: `input/abundances/D_PASTA/Melendez2025/A46_tablea1_raw.csv` and `A46_table2_raw.csv`, containing stellar Fe/H and element [X/Fe] respectively.
- External EW mode: a Bedell Table1-style wide table with wavelength, species, EP/loggf and one EW column per star. Supply a compatible fitted-linemask template and explicit file paths.

Use the reference formats expected by the loader; a PDF or arbitrary literature CSV is not interchangeable with these tables. Species matching is audited in the plotting directory.

## Main outputs

- `atmos_params_<target>.dump`: iSpec-style parameter payload; model MH comes from q2. Legacy repeated `_q2` suffixes are canonicalized.
- `q2_work/`: target/Sun EW inputs, initial parameters and q2 solution.
- `linemasks/`: full fitted masks, filtered abundance masks, `Lines_deleted.tsv`, `Lines_EW_modified.tsv`, and local-retry records/arrays/figures.
- `abundances_linebyline_<target>_raw.csv`: per-line target/Sun values, differential abundance and HFS status.
- `abundances_linebyline_<target>.csv`: per-species summary, retained line count and uncertainty components. Element-level plots combine these using retained-line counts.
- `HFS/`: component files, paired baselines, executable logs, detailed line decisions and summaries.
- Parameter/EW sensitivity outputs: `line_sensitivities.csv` and `assumptions.json` in the uncertainty directory returned by the pipeline.
- `abundance_plots/`: species and element plots, with and without references/HFS comparisons where available.

The raw/species summary field `std_[X/H]` can mean line scatter; final plotting tables use that legacy column name for the uncertainty of the combined mean. Consult `sem_[X/H]`, `stat_err_[X/H]`, `param_err_[X/H]`, `total_err_[X/H]`, `line_scatter_[X/H]` and `uncertainty_kind` rather than assuming every std column is the same quantity.

Every retained line has equal abundance weight. Parameter sensitivities use that same weighting; independent measurement errors combine in quadrature. A one-line element has no empirical scatter estimate, but may have a propagated measurement error. Missing parameter uncertainty remains unknown. The error model is a diagonal approximation, not a full atmosphere posterior.
