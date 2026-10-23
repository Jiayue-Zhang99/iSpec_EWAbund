# iSpec_EWAbund

[Data formats and outputs](docs/DATA_FORMATS.md)

An equivalent-width pipeline for high-resolution spectra of solar-type stars. It combines iSpec preprocessing and EW measurements, q2 differential atmospheric parameters, iSpec/MOOG element abundances, selected HFS corrections, human line review, and uncertainty propagation. This release packages the working line-by-line workflow as of 2026-10-22.

This repository is the **pipeline layer**. Install iSpec, q2, their atmosphere grids, MOOG, and (for HFS) linemake separately. It does not include a full iSpec distribution, raw 2D reduction, or the author's target survey. Two normalized MIKE spectra are provided for a worked example: **HD73350** and **Sun_MIKE_vesta_R45500**. HD73550 in earlier task descriptions refers here to the actual HD73350 files; no spectrum has been relabeled.

## Scientific conventions

- Atmospheric parameters: q2 fits Fe excitation/ionization balance relative to the solar reference. Its [Fe/H] sets the fixed atmosphere metallicity used in the element calculation.
- Element abundances, including Fe: use the existing `ispec.determine_abundances(code="moog")` workflow and line-by-line target minus solar abundances. The final Fe value need not equal q2's fitted metallicity because the estimators/backends and retained line samples differ.
- Element means: **equal weight per retained transition**, across ionization stages. Ion summaries are combined with `n_i/N` weights, not equal weights per ion.
- Default atmosphere/abundance resources: MARCS.GES and Grevesse (2007). The solar anchor adopts Teff=5772 K, logg=4.44, [M/H]=0, alpha=0, vt=1 km/s. Measured solar atmospheric parameters are diagnostic.
- HFS: V, Mn, Co, Cu, Ba. Compare matched target/Sun baselines with matching component provenance; missing or invalid pairs are not silently accepted. Microturbulence and model metallicity follow the adopted atmosphere.
- Oxygen: LTE through the ordinary backend by default. `run_O_triple=True` explicitly uses q2's O I 777 nm interface and NLTE correction for both target and Sun. All three matched transitions are required; invalid interpolation is not silently replaced by LTE.
- Errors: evaluate fixed-line responses to ±1-sigma changes in Teff, logg, [M/H], and vt. Parameter covariance is approximated as diagonal; shared parameter responses are retained when forming [X/Fe]. The statistical term uses the larger available estimate of empirical SEM and propagated target+Sun EW noise. Missing errors remain unknown, not zero. The solar atmosphere is fixed; this does not include all continuum, line-data, or model systematics.
- Same-instrument solar references are selected at the nearest available resolving power. This selection does **not** guarantee identical line-spread functions. Verify the resolution and continuum for your own observations.

## Installation

The reference environment uses Python 3.13, NumPy 2.3, SciPy 1.16, pandas 2.3, Matplotlib 3.10, and Astropy 7.1. `environment.yml` provides a convenient environment specification; `requirements.txt` lists Python dependencies, not compiled radiative-transfer software. Other versions are not certified by this packaging pass.

```bash
conda env create -f environment.yml
conda activate ispec-ewabund
```

1. Install a full [iSpec](https://github.com/marblestation/iSpec) checkout with its input resources, compiled extensions and compiled MOOG. Record the exact version used. The source installation for this release was based on commit `502974867b3a3313f561a3aafb31b1336eba5026`, with the explicit local compatibility delta in `patches/`. Review [patch instructions](patches/README.md) before using an unmodified engine.
2. Install the astronomical **q2** package and its MARCS/model data and MOOG-for-q2. Do not install an unrelated package merely named q2. The working q2 configuration expects `~/q2-tools/Data` and `~/q2-tools/MOOG-for-q2/MOOGSILENT`; consult your q2 installation's configuration if it differs. Importing some q2 versions can trigger downloads, so finish its installation first.
3. For HFS, install and compile [linemake](https://github.com/vmplacco/linemake), including its line databases. Set `MOOGSILENT_PATH` to the standalone MOOG executable used for blends. This is separate from iSpec's internal MOOG and q2's own configuration.
4. From this repository root, configure links to your installed engine:

```bash
python scripts/setup_workspace.py --ispec-root /path/to/iSpec --examples
export MOOGSILENT_PATH=/path/to/MOOG-for-q2/MOOGSILENT
# If linemake is not under the iSpec tools directory:
export ISPEC_LINEMAKE_BIN=/path/to/linemake/linemake.go
python scripts/check_environment.py --examples --hfs
```

`setup_workspace.py` links external runtime resources and copies the example metadata/library into ignored working locations. It never edits the external engine, and it refuses conflicting links. Existing example output files and library CSVs are preserved on repeated setup. `--linemake /path/to/linemake.go` can configure the local executable link.

The workspace root defaults to the parent of `Code/`. `ISPEC_EW_ROOT` can override it. **Do not point it at your external iSpec installation.** In this release, legacy `ISPEC_DIR` does not select the pipeline workspace. The setup links supply the external engine without mixing your results into that installation.

## Run the included MIKE example

The example spectra already contain normalized rest-frame flux. Do not reapply velocity correction, Angstrom-to-nm conversion, normalization, or resolution degradation to them. The FITS spectral arrays are unchanged from the selected source files; headers have been minimized to remove local processing metadata. Both configured resolving powers are 45500. See `examples/mike/provenance.json` for checksums and source filenames.

1. Run setup with `--examples`.
2. Start Jupyter in the production code directory:

   ```bash
   cd Code
   jupyter lab
   ```

3. Open `0_Sun_abundance.ipynb`. The default configuration selects the included R45500 Vesta FITS. Run the solar EW review, atmosphere diagnostic, element/HFS analysis, and solar-library export. Its preprocessing call is disabled for this already normalized example. This creates the solar masks and HFS provenance needed for a full target run; the small supplied library tables alone do not replace those products.
4. `inlist.py` defaults to `target="HD73350"`. Its metadata contains only the authorized MIKE example row. The prepared linelist is included, so skip notebooks 1 and 2 for this example.
5. Run `3a_atmos_params_iteration_lbl.ipynb`, then `4a_ele_abundance_lbl.ipynb`, then `5_ele_abundance_plot.ipynb`. Save every QA page to continue.

Alternatively, after generating the solar products, run from the repository root:

```bash
python Code/run_3a_atmos_lbl.py --target HD73350
python Code/run_4a_ele_lbl.py --target HD73350
python Code/run_5_ele_plot.py --target HD73350
```

`run_4a_ele_lbl.py --skip-hfs` disables HFS; `--hfs-dry-run` prepares a diagnostic dry run. The default includes HFS. Plotting references are disabled in the example because the full literature databases are not bundled. Human QA means a normal analysis is interactive, not an unattended benchmark.

## New observations and workflow files

- `0_Sun_abundance.ipynb`: construct the same-instrument solar reference. Use distinct labels for distinct resolution products.
- `1_data_processing.ipynb`: read 1D spectra, handle wavelength units/uncertainties, velocities, and continuum normalization. Raw 2D extraction and echelle order combination are external steps.
- `2_linelist_modify.ipynb`: rebuild the atomic linelist when required. Supply your VALD data at `input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv`; inspect crossmatches before replacing atomic parameters. Existing example users need not run this notebook.
- `3a_atmos_params_iteration_lbl.ipynb` / `run_3a_atmos_lbl.py`: Fe measurements, QA and q2 atmospheric fit.
- `4a_ele_abundance_lbl.ipynb` / `run_4a_ele_lbl.py`: fixed-atmosphere element abundances, element QA, HFS and parameter/EW errors.
- `5_ele_abundance_plot.ipynb` / `run_5_ele_plot.py`: species/element plots versus atomic number and condensation temperature.

Edit `input/spectra/input_spectra_info.xlsx` and `Code/inlist.py` for a new target. Use one unambiguous metadata row per target identifier; distinguish observations/instruments with separate identifiers. Provide initial Teff/logg/[Fe/H], instrument, resolving power, sky position, observation time and initial parameter errors. Paths may be overridden explicitly. Ensure the final normalized spectrum exists at `spectrum_norm_path` before step 3a. The normalized-input flag does not create or normalize this file for you.

## Configuration

`Code/config.py` defines defaults; `Code/inlist.py` sets target and per-run overrides. `PIPELINE_OPTIONS_OVERRIDE` takes precedence over defaults; `RUN_CONTEXT_OVERRIDE` supplies explicit paths, normalization settings and reference choice.

Frequently used options:

- `input_is_already_normalized`: input convention; verify the actual file supplied.
- `run_find_linemasks`: True measures/rebuilds masks; False reuses existing fitted masks for reruns.
- `line_review_enabled`, `line_review_ui_mode`: interactive QA, using a local browser or notebook view.
- `continuum_rescale_enabled`, `continuum_rescale_apply`, `continuum_rescale_tol`: optional continuum-window rescaling. The example retains the current inlist settings; this is distinct from local retry normalization.
- `input_has_external_ews`: optional Bedell Table1-style EW mode, requiring an existing linemask template. Configure `bedell2018_table1_path`, `bedell_ew_column` and template path. Fe retains pipeline EWs by default; non-Fe EWs may be replaced/appended.
- `REF_NAME`: None disables literature comparison; `Bedell2018` and `Melendez2025` use external literature tables under `input/abundances/D_PASTA/` (see data-format guide).

Switches near the top of `run_context.py`:

- `REF_MATCH_SPECIES=True`: match available reference species to measured diagnostics where possible, e.g. C I rather than combining C I and CH. Element-only references remain element-level comparisons.
- `run_O_triple=False`: optional q2 O I triplet NLTE interface.
- `run_local_ew_retry=True`: second-pass continuum/EW recovery.

Restart the notebook kernel after changing imported modules or switches to avoid retaining cached functions.

## Human QA and local recovery

First QA: choose `good`, a rejection reason, or `Re-normalize locally`. After Save (or advancing past the last image), explicitly requested lines receive a local continuum/Gaussian refit. Automatic recovery also applies to an element when all three conditions hold:

1. No usable line survives this QA round.
2. At least one current EW is finite and **strictly greater than 20 mAngstrom**. Exactly 20 does not qualify; missing values do not qualify on their own.
3. No line of that element was manually marked for local renormalization this round.

Automatic selection retries the element's current line set, not only its >20 mAngstrom line. A manual request has no 20 mAngstrom threshold. Lines removed before initial QA are not automatically restored.

Second QA compares the original fit with the local continuum and new Gaussian fit. `Origin EW ok` restores the original EW, uncertainty and fit; `new EW ok` accepts the new local normalization, EW and uncertainty. There is no ambiguous `good` option on the second page. A rejection reason cancels either acceptance choice. Failed new fits may still permit the original result when recoverable. Rejected second-pass lines are marked `failed_both_qa_rounds` and are not retried a third time. Cancel interrupts the workflow; a pending review can resume.

The global normalized FITS is not overwritten by local retry. Accepted local arrays and complete fit values are recorded and replayed in later abundance steps. Checksums guard against applying them to a different spectrum. External EW mode does not perform local spectral retries.

## Outputs, verification and limitations

Results live under ignored `output/<target>/` and `output/Sun/<solar_label>/`. Key products include the canonical atmosphere dump, q2 input/solution CSVs, raw per-line abundances, species summaries, uncertainty sensitivity tables, QA decisions/local-fit audit files, HFS tables and abundance plots. Detailed formats and error-field meanings are in [DATA_FORMATS.md](docs/DATA_FORMATS.md).

Regression tests are separate from production code:

```bash
PYTHONPATH=Code:. python -m unittest discover -s tests -v
```

Tests require the configured runtime and installed sample metadata. QA HTTP tests bind localhost temporary ports. They exercise numerical/data contracts and review-state behavior; they do not reproduce the full solar and target analysis. The migration did not rerun human QA or replace the author's scientific results. See docs/verification.json for the checks performed.

If a run fails, first check file paths and units, solar instrument/resolution selection, external grids/executables, and whether the preceding step produced its required output. Unknown uncertainties should remain visible as missing values. A small internal error bar is not evidence that normalization, resolution, blends, or the atmosphere model are free of systematics.

## Distribution and attribution

Only the two authorized example spectra and their minimal solar reference/target metadata are included. Survey data, old diagnostic outputs, local logs, compiled software and large grids are excluded. See [THIRD_PARTY.md](THIRD_PARTY.md) for resource attribution and licensing boundaries. No new blanket license is assigned to third-party line data or the author's code by this migration.

The existing Git repository's history contains old data objects. `.gitignore` does not erase history. A clean source archive is supplied in `dist/` for starting a fresh repository without that historical baggage; do not publish the old history if the intended release must exclude its old contents.
