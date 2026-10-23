# Authorized MIKE example inputs

- `HD73350_normalized.fits`: normalized, rest-frame target spectrum, configured R=45500.
- `Sun_MIKE_vesta_R45500_normalized.fits`: normalized Vesta spectrum previously degraded to configured R=45500.
- `input_spectra_info.xlsx`: only the MIKE HD73350 row, with fields needed by the pipeline. Survey rows and PI/contact/archive columns have been omitted.
- `solar_lines_instru.csv`, `solar_abund_instru.csv`: only the MIKE R45500 solar-label block. These are starting library examples, not all intermediate products required for HFS and errors. Run notebook 0 to regenerate the complete solar products before a full target run.
- `provenance.json`: source-relative filenames, source/export SHA256 hashes and header handling. FITS data columns are numerically unchanged; only headers were minimized.

Wavelength data are in nm. Do not reapply Angstrom conversion, radial-velocity correction, or resolution degradation. The parent README explains initialization and workflow order. These inputs are examples, not an independent scientific accuracy benchmark, and no other target spectra are included.
