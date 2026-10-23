# Attribution and distribution boundaries

This export separates pipeline code from external scientific engines and large data packages. It does not relicense the author's code or third-party materials. No repository-wide permissive LICENSE has been invented. Choose an appropriate code license with the copyright holder before advertising an open-source license grant.

- iSpec: https://github.com/marblestation/iSpec . The engine source identifies AGPL-3.0-or-later terms. The small compatibility patch in patches/ is derived from that engine; preserve its license. The original engine is installed separately, not bundled here.
- q2 (astronomical spectroscopic analysis by Ivan Ramirez and collaborators): install the correct scientific package and its own data and MOOG dependencies. The local environment used a configured package without a distribution version exposed; exact package-file hashes are recorded in docs/runtime_versions.json where available.
- MOOG: external Fortran radiative-transfer software. iSpec, q2, and HFS may call different installations. Preserve the release/damping/data choices when reproducing results.
- linemake: https://github.com/vmplacco/linemake ; cite the generator and the transition-data sources relevant to your selected components. Its executable and databases are not bundled.
- MARCS.GES, Grevesse (2007), VALD: obtain through their appropriate distribution channels. No full model grid, VALD database, or molecular-line archive is distributed in this package.
- The small Melendez2014-based pipeline linelist, HFS Iris templates, continuum mask and Lodders2003 condensation-temperature table are preserved from the source workspace. Filenames identify their scientific provenance but do not establish a new redistribution license. Review the underlying data attribution/permission when publicly distributing them; this migration does not claim ownership of those tables.
- MIKE example spectra: HD73350 and collaborator-provided Vesta, explicitly selected for this release by the project owner. The export retains numerical spectral columns and strips other FITS metadata. Their author/collaborator attribution and any publication-specific data statement remain the owner's responsibility.

Cite the engine/method papers and atomic/HFS/solar reference data actually used in scientific publications. The two examples are workflow inputs, not a claim of independently validated final abundances.
