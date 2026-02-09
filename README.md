# iSpec_EWAbund - Stellar Abundance Analysis Pipeline (EW-based, iSpec + q2)
This repository contains a line-by-line equivalent-width (EW)–based stellar abundance analysis pipeline, built around iSpec and q2, and primarily designed for solar analogs observed with high-resolution spectrographs.

## Important note:
- This repository is under active development.
- The current codebase reflects a working research pipeline, but has not yet been fully cleaned, modularized, or finalized.
- A more polished and documented version will be released in a future update.

## Overview

The pipeline performs:
    - Continuum normalization and spectrum preprocessing
    - Atomic linelist preparation and modification
    - Iterative determination of stellar atmospheric parameters
    - Line-by-line differential abundance analysis relative to the Sun
    - Condensation-temperature–dependent abundance visualization

The workflow follows a physically transparent, EW-based approach, suitable for high-precision differential abundance studies.