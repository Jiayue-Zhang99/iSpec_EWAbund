"""Explicit spectrum error semantics at the alexmods -> iSpec boundary.

New MIKE text products have the iSpec header ``waveobs flux err`` and contain
one-sigma errors. Legacy ``*_stitched_auto.txt`` and ``Sun_MIKE_vesta.txt``
contain inverse variances. Never infer the meaning from numerical magnitude.
Wavelength units remain the caller's responsibility (as before).
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
import numpy as np


def _text_layout(path):
    opener = gzip.open if str(path).lower().endswith('.gz') else open
    with opener(path, 'rt') as handle:
        for i, line in enumerate(handle):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            try:
                [float(value) for value in fields]
            except ValueError:
                return [value.lower() for value in fields], i + 1
            return None, i
    raise ValueError(f'Empty spectrum: {path}')


def read_spectrum_for_pipeline(ispec, filepath, *, error_kind='auto'):
    """Return (iSpec spectrum, ingestion report), converting ivar exactly once.

    ``error_kind`` is auto, ivar or sigma. Headerless text with an unknown
    provenance must specify ivar/sigma explicitly. Pixels with invalid ivar or
    sigma are removed rather than assigned zero uncertainty. An all-zero sigma
    column retains the established iSpec meaning of unavailable uncertainties.
    """
    path = Path(filepath)
    kind = str(error_kind).lower()
    if kind not in {'auto', 'ivar', 'sigma'}:
        raise ValueError('error_kind must be auto, ivar or sigma')
    name = path.name.lower()
    if name.endswith(('.fits', '.fit', '.fits.gz', '.fit.gz')):
        if kind == 'ivar':
            raise ValueError('Do not reinterpret FITS SIGMA as ivar; convert the original text product')
        spectrum = ispec.read_spectrum(str(path))
        return spectrum, {'source_error_kind': 'FITS-defined', 'converted_ivar': False,
                          'output_pixels': len(spectrum), 'invalid_pixels_removed': 0}

    header, skiprows = _text_layout(path)
    if header is not None:
        if len(header) != 3 or header[2] not in {'err', 'error', 'sigma', 'ivar'}:
            raise ValueError(f'Expected wavelength/flux/err or ivar header in {path}')
        declared = 'ivar' if header[2] == 'ivar' else 'sigma'
        if kind != 'auto' and kind != declared:
            raise ValueError(f'error_kind={kind} contradicts the {header[2]} header in {path}')
        kind = declared
    elif kind == 'auto':
        legacy_name = name.removesuffix('.gz')
        if legacy_name.endswith('_stitched_auto.txt') or legacy_name == 'sun_mike_vesta.txt':
            kind = 'ivar'
        else:
            raise ValueError(f'Headerless spectrum has ambiguous error semantics: {path}. '
                             'Specify error_kind="ivar" or "sigma".')

    values = np.loadtxt(path, skiprows=skiprows, ndmin=2)
    if values.shape[1] != 3:
        raise ValueError(f'Expected exactly 3 columns in {path}; found {values.shape[1]}')
    wave, flux, uncertainty = values.T
    valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(uncertainty)
    no_errors = kind == 'sigma' and np.all(uncertainty == 0)
    if not no_errors:
        valid &= uncertainty > 0
    sigma = np.zeros(len(values), dtype=float)
    if kind == 'ivar':
        sigma[valid] = 1.0 / np.sqrt(uncertainty[valid])
    else:
        sigma[valid] = uncertainty[valid]
    if not np.any(valid):
        raise ValueError(f'No valid pixels remain in {path}')
    spectrum = ispec.create_spectrum_structure(wave[valid], flux[valid], sigma[valid])
    order = np.argsort(spectrum['waveobs'], kind='stable')
    spectrum = spectrum[order]
    # Match iSpec's keep-first duplicate convention without dropping headerless
    # data rows or using its two-line legacy-header fallback.
    keep = np.r_[True, np.diff(spectrum['waveobs']) != 0]
    duplicates = int((~keep).sum())
    spectrum = spectrum[keep]
    report = {'source_error_kind': kind, 'converted_ivar': kind == 'ivar',
              'input_pixels': len(values), 'output_pixels': len(spectrum),
              'invalid_pixels_removed': int((~valid).sum()),
              'duplicate_pixels_removed': duplicates, 'header_present': header is not None}
    if kind == 'ivar':
        logging.info('Converted MIKE ivar to sigma=1/sqrt(ivar): %s', path)
    if report['invalid_pixels_removed']:
        logging.warning('Removed %d pixels with invalid flux/wavelength/uncertainty: %s',
                        report['invalid_pixels_removed'], path)
    return spectrum, report
