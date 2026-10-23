#!/usr/bin/env python3
"""Check local prerequisites without fitting spectra or changing outputs."""
import argparse
import importlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'Code'), str(ROOT)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples', action='store_true')
    p.add_argument('--hfs', action='store_true')
    args = p.parse_args()
    errors = []
    for name in ['numpy','scipy','pandas','matplotlib','astropy','openpyxl','ispec','q2']:
        try:
            if name == 'q2':
                qroot=Path.home()/'q2-tools'
                if not (qroot/'Data/ModelAtmospheres').is_dir() or not (qroot/'MOOG-for-q2/MOOGSILENT').is_file():
                    errors.append('q2 data/MOOG prerequisites missing under ~/q2-tools; import skipped to avoid automatic downloads. Check your q2 configuration if installed elsewhere.')
                    continue
            mod = importlib.import_module(name)
            print(f'OK import {name}: {getattr(mod,"__version__","version not exposed")}')
            if name == 'ispec' and not hasattr(mod, 'plot_spectra'):
                errors.append('iSpec lacks plot_spectra; review patches/README.md.')
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    paths=['input/atmospheres/MARCS.GES/parameters.tsv',
           'input/abundances/Grevesse.2007/stdatom.dat',
           'input/linelists/linelist_melendez2014/linelist_melendez2014_ispec.tsv',
           'input/regions/fe_lines_continuum.txt']
    for rel in paths:
        if not (ROOT/rel).exists():errors.append(f'Missing {rel}')
    # Check compiled iSpec MOOG, rather than treating a source directory as sufficient.
    moog=ROOT/'synthesizer/moog/MOOGSILENT'
    if not moog.is_file() or not os.access(moog,os.X_OK):
        errors.append('Missing executable synthesizer/moog/MOOGSILENT; compile iSpec MOOG.')
    if args.hfs:
        from HFSBlends import _find_moog_executable
        if not _find_moog_executable():errors.append('HFS requires MOOGSILENT_PATH or MOOGSILENT on PATH.')
        lm=os.environ.get('ISPEC_LINEMAKE_BIN') or os.environ.get('LINEMAKE_BIN') or str(ROOT/'tools/linemake/linemake.go')
        if not Path(lm).is_file() or not os.access(lm,os.X_OK):errors.append('HFS requires an executable linemake.')
    if args.examples:
        try:
            from run_context import load_run_context
            ctx=load_run_context('HD73350')
            for key in ['input_spectra_path','spectrum_norm_path','solar_lines_csv']:
                if not Path(ctx[key]).is_file():errors.append(f'Missing {key}: {ctx[key]}')
            if ctx['solar_label']!='Sun_MIKE_vesta_R45500' or ctx['solar_resolution']!=45500:
                errors.append('Example solar selection is not MIKE Vesta R45500.')
            import ispec
            for name in ['HD73350_normalized.fits','Sun_MIKE_vesta_R45500_normalized.fits']:
                s=ispec.read_spectrum(str(ROOT/'examples/mike'/name))
                print(f'OK example {name}: {len(s)} pixels')
        except Exception as exc:errors.append(f'Example inputs: {exc}')
    for err in errors:print('ERROR:',err)
    print(f'{len(errors)} prerequisite error(s). This is not an end-to-end abundance validation.')
    return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
