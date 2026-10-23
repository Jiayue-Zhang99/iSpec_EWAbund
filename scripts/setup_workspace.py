#!/usr/bin/env python3
"""Link an existing iSpec runtime and install the two authorized example inputs.

Never modify the external installation or replace existing local results.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(source, destination, required=True):
    if not source.exists():
        if required:
            raise FileNotFoundError(source)
        print(f'Optional resource absent: {source.name}')
        return
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f'Refusing to replace {destination}; relocate it or select a fresh checkout.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def install_sample(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f'Preserving existing {destination.relative_to(ROOT)}')
    else:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ispec-root', required=True, type=Path)
    parser.add_argument('--linemake', type=Path, help='Compiled linemake executable (optional).')
    parser.add_argument('--examples', action='store_true', help='Install MIKE metadata, solar tables and normalized target spectrum.')
    args = parser.parse_args()
    engine = args.ispec_root.expanduser().resolve()
    if engine == ROOT:
        parser.error('--ispec-root must point to a separate full iSpec installation.')
    required = ['ispec', 'synthesizer', 'input/atmospheres/MARCS.GES',
                'input/abundances/Grevesse.2007']
    missing = [p for p in required if not (engine/p).exists()]
    if missing:
        parser.error('Missing iSpec resources: '+', '.join(missing))
    for rel in required:
        link(engine/rel, ROOT/rel)
    for rel in ['input/abundances/Grevesse.1998', 'input/linelists/CCF',
                'input/spectra/templates', 'input/regions/strong_lines']:
        link(engine/rel, ROOT/rel, required=False)
    # VALD is optional for the prepared example linelist, required for rebuilding it.
    rel = 'input/linelists/transitions/VALD.300_1100nm'
    link(engine/rel, ROOT/rel, required=False)
    linemake = args.linemake or engine/'tools/linemake/linemake.go'
    link(linemake.expanduser(), ROOT/'tools/linemake/linemake.go', required=False)
    if args.examples:
        example = ROOT/'examples/mike'
        for name in ['solar_abund_instru.csv', 'solar_lines_instru.csv']:
            install_sample(example/name, ROOT/'input'/name)
        install_sample(example/'input_spectra_info.xlsx', ROOT/'input/spectra/input_spectra_info.xlsx')
        install_sample(example/'HD73350_normalized.fits', ROOT/'output/HD73350/spectra_HD73350_normalized_splines_segmts.fits')
    # Local-only configuration, explicitly ignored by Git.
    (ROOT/'.runtime.json').write_text(json.dumps({'ispec_root':str(engine)}, indent=2)+'\n')
    print('Setup complete. Run scripts/check_environment.py before the notebooks.')


if __name__ == '__main__':
    main()
