#!/usr/bin/env python3
"""Build a clean archive from the reviewed release file allowlist, without Git history."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=ROOT/'release_manifest.json'
    entries=json.loads(manifest.read_text())['files']
    for entry in entries:
        p=ROOT/entry['path']
        if p.is_symlink() or not p.is_file():raise RuntimeError(f'Invalid release file: {p}')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError(f'File changed since release review: {entry["path"]}. Review and update release_manifest.json before exporting.')
    destination=ROOT/'dist/iSpec_EWAbund_20261022.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for rel in [e['path'] for e in entries]+['release_manifest.json']:
            z.write(ROOT/rel,'iSpec_EWAbund/'+rel)
    print(destination)
    print('SHA256:',hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__=='__main__':main()
