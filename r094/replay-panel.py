#!/usr/bin/env python3
"""Replay the registered PHerc. 172 and 332 fibre benchmark."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from r103.formats import image8
from r103.panel import predict_start
from fibre_score import longest_run, reference_points

METHODS = ('VC3D', 'R094', 'R103')
FIELDS = ('revision', 'region', 'start', *METHODS)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_file(spec, directory, download):
    path = directory / spec['path']
    def verify(candidate):
        if candidate.stat().st_size != spec['bytes'] or file_hash(candidate) != spec['sha256']:
            raise ValueError(f'Official source size or SHA-256 mismatch: {path}')
    if path.exists() or not download:
        verify(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {spec['path']}", flush=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp) / 'download'
        with urllib.request.urlopen(spec['url'], timeout=120) as response, staged.open('wb') as dest:
            shutil.copyfileobj(response, dest)
        verify(staged)
        os.replace(staged, path)
    return path


def read_crop(image, side, origin):
    from PIL import Image
    z, y, x = origin
    with Image.open(image) as volume:
        if (volume.n_frames, *volume.size) != (side, side, side):
            raise ValueError(f'Unexpected TIFF dimensions for {image.name}')
        frames = []
        for page in range(z, z + 128):
            volume.seek(page)
            frames.append(np.array(volume.crop((x, y, x + 128, y + 128))))
    return image8(np.stack(frames))


def regenerate(blocks, args, stage):
    # Finish and verify every prediction before opening any held-back NML.
    for block in blocks:
        image = source_file(block['ct'], args.data_dir, args.download)
        for start in block['starts']:
            ident = start['id']
            print(f'Regenerating {ident}', flush=True)
            ct = read_crop(image, block['source_side'], start['crop_origin_zyx'])
            seed = np.array(start['seed_zyx'], dtype='<f4')
            with tempfile.TemporaryDirectory(dir=stage) as tmp:
                meshes = predict_start(ct, seed, start, Path(tmp), args.extender)
                np.savez_compressed(stage / (ident + '.npz'), **meshes)
    for block in blocks:
        nml = source_file(block['nml'], args.data_dir, args.download)
        for start in block['starts']:
            reference = reference_points(nml, block, start)
            with np.load(stage / (start['id'] + '.npz'), allow_pickle=False) as meshes:
                yield measure(block, start, reference, meshes)


def measure(block, start, reference, meshes):
    ident = start['id']
    reference = np.asarray(reference, dtype='<f8')
    if reference.shape != (41, 3) or hashlib.sha256(reference.tobytes()).hexdigest() != start['reference_sha256']:
        raise ValueError(f'Registered reference mismatch for {ident}')
    row = dict(revision=block['revision'], region=block['block'], start=ident)
    for method in METHODS:
        mesh = np.asarray(meshes[method], dtype='<f4')
        if hashlib.sha256(mesh.tobytes()).hexdigest() != start['mesh_sha256'][method]:
            raise ValueError(f'Registered {method} mesh mismatch for {ident}')
        row[method] = str(longest_run(reference, mesh))
    return row


def rescore_archives(register, archives):
    for revision, path in zip(('R115', 'R116'), archives):
        if file_hash(path) != register['archives'][revision]:
            raise ValueError(f'Expected the original {revision} result ZIP at {path}')
        with zipfile.ZipFile(path) as archive:
            references = json.loads(archive.read('PUNKTY.json'))['references_local_zyx']
            for block in register['blocks']:
                if block['revision'] != revision:
                    continue
                for start in block['starts']:
                    ident = start['id']
                    with np.load(io.BytesIO(archive.read(f'ZAMROZONE/{ident}.npz')), allow_pickle=False) as meshes:
                        yield measure(block, start, references[ident], meshes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data-dir', type=Path, help='Official dataset root with imagesTr/ and nml/')
    source.add_argument('--archives', type=Path, nargs=2, metavar=('R115_ZIP', 'R116_ZIP'),
                        help='Rescore previously generated meshes from the two original result ZIPs')
    parser.add_argument('--download', action='store_true', help='Fetch missing registered inputs to --data-dir')
    parser.add_argument('--extender', help='Pinned VC3D executable or wrapper (default: vc_extend_edge on PATH)')
    parser.add_argument('--out', required=True, type=Path, help='New output directory')
    args = parser.parse_args()
    if args.archives and (args.extender or args.download):
        parser.error('--extender and --download apply only to --data-dir')
    if args.data_dir:
        executable = shutil.which(args.extender or 'vc_extend_edge')
        if not executable:
            parser.error('Install the pinned VC3D extender or supply --extender /path/to/executable')
        args.extender = str(Path(executable).absolute())
    if args.out.exists() or args.out.is_symlink():
        parser.error('Choose a new output directory')

    register = json.loads((HERE / 'fibre-panel.json').read_text())
    with (HERE / 'paired-scores.csv').open(newline='') as table:
        expected = list(csv.DictReader(table))
    if len(expected) != 48 or len({r['start'] for r in expected}) != 48:
        raise ValueError('Expected 48 distinct published starts')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / 'output'
        stage.mkdir()
        rows = list(rescore_archives(register, args.archives) if args.archives
                    else regenerate(register['blocks'], args, stage))
        if len(rows) != len(expected):
            raise ValueError(f'Expected {len(expected)} rows, obtained {len(rows)}')
        for measured, published in zip(rows, expected):
            if measured != published:
                raise ValueError(f'Paired score mismatch: measured {measured}, published {published}')
        with (stage / 'paired-scores.csv').open('w', newline='') as table:
            writer = csv.DictWriter(table, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        stage.rename(args.out)
    mode = 'archived meshes rescored' if args.archives else 'meshes regenerated from official CT and scored'
    print(f'PASS: 48 starts, 144 scores match the published table ({mode}).')


if __name__ == '__main__':
    main()
