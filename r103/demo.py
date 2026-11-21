"""Rebuild the R103 before/after example locally from the official scan or local inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
from .formats import image8, surface_read, surface_write, volume_write
from .zarr_volume import crop_array
from .corrector import correct
from .inputs import nrrd, neutral_seed
from .views import page, midline_length
from .scoring import masks, classify, triangles
from .view_mesh import write_view

VOLUME_SHA = 'f848002604a61d02b8c8cfbaf0e992f4a412029901fd8f0139bdbc5368d4bf97'
LABEL_SHA = '272d857118eb56c3b36cb8ed9894de4dee25213bf6e98a7cf91d13e4b54a331f'
CROP_SHA = 'f43fb04828a6c3a6b8b49d1d071d5d3fecdc43de04a2663e7c4803e801b62e0a'
SEED_SHA = '30e18122b00c1a668eae97b447cd3687e0957ab123f736a7cac7542dbeecdbdc'
FORWARD_SHA = {
    'VC3D': '5d9f2cb497afc43be0fcd683efc92d61841211c44145a341134bad3df0a3894d',
    'R094': '584f37dcf34a77fb4c2903f4046c4955c200d1d1a8ea57914e10725960a092d7',
    'R103': '65e06307440a8743e038b74b24dae01834dd47d6b4195eb141418353b6e2f638',
}
VPKG = 'PHercParis4.volpkg'
CROP = (slice(11, 139), slice(18, 146), slice(18, 146))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def official_crop(path, expected):
    if path.stat().st_size > 40 * 1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('This is not the registered source NRRD. See README.md for the official file.')
    array, _ = nrrd(path)
    if array.shape != (256, 256, 256):
        raise ValueError('Unexpected source shape.')
    return array[CROP]


def mesh(path):
    grid, valid = surface_read(path)
    if not valid.all():
        raise ValueError(f'Incomplete coordinates in {path.name}.')
    return grid


def replay(ct, base, prior, seed):
    if not np.array_equal(base[:3], seed) or not np.array_equal(prior[:3], seed):
        raise ValueError('The two input meshes must both start at the registered edge.')
    after, _, _ = correct(ct, base, prior)
    return {'VC3D': base, 'R094': prior, 'R103': after}


def extend(vol, anchor, work, extender, reverse):
    dest = work / ('reverse' if reverse else 'forward')
    cmd = [sys.executable, '-B', '-m', 'r103.parents', '--ct', str(vol.resolve()),
           '--seed', str(anchor.resolve()), '--out', str(dest.resolve()), '--extender', extender]
    if reverse:
        cmd.append('--reverse')
    subprocess.run(cmd, cwd=ROOT, check=True)
    return mesh(dest/'VC3D.tifxyz'), mesh(dest/'R094.tifxyz')


def join(forward, reverse):
    names = ('VC3D', 'R103')
    if reverse is None:
        return {name: forward[name] for name in names}
    n = min(len(reverse[name]) for name in names)
    return {name: np.concatenate((reverse[name][3:n][::-1], forward[name])) for name in names}


def annotation_counts(grid, target, other):
    quadrature = np.eye(3) * 0.5 + np.ones((3, 3)) / 6
    points = np.einsum('qv,rcsvd->rcsqd', quadrature, triangles(grid))
    groups = classify(points, target, other).reshape(len(grid) - 1, -1)
    target_only = (groups == 1).sum(axis=1)
    other_only = (groups == 2).sum(axis=1)
    supported = (target_only * 2 >= groups.shape[1]) & (other_only == 0)
    return {'intervals': len(supported), 'supported_intervals': int(supported.sum()),
            'samples_per_interval': int(groups.shape[1])}


def build(args, stage, work):
    local = args.local_example
    ct = (crop_array(local/VPKG/'volumes'/'ct.zarr', None)[0] if local else
          image8(official_crop(args.volume, VOLUME_SHA)))
    if args.verify_source and not np.array_equal(ct, image8(official_crop(args.verify_source, VOLUME_SHA))):
        raise ValueError('The official NRRD and the archived local crop differ after normalization.')
    seed = neutral_seed(ct)
    if digest(ct) != CROP_SHA or digest(seed) != SEED_SHA:
        raise ValueError('The normalized CT crop or its starting edge differs from the reference.')
    pkg = stage / VPKG
    vol = pkg/'volumes'/'ct.zarr'
    volume_write(vol, ct)
    (pkg/'config.json').write_text(json.dumps({'name': 'R103 local demonstration', 'version': 7,
                                              'materialthickness': 150.0}) + '\n')
    anchor = work/'anchor.tifxyz'
    surface_write(anchor, seed)
    if local:
        paths = local/VPKG/'paths'
        base, prior = mesh(paths/'01_VC3D.tifxyz'), mesh(paths/'02_R094.tifxyz')
    else:
        base, prior = extend(vol, anchor, work, args.extender, False)
    forward = replay(ct, base, prior, seed)
    for name, grid in forward.items():
        if digest(grid) != FORWARD_SHA[name]:
            raise ValueError(f'{name} differs from the registered forward result. Check the VC3D build and inputs.')
    reverse = None
    if not local:
        b, p = extend(vol, anchor, work, args.extender, True)
        reverse = replay(ct, b, p, seed[::-1])
    elif (local/'REVERSE_INPUTS').is_dir():
        inputs = local/'REVERSE_INPUTS'
        reverse = replay(ct, mesh(inputs/'before.tifxyz'), mesh(inputs/'prior.tifxyz'), seed[::-1])
        if not np.array_equal(reverse['R103'], mesh(inputs/'after.tifxyz')):
            raise ValueError('The corrected reverse mesh differs from the local source.')
    results = join(forward, reverse)
    for name, grid in results.items():
        if local and reverse is not None:
            if not np.array_equal(grid, mesh(paths/(name+'_FULL.tifxyz'))):
                raise ValueError(f'{name} differs from the full local two-direction mesh.')
        surface_write(pkg/'paths'/(name+'.tifxyz'), grid, name)
    summary = {'forward_vertices_verified': int(forward['R103'].shape[0] * forward['R103'].shape[1]),
               'mode': 'archived_inputs' if local else 'official_extender',
               'normalized_from_official_nrrd': bool(args.volume or args.verify_source),
               'rows': int(len(results['R103'])),
               'measured_centerline_voxels': {k: midline_length(g) for k, g in results.items()}}
    write_view(stage/'MESH_3D.html', results['VC3D'], results['R103'], seed[1, 4])
    if args.mask:
        labels = official_crop(args.mask, LABEL_SHA)
        if int(labels[64,64,64]) != 91:
            raise ValueError('Unexpected instance at the shared start.')
        target, other = masks(labels)
        summary['annotation'] = {k: annotation_counts(g, target, other) for k,g in results.items()}
    explanation = ('Both directions share one edge. ' if reverse is not None else 'Forward direction only. ')
    explanation += 'The images and the longitudinal and transverse sections below are generated locally from CT.'
    page(ct, results['VC3D'], results['R103'], stage/'VIEW.html', explanation, measurements='<p><a href="MESH_3D.html">Rotate the paired meshes in 3D</a></p>')
    (stage/'MEASUREMENT.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--volume', type=Path, help='Authorized original region volume NRRD')
    source.add_argument('--local-example', type=Path, help='Extracted local example with PHercParis4.volpkg')
    parser.add_argument('--verify-source', type=Path, help='Also compare the archived crop to the official volume NRRD')
    parser.add_argument('--extender', default='vc_extend_edge', help='Executable or wrapper for the official VC3D extender')
    parser.add_argument('--mask', type=Path, help='Optional official mask, used after prediction for annotation checks')
    parser.add_argument('--out', required=True, type=Path, help='New local output directory')
    args = parser.parse_args()
    if args.verify_source and not args.local_example:
        parser.error('--verify-source is used with --local-example.')
    out = args.out.absolute()
    if out.exists() or out.is_symlink():
        parser.error('Choose a new output directory.')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        work = Path(tmp)
        stage = work/'output'
        stage.mkdir()
        summary = build(args, stage, work)
        stage.rename(out)
    print(json.dumps(summary, indent=2, allow_nan=False))
    print('VIEW=' + str(out/'VIEW.html'))


if __name__ == '__main__':
    main()
