"""Build the two parent ribbons used by R103 from VC3D and a shared 3×9 seed."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from .formats import surface_read, surface_write
from .uniform import correct_uniform
from .alignment import align_public
from .corrector import validate_grid
from .zarr_volume import crop_array


def parent_ribbons(ct, seed, extend):
    memo = {}
    def pair(seam):
        stamp = seam.tobytes()
        if stamp not in memo:
            mesh = extend(seam)
            validate_grid(mesh)
            if len(mesh) > 13 or not np.array_equal(mesh[:3], seam):
                raise ValueError('The extender must retain all three anchor rows.')
            corrected = correct_uniform(ct, seam, mesh, np.ones((13, 9), bool)) if len(mesh) == 13 else mesh
            memo[stamp] = mesh, corrected
        return memo[stamp]
    histories = []
    for arm in (0, 1):
        grid = seed.copy()
        for _ in range(3):
            addition = pair(grid[-3:])[arm]
            grid = np.concatenate((grid, addition[3:]))
            if len(addition) < 13:
                break
        validate_grid(grid)
        histories.append(grid)
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ct', 'seed', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--extender', default='vc_extend_edge')
    parser.add_argument('--reverse', action='store_true', help='Extend from the opposite edge of the same three anchor rows.')
    args = parser.parse_args()
    image, _ = crop_array(args.ct, None)
    seed, ok = surface_read(args.seed)
    validate_grid(seed)
    if len(seed) != 3 or not ok.all() or args.out.exists():
        parser.error('Supply a complete 3×9 seed and a new output directory.')
    if args.reverse:
        seed = np.ascontiguousarray(seed[::-1])
    def one(seam):
        with tempfile.TemporaryDirectory() as w:
            t = Path(w)
            surface_write(t/'seed.tifxyz', seam)
            subprocess.run([args.extender, '-i', str(t/'seed.tifxyz'), '-o', str(t/'extended.tifxyz'), '-n', str(args.ct.resolve()),
                            '--sides', 'down', '--cells', '10', '--rounds', '1'], check=True)
            grid, valid = align_public(*surface_read(t/'extended.tifxyz'), seam)
            rows = np.flatnonzero(~(valid.all(axis=1) & ((grid>=0) & (grid<=127)).all(axis=(1,2))))
            grid = grid[:int(rows[0]) if len(rows) else len(grid)].copy()
            if len(grid) < 3:
                raise ValueError('Incomplete starting edge.')
            grid[:3] = seam
            return grid
    base, prior = parent_ribbons(image, seed, one)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as w:
        stage = Path(w)/'parents'
        stage.mkdir()
        surface_write(stage/'VC3D.tifxyz', base)
        surface_write(stage/'R094.tifxyz', prior)
        stage.rename(args.out)


if __name__ == '__main__':
    main()
