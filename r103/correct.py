"""Correct a VC3D surface extension using its local CT crop and a second candidate surface."""
import argparse
import tempfile
from pathlib import Path

import numpy as np

from .corrector import correct
from .formats import surface_read, surface_write
from .zarr_volume import crop_array


def load_complete_strip(path):
    mesh, valid = surface_read(path)
    if not valid.all():
        raise ValueError("Input meshes require complete rows.")
    return mesh


def run(ct_path, base_path, prior_path, dest):
    if dest.exists() or dest.is_symlink():
        raise ValueError("Choose a new output path.")
    ct, origin = crop_array(ct_path, None)
    if np.any(origin):
        raise ValueError("The mesh coordinates must use the local crop origin.")
    surface, _, selection = correct(ct, load_complete_strip(base_path), load_complete_strip(prior_path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / "result"
        surface_write(stage, surface, ident=dest.name)
        actual, valid = surface_read(stage)
        if not valid.all() or not np.array_equal(actual, surface):
            raise ValueError("Coordinates did not survive the tifxyz round trip.")
        stage.rename(dest)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path, help="Local 128-cubed uint8 level-0 Zarr")
    parser.add_argument("--vc3d", required=True, type=Path, help="VC3D tifxyz strip")
    parser.add_argument("--r094", required=True, type=Path, help="Uniform correction tifxyz strip on the same starting edge")
    parser.add_argument("--out", required=True, type=Path, help="New output tifxyz directory")
    args = parser.parse_args()
    selection = run(args.ct, args.vc3d, args.r094, args.out)
    print(f"Selected {selection['selected']}. Wrote {args.out}")


if __name__ == "__main__":
    main()
