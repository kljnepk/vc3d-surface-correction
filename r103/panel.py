"""Replay registered starts on PHerc. Paris. 4 from local official NRRD files."""
import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .corrector import correct
from .demo import extend
from .views import page
from .view_mesh import write_view
from .formats import image8, volume_write, surface_write
from .inputs import neutral_seed, nrrd
from .scoring import masks, score

BASE = Path(__file__).resolve().parent.parent / "results"
METRICS = {"target_area_vox2":"goal", "other_sheet_area_vox2":"other",
           "unmarked_area_vox2":"unmarked", "ambiguous_area_vox2":"ambiguous",
           "switch_columns":"switches","complete":"complete","added_rows":"rows"}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_cube(root, region, kind, expected_sha):
    name = f"{region}_{kind}.nrrd"
    candidates = (root / name, root / region / name, root / f"{region}_{kind}" / name)
    for path in candidates:
        if path.is_file():
            if path.stat().st_size > 40 * 1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
                raise ValueError(f"Unexpected source bytes in {path}")
            array, _ = nrrd(path)
            if array.shape != (256, 256, 256):
                raise ValueError(f"Unexpected shape in {path}")
            return array
    raise FileNotFoundError(f"Place the official {name} in {root}.")


def predict_start(ct, seed, reference, work, extender):
    if digest(ct) != reference["crop_sha256"] or digest(seed) != reference["seed_sha256"]:
        raise ValueError("Crop or starting edge differs from the registered input.")
    vol = work / "ct.zarr"
    volume_write(vol, ct)
    anchor = work / "seed.tifxyz"
    surface_write(anchor, seed)
    before, prior = extend(vol, anchor, work, extender, False)
    final = correct(ct, before, prior)[0]
    meshes = {"VC3D": before, "R094": prior, "R103": final}
    for method, mesh in meshes.items():
        if digest(mesh.astype("<f4", copy=False)) != reference["mesh_sha256"][method]:
            raise ValueError(f"{method} differs from the registered surface. Check the VC3D build and local inputs.")
    return meshes


def compare_scores(meshes, seed, labels, instance, expected):
    if int(labels[64,64,64]) != instance:
        raise ValueError("The annotation at the registered starting point has changed.")
    target, other = masks(labels)
    rows = []
    for method, mesh in meshes.items():
        result = score(mesh, seed, target, other)
        measured = {col: result[key] for col, key in METRICS.items()}
        for index, block in enumerate(result["blocks"], 1):
            measured[f"target_area_by_step_{index}"] = block[1]
        for col, value in measured.items():
            if abs(float(value) - float(expected[method][col])) > 1e-6:
                raise ValueError(f"{method} differs from the archived {col}.")
        measured["complete"] = int(measured["complete"])
        rows.append({"fixed_start":expected[method]["fixed_start"], "method":method, **measured})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory of unmodified official volumes and masks")
    parser.add_argument("--extender", default="vc_extend_edge", help="Path to the VC3D binary or executable wrapper")
    parser.add_argument("--case", help="Optional fixed start ID from results/paired_scores.csv")
    parser.add_argument("--out", type=Path, required=True, help="New directory for the validated scores")
    args = parser.parse_args()
    out = args.out.absolute()
    if out.exists() or out.is_symlink():
        parser.error("Choose a new output directory.")
    with (BASE / "paired_scores.csv").open(newline="") as src:
        table = csv.DictReader(src)
        fields = table.fieldnames
        expected = {}
        for row in table:
            expected.setdefault(row["fixed_start"], {})[row["method"]] = row
    if args.case and args.case not in expected:
        parser.error("Unknown --case. Use a fixed_start value from results/paired_scores.csv.")
    registry = json.loads((BASE / "starts.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temporary:
        work = Path(temporary)
        stage = work / "output"
        stage.mkdir()
        count = 0
        with (stage / "verified_scores.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for region, entry in registry.items():
                selected = [(f"{region}_seed_{k}", data) for k, data in entry["starts"].items()
                            if args.case in (None, f"{region}_seed_{k}")]
                if not selected:
                    continue
                raw = read_cube(args.data_dir, region, "volume", entry["volume_sha256"])
                frozen = {}
                for key, data in selected:
                    crop = tuple(slice(v, v + 128) for v in data["origin_zyx"])
                    ct = image8(raw[crop])
                    seed = neutral_seed(ct)
                    with tempfile.TemporaryDirectory(dir=work) as one:
                        meshes = predict_start(ct, seed, data, Path(one), args.extender)
                    frozen[key] = (meshes, seed, crop, data["target_instance"])
                labels = read_cube(args.data_dir, region, "mask", entry["mask_sha256"])
                for key, (meshes, seed, crop, instance) in frozen.items():
                    writer.writerows(compare_scores(meshes, seed, labels[crop], instance, expected[key]))
                    if args.case:
                        label = f"PHerc. Paris. 4 · region {region} · start {key.rsplit('_', 1)[1]}"
                        before, after = meshes["VC3D"], meshes["R103"]
                        write_view(stage / "MESH_3D.html", before, after, seed[1, 4], label)
                        page(image8(raw[crop]), before, after, stage / "VIEW.html",
                             "Forward extension from the registered starting edge.",
                             measurements='<p><a href="MESH_3D.html">Rotate the paired meshes in 3D</a></p>',
                             label=label)
                    count += 1
                    print(f"{key} matches all three registered surfaces and scores.", flush=True)
        if count != (1 if args.case else len(expected)):
            raise ValueError("Incomplete replay.")
        stage.rename(out)
    print(f"Verified {count} starts. Results in {out / 'verified_scores.csv'}")


if __name__ == "__main__":
    main()
