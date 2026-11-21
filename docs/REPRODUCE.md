# Reproduce the surface comparison

Run the commands from the root of this repository. The benchmark used Python 3.10.11, NumPy 1.26.4 and the Linux VC3D executable `vc_extend_edge`. The correction itself uses NumPy on CPU. A matching VC3D executable, or an executable wrapper that forwards its arguments to that build, is needed to regenerate the parent surfaces. The bundled TIFF reader can use system libtiff. For other compressed TIFF encodings install `tifffile` with `imagecodecs`.

[VC3D benchmark release](https://github.com/ScrollPrize/villa/releases/tag/vc3d-archive-latest-444656323) · [VC3D source at the benchmark commit](https://github.com/ScrollPrize/villa/tree/97fb9336c8898be27e78eff8cd23ce5e7185ec0a)

The Linux executable used in the study had SHA-256 `7449f2b1906a822074562e2ab3298534437e0dfbe826b28d38660c1199a29d37`. The examples below assume it is on your PATH. Alternatively, add `--extender /absolute/path/to/vc_extend_edge` to the command. A wrapper running inside a container must expose input, temporary and output paths at the same absolute locations. For a clean dependency install on Python 3.10, use `python3 -m venv .venv`, `source .venv/bin/activate`, then `python3 -m pip install numpy==1.26.4`.

## One real example

Obtain the volume NRRD (14.3 MiB) and the mask NRRD (458 KiB) from the official directory for region `01744_02256_04816`. Place them in a local folder called `input`. The mask enables the interval check. Mesh prediction uses the volume and a fixed starting point. [Official example files](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels-harmonized/01744_02256_04816/) · [Data terms](https://dl.ash2txt.org/LICENSE.txt)

```sh
python3 -m r103.demo \
  --volume input/01744_02256_04816_volume.nrrd \
  --mask input/01744_02256_04816_mask.nrrd \
  --out local_example
```

Use a new output directory. Open `local_example/VIEW.html` in a browser. Inspect the surface texture on each strip with the same depth and grayscale. Look down the centre and near both edges in the longitudinal sections. Change the position for a transverse section and toggle the mesh lines to see the material beneath them. Follow the 3D link to rotate the two meshes or show them one at a time. For the native application, open `local_example/PHercParis4.volpkg` in VC3D.

The first 33 × 9 vertices of all three forward strips have registered coordinate hashes. The two-direction demonstration uses one shared starting patch and 30 added rows in each direction. The saved local result had 63 × 9 vertices and an R103 centreline length of 124.4 voxels. `MEASUREMENT.json` records counts and lengths. With the official mask the two-direction example returned 55 of 62 supported intervals for VC3D and 62 for R103. An interval is supported when at least 24 of its 48 sample positions land exclusively near the target and none land exclusively near another labeled sheet. Inspect the CT separately for fiber visibility.

You can also replay the correction if you already have a local example containing the two parent meshes and the normalized crop. Pass its containing directory to `python3 -m r103.demo --local-example /path/to/example --out local_replay`. A forward-only fixture produces 33 rows. The extended local fixture also verifies the two-direction meshes. All images are produced locally from the authorized CT.

To generate the two central-layer TIFFs with the native VC3D renderer, run the following in a shell with `vc_render_tifxyz` on its PATH.

```sh
pkg=local_example/PHercParis4.volpkg
for arm in VC3D R103; do
  mkdir -p "local_example/native_$arm"
  vc_render_tifxyz -v "$pkg/volumes/ct.zarr" -s "$pkg/paths/$arm.tifxyz" \
    --scale 1 -g 0 --num-slices 1 --tif-output "local_example/native_$arm"
done
```

## The 28 forward comparisons

The seven region IDs, exact source digests, crop origins, starting instances and frozen coordinate hashes are in [starts.json](../results/starts.json). In the official collection choose each region by its ID. Download its `_volume.nrrd` and `_mask.nrrd` into `input/` or `input/<region>/`. For a single-start replay, only that region's two files are needed. [Official region catalogue](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/instance-labels-harmonized/)

```sh
python3 -m r103.panel --data-dir input --out all_starts

python3 -m r103.panel --data-dir input \
  --case 10192_04048_03280_seed_2 --out example_loss
```

The first command compares all 84 method-by-start results with [the trial table](../results/paired_scores.csv). The second writes `example_loss/VIEW.html` and its 3D overlay for the largest observed decline in target area. In each region, surfaces are computed and checked against their recorded hashes before the mask is opened for scoring.

Each start uses a 128³ crop. CT intensities are normalized using the 2nd and 98th percentiles of every third voxel along each axis, then clipped and rounded to uint8. A 3 × 9 starting edge is oriented from CT at the registered point. Each forward strip may grow by three blocks of ten rows. For area scoring, each quadrilateral is split into two triangles with three sampling positions per triangle. Each position receives a fixed share of the corresponding starting-edge triangle area. The nearest voxel is classified within a two-voxel radius as target only, other sheet only, both, or neither. These four classes are counted separately. Means give each of the 28 starts equal weight.

## Apply R103 to an existing surface

Supply a normalized local 128³ uint8 Z,Y,X Zarr crop and two complete `tifxyz` strips (3–33 rows, nine columns) in that coordinate frame. Both strips must have the same first three rows. The typical inputs use an uncompressed, gzip or zlib Zarr v2 level 0. Other encodings may require compatible `zarr` and `numcodecs` packages.

```sh
python3 -m r103.parents --ct crop.zarr --seed seed.tifxyz --out parents
python3 -m r103.correct --ct crop.zarr --vc3d parents/VC3D.tifxyz \
  --r094 parents/R094.tifxyz --out corrected.tifxyz
```

If you already have the VC3D and uniform-correction ribbons, only the second command is needed. The output uses the input crop's local voxel coordinates.
