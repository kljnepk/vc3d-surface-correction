# R094 · row-uniform CT-guided correction

Supplement to the R103 submission (21 November 2026, November Progress Prizes). R094 was included in the original repository and is a candidate used by R103. It can be run without executing the R103 correction, but requires the VC3D extender.

## Results

| Scroll | Starts / regions | VC3D | R094 | R103 | R094 vs VC3D |
|---|---:|---:|---:|---:|---:|
| PHerc. 172 | 24 / 8 | 14.42 | 20.25 | 20.67 | +40.5% |
| PHerc. 332 | 24 / 8 | 14.54 | 21.04 | 19.92 | +44.7% |
| Both panels | 48 / 16 | 14.48 | 20.65 | 20.29 | +42.6% |

The score is the longest uninterrupted sequence of reference stations within two voxels of a triangle of the newly extended mesh. There are 41 stations, sampled at +10 through +50 voxels along the held-back manual continuation. Positions outside the CT crop break the sequence for all methods. The first eight voxels of the manual trace supply the starting direction. Each region contains three starts and receives equal weight.

R094 improves the regional mean over VC3D in six of eight regions on PHerc. 172 and eight of eight on PHerc. 332. Across both panels R094 wins against R103 at 19 starts, ties at 16 and loses at 13. The combined mean difference is only 0.354 points; this is not evidence of statistically established superiority.

The original PHerc. Paris. 4 benchmark measures a different property. R094 increased mean target-sheet area from 532.46 to 559.36 voxel², but also increased mean other-sheet area from 0.89 to 1.89 voxel². R103 achieved 589.64 and 0.11 voxel² respectively. Neither fibre proximity nor these area measurements establish ink legibility or operator time saved.

[All 48 paired scores](paired-scores.csv) · [Original benchmark and full protocol](../r103-external-fibres.md)

## Method

The [uniform correction](../r103/uniform.py) estimates a scalar normal displacement for each longitudinal row. That scalar is shared across the width of the row and applied along each vertex's local normal. The [profile-path code](../r103/profile_path.py) searches discrete offsets from −3 to +3 voxels in 0.5-voxel steps using dynamic programming with profile similarity and transition penalties. The uniform correction evaluates candidate paths, including an unchanged strip, and selects by CT-profile fit and displacement and regularity costs. A reverse path and a round-trip-checked candidate are included; not every selected candidate is required to pass a round-trip check. Degenerate or unsuitable frames can be left unchanged. The correction needs no training or sheet mask at inference.

The [parent-ribbon builder](../r103/parents.py) runs at most three ten-row extensions, using the last three rows as the next anchor. From the repository root, provide a normalized 128³ uint8 Zarr crop in Z,Y,X order, a complete 3×9 seed in the same coordinate frame, and the matching VC3D executable:

```sh
python3 -m r103.parents --ct crop.zarr --seed seed.tifxyz \
  --out parents --extender /absolute/path/to/vc_extend_edge
```

The command produces both `parents/VC3D.tifxyz` and `parents/R094.tifxyz`, without running `r103.correct`. Use a new output directory. See [reproduction instructions](../docs/REPRODUCE.md) for dependencies and the pinned VC3D build.

## Fresh replay of five illustrative cases (26 November 2026)

| Case | Scroll | VC3D | R094 | R103 (archived) |
|---|---|---:|---:|---:|
| 01 | 172 | 4 | 29 | 17 |
| 02 | 172 | 7 | 1 | 26 |
| 03 | 332 | 6 | 41 | 41 |
| 04 | 332 | 28 | 41 | 23 |
| 05 | 332 | 41 | 27 | 15 |

These examples were selected after scoring to illustrate gains, losses and equal results, not to form a new benchmark. Case 03 has identical R094 and R103 mesh coordinates. Case 05 is a baseline win.

The baseline and uniform correction were rerun from the five verified source TIFFs using the original pinned VC3D build, Python 3.10.11 and NumPy 1.26.4. The ten regenerated mesh coordinate arrays matched their archived float32 byte digests exactly, with 25 native extender calls and no reuse of stored extension results. R103 was not rerun. This is a replay of five selected cases in the same numerical environment, not a repeat of all 48 starts.

[Replay verification and coordinate hashes](replay-verification.json)

The archived numerical recheck also verified 84 manifest entries and recomputed 5,868 in-crop point-to-triangle distances (36 out-of-crop entries) for the three methods across all 48 starts. Every longest-run score was reproduced; maximum distance difference was 3.55 × 10⁻¹⁵ voxels. These checks validate the saved computations, not the manual references or complete sheet identity.

## Data provenance

The external fibre experiments use Vesuvius Challenge [fiber-skeletons (2025-07)](https://scrollprize.org/data_datasets#fiber-skeletons-2025-07), Dataset004_sk-fibers-binary-20250728 (Scroll 5 and Scroll 3), including manually traced WEBKNOSSOS annotations. [Dataset README](https://dl.ash2txt.org/datasets/fiber-skeletons/README.txt). The original Paris. 4 benchmark uses EduceLab-Scrolls, EduceLab / University of Kentucky ([Parsons et al., 2023](https://doi.org/10.48550/arXiv.2304.02084)). See [official citation guidance](https://scrollprize.org/data) and [data-server terms](https://dl.ash2txt.org/LICENSE.txt).

Original result archive SHA-256:

| Panel | SHA-256 |
|---|---|
| PHerc. 172 (R115) | `4f3444256991c5e4dd82d44e8c3fe445279e29ce90114a11848af0a8b4c64848` |
| PHerc. 332 (R116) | `bce5c5ceefd7eacc46d6f5ebbc53b4a9031a2d243861cb8d175a5f8a61460e98` |

The algorithm and original results remain unchanged. No source TIFFs or original annotation files are included here.
