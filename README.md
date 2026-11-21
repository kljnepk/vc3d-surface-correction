# R103 · CT-guided correction for VC3D edge extensions

R103 adjusts a narrow surface strip after VC3D has extended it. If the mesh strays onto a neighbouring sheet, any downstream ink detector can sample the wrong page. R103 uses the local CT signal and the original starting edge to choose a correction. It writes a standard `tifxyz` surface that can be reopened in VC3D.

## Results on PHerc. Paris. 4

The correction was fixed before testing in seven previously unused regions of this scroll. All three methods completed the same 28 starts, four in each region.

| Method | Mean area on the target sheet (voxel²) | Mean area on another sheet (voxel²) |
| --- | ---: | ---: |
| VC3D edge extension | 532.46 | 0.89 |
| Uniform correction (R094) | 559.36 | 1.89 |
| R103 | 589.64 | 0.11 |

![Paired changes in target area for all 28 starts. R103 improved 20, lost area on seven and tied on one.](results/paired_changes.png)

Compared with VC3D, R103 gained **57.18 voxel² per start on average (+10.7%)**. Six of the seven regional means increased. The figure shows each individual start, including the largest loss (56.00 voxel², region `10192_04048_03280`, start 2). The metric measures where the mesh lies relative to sheet annotations. It does not measure ink legibility or time saved by an operator.

[All paired scores](results/paired_scores.csv) · [Reproduction instructions and metric](docs/REPRODUCE.md)

## Inspect the result

- **Check the numbers immediately.** From this repository, run `python3 results/summarize.py`. This calculation requires only Python.
- **Inspect the CT.** Rebuild one real example from an official volume and its annotation. The local viewer shows both surfaces on the same grayscale, longitudinal and transverse CT sections, and a rotatable 3D overlay. You can also open the generated volume package in VC3D.
- **Inspect a failure.** The instructions include a command for the start that lost 56 voxel², using the same display as the successful example.

In the two-direction extension of region `01744_02256_04816`, start 0, the R103 centreline measured 124.4 voxels. Of 62 intervals along this strip, 55 met the annotation-overlap rule with VC3D and 62 with R103. This is a second look at one of the 28 starts.

## How it works

Five longitudinal tracks estimate local shifts from CT profiles. The algorithm smooths these shifts across the width of the strip and joins compatible left and right halves. Four interleaved tracks assess the CT fit and decide between that surface and the uniform correction. Both candidates have the same starting edge. The correction requires no training or sheet mask at inference. The benchmark used annotated locations to place its starting windows.

[Correction source](r103/corrector.py) · [CT profiles and scoring of candidate surfaces](r103/profiles.py)

## Data and software

Data were obtained from **EduceLab-Scrolls**, EduceLab / University of Kentucky. The Vesuvius Challenge provides the volumetric instance labels and VC3D. Obtain the scans from the official source under its data terms.

[Parsons et al. (2023), *EduceLab-Scrolls*](https://doi.org/10.48550/arXiv.2304.02084) · [PHerc. Paris. 4](https://scrollprize.org/data_browser/PHercParis4) · [Data terms](https://dl.ash2txt.org/LICENSE.txt) · [VC3D software](https://github.com/ScrollPrize/villa)
