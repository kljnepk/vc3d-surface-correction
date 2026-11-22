# How far does a corrected strip follow a human-traced fibre?

The R103 submission measured surface area on the target sheet of PHerc. Paris. 4. These follow-up experiments test a different property on two further scrolls. A human trace supplies the first eight voxels of direction. The programme then predicts a strip from the local CT. The held-back continuation of the trace measures how far the strip follows that fibre.

| Scroll | VC3D | Uniform correction R094 | Submitted R103 | R103 change vs VC3D | Starts longer / tied / shorter | Regions better / tied / worse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PHerc. 172 | 14.42 | 20.25 | 20.67 | +6.250 | 18 / 4 / 2 | 7 / 0 / 1 |
| PHerc. 332 | 14.54 | 21.04 | 19.92 | +5.375 | 16 / 2 / 6 | 6 / 0 / 2 |

Each method started from the same three mesh rows and the same 128³ CT crop. We sampled 41 consecutive positions, +10 through +50 voxels of arc length along the manual continuation. A position counted if its Euclidean distance to a triangle of the newly extended mesh was at most two voxels. The score is the longest uninterrupted run. Positions outside the CT crop break the run for every method. Each scroll contributed eight regions with three starts per region. We averaged within each region and gave each region equal weight. All meshes were saved before scoring. The two-region PHerc. 172 pilot is excluded from this table.

R103 improved on VC3D in 34 of 48 starts, with six ties and eight losses. R094 achieved the highest mean of these three methods on PHerc. 332 and beat VC3D on the mean in all eight regions there. R103's added value over R094 varies by scroll. All 24 PHerc. 172 starts and 23 of 24 PHerc. 332 starts had all 41 positions in view. On PHerc. 332 the R103 gain over VC3D was +5.67 points at 1.5 voxels and +5.50 at 2.5 voxels. These follow-ups quantify fibre proximity. Whole-sheet identity, operator time and letter legibility require separate measurements.

<details>
<summary>Paired scores for all 48 starts</summary>

The start ID includes the CT cube, annotation tree and node. The three scores are lengths in points. The visible count is shared by all methods. To reproduce each mean, average the three starts within each cube, then average the eight cube means from that scroll.

| Scroll | Start ID | Visible | VC3D | R094 | R103 |
| --- | --- | ---: | ---: | ---: | ---: |
| 172 | `s5_05497_02497_02497_256_t1_n1` | 41 | 9 | 11 | 10 |
| 172 | `s5_05497_02497_02497_256_t2_n33` | 41 | 4 | 29 | 17 |
| 172 | `s5_05497_02497_02497_256_t4_n105` | 41 | 7 | 1 | 26 |
| 172 | `s5_07997_02997_05497_256_t1_n1` | 41 | 0 | 4 | 14 |
| 172 | `s5_07997_02997_05497_256_t2_n17` | 41 | 22 | 28 | 28 |
| 172 | `s5_07997_02997_05497_256_t5_n78` | 41 | 14 | 41 | 33 |
| 172 | `s5_08997_02497_04997_256_t4_n56` | 41 | 9 | 23 | 12 |
| 172 | `s5_08997_02497_04997_256_t12_n220` | 41 | 5 | 26 | 8 |
| 172 | `s5_08997_02497_04997_256_t15_n413` | 41 | 11 | 27 | 26 |
| 172 | `s5_11497_05497_03997_256_t1_n1` | 41 | 0 | 0 | 0 |
| 172 | `s5_11497_05497_03997_256_t2_n48` | 41 | 24 | 25 | 24 |
| 172 | `s5_11497_05497_03997_256_t4_n164` | 41 | 10 | 1 | 1 |
| 172 | `s5_13497_01497_02497_256_t1_n1` | 41 | 22 | 18 | 22 |
| 172 | `s5_13497_01497_02497_256_t2_n41` | 41 | 41 | 41 | 41 |
| 172 | `s5_13497_01497_02497_256_t3_n73` | 41 | 5 | 10 | 7 |
| 172 | `s5_14997_01497_01497_256_t1_n1` | 41 | 9 | 16 | 21 |
| 172 | `s5_14997_01497_01497_256_t2_n45` | 41 | 23 | 13 | 6 |
| 172 | `s5_14997_01497_01497_256_t3_n79` | 41 | 20 | 13 | 29 |
| 172 | `s5_16997_03497_03997_256_t1_n1` | 41 | 15 | 25 | 25 |
| 172 | `s5_16997_03497_03997_256_t2_n47` | 41 | 34 | 41 | 41 |
| 172 | `s5_16997_03497_03997_256_t3_n48` | 41 | 26 | 28 | 27 |
| 172 | `s5_16997_03997_04497_256_t1_n1` | 41 | 8 | 27 | 41 |
| 172 | `s5_16997_03997_04497_256_t5_n174` | 41 | 10 | 16 | 15 |
| 172 | `s5_16997_03997_04497_256_t7_n292` | 41 | 18 | 22 | 22 |
| 332 | `s3_00994_00494_01994_512_t4_n471` | 41 | 8 | 9 | 31 |
| 332 | `s3_00994_00494_01994_512_t28_n1831` | 41 | 11 | 19 | 21 |
| 332 | `s3_00994_00494_01994_512_t35_n2488` | 41 | 21 | 20 | 14 |
| 332 | `s3_00994_01994_00494_512_t18_n1513` | 41 | 0 | 0 | 0 |
| 332 | `s3_00994_01994_00494_512_t38_n3427` | 41 | 25 | 41 | 41 |
| 332 | `s3_00994_01994_00494_512_t43_n3959` | 41 | 6 | 41 | 41 |
| 332 | `s3_00994_02494_01494_512_t5_n309` | 41 | 15 | 20 | 17 |
| 332 | `s3_00994_02494_01494_512_t15_n852` | 41 | 10 | 26 | 13 |
| 332 | `s3_00994_02494_01494_512_t16_n62680` | 41 | 36 | 21 | 23 |
| 332 | `s3_01994_01494_00994_512_t23_n2334` | 41 | 1 | 1 | 6 |
| 332 | `s3_01994_01494_00994_512_t43_n3808` | 41 | 13 | 31 | 31 |
| 332 | `s3_01994_01494_00994_512_t53_n4536` | 41 | 3 | 13 | 5 |
| 332 | `s3_01994_01994_01494_512_t10_n408` | 41 | 0 | 0 | 2 |
| 332 | `s3_01994_01994_01494_512_t13_n529` | 41 | 28 | 41 | 23 |
| 332 | `s3_01994_01994_01494_512_t14_n558` | 41 | 40 | 30 | 31 |
| 332 | `s3_02994_02494_01994_512_t22_n691` | 41 | 5 | 14 | 14 |
| 332 | `s3_02994_02494_01994_512_t31_n54005` | 41 | 3 | 6 | 6 |
| 332 | `s3_02994_02494_01994_512_t60_n2939` | 41 | 13 | 41 | 41 |
| 332 | `s3_03494_01494_00494_512_t5_n503` | 29 | 20 | 18 | 18 |
| 332 | `s3_03494_01494_00494_512_t6_n596` | 41 | 14 | 13 | 21 |
| 332 | `s3_03494_01494_00494_512_t9_n990` | 41 | 10 | 18 | 10 |
| 332 | `s3_04494_00994_01494_512_t1_n1` | 41 | 7 | 33 | 33 |
| 332 | `s3_04494_00994_01494_512_t14_n834` | 41 | 41 | 27 | 15 |
| 332 | `s3_04494_00994_01494_512_t16_n980` | 41 | 19 | 22 | 21 |

</details>

Data used for the fibre references and corresponding CT came from Vesuvius Challenge Dataset004_sk-fibers-binary-20250728 (Scroll 5 and Scroll 3). Scroll 3 CT is from EduceLab-Scrolls, EduceLab / University of Kentucky. [Fibre data and annotations](https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset004_sk-fibers-binary-20250728/) · [PHerc. 172](https://scrollprize.org/data_browser/PHerc0172) · [PHerc. 332](https://scrollprize.org/data_browser/PHerc0332) · [Parsons et al., EduceLab-Scrolls](https://doi.org/10.48550/arXiv.2304.02084)

[Original correction and reproduction instructions](docs/REPRODUCE.md). The comparison used the same VC3D edge-extension executable as the original benchmark. SHA-256 of the two follow-up result archives:

| Scroll | Archive SHA-256 |
| --- | --- |
| PHerc. 172 | `4f3444256991c5e4dd82d44e8c3fe445279e29ce90114a11848af0a8b4c64848` |
| PHerc. 332 | `bce5c5ceefd7eacc46d6f5ebbc53b4a9031a2d243861cb8d175a5f8a61460e98` |
