# Sharpness of the SANPO frames

Score = variance of the Laplacian (grayscale frame resized to 640 px wide). Higher = sharper, lower = blurrier. There is no built-in line between blurry and sharp: look at `example_sheets/` to choose one.

## Score at each percentile (percentile 10 = 10% of the frames score below this value)

| Frames | Count | p1 | p5 | p10 | p20 | p30 | p50 | p75 | p90 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 23,853 | 68 | 203 | 312 | 490 | 651 | 941 | 1430 | 2119 | 3640 |
| train | 16,449 | 57 | 196 | 300 | 485 | 646 | 941 | 1380 | 2012 | 3605 |
| val | 2,297 | 207 | 348 | 448 | 594 | 757 | 1113 | 2048 | 2662 | 4118 |
| test | 5,107 | 72 | 193 | 294 | 452 | 615 | 890 | 1413 | 2008 | 3391 |

## Frames below each score, per split

| Score below | all | train | val | test |
| --- | --- | --- | --- | --- |
| 68 | 239 (1.0%) | 190 (1.2%) | 4 (0.2%) | 45 (0.9%) |
| 203 | 1,193 (5.0%) | 887 (5.4%) | 23 (1.0%) | 283 (5.5%) |
| 312 | 2,386 (10.0%) | 1,739 (10.6%) | 86 (3.7%) | 561 (11.0%) |
| 490 | 4,771 (20.0%) | 3,329 (20.2%) | 301 (13.1%) | 1,141 (22.3%) |
| 651 | 7,156 (30.0%) | 4,975 (30.2%) | 532 (23.2%) | 1,649 (32.3%) |
| 941 | 11,926 (50.0%) | 8,231 (50.0%) | 952 (41.4%) | 2,743 (53.7%) |
