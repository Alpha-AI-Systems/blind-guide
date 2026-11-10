# Summary: yoloe26n_seg_prompts_v1_img1280

- **Evaluation:** sanpo_masks_v1_yoloe26n_seg_text_prompts
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 16
- **Model input size (long side):** 1280
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.102** | 0.051 | 0.119 | 0.451 | 0.110 | 2.08 |
| test | 5,107 | **0.085** | 0.043 | 0.118 | 0.417 | 0.123 | 2.26 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 3,298 | **0.111** | 0.127 |
| test | 5 m | 8,043 | **0.106** | 0.196 |
| val | 10 m | 11,403 | **0.139** | 0.159 |
| test | 10 m | 22,470 | **0.107** | 0.187 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| pedestrian | 8,720 | 0.623 |
| vehicle | 11,023 | 0.388 |
| crosswalk | 1,569 | 0.103 |
| rider | 209 | 0.037 |
| traffic light | 1,450 | 0.036 |
| pole | 5,324 | 0.033 |
| wall/fence | 9,173 | 0.027 |
| traffic sign | 1,009 | 0.026 |
| tree | 6,439 | 0.024 |
| obstacle | 19,302 | 0.024 |
| bike rack | 883 | 0.014 |
| animal | 25 | 0.011 |
| opening-door | 546 | 0.011 |
| stairs | 233 | 0.001 |
| guard rail/road barrier | 949 | 0.000 |
| water body | 378 | 0.000 |
