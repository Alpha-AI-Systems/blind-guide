# Summary: yoloe26n_prompts_v1_img640

- **Evaluation:** sanpo_boxes_v1_yoloe26n_text_prompts
- **GPU:** NVIDIA L4
- **Classes evaluated:** 16
- **Model input size (long side):** 640
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.079** | 0.044 | 0.129 | 0.536 | 0.066 | 0.89 |
| test | 5,107 | **0.080** | 0.046 | 0.148 | 0.562 | 0.088 | 0.91 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 3,298 | **0.127** | 0.124 |
| test | 5 m | 8,043 | **0.116** | 0.200 |
| val | 10 m | 11,403 | **0.134** | 0.133 |
| test | 10 m | 22,470 | **0.110** | 0.167 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| pedestrian | 8,720 | 0.537 |
| vehicle | 11,023 | 0.329 |
| crosswalk | 1,569 | 0.158 |
| tree | 6,439 | 0.073 |
| wall/fence | 9,173 | 0.048 |
| rider | 209 | 0.027 |
| traffic light | 1,450 | 0.026 |
| obstacle | 19,302 | 0.020 |
| pole | 5,324 | 0.019 |
| opening-door | 546 | 0.014 |
| traffic sign | 1,009 | 0.012 |
| bike rack | 883 | 0.008 |
| stairs | 233 | 0.004 |
| guard rail/road barrier | 949 | 0.001 |
| animal | 25 | 0.000 |
| water body | 378 | 0.000 |
