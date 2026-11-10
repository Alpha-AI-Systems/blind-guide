# Summary: yoloe26n_seg_prompts_v1_img640

- **Evaluation:** sanpo_masks_v1_yoloe26n_seg_text_prompts
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 16
- **Model input size (long side):** 640
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.082** | 0.043 | 0.098 | 0.503 | 0.072 | 1.11 |
| test | 5,107 | **0.079** | 0.043 | 0.105 | 0.514 | 0.094 | 1.17 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 3,298 | **0.142** | 0.141 |
| test | 5 m | 8,043 | **0.125** | 0.212 |
| val | 10 m | 11,403 | **0.145** | 0.146 |
| test | 10 m | 22,470 | **0.114** | 0.177 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| pedestrian | 8,720 | 0.546 |
| vehicle | 11,023 | 0.314 |
| crosswalk | 1,569 | 0.171 |
| tree | 6,439 | 0.046 |
| rider | 209 | 0.045 |
| wall/fence | 9,173 | 0.040 |
| obstacle | 19,302 | 0.023 |
| pole | 5,324 | 0.019 |
| traffic sign | 1,009 | 0.018 |
| traffic light | 1,450 | 0.016 |
| stairs | 233 | 0.016 |
| opening-door | 546 | 0.012 |
| bike rack | 883 | 0.005 |
| guard rail/road barrier | 949 | 0.000 |
| animal | 25 | 0.000 |
| water body | 378 | 0.000 |
