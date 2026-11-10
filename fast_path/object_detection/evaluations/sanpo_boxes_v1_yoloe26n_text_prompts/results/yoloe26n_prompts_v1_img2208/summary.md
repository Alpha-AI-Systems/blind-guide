# Summary: yoloe26n_prompts_v1_img2208

- **Evaluation:** sanpo_boxes_v1_yoloe26n_text_prompts
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 16
- **Model input size (long side):** 2208
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.094** | 0.047 | 0.145 | 0.449 | 0.108 | 2.05 |
| test | 5,107 | **0.076** | 0.039 | 0.148 | 0.398 | 0.118 | 2.36 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 3,298 | **0.021** | 0.062 |
| test | 5 m | 8,043 | **0.020** | 0.123 |
| val | 10 m | 11,403 | **0.100** | 0.127 |
| test | 10 m | 22,470 | **0.073** | 0.149 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| pedestrian | 8,720 | 0.483 |
| vehicle | 11,023 | 0.416 |
| animal | 25 | 0.082 |
| rider | 209 | 0.042 |
| tree | 6,439 | 0.039 |
| pole | 5,324 | 0.030 |
| obstacle | 19,302 | 0.025 |
| traffic light | 1,450 | 0.023 |
| traffic sign | 1,009 | 0.018 |
| bike rack | 883 | 0.016 |
| wall/fence | 9,173 | 0.014 |
| crosswalk | 1,569 | 0.013 |
| opening-door | 546 | 0.010 |
| guard rail/road barrier | 949 | 0.010 |
| stairs | 233 | 0.000 |
| water body | 378 | 0.000 |
