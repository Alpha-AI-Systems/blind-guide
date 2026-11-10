# Summary: yoloe26n_prompts_v1_img1280

- **Evaluation:** sanpo_boxes_v1_yoloe26n_text_prompts
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 16
- **Model input size (long side):** 1280
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.100** | 0.052 | 0.159 | 0.480 | 0.102 | 1.71 |
| test | 5,107 | **0.088** | 0.049 | 0.169 | 0.452 | 0.118 | 1.87 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 3,298 | **0.104** | 0.116 |
| test | 5 m | 8,043 | **0.099** | 0.190 |
| val | 10 m | 11,403 | **0.137** | 0.149 |
| test | 10 m | 22,470 | **0.110** | 0.183 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| pedestrian | 8,720 | 0.592 |
| vehicle | 11,023 | 0.407 |
| crosswalk | 1,569 | 0.082 |
| tree | 6,439 | 0.068 |
| rider | 209 | 0.060 |
| bike rack | 883 | 0.038 |
| wall/fence | 9,173 | 0.035 |
| pole | 5,324 | 0.032 |
| traffic light | 1,450 | 0.024 |
| obstacle | 19,302 | 0.024 |
| traffic sign | 1,009 | 0.021 |
| opening-door | 546 | 0.017 |
| animal | 25 | 0.007 |
| guard rail/road barrier | 949 | 0.001 |
| stairs | 233 | 0.000 |
| water body | 378 | 0.000 |
