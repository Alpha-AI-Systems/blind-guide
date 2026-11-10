# Summary: yolo26n_img1280

- **Evaluation:** sanpo_boxes_v1_yolo26n_coco_zero_shot
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 4
- **Model input size (long side):** 1280
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.425** | 0.231 | 0.449 | 0.646 | 0.473 | 1.56 |
| test | 5,107 | **0.320** | 0.195 | 0.392 | 0.548 | 0.533 | 1.85 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 460 | **0.311** | 0.770 |
| test | 5 m | 1,855 | **0.288** | 0.781 |
| val | 10 m | 2,498 | **0.433** | 0.732 |
| test | 10 m | 5,900 | **0.349** | 0.717 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| person | 8,929 | 0.621 |
| vehicle | 11,023 | 0.516 |
| animal | 25 | 0.095 |
| traffic light | 1,450 | 0.048 |
