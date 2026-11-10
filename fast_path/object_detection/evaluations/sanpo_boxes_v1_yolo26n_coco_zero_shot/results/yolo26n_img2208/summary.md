# Summary: yolo26n_img2208

- **Evaluation:** sanpo_boxes_v1_yolo26n_coco_zero_shot
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 4
- **Model input size (long side):** 2208
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.410** | 0.211 | 0.429 | 0.611 | 0.459 | 1.77 |
| test | 5,107 | **0.335** | 0.179 | 0.363 | 0.489 | 0.512 | 2.24 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 460 | **0.107** | 0.537 |
| test | 5 m | 1,855 | **0.075** | 0.647 |
| val | 10 m | 2,498 | **0.329** | 0.634 |
| test | 10 m | 5,900 | **0.295** | 0.618 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| person | 8,929 | 0.540 |
| vehicle | 11,023 | 0.476 |
| animal | 25 | 0.282 |
| traffic light | 1,450 | 0.041 |
