# Summary: yolo26n_img640

- **Evaluation:** sanpo_boxes_v1_yolo26n_coco_zero_shot
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 4
- **Model input size (long side):** 640
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.298** | 0.167 | 0.370 | 0.733 | 0.334 | 0.74 |
| test | 5,107 | **0.266** | 0.163 | 0.327 | 0.671 | 0.426 | 0.88 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 460 | **0.384** | 0.783 |
| test | 5 m | 1,855 | **0.330** | 0.783 |
| val | 10 m | 2,498 | **0.416** | 0.695 |
| test | 10 m | 5,900 | **0.329** | 0.696 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| person | 8,929 | 0.583 |
| vehicle | 11,023 | 0.447 |
| traffic light | 1,450 | 0.033 |
| animal | 25 | 0.001 |
