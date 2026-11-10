# Summary: yolo26n_seg_img640

- **Evaluation:** sanpo_masks_v1_yolo26n_seg_coco_zero_shot
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 4
- **Model input size (long side):** 640
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.304** | 0.155 | 0.255 | 0.699 | 0.362 | 0.94 |
| test | 5,107 | **0.273** | 0.144 | 0.254 | 0.647 | 0.453 | 1.04 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 460 | **0.418** | 0.822 |
| test | 5 m | 1,855 | **0.348** | 0.798 |
| val | 10 m | 2,498 | **0.460** | 0.727 |
| test | 10 m | 5,900 | **0.341** | 0.719 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| person | 8,929 | 0.580 |
| vehicle | 11,023 | 0.450 |
| traffic light | 1,450 | 0.050 |
| animal | 25 | 0.009 |
