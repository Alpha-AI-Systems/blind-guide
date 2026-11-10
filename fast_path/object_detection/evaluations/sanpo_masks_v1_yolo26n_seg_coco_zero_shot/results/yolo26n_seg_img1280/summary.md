# Summary: yolo26n_seg_img1280

- **Evaluation:** sanpo_masks_v1_yolo26n_seg_coco_zero_shot
- **GPU:** NVIDIA H100 80GB HBM3
- **Classes evaluated:** 4
- **Model input size (long side):** 1280
- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. Examples: `example_images/`.

## Main scores

| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |
|---|---|---|---|---|---|---|---|
| val | 2,297 | **0.411** | 0.200 | 0.328 | 0.628 | 0.487 | 1.74 |
| test | 5,107 | **0.363** | 0.180 | 0.311 | 0.530 | 0.556 | 2.07 |

## Near objects (farther objects ignored)

| Split | Within | GT boxes | AP50 | Recall @0.25 |
|---|---|---|---|---|
| val | 5 m | 460 | **0.370** | 0.791 |
| test | 5 m | 1,855 | **0.326** | 0.792 |
| val | 10 m | 2,498 | **0.491** | 0.763 |
| test | 10 m | 5,900 | **0.424** | 0.735 |

## Per class (test, sorted by AP50)

| Class | GT boxes | AP50 |
|---|---|---|
| person | 8,929 | 0.638 |
| vehicle | 11,023 | 0.518 |
| animal | 25 | 0.231 |
| traffic light | 1,450 | 0.065 |
