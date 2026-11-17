# Test: exp4_frames_15fps / best.pt

2026-11-17 07:21:57 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3236** | 0.7263 | 0.6385 |
| val_human | 935 | **0.3190** | 0.7219 | 0.6357 |
| test_human | 2,561 | **0.3696** | 0.7392 | 0.7480 |
| test_all | 5,107 | **0.3757** | 0.7441 | 0.7533 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7837 | bike rack | 0.3324 |
| water body | 0.7541 | tree | 0.3138 |
| pedestrian | 0.7340 | stairs | 0.2903 |
| sidewalk | 0.7125 | other walkable surface | 0.2674 |
| paved trail | 0.6750 | hand rail | 0.2519 |
| building | 0.6676 | curb | 0.2209 |
| vegetation | 0.6372 | traffic sign | 0.1696 |
| vehicle | 0.6002 | opening-door | 0.0624 |
| terrain | 0.5651 | animal | 0.0267 |
| crosswalk | 0.5232 | traffic light | 0.0124 |
| wall/fence | 0.4952 | opening-gate | 0.0000 |
| road | 0.4889 | rider | 0.0000 |
| obstacle | 0.4505 | inaccessible surface | 0.0000 |
| guard rail/road barrier | 0.3478 | railway track | 0.0000 |
| pole | 0.3359 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
