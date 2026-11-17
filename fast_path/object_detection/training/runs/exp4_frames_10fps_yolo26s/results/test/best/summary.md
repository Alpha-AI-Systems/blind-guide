# Test: exp4_frames_10fps_yolo26s / best.pt

2026-11-17 10:10:25 UTC · imgsz 352 (training scale) · trained from yolo26s-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3507** | 0.7399 | 0.6484 |
| val_human | 935 | **0.3435** | 0.7350 | 0.6451 |
| test_human | 2,561 | **0.3897** | 0.7475 | 0.7593 |
| test_all | 5,107 | **0.3957** | 0.7528 | 0.7660 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8802 | pole | 0.3607 |
| pedestrian | 0.7924 | tree | 0.2977 |
| sky | 0.7683 | bike rack | 0.2791 |
| sidewalk | 0.7098 | curb | 0.2484 |
| vehicle | 0.6769 | other walkable surface | 0.1942 |
| paved trail | 0.6658 | guard rail/road barrier | 0.1570 |
| vegetation | 0.6560 | traffic sign | 0.1460 |
| building | 0.6545 | opening-door | 0.0761 |
| stairs | 0.6106 | traffic light | 0.0569 |
| terrain | 0.5929 | opening-gate | 0.0019 |
| crosswalk | 0.5515 | rider | 0.0000 |
| road | 0.5338 | animal | 0.0000 |
| wall/fence | 0.4915 | inaccessible surface | 0.0000 |
| obstacle | 0.4888 | railway track | 0.0000 |
| hand rail | 0.4111 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
