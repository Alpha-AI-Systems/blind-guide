# Test: exp4_frames_10fps / best.pt

2026-11-17 07:21:33 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3259** | 0.7222 | 0.6445 |
| val_human | 935 | **0.3234** | 0.7174 | 0.6411 |
| test_human | 2,561 | **0.3646** | 0.7421 | 0.7546 |
| test_all | 5,107 | **0.3716** | 0.7471 | 0.7611 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8354 | tree | 0.3033 |
| sky | 0.7753 | bike rack | 0.2966 |
| pedestrian | 0.7374 | curb | 0.2473 |
| sidewalk | 0.7190 | other walkable surface | 0.2360 |
| building | 0.6702 | hand rail | 0.1484 |
| paved trail | 0.6618 | traffic sign | 0.1466 |
| vegetation | 0.6438 | guard rail/road barrier | 0.1315 |
| vehicle | 0.6303 | opening-door | 0.0651 |
| crosswalk | 0.5811 | traffic light | 0.0551 |
| terrain | 0.5624 | opening-gate | 0.0000 |
| road | 0.5132 | rider | 0.0000 |
| wall/fence | 0.4993 | animal | 0.0000 |
| obstacle | 0.4706 | inaccessible surface | 0.0000 |
| pole | 0.3259 | railway track | 0.0000 |
| stairs | 0.3185 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
