# Test: exp1_baseline_lr0005 / best.pt

2026-11-15 08:41:30 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.0005

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3052** | 0.7208 | 0.6374 |
| test_human | 2,561 | **0.3158** | 0.7164 | 0.7297 |
| test_all | 5,107 | **0.3230** | 0.7211 | 0.7346 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7710 | water body | 0.2294 |
| pedestrian | 0.7314 | curb | 0.2174 |
| sidewalk | 0.6668 | other walkable surface | 0.2142 |
| vegetation | 0.6595 | traffic sign | 0.1640 |
| building | 0.6533 | bike rack | 0.1596 |
| paved trail | 0.6118 | hand rail | 0.0631 |
| vehicle | 0.5925 | guard rail/road barrier | 0.0603 |
| terrain | 0.5419 | opening-door | 0.0398 |
| crosswalk | 0.5280 | traffic light | 0.0301 |
| road | 0.4531 | opening-gate | 0.0000 |
| wall/fence | 0.4474 | rider | 0.0000 |
| obstacle | 0.4322 | animal | 0.0000 |
| pole | 0.3302 | inaccessible surface | 0.0000 |
| tree | 0.3010 | railway track | 0.0000 |
| stairs | 0.2594 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
