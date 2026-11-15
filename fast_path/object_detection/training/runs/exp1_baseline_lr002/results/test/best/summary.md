# Test: exp1_baseline_lr002 / best.pt

2026-11-15 08:41:27 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3139** | 0.7264 | 0.6384 |
| test_human | 2,561 | **0.3276** | 0.7280 | 0.7487 |
| test_all | 5,107 | **0.3338** | 0.7328 | 0.7546 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7818 | stairs | 0.2789 |
| pedestrian | 0.7403 | curb | 0.2299 |
| sidewalk | 0.6823 | other walkable surface | 0.2197 |
| building | 0.6547 | bike rack | 0.1628 |
| vegetation | 0.6543 | traffic sign | 0.1565 |
| vehicle | 0.6315 | hand rail | 0.0542 |
| paved trail | 0.6232 | opening-door | 0.0199 |
| crosswalk | 0.5737 | guard rail/road barrier | 0.0128 |
| terrain | 0.5660 | traffic light | 0.0113 |
| road | 0.4962 | opening-gate | 0.0000 |
| wall/fence | 0.4638 | rider | 0.0000 |
| obstacle | 0.4294 | animal | 0.0000 |
| water body | 0.4196 | inaccessible surface | 0.0000 |
| pole | 0.3451 | railway track | 0.0000 |
| tree | 0.2922 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
