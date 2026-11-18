# Test: exp1_baseline_lr004 / best.pt

2026-11-18 16:40:37 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.004

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3052** | 0.7177 | 0.6364 |
| val_human | 935 | **0.3008** | 0.7130 | 0.6339 |
| test_human | 2,561 | **0.3365** | 0.7271 | 0.7519 |
| test_all | 5,107 | **0.3431** | 0.7311 | 0.7571 |
| test_head_human | 1,289 | **0.2834** | 0.6514 | 0.6730 |
| test_head_all | 2,603 | **0.2905** | 0.6568 | 0.6774 |
| test_paper_human | 3,850 | **0.3276** | 0.7017 | 0.7256 |
| test_paper_all | 7,710 | **0.3346** | 0.7059 | 0.7304 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7885 | tree | 0.2765 |
| pedestrian | 0.7262 | curb | 0.2451 |
| sidewalk | 0.6789 | other walkable surface | 0.1811 |
| building | 0.6589 | bike rack | 0.1530 |
| vegetation | 0.6502 | traffic sign | 0.1484 |
| water body | 0.6246 | hand rail | 0.0527 |
| paved trail | 0.6191 | opening-door | 0.0373 |
| vehicle | 0.6189 | guard rail/road barrier | 0.0334 |
| crosswalk | 0.5662 | traffic light | 0.0240 |
| terrain | 0.5548 | opening-gate | 0.0000 |
| road | 0.5187 | rider | 0.0000 |
| wall/fence | 0.4775 | animal | 0.0000 |
| obstacle | 0.4282 | inaccessible surface | 0.0000 |
| stairs | 0.3676 | railway track | 0.0000 |
| pole | 0.3279 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
