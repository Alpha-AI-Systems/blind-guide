# Test: exp1_baseline_lr0005 / best.pt

2026-11-17 21:24:36 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.0005

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3053** | 0.7207 | 0.6374 |
| val_human | 935 | **0.3012** | 0.7161 | 0.6342 |
| test_human | 2,561 | **0.3158** | 0.7164 | 0.7297 |
| test_all | 5,107 | **0.3230** | 0.7211 | 0.7346 |
| test_head_human | 1,289 | **0.3103** | 0.6931 | 0.7262 |
| test_head_all | 2,603 | **0.3129** | 0.6987 | 0.7296 |
| test_paper_human | 3,850 | **0.3182** | 0.7086 | 0.7356 |
| test_paper_all | 7,710 | **0.3252** | 0.7135 | 0.7400 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7709 | water body | 0.2297 |
| pedestrian | 0.7314 | curb | 0.2174 |
| sidewalk | 0.6669 | other walkable surface | 0.2142 |
| vegetation | 0.6596 | traffic sign | 0.1641 |
| building | 0.6533 | bike rack | 0.1595 |
| paved trail | 0.6117 | hand rail | 0.0630 |
| vehicle | 0.5927 | guard rail/road barrier | 0.0603 |
| terrain | 0.5421 | opening-door | 0.0401 |
| crosswalk | 0.5280 | traffic light | 0.0301 |
| road | 0.4530 | opening-gate | 0.0000 |
| wall/fence | 0.4475 | rider | 0.0000 |
| obstacle | 0.4321 | animal | 0.0000 |
| pole | 0.3301 | inaccessible surface | 0.0000 |
| tree | 0.3015 | railway track | 0.0000 |
| stairs | 0.2588 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
