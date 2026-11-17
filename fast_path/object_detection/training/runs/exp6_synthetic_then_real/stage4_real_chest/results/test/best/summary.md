# Test: exp6_synthetic_then_real/stage4_real_chest / best.pt

2026-11-17 22:22:18 UTC · imgsz 352 (training scale) · trained from stage3_synthetic_chest_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3280** | 0.7333 | 0.6504 |
| val_human | 935 | **0.3236** | 0.7293 | 0.6493 |
| test_human | 2,561 | **0.3638** | 0.7354 | 0.7545 |
| test_all | 5,107 | **0.3689** | 0.7398 | 0.7604 |
| test_head_human | 1,289 | **0.3287** | 0.6903 | 0.7147 |
| test_head_all | 2,603 | **0.3341** | 0.6943 | 0.7176 |
| test_paper_human | 3,850 | **0.3644** | 0.7202 | 0.7457 |
| test_paper_all | 7,710 | **0.3706** | 0.7244 | 0.7503 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.7996 | tree | 0.3058 |
| sky | 0.7790 | curb | 0.2600 |
| pedestrian | 0.7511 | bike rack | 0.2234 |
| sidewalk | 0.7246 | other walkable surface | 0.2173 |
| building | 0.6682 | guard rail/road barrier | 0.1997 |
| paved trail | 0.6486 | hand rail | 0.1581 |
| vehicle | 0.6474 | traffic sign | 0.1180 |
| vegetation | 0.6454 | opening-door | 0.0605 |
| terrain | 0.5750 | rider | 0.0445 |
| crosswalk | 0.5307 | traffic light | 0.0390 |
| road | 0.5214 | opening-gate | 0.0000 |
| wall/fence | 0.4454 | animal | 0.0000 |
| obstacle | 0.4240 | inaccessible surface | 0.0000 |
| stairs | 0.4165 | railway track | 0.0000 |
| pole | 0.3464 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
