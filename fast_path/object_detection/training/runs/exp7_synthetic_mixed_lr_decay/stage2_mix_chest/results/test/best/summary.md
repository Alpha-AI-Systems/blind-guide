# Test: exp7_synthetic_mixed_lr_decay/stage2_mix_chest / best.pt

2026-11-18 19:08:24 UTC · imgsz 352 (training scale) · trained from stage1_mix_head_best, lr0 0.001

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3332** | 0.7381 | 0.6392 |
| val_human | 935 | **0.3274** | 0.7347 | 0.6387 |
| test_human | 2,561 | **0.3643** | 0.7330 | 0.7552 |
| test_all | 5,107 | **0.3719** | 0.7382 | 0.7610 |
| test_head_human | 1,289 | **0.3769** | 0.7562 | 0.7814 |
| test_head_all | 2,603 | **0.3831** | 0.7601 | 0.7842 |
| test_paper_human | 3,850 | **0.3809** | 0.7408 | 0.7752 |
| test_paper_all | 7,710 | **0.3902** | 0.7456 | 0.7799 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8379 | bike rack | 0.2990 |
| sky | 0.7698 | tree | 0.2882 |
| pedestrian | 0.7524 | guard rail/road barrier | 0.2373 |
| sidewalk | 0.7019 | curb | 0.2283 |
| building | 0.6608 | other walkable surface | 0.2177 |
| paved trail | 0.6502 | traffic sign | 0.1880 |
| vehicle | 0.6480 | hand rail | 0.0978 |
| vegetation | 0.6426 | rider | 0.0487 |
| terrain | 0.5896 | traffic light | 0.0442 |
| crosswalk | 0.5474 | opening-door | 0.0220 |
| road | 0.5220 | opening-gate | 0.0000 |
| wall/fence | 0.4359 | animal | 0.0000 |
| obstacle | 0.4358 | inaccessible surface | 0.0000 |
| pole | 0.3556 | railway track | 0.0000 |
| stairs | 0.3440 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
