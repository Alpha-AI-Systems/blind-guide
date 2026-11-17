# Test: exp7_synthetic_mixed/stage2_mix_chest / best.pt

2026-11-17 22:21:32 UTC · imgsz 352 (training scale) · trained from stage1_mix_head_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3312** | 0.7426 | 0.6595 |
| val_human | 935 | **0.3270** | 0.7387 | 0.6610 |
| test_human | 2,561 | **0.3884** | 0.7391 | 0.7626 |
| test_all | 5,107 | **0.3926** | 0.7434 | 0.7685 |
| test_head_human | 1,289 | **0.3625** | 0.7447 | 0.7772 |
| test_head_all | 2,603 | **0.3652** | 0.7479 | 0.7800 |
| test_paper_human | 3,850 | **0.3973** | 0.7410 | 0.7777 |
| test_paper_all | 7,710 | **0.4019** | 0.7449 | 0.7826 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8322 | pole | 0.3656 |
| sky | 0.7682 | guard rail/road barrier | 0.3517 |
| pedestrian | 0.7424 | tree | 0.2710 |
| sidewalk | 0.7116 | curb | 0.2534 |
| building | 0.6594 | other walkable surface | 0.2072 |
| vehicle | 0.6452 | hand rail | 0.1834 |
| paved trail | 0.6443 | traffic sign | 0.1547 |
| vegetation | 0.6344 | animal | 0.0546 |
| crosswalk | 0.5974 | traffic light | 0.0546 |
| terrain | 0.5896 | rider | 0.0449 |
| stairs | 0.5440 | opening-door | 0.0432 |
| road | 0.5347 | opening-gate | 0.0000 |
| wall/fence | 0.4858 | inaccessible surface | 0.0000 |
| bike rack | 0.4503 | railway track | 0.0000 |
| obstacle | 0.4403 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
