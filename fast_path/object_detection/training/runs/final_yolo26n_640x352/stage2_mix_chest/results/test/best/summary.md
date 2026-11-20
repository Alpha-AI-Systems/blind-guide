# Test: final_yolo26n_640x352/stage2_mix_chest / best.pt

2026-11-19 20:52:04 UTC · imgsz 352 (training scale) · trained from stage1_mix_head_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3380** | 0.7355 | 0.6552 |
| val_human | 935 | **0.3336** | 0.7291 | 0.6532 |
| test_human | 2,561 | **0.3705** | 0.7323 | 0.7480 |
| test_all | 5,107 | **0.3770** | 0.7380 | 0.7537 |
| test_head_human | 1,289 | **0.3368** | 0.7389 | 0.7612 |
| test_head_all | 2,603 | **0.3436** | 0.7428 | 0.7640 |
| test_paper_human | 3,850 | **0.3772** | 0.7345 | 0.7633 |
| test_paper_all | 7,710 | **0.3849** | 0.7396 | 0.7679 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.7694 | bike rack | 0.3234 |
| sky | 0.7460 | tree | 0.3106 |
| pedestrian | 0.7303 | guard rail/road barrier | 0.3057 |
| sidewalk | 0.6925 | hand rail | 0.2987 |
| paved trail | 0.6620 | other walkable surface | 0.2377 |
| building | 0.6551 | curb | 0.2363 |
| vegetation | 0.6482 | traffic sign | 0.1204 |
| vehicle | 0.5920 | traffic light | 0.0654 |
| crosswalk | 0.5898 | opening-door | 0.0423 |
| terrain | 0.5619 | rider | 0.0250 |
| road | 0.5097 | opening-gate | 0.0000 |
| wall/fence | 0.5023 | animal | 0.0000 |
| obstacle | 0.4258 | inaccessible surface | 0.0000 |
| pole | 0.3512 | railway track | 0.0000 |
| stairs | 0.3424 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
