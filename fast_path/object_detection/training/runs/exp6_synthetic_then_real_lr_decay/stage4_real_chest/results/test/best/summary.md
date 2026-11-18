# Test: exp6_synthetic_then_real_lr_decay/stage4_real_chest / best.pt

2026-11-18 19:08:16 UTC · imgsz 352 (training scale) · trained from stage3_synthetic_chest_best, lr0 0.00070710678

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3297** | 0.7336 | 0.6606 |
| val_human | 935 | **0.3239** | 0.7289 | 0.6616 |
| test_human | 2,561 | **0.3812** | 0.7440 | 0.7527 |
| test_all | 5,107 | **0.3856** | 0.7481 | 0.7590 |
| test_head_human | 1,289 | **0.3382** | 0.6961 | 0.7209 |
| test_head_all | 2,603 | **0.3455** | 0.7017 | 0.7250 |
| test_paper_human | 3,850 | **0.3749** | 0.7279 | 0.7488 |
| test_paper_all | 7,710 | **0.3813** | 0.7324 | 0.7542 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8227 | pole | 0.3518 |
| sky | 0.7826 | tree | 0.3448 |
| pedestrian | 0.7482 | curb | 0.2648 |
| sidewalk | 0.7207 | other walkable surface | 0.2580 |
| building | 0.6790 | hand rail | 0.1859 |
| vegetation | 0.6650 | guard rail/road barrier | 0.1536 |
| vehicle | 0.6604 | traffic sign | 0.1395 |
| paved trail | 0.6466 | rider | 0.0430 |
| terrain | 0.5487 | opening-door | 0.0408 |
| crosswalk | 0.5469 | traffic light | 0.0348 |
| stairs | 0.5313 | opening-gate | 0.0000 |
| wall/fence | 0.5053 | animal | 0.0000 |
| road | 0.5038 | inaccessible surface | 0.0000 |
| obstacle | 0.4512 | railway track | 0.0000 |
| bike rack | 0.4243 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
