# Test: final_yolo26s_640x352/stage2_mix_chest / best.pt

2026-11-19 20:59:36 UTC · imgsz 352 (training scale) · trained from stage1_mix_head_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3621** | 0.7506 | 0.6656 |
| val_human | 935 | **0.3573** | 0.7459 | 0.6631 |
| test_human | 2,561 | **0.3994** | 0.7474 | 0.7571 |
| test_all | 5,107 | **0.4038** | 0.7523 | 0.7631 |
| test_head_human | 1,289 | **0.3768** | 0.7584 | 0.7759 |
| test_head_all | 2,603 | **0.3836** | 0.7621 | 0.7786 |
| test_paper_human | 3,850 | **0.4129** | 0.7511 | 0.7740 |
| test_paper_all | 7,710 | **0.4187** | 0.7557 | 0.7788 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8709 | pole | 0.4091 |
| sky | 0.7733 | guard rail/road barrier | 0.3551 |
| pedestrian | 0.7714 | tree | 0.2877 |
| sidewalk | 0.7220 | curb | 0.2692 |
| vehicle | 0.6818 | other walkable surface | 0.2222 |
| building | 0.6678 | traffic sign | 0.1843 |
| paved trail | 0.6600 | hand rail | 0.1663 |
| vegetation | 0.6582 | rider | 0.0944 |
| terrain | 0.5854 | traffic light | 0.0737 |
| crosswalk | 0.5848 | opening-door | 0.0439 |
| stairs | 0.5697 | animal | 0.0103 |
| wall/fence | 0.5085 | opening-gate | 0.0000 |
| road | 0.4939 | inaccessible surface | 0.0000 |
| obstacle | 0.4795 | railway track | 0.0000 |
| bike rack | 0.4392 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
