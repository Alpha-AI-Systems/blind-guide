# Test: final_yolo26n_960x544/stage2_mix_chest / best.pt

2026-11-20 15:00:15 UTC · imgsz 544 (training scale) · trained from stage1_mix_head_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3504** | 0.7456 | 0.6602 |
| val_human | 935 | **0.3440** | 0.7411 | 0.6594 |
| test_human | 2,561 | **0.4089** | 0.7547 | 0.7769 |
| test_all | 5,107 | **0.4134** | 0.7597 | 0.7828 |
| test_head_human | 1,289 | **0.3927** | 0.7598 | 0.7843 |
| test_head_all | 2,603 | **0.3985** | 0.7637 | 0.7882 |
| test_paper_human | 3,850 | **0.4256** | 0.7564 | 0.7890 |
| test_paper_all | 7,710 | **0.4324** | 0.7611 | 0.7941 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8775 | pole | 0.3945 |
| sky | 0.7802 | hand rail | 0.3289 |
| pedestrian | 0.7648 | guard rail/road barrier | 0.3238 |
| sidewalk | 0.7285 | curb | 0.2936 |
| paved trail | 0.6750 | tree | 0.2925 |
| building | 0.6687 | other walkable surface | 0.2691 |
| vehicle | 0.6667 | traffic sign | 0.2019 |
| vegetation | 0.6488 | rider | 0.1062 |
| crosswalk | 0.6138 | opening-door | 0.0709 |
| terrain | 0.6004 | traffic light | 0.0702 |
| road | 0.5768 | opening-gate | 0.0000 |
| stairs | 0.5218 | animal | 0.0000 |
| wall/fence | 0.5003 | inaccessible surface | 0.0000 |
| obstacle | 0.4826 | railway track | 0.0000 |
| bike rack | 0.4009 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
