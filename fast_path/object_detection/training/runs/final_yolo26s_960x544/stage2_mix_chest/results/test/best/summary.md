# Test: final_yolo26s_960x544/stage2_mix_chest / best.pt

2026-11-20 16:01:02 UTC · imgsz 544 (training scale) · trained from stage1_mix_head_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3755** | 0.7637 | 0.6629 |
| val_human | 935 | **0.3674** | 0.7598 | 0.6612 |
| test_human | 2,561 | **0.4118** | 0.7550 | 0.7623 |
| test_all | 5,107 | **0.4164** | 0.7593 | 0.7690 |
| test_head_human | 1,289 | **0.4159** | 0.7842 | 0.8017 |
| test_head_all | 2,603 | **0.4232** | 0.7878 | 0.8048 |
| test_paper_human | 3,850 | **0.4276** | 0.7648 | 0.7868 |
| test_paper_all | 7,710 | **0.4342** | 0.7690 | 0.7923 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8953 | pole | 0.4135 |
| pedestrian | 0.7923 | tree | 0.3136 |
| sky | 0.7749 | curb | 0.3112 |
| sidewalk | 0.7193 | hand rail | 0.2578 |
| vehicle | 0.6870 | guard rail/road barrier | 0.2184 |
| paved trail | 0.6742 | other walkable surface | 0.2181 |
| vegetation | 0.6616 | animal | 0.1749 |
| building | 0.6615 | traffic sign | 0.1648 |
| crosswalk | 0.6132 | rider | 0.1316 |
| terrain | 0.5859 | traffic light | 0.0931 |
| stairs | 0.5449 | opening-door | 0.0453 |
| road | 0.5199 | opening-gate | 0.0000 |
| wall/fence | 0.5165 | inaccessible surface | 0.0000 |
| obstacle | 0.5094 | railway track | 0.0000 |
| bike rack | 0.4448 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
