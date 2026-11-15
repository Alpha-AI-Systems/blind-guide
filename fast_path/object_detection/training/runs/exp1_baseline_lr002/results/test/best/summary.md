# Test: exp1_baseline_lr002 / best.pt

2026-11-15 11:03:43 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3138** | 0.7263 | 0.6384 |
| val_human | 935 | **0.3091** | 0.7226 | 0.6364 |
| test_human | 2,561 | **0.3276** | 0.7281 | 0.7487 |
| test_all | 5,107 | **0.3338** | 0.7329 | 0.7547 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7816 | stairs | 0.2781 |
| pedestrian | 0.7404 | curb | 0.2294 |
| sidewalk | 0.6826 | other walkable surface | 0.2193 |
| building | 0.6548 | bike rack | 0.1633 |
| vegetation | 0.6545 | traffic sign | 0.1564 |
| vehicle | 0.6319 | hand rail | 0.0544 |
| paved trail | 0.6233 | opening-door | 0.0200 |
| crosswalk | 0.5741 | guard rail/road barrier | 0.0126 |
| terrain | 0.5660 | traffic light | 0.0114 |
| road | 0.4967 | opening-gate | 0.0000 |
| wall/fence | 0.4641 | rider | 0.0000 |
| obstacle | 0.4293 | animal | 0.0000 |
| water body | 0.4189 | inaccessible surface | 0.0000 |
| pole | 0.3450 | railway track | 0.0000 |
| tree | 0.2919 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
