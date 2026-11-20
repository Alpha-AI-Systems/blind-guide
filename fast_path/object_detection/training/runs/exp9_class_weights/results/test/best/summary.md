# Test: exp9_class_weights / best.pt

2026-11-19 13:03:38 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3257** | 0.7240 | 0.6470 |
| val_human | 935 | **0.3195** | 0.7173 | 0.6426 |
| test_human | 2,561 | **0.3641** | 0.7319 | 0.7423 |
| test_all | 5,107 | **0.3714** | 0.7371 | 0.7485 |
| test_head_human | 1,289 | **0.3192** | 0.7076 | 0.7321 |
| test_head_all | 2,603 | **0.3228** | 0.7122 | 0.7361 |
| test_paper_human | 3,850 | **0.3652** | 0.7237 | 0.7460 |
| test_paper_all | 7,710 | **0.3725** | 0.7287 | 0.7514 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8047 | bike rack | 0.3070 |
| sky | 0.7820 | guard rail/road barrier | 0.2873 |
| pedestrian | 0.7299 | tree | 0.2842 |
| sidewalk | 0.7194 | curb | 0.2582 |
| building | 0.6599 | other walkable surface | 0.2209 |
| paved trail | 0.6488 | traffic sign | 0.1420 |
| vehicle | 0.6308 | hand rail | 0.0944 |
| vegetation | 0.6208 | opening-door | 0.0718 |
| terrain | 0.5755 | rider | 0.0367 |
| crosswalk | 0.5517 | traffic light | 0.0322 |
| wall/fence | 0.4801 | opening-gate | 0.0000 |
| road | 0.4584 | animal | 0.0000 |
| obstacle | 0.4567 | inaccessible surface | 0.0000 |
| pole | 0.3755 | railway track | 0.0000 |
| stairs | 0.3303 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
