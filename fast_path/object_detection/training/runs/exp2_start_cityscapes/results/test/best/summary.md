# Test: exp2_start_cityscapes / best.pt

2026-11-15 19:20:43 UTC · imgsz 352 (training scale) · trained from yolo26n-sem.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3209** | 0.7353 | 0.6511 |
| val_human | 935 | **0.3189** | 0.7296 | 0.6492 |
| test_human | 2,561 | **0.3518** | 0.7265 | 0.7432 |
| test_all | 5,107 | **0.3585** | 0.7327 | 0.7503 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8159 | tree | 0.2821 |
| sky | 0.7940 | other walkable surface | 0.2433 |
| pedestrian | 0.7604 | curb | 0.2345 |
| sidewalk | 0.6772 | bike rack | 0.1766 |
| vegetation | 0.6600 | guard rail/road barrier | 0.1256 |
| building | 0.6592 | traffic sign | 0.1051 |
| paved trail | 0.6243 | opening-door | 0.0568 |
| vehicle | 0.6225 | hand rail | 0.0513 |
| terrain | 0.5670 | rider | 0.0379 |
| crosswalk | 0.5597 | traffic light | 0.0362 |
| road | 0.4763 | opening-gate | 0.0000 |
| wall/fence | 0.4663 | animal | 0.0000 |
| obstacle | 0.4323 | inaccessible surface | 0.0000 |
| stairs | 0.3741 | railway track | 0.0000 |
| pole | 0.3637 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
