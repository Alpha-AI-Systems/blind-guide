# Test: exp3_imgsz736 / best.pt

2026-11-17 22:05:41 UTC · imgsz 736 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3617** | 0.7494 | 0.6495 |
| val_human | 935 | **0.3552** | 0.7440 | 0.6457 |
| test_human | 2,561 | **0.3926** | 0.7578 | 0.7690 |
| test_all | 5,107 | **0.3982** | 0.7629 | 0.7752 |
| test_head_human | 1,289 | **0.3588** | 0.7217 | 0.7369 |
| test_head_all | 2,603 | **0.3631** | 0.7259 | 0.7396 |
| test_paper_human | 3,850 | **0.3988** | 0.7456 | 0.7629 |
| test_paper_all | 7,710 | **0.4052** | 0.7504 | 0.7678 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8856 | pole | 0.4212 |
| sky | 0.7892 | tree | 0.2984 |
| pedestrian | 0.7678 | curb | 0.2572 |
| sidewalk | 0.7260 | other walkable surface | 0.2355 |
| paved trail | 0.6880 | traffic sign | 0.1919 |
| building | 0.6847 | hand rail | 0.1260 |
| vehicle | 0.6644 | guard rail/road barrier | 0.1171 |
| vegetation | 0.6446 | opening-door | 0.0687 |
| terrain | 0.6094 | traffic light | 0.0642 |
| crosswalk | 0.5900 | rider | 0.0401 |
| road | 0.5500 | opening-gate | 0.0256 |
| wall/fence | 0.5291 | animal | 0.0000 |
| obstacle | 0.4955 | inaccessible surface | 0.0000 |
| bike rack | 0.4763 | railway track | 0.0000 |
| stairs | 0.4379 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
