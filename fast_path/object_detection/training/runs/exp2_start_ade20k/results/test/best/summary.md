# Test: exp2_start_ade20k / best.pt

2026-11-15 19:20:31 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3222** | 0.7254 | 0.6353 |
| val_human | 935 | **0.3178** | 0.7210 | 0.6323 |
| test_human | 2,561 | **0.3483** | 0.7239 | 0.7407 |
| test_all | 5,107 | **0.3556** | 0.7288 | 0.7471 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7673 | bike rack | 0.3234 |
| pedestrian | 0.7368 | tree | 0.2697 |
| sidewalk | 0.6834 | curb | 0.2368 |
| building | 0.6673 | other walkable surface | 0.2052 |
| vegetation | 0.6319 | guard rail/road barrier | 0.1723 |
| paved trail | 0.6313 | hand rail | 0.1567 |
| vehicle | 0.6148 | traffic sign | 0.1372 |
| water body | 0.6079 | opening-door | 0.0377 |
| crosswalk | 0.5860 | traffic light | 0.0194 |
| terrain | 0.5494 | opening-gate | 0.0000 |
| road | 0.4825 | rider | 0.0000 |
| wall/fence | 0.4791 | animal | 0.0000 |
| obstacle | 0.4209 | inaccessible surface | 0.0000 |
| pole | 0.3553 | railway track | 0.0000 |
| stairs | 0.3289 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
