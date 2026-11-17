# Test: exp4_frames_human_only / best.pt

2026-11-17 07:21:30 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3185** | 0.7247 | 0.6426 |
| val_human | 935 | **0.3150** | 0.7200 | 0.6394 |
| test_human | 2,561 | **0.3616** | 0.7405 | 0.7566 |
| test_all | 5,107 | **0.3663** | 0.7442 | 0.7611 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7781 | stairs | 0.3275 |
| pedestrian | 0.7404 | tree | 0.2953 |
| sidewalk | 0.7182 | curb | 0.2290 |
| water body | 0.6804 | other walkable surface | 0.2275 |
| building | 0.6652 | traffic sign | 0.1508 |
| paved trail | 0.6606 | hand rail | 0.1426 |
| vehicle | 0.6494 | guard rail/road barrier | 0.0610 |
| vegetation | 0.6368 | opening-door | 0.0537 |
| crosswalk | 0.5914 | traffic light | 0.0403 |
| terrain | 0.5778 | rider | 0.0089 |
| road | 0.5353 | opening-gate | 0.0000 |
| wall/fence | 0.4712 | animal | 0.0000 |
| obstacle | 0.4544 | inaccessible surface | 0.0000 |
| bike rack | 0.4328 | railway track | 0.0000 |
| pole | 0.3563 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
