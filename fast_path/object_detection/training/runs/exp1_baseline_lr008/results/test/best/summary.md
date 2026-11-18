# Test: exp1_baseline_lr008 / best.pt

2026-11-18 16:40:45 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.008

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.2975** | 0.7140 | 0.6316 |
| val_human | 935 | **0.2938** | 0.7094 | 0.6291 |
| test_human | 2,561 | **0.3359** | 0.7177 | 0.7402 |
| test_all | 5,107 | **0.3404** | 0.7221 | 0.7442 |
| test_head_human | 1,289 | **0.2760** | 0.6573 | 0.6703 |
| test_head_all | 2,603 | **0.2821** | 0.6618 | 0.6731 |
| test_paper_human | 3,850 | **0.3276** | 0.6974 | 0.7167 |
| test_paper_all | 7,710 | **0.3322** | 0.7017 | 0.7201 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7819 | tree | 0.2629 |
| water body | 0.7510 | bike rack | 0.2528 |
| pedestrian | 0.7349 | curb | 0.2522 |
| sidewalk | 0.6635 | other walkable surface | 0.2176 |
| building | 0.6547 | traffic sign | 0.0951 |
| vehicle | 0.6320 | guard rail/road barrier | 0.0789 |
| paved trail | 0.6302 | hand rail | 0.0642 |
| vegetation | 0.6163 | opening-door | 0.0283 |
| terrain | 0.5435 | traffic light | 0.0151 |
| crosswalk | 0.5393 | opening-gate | 0.0000 |
| road | 0.4841 | rider | 0.0000 |
| wall/fence | 0.4647 | animal | 0.0000 |
| obstacle | 0.3954 | inaccessible surface | 0.0000 |
| pole | 0.3095 | railway track | 0.0000 |
| stairs | 0.2739 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
