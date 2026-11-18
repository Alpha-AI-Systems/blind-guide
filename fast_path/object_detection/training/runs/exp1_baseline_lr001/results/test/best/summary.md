# Test: exp1_baseline_lr001 / best.pt

2026-11-17 21:35:10 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.001

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3066** | 0.7122 | 0.6421 |
| val_human | 935 | **0.3037** | 0.7063 | 0.6388 |
| test_human | 2,561 | **0.3355** | 0.7261 | 0.7410 |
| test_all | 5,107 | **0.3434** | 0.7307 | 0.7459 |
| test_head_human | 1,289 | **0.3158** | 0.7021 | 0.7301 |
| test_head_all | 2,603 | **0.3210** | 0.7081 | 0.7337 |
| test_paper_human | 3,850 | **0.3387** | 0.7180 | 0.7432 |
| test_paper_all | 7,710 | **0.3469** | 0.7230 | 0.7477 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7675 | curb | 0.2333 |
| pedestrian | 0.7313 | other walkable surface | 0.2317 |
| sidewalk | 0.7039 | stairs | 0.2223 |
| building | 0.6610 | bike rack | 0.2073 |
| vegetation | 0.6536 | traffic sign | 0.1861 |
| water body | 0.6376 | guard rail/road barrier | 0.0605 |
| paved trail | 0.6331 | hand rail | 0.0430 |
| vehicle | 0.6103 | opening-door | 0.0393 |
| crosswalk | 0.5444 | traffic light | 0.0287 |
| terrain | 0.5363 | opening-gate | 0.0001 |
| road | 0.4823 | rider | 0.0000 |
| wall/fence | 0.4674 | animal | 0.0000 |
| obstacle | 0.4117 | inaccessible surface | 0.0000 |
| pole | 0.3260 | railway track | 0.0000 |
| tree | 0.3104 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
