# Test: exp1_baseline_lr001 / best.pt

2026-11-15 08:41:32 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.001

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3065** | 0.7121 | 0.6421 |
| test_human | 2,561 | **0.3356** | 0.7261 | 0.7411 |
| test_all | 5,107 | **0.3435** | 0.7307 | 0.7461 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.7676 | curb | 0.2333 |
| pedestrian | 0.7312 | other walkable surface | 0.2321 |
| sidewalk | 0.7041 | stairs | 0.2227 |
| building | 0.6608 | bike rack | 0.2071 |
| vegetation | 0.6534 | traffic sign | 0.1863 |
| water body | 0.6401 | guard rail/road barrier | 0.0606 |
| paved trail | 0.6332 | hand rail | 0.0431 |
| vehicle | 0.6106 | opening-door | 0.0392 |
| crosswalk | 0.5446 | traffic light | 0.0287 |
| terrain | 0.5364 | opening-gate | 0.0001 |
| road | 0.4827 | rider | 0.0000 |
| wall/fence | 0.4673 | animal | 0.0000 |
| obstacle | 0.4117 | inaccessible surface | 0.0000 |
| pole | 0.3257 | railway track | 0.0000 |
| tree | 0.3110 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
