# Test: local_chest_from_head_ade20k / best.pt

2026-11-17 19:11:40 UTC · imgsz 352 (training scale) · trained from stage1_real_head_best, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3310** | 0.7411 | 0.6515 |
| val_human | 935 | **0.3239** | 0.7352 | 0.6491 |
| test_human | 2,561 | **0.3815** | 0.7426 | 0.7614 |
| test_all | 5,107 | **0.3872** | 0.7473 | 0.7671 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8304 | bike rack | 0.3340 |
| sky | 0.7929 | tree | 0.2861 |
| pedestrian | 0.7602 | hand rail | 0.2859 |
| sidewalk | 0.6976 | curb | 0.2457 |
| building | 0.6729 | other walkable surface | 0.2206 |
| vegetation | 0.6430 | guard rail/road barrier | 0.1663 |
| vehicle | 0.6383 | traffic sign | 0.1152 |
| paved trail | 0.6381 | opening-door | 0.0814 |
| crosswalk | 0.6238 | traffic light | 0.0341 |
| terrain | 0.5898 | rider | 0.0273 |
| road | 0.5381 | opening-gate | 0.0000 |
| stairs | 0.5325 | animal | 0.0000 |
| wall/fence | 0.5190 | inaccessible surface | 0.0000 |
| obstacle | 0.4539 | railway track | 0.0000 |
| pole | 0.3366 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
