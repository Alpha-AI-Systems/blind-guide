# Test: exp3_imgsz544 / best.pt

2026-11-17 22:00:53 UTC · imgsz 544 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.3430** | 0.7368 | 0.6459 |
| val_human | 935 | **0.3382** | 0.7314 | 0.6428 |
| test_human | 2,561 | **0.3780** | 0.7501 | 0.7635 |
| test_all | 5,107 | **0.3852** | 0.7551 | 0.7699 |
| test_head_human | 1,289 | **0.3471** | 0.7329 | 0.7665 |
| test_head_all | 2,603 | **0.3529** | 0.7386 | 0.7709 |
| test_paper_human | 3,850 | **0.3864** | 0.7444 | 0.7722 |
| test_paper_all | 7,710 | **0.3938** | 0.7495 | 0.7778 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| water body | 0.8203 | bike rack | 0.3559 |
| sky | 0.7820 | tree | 0.3008 |
| pedestrian | 0.7557 | curb | 0.2481 |
| sidewalk | 0.7266 | other walkable surface | 0.2262 |
| building | 0.6695 | traffic sign | 0.2184 |
| paved trail | 0.6666 | hand rail | 0.1542 |
| vehicle | 0.6633 | guard rail/road barrier | 0.0941 |
| vegetation | 0.6463 | opening-door | 0.0452 |
| crosswalk | 0.6072 | traffic light | 0.0397 |
| terrain | 0.6016 | rider | 0.0232 |
| road | 0.5311 | opening-gate | 0.0013 |
| wall/fence | 0.5106 | animal | 0.0000 |
| obstacle | 0.4795 | inaccessible surface | 0.0000 |
| stairs | 0.4137 | railway track | 0.0000 |
| pole | 0.3817 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
