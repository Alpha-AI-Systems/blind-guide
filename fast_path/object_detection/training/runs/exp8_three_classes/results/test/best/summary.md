# Test: exp8_three_classes / best.pt

2026-11-18 18:32:33 UTC · imgsz 352 (training scale) · trained from yolo26n-sem-ade20k.pt, lr0 0.002

| Split | Frames | mIoU | Pixel acc | sanpo3 mIoU |
|---|---|---|---|---|
| val | 1,864 | **0.6416** | 0.8730 | 0.6416 |
| val_human | 935 | **0.6388** | 0.8718 | 0.6388 |
| test_human | 2,561 | **0.7691** | 0.9137 | 0.7691 |
| test_all | 5,107 | **0.7739** | 0.9166 | 0.7739 |
| test_head_human | 1,289 | **0.6932** | 0.8513 | 0.6932 |
| test_head_all | 2,603 | **0.6980** | 0.8546 | 0.6980 |
| test_paper_human | 3,850 | **0.7444** | 0.8927 | 0.7444 |
| test_paper_all | 7,710 | **0.7491** | 0.8956 | 0.7491 |

## IoU per class (test_human)

| Class | IoU | Class | IoU |
|---|---|---|---|
| obstacle | 0.8710 | not_safe | 0.5714 |
| safe | 0.8651 |  |  |

test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.
