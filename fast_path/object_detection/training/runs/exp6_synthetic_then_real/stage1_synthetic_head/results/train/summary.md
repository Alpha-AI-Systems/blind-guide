# exp6_synthetic_then_real/stage1_synthetic_head

**finished** · 8/8 epochs · launches run001 · updated 2026-11-17 18:32:26 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.7238** (epoch 6.96, step 7584) |
| best.pt re-validated | 0.7238 |
| Pixel accuracy (best epoch) | 0.9554 |
| sanpo3 mIoU (best epoch) | 0.9484 |
| Last validation mIoU | 0.7233 (epoch 7.25) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | train_head_5fps.txt |
| validation | every 316 steps |
| epochs | 8 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 0.58 |
| mosaic | 0.05 |
| close_mosaic | 15 |
| scale | 0.25 |
| degrees | 3.0 |
| fliplr | 0.5 |
| seed | 42 |
| deterministic | False |
| amp | True |

Full config: `train_config/<launch>_config.yaml`, `training_summary.json`.

## Model and cost

| Item | Value |
|---|---|
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 2.4093 at 640x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 6.67 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 48 GB (modal-container) |
| Training time | 24m01s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 203 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 0.29 ⭐ | 316 | 1.4604 | 0.5357 | 0.9206 | 0.2596 | 0.7249 | 0.9054 |
| 0.58 ⭐ | 632 | 0.7455 | 0.6136 | 0.9345 | 0.2010 | 0.3188 | 0.9259 |
| 0.87 ⭐ | 948 | 0.4843 | 0.6462 | 0.9409 | 0.1816 | 0.3039 | 0.9323 |
| 1.16 ⭐ | 1264 | 0.4361 | 0.6633 | 0.9427 | 0.1768 | 0.2966 | 0.9342 |
| 1.45 ⭐ | 1580 | 0.4094 | 0.6778 | 0.9455 | 0.1653 | 0.2916 | 0.9360 |
| 1.74 ⭐ | 1896 | 0.3917 | 0.6807 | 0.9471 | 0.1633 | 0.2862 | 0.9388 |
| 2.03 ⭐ | 2212 | 0.3792 | 0.6917 | 0.9481 | 0.1618 | 0.2829 | 0.9395 |
| 2.32 ⭐ | 2528 | 0.3691 | 0.6990 | 0.9495 | 0.1541 | 0.2794 | 0.9409 |
| 2.61 ⭐ | 2844 | 0.3587 | 0.7007 | 0.9511 | 0.1487 | 0.2771 | 0.9438 |
| 2.90 ⭐ | 3160 | 0.3533 | 0.7040 | 0.9514 | 0.1475 | 0.2736 | 0.9436 |
| 3.19 | 3476 | 0.3442 | 0.7033 | 0.9521 | 0.1479 | 0.2744 | 0.9444 |
| 3.48 ⭐ | 3792 | 0.3386 | 0.7098 | 0.9527 | 0.1434 | 0.2726 | 0.9450 |
| 3.77 ⭐ | 4108 | 0.3360 | 0.7103 | 0.9531 | 0.1412 | 0.2704 | 0.9460 |
| 4.06 ⭐ | 4424 | 0.3309 | 0.7123 | 0.9535 | 0.1418 | 0.2700 | 0.9460 |
| 4.35 ⭐ | 4740 | 0.3277 | 0.7135 | 0.9536 | 0.1424 | 0.2676 | 0.9464 |
| 4.64 ⭐ | 5056 | 0.3238 | 0.7160 | 0.9542 | 0.1384 | 0.2663 | 0.9472 |
| 4.93 | 5372 | 0.3229 | 0.7146 | 0.9543 | 0.1400 | 0.2653 | 0.9472 |
| 5.22 ⭐ | 5688 | 0.3197 | 0.7165 | 0.9545 | 0.1387 | 0.2661 | 0.9475 |
| 5.51 ⭐ | 6004 | 0.3181 | 0.7209 | 0.9548 | 0.1374 | 0.2653 | 0.9476 |
| 5.80 ⭐ | 6320 | 0.3155 | 0.7210 | 0.9551 | 0.1373 | 0.2634 | 0.9481 |
| 6.09 ⭐ | 6636 | 0.3127 | 0.7215 | 0.9551 | 0.1371 | 0.2646 | 0.9481 |
| 6.38 ⭐ | 6952 | 0.3136 | 0.7221 | 0.9553 | 0.1361 | 0.2636 | 0.9482 |
| 6.67 ⭐ | 7268 | 0.3103 | 0.7238 | 0.9554 | 0.1354 | 0.2629 | 0.9483 |
| 6.96 ⭐ | 7584 | 0.3121 | 0.7238 | 0.9554 | 0.1356 | 0.2626 | 0.9484 |
| 7.25 | 7900 | 0.3124 | 0.7233 | 0.9554 | 0.1357 | 0.2633 | 0.9484 |

⭐ = new best at that epoch.

## IoU per class (best epoch 6.96, step 7584)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sidewalk | 0.9559 | pole | 0.4769 |
| building | 0.9514 | traffic light | 0.4698 |
| sky | 0.9378 | rider | 0.3608 |
| road | 0.9281 | guard rail/road barrier | absent |
| pedestrian | 0.8804 | paved trail | absent |
| terrain | 0.8698 | hand rail | absent |
| vehicle | 0.8499 | opening-door | absent |
| vegetation | 0.8343 | opening-gate | absent |
| obstacle | 0.7627 | animal | absent |
| tree | 0.7590 | stairs | absent |
| crosswalk | 0.7539 | water body | absent |
| curb | 0.6616 | other walkable surface | absent |
| traffic sign | 0.5990 | inaccessible surface | absent |
| wall/fence | 0.4995 | railway track | absent |
| bike rack | 0.4777 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.9475 |
| sanpo3_not_safe | 0.9345 |
| sanpo3_obstacle | 0.9633 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 3.2363 / 0.3084 |
| Gradient norm, median | 16.96 (clipped at 10; 95% of steps above) |
| Mixed-precision overflow steps | 6 |
| Update/weight ratio, log10 median, first / last | -3.43 / -4.74 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (16 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
