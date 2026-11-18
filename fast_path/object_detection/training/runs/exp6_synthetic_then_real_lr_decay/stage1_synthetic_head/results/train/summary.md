# exp6_synthetic_then_real_lr_decay/stage1_synthetic_head

**finished** · 8/8 epochs · launches run001 · updated 2026-11-18 17:18:40 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.7267** (epoch 6.67, step 7268) |
| best.pt re-validated | 0.7267 |
| Pixel accuracy (best epoch) | 0.9552 |
| sanpo3 mIoU (best epoch) | 0.9479 |
| Last validation mIoU | 0.7264 (epoch 7.25) |

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
| Training time | 25m28s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 192 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 0.29 ⭐ | 316 | 1.4605 | 0.5348 | 0.9203 | 0.2604 | 0.7252 | 0.9050 |
| 0.58 ⭐ | 632 | 0.7452 | 0.6149 | 0.9343 | 0.2012 | 0.3187 | 0.9256 |
| 0.87 ⭐ | 948 | 0.4823 | 0.6511 | 0.9415 | 0.1808 | 0.3029 | 0.9323 |
| 1.16 ⭐ | 1264 | 0.4357 | 0.6678 | 0.9424 | 0.1782 | 0.2963 | 0.9337 |
| 1.45 ⭐ | 1580 | 0.4102 | 0.6781 | 0.9458 | 0.1664 | 0.2905 | 0.9377 |
| 1.74 ⭐ | 1896 | 0.3915 | 0.6838 | 0.9474 | 0.1598 | 0.2846 | 0.9374 |
| 2.03 ⭐ | 2212 | 0.3797 | 0.6929 | 0.9483 | 0.1600 | 0.2823 | 0.9399 |
| 2.32 ⭐ | 2528 | 0.3693 | 0.6965 | 0.9494 | 0.1550 | 0.2804 | 0.9410 |
| 2.61 ⭐ | 2844 | 0.3595 | 0.7053 | 0.9507 | 0.1490 | 0.2756 | 0.9430 |
| 2.90 ⭐ | 3160 | 0.3530 | 0.7056 | 0.9513 | 0.1490 | 0.2746 | 0.9429 |
| 3.19 ⭐ | 3476 | 0.3439 | 0.7086 | 0.9525 | 0.1456 | 0.2727 | 0.9448 |
| 3.48 ⭐ | 3792 | 0.3388 | 0.7123 | 0.9526 | 0.1441 | 0.2720 | 0.9451 |
| 3.77 ⭐ | 4108 | 0.3362 | 0.7171 | 0.9531 | 0.1406 | 0.2697 | 0.9456 |
| 4.06 | 4424 | 0.3310 | 0.7171 | 0.9534 | 0.1421 | 0.2692 | 0.9452 |
| 4.35 ⭐ | 4740 | 0.3278 | 0.7209 | 0.9535 | 0.1413 | 0.2678 | 0.9457 |
| 4.64 | 5056 | 0.3237 | 0.7208 | 0.9542 | 0.1380 | 0.2665 | 0.9469 |
| 4.93 ⭐ | 5372 | 0.3229 | 0.7227 | 0.9544 | 0.1378 | 0.2661 | 0.9470 |
| 5.22 | 5688 | 0.3197 | 0.7200 | 0.9545 | 0.1381 | 0.2660 | 0.9473 |
| 5.51 | 6004 | 0.3182 | 0.7212 | 0.9546 | 0.1380 | 0.2653 | 0.9474 |
| 5.80 ⭐ | 6320 | 0.3155 | 0.7247 | 0.9549 | 0.1368 | 0.2631 | 0.9477 |
| 6.09 ⭐ | 6636 | 0.3128 | 0.7257 | 0.9550 | 0.1362 | 0.2641 | 0.9478 |
| 6.38 ⭐ | 6952 | 0.3139 | 0.7262 | 0.9551 | 0.1357 | 0.2639 | 0.9479 |
| 6.67 ⭐ | 7268 | 0.3106 | 0.7267 | 0.9552 | 0.1352 | 0.2627 | 0.9479 |
| 6.96 | 7584 | 0.3121 | 0.7261 | 0.9552 | 0.1353 | 0.2626 | 0.9481 |
| 7.25 | 7900 | 0.3124 | 0.7264 | 0.9552 | 0.1355 | 0.2632 | 0.9481 |

⭐ = new best at that epoch.

## IoU per class (best epoch 6.67, step 7268)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sidewalk | 0.9557 | traffic light | 0.4735 |
| building | 0.9514 | pole | 0.4732 |
| sky | 0.9373 | rider | 0.3621 |
| road | 0.9270 | guard rail/road barrier | absent |
| pedestrian | 0.8802 | paved trail | absent |
| terrain | 0.8674 | hand rail | absent |
| vehicle | 0.8505 | opening-door | absent |
| vegetation | 0.8337 | opening-gate | absent |
| obstacle | 0.7602 | animal | absent |
| tree | 0.7587 | stairs | absent |
| crosswalk | 0.7516 | water body | absent |
| curb | 0.6613 | other walkable surface | absent |
| traffic sign | 0.6009 | inaccessible surface | absent |
| wall/fence | 0.5523 | railway track | absent |
| bike rack | 0.4831 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.9470 |
| sanpo3_not_safe | 0.9338 |
| sanpo3_obstacle | 0.9630 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 3.2363 / 0.3088 |
| Gradient norm, median | 17.00 (clipped at 10; 95% of steps above) |
| Mixed-precision overflow steps | 6 |
| Update/weight ratio, log10 median, first / last | -3.43 / -4.73 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (16 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
