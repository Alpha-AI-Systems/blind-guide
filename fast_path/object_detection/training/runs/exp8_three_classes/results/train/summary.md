# exp8_three_classes

**finished** · 18/18 epochs · launches run001 · updated 2026-11-18 18:13:38 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.6416** (epoch 14, step 7392) |
| best.pt re-validated | 0.6416 |
| Pixel accuracy (best epoch) | 0.8730 |
| val_human mIoU (best epoch, human-labeled frames) | 0.6388 |
| sanpo3 mIoU (best epoch) | 0.6416 |
| Last validation mIoU | 0.6399 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps |
| classes | the groups of sanpo3 (data.class_merge) |
| validation | every epoch |
| epochs | 18 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 2 |
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
| Parameters | 1,630,822 (1.63 M) |
| Forward GFLOPs per image | 2.3976 at 640x352 |
| best.pt size | 3.43 MB |
| Peak GPU memory | 2.297 GB |
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 29m20s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $0.51 at $1.04/h |
| Speed (median) | 187 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 528 | 0.7030 | 0.6182 | 0.8601 | 0.6690 | 0.2975 | 0.6168 | 0.6182 |
| 2 | 1056 | 0.3622 | 0.6117 | 0.8556 | 0.8422 | 0.2980 | 0.6084 | 0.6117 |
| 3 | 1584 | 0.2906 | 0.6068 | 0.8589 | 0.9167 | 0.2891 | 0.6031 | 0.6068 |
| 4 ⭐ | 2112 | 0.2417 | 0.6294 | 0.8666 | 0.8937 | 0.2639 | 0.6260 | 0.6294 |
| 5 ⭐ | 2640 | 0.2136 | 0.6386 | 0.8742 | 0.9421 | 0.2535 | 0.6363 | 0.6386 |
| 6 ⭐ | 3168 | 0.1925 | 0.6388 | 0.8726 | 0.8122 | 0.2550 | 0.6352 | 0.6388 |
| 7 ⭐ | 3696 | 0.1777 | 0.6397 | 0.8710 | 0.8835 | 0.2528 | 0.6367 | 0.6397 |
| 8 ⭐ | 4224 | 0.1646 | 0.6398 | 0.8714 | 0.9057 | 0.2514 | 0.6365 | 0.6397 |
| 9 ⭐ | 4752 | 0.1554 | 0.6415 | 0.8711 | 1.0084 | 0.2464 | 0.6376 | 0.6415 |
| 10 | 5280 | 0.1469 | 0.6376 | 0.8714 | 1.1107 | 0.2507 | 0.6343 | 0.6376 |
| 11 | 5808 | 0.1407 | 0.6411 | 0.8715 | 1.0342 | 0.2470 | 0.6379 | 0.6411 |
| 12 | 6336 | 0.1351 | 0.6396 | 0.8710 | 1.0901 | 0.2467 | 0.6368 | 0.6396 |
| 13 | 6864 | 0.1311 | 0.6401 | 0.8720 | 1.0867 | 0.2464 | 0.6367 | 0.6401 |
| 14 ⭐ | 7392 | 0.1280 | 0.6416 | 0.8730 | 1.0612 | 0.2459 | 0.6388 | 0.6416 |
| 15 | 7920 | 0.1254 | 0.6399 | 0.8718 | 1.1017 | 0.2468 | 0.6370 | 0.6399 |
| 16 | 8448 | 0.1229 | 0.6400 | 0.8719 | 1.1203 | 0.2461 | 0.6372 | 0.6400 |
| 17 | 8976 | 0.1217 | 0.6403 | 0.8719 | 1.1279 | 0.2462 | 0.6375 | 0.6403 |
| 18 | 9504 | 0.1206 | 0.6399 | 0.8719 | 1.1222 | 0.2466 | 0.6370 | 0.6399 |

⭐ = new best at that epoch.

## IoU per class (best epoch 14, step 7392)

| Class | IoU | Class | IoU |
|---|---|---|---|
| obstacle | 0.8817 | not_safe | 0.2722 |
| safe | 0.7709 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7709 |
| sanpo3_not_safe | 0.2722 |
| sanpo3_obstacle | 0.8817 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 2.1203 / 0.1713 |
| Gradient norm, median | 21.75 (clipped at 10; 98% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.64 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
