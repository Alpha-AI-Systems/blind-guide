# exp6_synthetic_then_real_lr_decay/stage3_synthetic_chest

**finished** · 9/9 epochs · launches run001 · updated 2026-11-18 18:18:03 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.7400** (epoch 8.94, step 9504) |
| best.pt re-validated | 0.7401 |
| Pixel accuracy (best epoch) | 0.9544 |
| sanpo3 mIoU (best epoch) | 0.9522 |
| Last validation mIoU | 0.7400 (epoch 8.94) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | train_chest_5fps.txt |
| validation | every 528 steps |
| epochs | 9 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.001 |
| lrf | 0.01 |
| warmup_epochs | 0.9934 |
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
| Peak GPU memory | 6.713 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 48 GB (modal-container) |
| Training time | 28m55s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 191 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 528 | 0.7391 | 0.6619 | 0.9393 | 0.1904 | 0.3187 | 0.9355 |
| 0.99 ⭐ | 1056 | 0.4271 | 0.6978 | 0.9464 | 0.1617 | 0.2709 | 0.9437 |
| 1.49 ⭐ | 1584 | 0.3785 | 0.7114 | 0.9493 | 0.1533 | 0.2638 | 0.9462 |
| 1.99 ⭐ | 2112 | 0.3598 | 0.7156 | 0.9499 | 0.1472 | 0.2670 | 0.9477 |
| 2.48 ⭐ | 2640 | 0.3458 | 0.7238 | 0.9516 | 0.1426 | 0.2587 | 0.9490 |
| 2.98 ⭐ | 3168 | 0.3373 | 0.7274 | 0.9520 | 0.1417 | 0.2622 | 0.9499 |
| 3.48 ⭐ | 3696 | 0.3318 | 0.7304 | 0.9529 | 0.1394 | 0.2568 | 0.9500 |
| 3.97 ⭐ | 4224 | 0.3265 | 0.7319 | 0.9529 | 0.1393 | 0.2572 | 0.9503 |
| 4.47 ⭐ | 4752 | 0.3218 | 0.7335 | 0.9530 | 0.1386 | 0.2548 | 0.9508 |
| 4.97 ⭐ | 5280 | 0.3194 | 0.7336 | 0.9533 | 0.1371 | 0.2548 | 0.9511 |
| 5.46 ⭐ | 5808 | 0.3162 | 0.7373 | 0.9540 | 0.1353 | 0.2526 | 0.9516 |
| 5.96 ⭐ | 6336 | 0.3123 | 0.7390 | 0.9540 | 0.1356 | 0.2536 | 0.9515 |
| 6.46 | 6864 | 0.3094 | 0.7387 | 0.9541 | 0.1353 | 0.2519 | 0.9519 |
| 6.95 ⭐ | 7392 | 0.3086 | 0.7392 | 0.9542 | 0.1342 | 0.2512 | 0.9519 |
| 7.45 | 7920 | 0.3069 | 0.7391 | 0.9542 | 0.1352 | 0.2519 | 0.9520 |
| 7.95 ⭐ | 8448 | 0.3071 | 0.7396 | 0.9543 | 0.1347 | 0.2516 | 0.9521 |
| 8.44 ⭐ | 8976 | 0.3051 | 0.7398 | 0.9544 | 0.1345 | 0.2514 | 0.9521 |
| 8.94 ⭐ | 9504 | 0.3040 | 0.7400 | 0.9544 | 0.1343 | 0.2512 | 0.9522 |

⭐ = new best at that epoch.

## IoU per class (best epoch 8.94, step 9504)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sidewalk | 0.9615 | traffic light | 0.5346 |
| pedestrian | 0.9462 | bike rack | 0.4472 |
| building | 0.9414 | rider | 0.3917 |
| sky | 0.9372 | guard rail/road barrier | absent |
| road | 0.9287 | paved trail | absent |
| vehicle | 0.8781 | hand rail | absent |
| terrain | 0.8613 | opening-door | absent |
| vegetation | 0.8485 | opening-gate | absent |
| tree | 0.7664 | animal | absent |
| crosswalk | 0.7614 | stairs | absent |
| obstacle | 0.7045 | water body | absent |
| curb | 0.6474 | other walkable surface | absent |
| traffic sign | 0.5954 | inaccessible surface | absent |
| wall/fence | 0.5910 | railway track | absent |
| pole | 0.5783 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.9524 |
| sanpo3_not_safe | 0.9351 |
| sanpo3_obstacle | 0.9689 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 2.3752 / 0.2870 |
| Gradient norm, median | 18.86 (clipped at 10; 98% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.94 / -4.94 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
