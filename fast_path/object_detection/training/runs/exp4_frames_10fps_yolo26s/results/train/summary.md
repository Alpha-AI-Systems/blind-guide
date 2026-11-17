# exp4_frames_10fps_yolo26s

**finished** · 9/9 epochs · launches run001 · updated 2026-11-17 09:56:37 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3507** (epoch 3.51, step 3696) |
| best.pt re-validated | 0.3507 |
| Pixel accuracy (best epoch) | 0.7398 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3436 |
| sanpo3 mIoU (best epoch) | 0.6484 |
| Last validation mIoU | 0.3424 (epoch 9) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26s-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 10 fps |
| validation | every 528 steps |
| epochs | 9 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 1 |
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
| Parameters | 6,505,722 (6.51 M) |
| Forward GFLOPs per image | 9.4756 at 640x352 |
| best.pt size | 13.2 MB |
| Peak GPU memory | 8.19 GB |
| Hardware | NVIDIA A10G, 4 CPU cores, 48 GB (modal-container) |
| Training time | 46m54s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.31 at $1.67/h |
| Speed (median) | 115 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 528 | 2.6458 | 0.2969 | 0.7233 | 1.0430 | 0.8151 | 0.2897 | 0.6319 |
| 1 ⭐ | 1056 | 1.0707 | 0.3016 | 0.7230 | 1.3429 | 0.8038 | 0.2982 | 0.6235 |
| 1.50 ⭐ | 1584 | 0.7929 | 0.3166 | 0.7228 | 1.2585 | 0.7474 | 0.3112 | 0.6558 |
| 2 ⭐ | 2112 | 0.6364 | 0.3351 | 0.7252 | 1.5051 | 0.6542 | 0.3283 | 0.6511 |
| 2.50 | 2640 | 0.5445 | 0.3324 | 0.7304 | 1.3587 | 0.5914 | 0.3236 | 0.6449 |
| 3.01 | 3168 | 0.4917 | 0.3276 | 0.7377 | 1.3773 | 0.5551 | 0.3215 | 0.6476 |
| 3.51 ⭐ | 3696 | 0.4493 | 0.3507 | 0.7398 | 1.4052 | 0.5401 | 0.3436 | 0.6484 |
| 4.01 | 4224 | 0.4191 | 0.3426 | 0.7396 | 1.4204 | 0.5392 | 0.3367 | 0.6507 |
| 4.51 | 4752 | 0.3967 | 0.3383 | 0.7387 | 1.4379 | 0.5330 | 0.3311 | 0.6521 |
| 5.01 | 5280 | 0.3734 | 0.3367 | 0.7397 | 1.5299 | 0.5185 | 0.3308 | 0.6478 |
| 5.51 | 5808 | 0.3578 | 0.3394 | 0.7395 | 1.4743 | 0.5190 | 0.3346 | 0.6533 |
| 6.01 | 6336 | 0.3439 | 0.3435 | 0.7434 | 1.5191 | 0.5104 | 0.3383 | 0.6492 |
| 6.51 | 6864 | 0.3312 | 0.3437 | 0.7415 | 1.5226 | 0.5080 | 0.3401 | 0.6516 |
| 7.01 | 7392 | 0.3248 | 0.3416 | 0.7408 | 1.5545 | 0.5046 | 0.3373 | 0.6495 |
| 7.51 | 7920 | 0.3107 | 0.3417 | 0.7419 | 1.5536 | 0.5034 | 0.3373 | 0.6509 |
| 8.02 | 8448 | 0.3092 | 0.3416 | 0.7419 | 1.5478 | 0.4991 | 0.3373 | 0.6529 |
| 8.52 | 8976 | 0.3034 | 0.3415 | 0.7431 | 1.5509 | 0.4967 | 0.3369 | 0.6521 |
| 9 | 9486 | 0.3013 | 0.3424 | 0.7430 | 1.5523 | 0.4966 | 0.3382 | 0.6523 |

⭐ = new best at that epoch.

## IoU per class (best epoch 3.51, step 3696)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8666 | terrain | 0.2995 |
| pedestrian | 0.7991 | road | 0.2575 |
| vehicle | 0.7816 | stairs | 0.2025 |
| building | 0.7719 | bike rack | 0.1107 |
| sidewalk | 0.7404 | other walkable surface | 0.0743 |
| sky | 0.6422 | opening-door | 0.0634 |
| paved trail | 0.6202 | traffic light | 0.0562 |
| wall/fence | 0.5864 | traffic sign | 0.0502 |
| hand rail | 0.5645 | guard rail/road barrier | 0.0000 |
| vegetation | 0.5209 | rider | 0.0000 |
| opening-gate | 0.3909 | animal | 0.0000 |
| pole | 0.3760 | water body | 0.0000 |
| curb | 0.3588 | inaccessible surface | 0.0000 |
| obstacle | 0.3568 | railway track | absent |
| tree | 0.3284 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7708 |
| sanpo3_not_safe | 0.2984 |
| sanpo3_obstacle | 0.8759 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.4128 / 0.2991 |
| Gradient norm, median | 59.86 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.67 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (19 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
