# exp6_synthetic_then_real_lr_decay/stage4_real_chest

**finished** · 18/18 epochs · launches run001 · updated 2026-11-18 18:49:28 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3297** (epoch 13, step 6864) |
| best.pt re-validated | 0.3298 |
| Pixel accuracy (best epoch) | 0.7335 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3239 |
| sanpo3 mIoU (best epoch) | 0.6606 |
| Last validation mIoU | 0.3279 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps |
| validation | every epoch |
| epochs | 18 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.00070710678 |
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
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 2.4093 at 640x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 6.713 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 48 GB (modal-container) |
| Training time | 28m25s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 191 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 528 | 2.9623 | 0.2804 | 0.7158 | 0.9928 | 0.8050 | 0.2762 | 0.6576 |
| 2 ⭐ | 1056 | 1.4805 | 0.3099 | 0.7264 | 1.0025 | 0.7715 | 0.3052 | 0.6846 |
| 3 ⭐ | 1584 | 1.1228 | 0.3183 | 0.7303 | 1.1571 | 0.7354 | 0.3156 | 0.6714 |
| 4 | 2112 | 0.9337 | 0.3163 | 0.7228 | 1.2260 | 0.7129 | 0.3115 | 0.6539 |
| 5 ⭐ | 2640 | 0.8351 | 0.3226 | 0.7239 | 1.3365 | 0.6940 | 0.3186 | 0.6678 |
| 6 ⭐ | 3168 | 0.7688 | 0.3261 | 0.7282 | 1.3023 | 0.6838 | 0.3225 | 0.6639 |
| 7 | 3696 | 0.7243 | 0.3151 | 0.7336 | 1.3223 | 0.6654 | 0.3112 | 0.6607 |
| 8 | 4224 | 0.6870 | 0.3261 | 0.7345 | 1.3304 | 0.6589 | 0.3221 | 0.6608 |
| 9 ⭐ | 4752 | 0.6595 | 0.3277 | 0.7386 | 1.3362 | 0.6506 | 0.3218 | 0.6579 |
| 10 | 5280 | 0.6361 | 0.3252 | 0.7293 | 1.4012 | 0.6506 | 0.3192 | 0.6515 |
| 11 | 5808 | 0.6173 | 0.3262 | 0.7356 | 1.3532 | 0.6492 | 0.3206 | 0.6588 |
| 12 | 6336 | 0.6007 | 0.3272 | 0.7310 | 1.3906 | 0.6458 | 0.3219 | 0.6547 |
| 13 ⭐ | 6864 | 0.5891 | 0.3297 | 0.7335 | 1.3904 | 0.6430 | 0.3239 | 0.6606 |
| 14 | 7392 | 0.5811 | 0.3273 | 0.7320 | 1.4195 | 0.6423 | 0.3216 | 0.6562 |
| 15 | 7920 | 0.5720 | 0.3274 | 0.7313 | 1.4265 | 0.6383 | 0.3221 | 0.6573 |
| 16 | 8448 | 0.5697 | 0.3272 | 0.7310 | 1.4225 | 0.6364 | 0.3219 | 0.6567 |
| 17 | 8976 | 0.5637 | 0.3275 | 0.7312 | 1.4312 | 0.6355 | 0.3223 | 0.6600 |
| 18 | 9504 | 0.5649 | 0.3279 | 0.7317 | 1.4217 | 0.6354 | 0.3227 | 0.6582 |

⭐ = new best at that epoch.

## IoU per class (best epoch 13, step 6864)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8384 | road | 0.2927 |
| pedestrian | 0.7686 | stairs | 0.1193 |
| building | 0.7626 | traffic sign | 0.1028 |
| sidewalk | 0.7609 | traffic light | 0.0987 |
| vehicle | 0.7133 | bike rack | 0.0834 |
| paved trail | 0.6176 | other walkable surface | 0.0759 |
| hand rail | 0.5907 | opening-door | 0.0683 |
| wall/fence | 0.5508 | opening-gate | 0.0322 |
| sky | 0.5486 | animal | 0.0112 |
| vegetation | 0.5083 | guard rail/road barrier | 0.0069 |
| pole | 0.3694 | rider | 0.0000 |
| curb | 0.3416 | water body | 0.0000 |
| obstacle | 0.3408 | inaccessible surface | 0.0000 |
| terrain | 0.3250 | railway track | absent |
| tree | 0.3033 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7823 |
| sanpo3_not_safe | 0.3201 |
| sanpo3_obstacle | 0.8792 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 12.9470 / 0.7289 |
| Gradient norm, median | 98.66 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -4.10 / -5.09 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
