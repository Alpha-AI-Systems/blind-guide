# exp7_synthetic_mixed_lr_decay/stage2_mix_chest

**finished** · 18/18 epochs · launches run001 · updated 2026-11-18 18:26:55 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3332** (epoch 8, step 8448) |
| best.pt re-validated | 0.3332 |
| Pixel accuracy (best epoch) | 0.7371 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3271 |
| sanpo3 mIoU (best epoch) | 0.6356 |
| Last validation mIoU | 0.3306 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps + train_chest_5fps (16+16 per batch) |
| validation | every epoch |
| epochs | 18 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.001 |
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
| Peak GPU memory | 6.67 GB |
| Hardware | NVIDIA A100-SXM4-80GB, 8 CPU cores, 64 GB (modal-container) |
| Training time | 49m35s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 213 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 1056 | 1.3245 | 0.2994 | 0.7215 | 0.9649 | 0.6275 | 0.2951 | 0.6499 |
| 2 ⭐ | 2112 | 0.8907 | 0.3145 | 0.7203 | 1.1669 | 0.6207 | 0.3101 | 0.6493 |
| 3 ⭐ | 3168 | 0.7461 | 0.3210 | 0.7355 | 1.1565 | 0.6171 | 0.3162 | 0.6711 |
| 4 ⭐ | 4224 | 0.6605 | 0.3269 | 0.7344 | 1.3522 | 0.6022 | 0.3203 | 0.6391 |
| 5 | 5280 | 0.6139 | 0.3217 | 0.7316 | 1.4126 | 0.5918 | 0.3155 | 0.6404 |
| 6 | 6336 | 0.5808 | 0.3236 | 0.7306 | 1.4823 | 0.5525 | 0.3182 | 0.6303 |
| 7 ⭐ | 7392 | 0.5519 | 0.3330 | 0.7389 | 1.4407 | 0.5342 | 0.3278 | 0.6371 |
| 8 ⭐ | 8448 | 0.5294 | 0.3332 | 0.7371 | 1.5413 | 0.5331 | 0.3271 | 0.6356 |
| 9 | 9504 | 0.5121 | 0.3318 | 0.7353 | 1.5297 | 0.5314 | 0.3254 | 0.6385 |
| 10 | 10560 | 0.4963 | 0.3309 | 0.7369 | 1.5322 | 0.5292 | 0.3242 | 0.6363 |
| 11 | 11616 | 0.4853 | 0.3332 | 0.7382 | 1.5100 | 0.5264 | 0.3274 | 0.6393 |
| 12 | 12672 | 0.4751 | 0.3316 | 0.7358 | 1.5410 | 0.5266 | 0.3253 | 0.6373 |
| 13 | 13728 | 0.4667 | 0.3305 | 0.7362 | 1.5559 | 0.5267 | 0.3246 | 0.6368 |
| 14 | 14784 | 0.4608 | 0.3310 | 0.7371 | 1.5490 | 0.5261 | 0.3252 | 0.6370 |
| 15 | 15840 | 0.4548 | 0.3309 | 0.7372 | 1.5432 | 0.5252 | 0.3251 | 0.6377 |
| 16 | 16896 | 0.4505 | 0.3307 | 0.7374 | 1.5419 | 0.5247 | 0.3249 | 0.6379 |
| 17 | 17952 | 0.4468 | 0.3307 | 0.7379 | 1.5398 | 0.5239 | 0.3249 | 0.6385 |
| 18 | 19008 | 0.4484 | 0.3306 | 0.7380 | 1.5384 | 0.5238 | 0.3249 | 0.6386 |

⭐ = new best at that epoch.

## IoU per class (best epoch 8, step 8448)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8634 | road | 0.2397 |
| sidewalk | 0.8188 | opening-gate | 0.1562 |
| pedestrian | 0.7694 | traffic light | 0.1499 |
| building | 0.7520 | traffic sign | 0.1461 |
| vehicle | 0.7316 | stairs | 0.1073 |
| paved trail | 0.6363 | bike rack | 0.0787 |
| wall/fence | 0.5619 | other walkable surface | 0.0710 |
| hand rail | 0.5564 | opening-door | 0.0413 |
| sky | 0.5145 | rider | 0.0061 |
| vegetation | 0.5100 | guard rail/road barrier | 0.0015 |
| pole | 0.3945 | animal | 0.0003 |
| curb | 0.3299 | water body | 0.0000 |
| obstacle | 0.3079 | inaccessible surface | 0.0000 |
| terrain | 0.2984 | railway track | absent |
| tree | 0.2847 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7652 |
| sanpo3_not_safe | 0.2734 |
| sanpo3_obstacle | 0.8684 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 2.4296 / 0.3823 |
| Gradient norm, median | 70.46 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 12 |
| Update/weight ratio, log10 median, first / last | -4.25 / -4.93 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (39 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
