# exp7_synthetic_mixed/stage2_mix_chest

**finished** · 18/18 epochs · launches run001 · updated 2026-11-17 21:05:29 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3313** (epoch 10, step 10560) |
| best.pt re-validated | 0.3312 |
| Pixel accuracy (best epoch) | 0.7427 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3271 |
| sanpo3 mIoU (best epoch) | 0.6595 |
| Last validation mIoU | 0.3273 (epoch 18) |

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
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 2.4093 at 640x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 6.67 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 64 GB (modal-container) |
| Training time | 49m50s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 211 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 1056 | 1.2455 | 0.3020 | 0.7210 | 0.9965 | 0.6176 | 0.2971 | 0.6475 |
| 2 ⭐ | 2112 | 0.8421 | 0.3051 | 0.7055 | 1.3125 | 0.6187 | 0.3023 | 0.6328 |
| 3 ⭐ | 3168 | 0.7436 | 0.3104 | 0.7208 | 1.3143 | 0.6148 | 0.3062 | 0.6267 |
| 4 ⭐ | 4224 | 0.6669 | 0.3151 | 0.7207 | 1.4131 | 0.6054 | 0.3099 | 0.6351 |
| 5 ⭐ | 5280 | 0.6160 | 0.3196 | 0.7282 | 1.4158 | 0.5752 | 0.3152 | 0.6508 |
| 6 ⭐ | 6336 | 0.5809 | 0.3252 | 0.7392 | 1.3685 | 0.5495 | 0.3193 | 0.6757 |
| 7 | 7392 | 0.5508 | 0.3250 | 0.7296 | 1.4798 | 0.5399 | 0.3200 | 0.6468 |
| 8 | 8448 | 0.5242 | 0.3206 | 0.7360 | 1.4682 | 0.5279 | 0.3162 | 0.6464 |
| 9 ⭐ | 9504 | 0.5038 | 0.3302 | 0.7384 | 1.4707 | 0.5305 | 0.3245 | 0.6511 |
| 10 ⭐ | 10560 | 0.4828 | 0.3313 | 0.7427 | 1.4719 | 0.5230 | 0.3271 | 0.6595 |
| 11 | 11616 | 0.4666 | 0.3289 | 0.7415 | 1.4810 | 0.5182 | 0.3246 | 0.6593 |
| 12 | 12672 | 0.4534 | 0.3273 | 0.7401 | 1.5059 | 0.5179 | 0.3220 | 0.6562 |
| 13 | 13728 | 0.4420 | 0.3276 | 0.7394 | 1.5247 | 0.5190 | 0.3224 | 0.6504 |
| 14 | 14784 | 0.4338 | 0.3275 | 0.7398 | 1.5183 | 0.5186 | 0.3220 | 0.6489 |
| 15 | 15840 | 0.4248 | 0.3280 | 0.7405 | 1.5133 | 0.5184 | 0.3226 | 0.6509 |
| 16 | 16896 | 0.4194 | 0.3275 | 0.7408 | 1.5121 | 0.5179 | 0.3221 | 0.6519 |
| 17 | 17952 | 0.4148 | 0.3274 | 0.7412 | 1.5135 | 0.5177 | 0.3221 | 0.6526 |
| 18 | 19008 | 0.4152 | 0.3273 | 0.7416 | 1.5130 | 0.5177 | 0.3220 | 0.6531 |

⭐ = new best at that epoch.

## IoU per class (best epoch 10, step 10560)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8750 | tree | 0.2963 |
| sidewalk | 0.8278 | traffic sign | 0.1271 |
| pedestrian | 0.7750 | stairs | 0.1206 |
| building | 0.7574 | traffic light | 0.0897 |
| vehicle | 0.7398 | bike rack | 0.0704 |
| paved trail | 0.6210 | other walkable surface | 0.0506 |
| hand rail | 0.5650 | opening-door | 0.0438 |
| wall/fence | 0.5608 | animal | 0.0248 |
| sky | 0.5276 | guard rail/road barrier | 0.0199 |
| vegetation | 0.5190 | opening-gate | 0.0106 |
| pole | 0.3887 | rider | 0.0027 |
| curb | 0.3261 | water body | 0.0000 |
| road | 0.3155 | inaccessible surface | 0.0000 |
| obstacle | 0.3121 | railway track | absent |
| terrain | 0.3081 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7741 |
| sanpo3_not_safe | 0.3369 |
| sanpo3_obstacle | 0.8675 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 2.3832 / 0.3758 |
| Gradient norm, median | 59.12 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 12 |
| Update/weight ratio, log10 median, first / last | -3.96 / -4.64 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (39 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
