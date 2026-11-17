# exp4_frames_10fps

**finished** · 9/9 epochs · launches run001 · updated 2026-11-16 22:32:28 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3259** (epoch 4.51, step 4752) |
| best.pt re-validated | 0.3259 |
| Pixel accuracy (best epoch) | 0.7222 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3235 |
| sanpo3 mIoU (best epoch) | 0.6446 |
| Last validation mIoU | 0.3218 (epoch 9) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
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
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 2.4093 at 640x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 6.67 GB |
| Hardware | Tesla T4, 4 CPU cores, 48 GB (modal-container) |
| Training time | 1h27m44s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.70 at $1.16/h |
| Speed (median) | 62 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 528 | 2.7736 | 0.2681 | 0.7049 | 1.1702 | 0.8411 | 0.2625 | 0.6222 |
| 1 ⭐ | 1056 | 1.3501 | 0.2843 | 0.6775 | 1.7398 | 0.8211 | 0.2806 | 0.6145 |
| 1.50 ⭐ | 1584 | 1.0361 | 0.2955 | 0.7119 | 1.2766 | 0.7739 | 0.2914 | 0.6298 |
| 2 ⭐ | 2112 | 0.8592 | 0.3123 | 0.7207 | 1.3795 | 0.7473 | 0.3081 | 0.6522 |
| 2.50 ⭐ | 2640 | 0.7475 | 0.3129 | 0.7257 | 1.4128 | 0.6960 | 0.3081 | 0.6420 |
| 3.01 | 3168 | 0.6917 | 0.3119 | 0.7209 | 1.3836 | 0.6916 | 0.3072 | 0.6425 |
| 3.51 ⭐ | 3696 | 0.6368 | 0.3245 | 0.7221 | 1.5504 | 0.6202 | 0.3198 | 0.6392 |
| 4.01 | 4224 | 0.5845 | 0.3196 | 0.7179 | 1.5033 | 0.6170 | 0.3156 | 0.6438 |
| 4.51 ⭐ | 4752 | 0.5569 | 0.3259 | 0.7222 | 1.5985 | 0.5862 | 0.3235 | 0.6446 |
| 5.01 | 5280 | 0.5277 | 0.3220 | 0.7255 | 1.5659 | 0.5712 | 0.3179 | 0.6419 |
| 5.51 | 5808 | 0.5082 | 0.3164 | 0.7278 | 1.4120 | 0.5622 | 0.3118 | 0.6527 |
| 6.01 | 6336 | 0.4908 | 0.3248 | 0.7320 | 1.4577 | 0.5668 | 0.3198 | 0.6540 |
| 6.51 | 6864 | 0.4757 | 0.3236 | 0.7317 | 1.5155 | 0.5638 | 0.3179 | 0.6471 |
| 7.01 | 7392 | 0.4653 | 0.3215 | 0.7303 | 1.5109 | 0.5605 | 0.3156 | 0.6479 |
| 7.51 | 7920 | 0.4502 | 0.3232 | 0.7314 | 1.5193 | 0.5590 | 0.3178 | 0.6472 |
| 8.02 | 8448 | 0.4486 | 0.3229 | 0.7298 | 1.5138 | 0.5589 | 0.3173 | 0.6482 |
| 8.52 | 8976 | 0.4403 | 0.3225 | 0.7304 | 1.5159 | 0.5571 | 0.3167 | 0.6482 |
| 9 | 9486 | 0.4392 | 0.3218 | 0.7305 | 1.5142 | 0.5569 | 0.3163 | 0.6476 |

⭐ = new best at that epoch.

## IoU per class (best epoch 4.51, step 4752)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8502 | opening-gate | 0.2499 |
| sidewalk | 0.7637 | road | 0.2374 |
| pedestrian | 0.7503 | stairs | 0.1165 |
| building | 0.7498 | traffic sign | 0.0941 |
| vehicle | 0.7151 | opening-door | 0.0913 |
| sky | 0.6162 | other walkable surface | 0.0740 |
| paved trail | 0.5771 | bike rack | 0.0387 |
| wall/fence | 0.5415 | traffic light | 0.0322 |
| vegetation | 0.5035 | guard rail/road barrier | 0.0034 |
| hand rail | 0.4162 | rider | 0.0000 |
| pole | 0.3735 | animal | 0.0000 |
| obstacle | 0.3683 | water body | 0.0000 |
| curb | 0.3400 | inaccessible surface | 0.0000 |
| terrain | 0.3260 | railway track | absent |
| tree | 0.2972 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7729 |
| sanpo3_not_safe | 0.2800 |
| sanpo3_obstacle | 0.8807 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.6890 / 0.4666 |
| Gradient norm, median | 68.85 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.63 / -4.64 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (19 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
