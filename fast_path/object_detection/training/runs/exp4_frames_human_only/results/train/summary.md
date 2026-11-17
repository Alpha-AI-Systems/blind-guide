# exp4_frames_human_only

**finished** · 36/36 epochs · launches run001 · updated 2026-11-16 22:29:29 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3184** (epoch 29.89, step 7920) |
| best.pt re-validated | 0.3184 |
| Pixel accuracy (best epoch) | 0.7246 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3149 |
| sanpo3 mIoU (best epoch) | 0.6426 |
| Last validation mIoU | 0.3182 (epoch 36) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | human (5 fps) |
| validation | every 528 steps |
| epochs | 36 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 4 |
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
| Peak GPU memory | 6.672 GB |
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 1h27m48s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.51 at $1.04/h |
| Speed (median) | 62 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1.99 ⭐ | 528 | 2.8138 | 0.2625 | 0.7114 | 1.1314 | 0.8391 | 0.2578 | 0.6128 |
| 3.98 ⭐ | 1056 | 1.3909 | 0.2640 | 0.7033 | 1.2980 | 0.8335 | 0.2608 | 0.6091 |
| 5.98 ⭐ | 1584 | 1.0908 | 0.3027 | 0.7179 | 1.3064 | 0.7946 | 0.2995 | 0.6262 |
| 7.97 ⭐ | 2112 | 0.8889 | 0.3129 | 0.7201 | 1.2053 | 0.7650 | 0.3094 | 0.6611 |
| 9.96 | 2640 | 0.7975 | 0.2964 | 0.7079 | 1.5814 | 0.7299 | 0.2934 | 0.6356 |
| 11.95 ⭐ | 3168 | 0.7335 | 0.3131 | 0.7156 | 1.5557 | 0.7178 | 0.3100 | 0.6411 |
| 13.95 | 3696 | 0.6738 | 0.3120 | 0.7067 | 1.5871 | 0.6591 | 0.3074 | 0.6332 |
| 15.94 | 4224 | 0.6186 | 0.3131 | 0.7238 | 1.6181 | 0.6015 | 0.3098 | 0.6411 |
| 17.93 | 4752 | 0.5833 | 0.3127 | 0.7217 | 1.5636 | 0.5881 | 0.3080 | 0.6339 |
| 19.92 ⭐ | 5280 | 0.5584 | 0.3154 | 0.7258 | 1.4911 | 0.5841 | 0.3128 | 0.6423 |
| 21.92 | 5808 | 0.5345 | 0.3140 | 0.7238 | 1.5678 | 0.5767 | 0.3114 | 0.6405 |
| 23.91 ⭐ | 6336 | 0.5157 | 0.3171 | 0.7206 | 1.6109 | 0.5790 | 0.3136 | 0.6387 |
| 25.90 ⭐ | 6864 | 0.5005 | 0.3179 | 0.7240 | 1.5660 | 0.5751 | 0.3145 | 0.6419 |
| 27.89 | 7392 | 0.4876 | 0.3160 | 0.7237 | 1.5590 | 0.5699 | 0.3127 | 0.6427 |
| 29.89 ⭐ | 7920 | 0.4786 | 0.3184 | 0.7246 | 1.5884 | 0.5709 | 0.3149 | 0.6426 |
| 31.88 | 8448 | 0.4713 | 0.3181 | 0.7241 | 1.5945 | 0.5697 | 0.3145 | 0.6417 |
| 33.87 | 8976 | 0.4665 | 0.3177 | 0.7245 | 1.5884 | 0.5698 | 0.3140 | 0.6426 |
| 35.86 | 9504 | 0.4624 | 0.3182 | 0.7251 | 1.5885 | 0.5693 | 0.3144 | 0.6423 |
| 36 | 9540 | 0.4697 | 0.3182 | 0.7252 | 1.5892 | 0.5693 | 0.3145 | 0.6422 |

⭐ = new best at that epoch.

## IoU per class (best epoch 29.89, step 7920)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8472 | road | 0.2397 |
| sidewalk | 0.7768 | stairs | 0.1353 |
| pedestrian | 0.7615 | traffic sign | 0.0654 |
| building | 0.7561 | other walkable surface | 0.0571 |
| vehicle | 0.7176 | bike rack | 0.0385 |
| sky | 0.6075 | traffic light | 0.0274 |
| paved trail | 0.5825 | opening-gate | 0.0186 |
| hand rail | 0.5467 | opening-door | 0.0145 |
| wall/fence | 0.5420 | guard rail/road barrier | 0.0073 |
| vegetation | 0.5061 | animal | 0.0029 |
| curb | 0.3481 | rider | 0.0000 |
| pole | 0.3470 | water body | 0.0000 |
| terrain | 0.3319 | inaccessible surface | 0.0000 |
| obstacle | 0.3211 | railway track | absent |
| tree | 0.3171 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7693 |
| sanpo3_not_safe | 0.2823 |
| sanpo3_obstacle | 0.8764 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.7142 / 0.4828 |
| Gradient norm, median | 65.37 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.64 / -4.64 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
