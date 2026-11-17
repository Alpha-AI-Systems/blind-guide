# exp4_frames_15fps

**finished** · 6/6 epochs · launches run001 · updated 2026-11-16 22:34:03 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3235** (epoch 2.01, step 3168) |
| best.pt re-validated | 0.3236 |
| Pixel accuracy (best epoch) | 0.7264 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3190 |
| sanpo3 mIoU (best epoch) | 0.6385 |
| Last validation mIoU | 0.3160 (epoch 6) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 15 fps |
| validation | every 528 steps |
| epochs | 6 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 0.67 |
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
| Peak GPU memory | 6.671 GB |
| Hardware | Tesla T4, 4 CPU cores, 64 GB (modal-container) |
| Training time | 1h27m03s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.87 at $1.29/h |
| Speed (median) | 62 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.33 ⭐ | 528 | 2.7812 | 0.2629 | 0.6994 | 1.1236 | 0.8387 | 0.2554 | 0.6340 |
| 0.67 ⭐ | 1056 | 1.3434 | 0.2663 | 0.6846 | 1.2807 | 0.8212 | 0.2622 | 0.6288 |
| 1 ⭐ | 1584 | 1.0365 | 0.2920 | 0.7068 | 1.4358 | 0.7745 | 0.2902 | 0.6287 |
| 1.34 ⭐ | 2112 | 0.8413 | 0.3085 | 0.7197 | 1.3904 | 0.7263 | 0.3026 | 0.6324 |
| 1.67 ⭐ | 2640 | 0.7392 | 0.3110 | 0.7107 | 1.3468 | 0.6972 | 0.3058 | 0.6509 |
| 2.01 ⭐ | 3168 | 0.6865 | 0.3235 | 0.7264 | 1.3800 | 0.6857 | 0.3190 | 0.6385 |
| 2.34 | 3696 | 0.6223 | 0.3125 | 0.7188 | 1.4786 | 0.6344 | 0.3069 | 0.6374 |
| 2.67 | 4224 | 0.5795 | 0.3125 | 0.7116 | 1.6123 | 0.6058 | 0.3058 | 0.6355 |
| 3.01 | 4752 | 0.5469 | 0.3178 | 0.7281 | 1.4324 | 0.5800 | 0.3116 | 0.6409 |
| 3.34 | 5280 | 0.5212 | 0.3110 | 0.7291 | 1.4417 | 0.5695 | 0.3057 | 0.6446 |
| 3.68 | 5808 | 0.4973 | 0.3170 | 0.7215 | 1.5130 | 0.5753 | 0.3108 | 0.6405 |
| 4.01 | 6336 | 0.4824 | 0.3181 | 0.7288 | 1.4983 | 0.5648 | 0.3123 | 0.6410 |
| 4.34 | 6864 | 0.4614 | 0.3184 | 0.7234 | 1.5667 | 0.5636 | 0.3116 | 0.6379 |
| 4.68 | 7392 | 0.4576 | 0.3192 | 0.7246 | 1.5422 | 0.5659 | 0.3121 | 0.6384 |
| 5.01 | 7920 | 0.4493 | 0.3211 | 0.7288 | 1.5052 | 0.5647 | 0.3141 | 0.6412 |
| 5.35 | 8448 | 0.4394 | 0.3157 | 0.7259 | 1.5175 | 0.5619 | 0.3089 | 0.6405 |
| 5.68 | 8976 | 0.4346 | 0.3163 | 0.7253 | 1.5293 | 0.5636 | 0.3091 | 0.6401 |
| 6 | 9480 | 0.4311 | 0.3160 | 0.7261 | 1.5236 | 0.5624 | 0.3090 | 0.6400 |

⭐ = new best at that epoch.

## IoU per class (best epoch 2.01, step 3168)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8038 | opening-gate | 0.2484 |
| sidewalk | 0.7723 | road | 0.2400 |
| building | 0.7496 | stairs | 0.1257 |
| pedestrian | 0.7357 | opening-door | 0.0941 |
| vehicle | 0.6923 | traffic sign | 0.0913 |
| paved trail | 0.6360 | bike rack | 0.0578 |
| sky | 0.5970 | other walkable surface | 0.0457 |
| wall/fence | 0.5319 | guard rail/road barrier | 0.0278 |
| vegetation | 0.4954 | animal | 0.0161 |
| hand rail | 0.4893 | traffic light | 0.0159 |
| curb | 0.3523 | rider | 0.0000 |
| pole | 0.3357 | water body | 0.0000 |
| terrain | 0.3058 | inaccessible surface | 0.0000 |
| tree | 0.3008 | railway track | absent |
| obstacle | 0.2983 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7626 |
| sanpo3_not_safe | 0.2817 |
| sanpo3_obstacle | 0.8712 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 7.7342 / 0.5017 |
| Gradient norm, median | 69.01 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.65 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (19 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
