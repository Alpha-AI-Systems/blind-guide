# local_chest_from_head_ade20k

**finished** · 18/18 epochs · launches run001 · updated 2026-11-17 17:03:48 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3311** (epoch 15) |
| best.pt re-validated | 0.3311 |
| Pixel accuracy (best epoch) | 0.7411 |
| sanpo3 mIoU (best epoch) | 0.6515 |
| Last epoch mIoU | 0.3292 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | C:\Projects\blind-guide\fast_path\object_detection\training\runs\local_head_pretrain_ade20k\results\train\ultralytics\weights\best.pt |
| data | inputs |
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
| Peak GPU memory | 6.724 GB |
| Hardware | NVIDIA GeForce RTX 4060 Ti, 4 CPU cores, 32 GB (local) |
| Training time | 1h48m10s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 51 img/s, data wait 48% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss | sanpo3 mIoU |
|---|---|---|---|---|---|---|
| 1 ⭐ | 2.0454 | 0.3099 | 0.7308 | 1.0361 | 0.6716 | 0.6512 |
| 2 | 1.1084 | 0.2987 | 0.7006 | 1.1995 | 0.6770 | 0.6484 |
| 3 ⭐ | 0.9048 | 0.3251 | 0.7243 | 1.3231 | 0.6616 | 0.6446 |
| 4 | 0.7725 | 0.3072 | 0.7218 | 1.4039 | 0.6462 | 0.6318 |
| 5 | 0.7002 | 0.3121 | 0.7166 | 1.3429 | 0.6288 | 0.6515 |
| 6 | 0.6440 | 0.3189 | 0.7217 | 1.5151 | 0.5828 | 0.6491 |
| 7 ⭐ | 0.6039 | 0.3278 | 0.7295 | 1.4778 | 0.5787 | 0.6482 |
| 8 | 0.5687 | 0.3136 | 0.7283 | 1.4898 | 0.5707 | 0.6470 |
| 9 | 0.5377 | 0.3238 | 0.7338 | 1.4723 | 0.5694 | 0.6473 |
| 10 | 0.5163 | 0.3231 | 0.7313 | 1.4852 | 0.5689 | 0.6496 |
| 11 | 0.4990 | 0.3232 | 0.7337 | 1.5379 | 0.5573 | 0.6511 |
| 12 | 0.4794 | 0.3219 | 0.7359 | 1.5297 | 0.5514 | 0.6460 |
| 13 ⭐ | 0.4658 | 0.3298 | 0.7415 | 1.5095 | 0.5520 | 0.6494 |
| 14 | 0.4583 | 0.3288 | 0.7403 | 1.5192 | 0.5522 | 0.6484 |
| 15 ⭐ | 0.4483 | 0.3311 | 0.7411 | 1.4992 | 0.5495 | 0.6515 |
| 16 | 0.4389 | 0.3271 | 0.7399 | 1.5168 | 0.5510 | 0.6499 |
| 17 | 0.4365 | 0.3290 | 0.7410 | 1.5132 | 0.5500 | 0.6496 |
| 18 | 0.4356 | 0.3292 | 0.7410 | 1.5121 | 0.5505 | 0.6501 |

⭐ = new best at that epoch.

## IoU per class (best epoch 15)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8571 | road | 0.2545 |
| sidewalk | 0.7950 | traffic sign | 0.1198 |
| pedestrian | 0.7659 | stairs | 0.1157 |
| building | 0.7567 | bike rack | 0.0919 |
| vehicle | 0.7094 | traffic light | 0.0723 |
| paved trail | 0.6343 | other walkable surface | 0.0694 |
| sky | 0.5749 | animal | 0.0632 |
| wall/fence | 0.5494 | opening-gate | 0.0583 |
| vegetation | 0.5198 | opening-door | 0.0211 |
| hand rail | 0.4836 | guard rail/road barrier | 0.0112 |
| curb | 0.3857 | rider | 0.0000 |
| pole | 0.3608 | water body | 0.0000 |
| obstacle | 0.3589 | inaccessible surface | 0.0000 |
| terrain | 0.3281 | railway track | absent |
| tree | 0.3123 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7785 |
| sanpo3_not_safe | 0.2942 |
| sanpo3_obstacle | 0.8817 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 5.8887 / 0.4047 |
| Gradient norm, median | 67.12 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.63 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
