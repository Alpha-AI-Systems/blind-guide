# exp2_start_cityscapes

**finished** · 18/18 epochs · launches run001 · updated 2026-11-15 19:05:17 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3210** (epoch 9) |
| best.pt re-validated | 0.3210 |
| Pixel accuracy (best epoch) | 0.7354 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3190 |
| sanpo3 mIoU (best epoch) | 0.6512 |
| Last epoch mIoU | 0.3182 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem.pt |
| data | semantic_v1_short352 |
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
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 1h26m08s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.49 at $1.04/h |
| Speed (median) | 63 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 1 ⭐ | 3.1045 | 0.2535 | 0.6964 | 1.1398 | 0.8426 | 0.2515 | 0.6177 |
| 2 ⭐ | 1.6470 | 0.2788 | 0.7200 | 1.0570 | 0.8170 | 0.2747 | 0.6287 |
| 3 ⭐ | 1.2388 | 0.2908 | 0.7056 | 1.2003 | 0.8021 | 0.2893 | 0.6217 |
| 4 ⭐ | 1.0233 | 0.3050 | 0.7202 | 1.3706 | 0.7349 | 0.3027 | 0.6258 |
| 5 ⭐ | 0.8686 | 0.3143 | 0.7219 | 1.2754 | 0.7254 | 0.3115 | 0.6498 |
| 6 ⭐ | 0.7864 | 0.3177 | 0.7331 | 1.2817 | 0.7205 | 0.3147 | 0.6476 |
| 7 | 0.7313 | 0.3072 | 0.7280 | 1.2974 | 0.7064 | 0.3044 | 0.6448 |
| 8 | 0.6840 | 0.3129 | 0.7306 | 1.3857 | 0.6883 | 0.3109 | 0.6387 |
| 9 ⭐ | 0.6499 | 0.3210 | 0.7354 | 1.3843 | 0.6785 | 0.3190 | 0.6512 |
| 10 | 0.6208 | 0.3197 | 0.7327 | 1.3582 | 0.6766 | 0.3173 | 0.6551 |
| 11 | 0.5960 | 0.3175 | 0.7344 | 1.3995 | 0.6750 | 0.3135 | 0.6498 |
| 12 | 0.5774 | 0.3146 | 0.7329 | 1.4276 | 0.6544 | 0.3113 | 0.6475 |
| 13 | 0.5556 | 0.3176 | 0.7382 | 1.4231 | 0.6459 | 0.3140 | 0.6491 |
| 14 | 0.5468 | 0.3168 | 0.7349 | 1.4338 | 0.6407 | 0.3130 | 0.6472 |
| 15 | 0.5365 | 0.3189 | 0.7375 | 1.4010 | 0.6322 | 0.3152 | 0.6521 |
| 16 | 0.5254 | 0.3194 | 0.7361 | 1.4218 | 0.6168 | 0.3150 | 0.6512 |
| 17 | 0.5146 | 0.3191 | 0.7360 | 1.4485 | 0.6092 | 0.3149 | 0.6492 |
| 18 | 0.5094 | 0.3182 | 0.7350 | 1.4397 | 0.6067 | 0.3140 | 0.6477 |

⭐ = new best at that epoch.

## IoU per class (best epoch 9)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8258 | curb | 0.2699 |
| sidewalk | 0.7874 | stairs | 0.1153 |
| pedestrian | 0.7790 | other walkable surface | 0.0844 |
| building | 0.7463 | opening-gate | 0.0780 |
| vehicle | 0.7299 | traffic light | 0.0391 |
| paved trail | 0.6233 | bike rack | 0.0377 |
| sky | 0.6121 | animal | 0.0293 |
| wall/fence | 0.5487 | opening-door | 0.0224 |
| vegetation | 0.5235 | guard rail/road barrier | 0.0191 |
| hand rail | 0.4721 | traffic sign | 0.0171 |
| pole | 0.3994 | rider | 0.0000 |
| terrain | 0.3268 | water body | 0.0000 |
| obstacle | 0.3129 | inaccessible surface | 0.0000 |
| tree | 0.3081 | railway track | absent |
| road | 0.2794 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7749 |
| sanpo3_not_safe | 0.3061 |
| sanpo3_obstacle | 0.8727 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 12.3590 / 0.7012 |
| Gradient norm, median | 55.53 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.75 / -4.75 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
