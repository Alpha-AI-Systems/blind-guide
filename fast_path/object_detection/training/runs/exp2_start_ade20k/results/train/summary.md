# exp2_start_ade20k

**finished** · 18/18 epochs · launches run001 · updated 2026-11-15 19:12:17 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3222** (epoch 6) |
| best.pt re-validated | 0.3222 |
| Pixel accuracy (best epoch) | 0.7255 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3178 |
| sanpo3 mIoU (best epoch) | 0.6354 |
| Last epoch mIoU | 0.3186 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
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
| Training time | 1h12m29s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.25 at $1.04/h |
| Speed (median) | 62 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 1 ⭐ | 2.7929 | 0.2616 | 0.7028 | 1.2549 | 0.8390 | 0.2547 | 0.6170 |
| 2 ⭐ | 1.3639 | 0.3020 | 0.7193 | 1.2300 | 0.8117 | 0.2980 | 0.6344 |
| 3 ⭐ | 1.0603 | 0.3104 | 0.7071 | 1.2609 | 0.7829 | 0.3072 | 0.6567 |
| 4 ⭐ | 0.8782 | 0.3122 | 0.7221 | 1.4467 | 0.7525 | 0.3093 | 0.6368 |
| 5 | 0.7711 | 0.3011 | 0.7047 | 1.4865 | 0.7037 | 0.2981 | 0.6368 |
| 6 ⭐ | 0.7078 | 0.3222 | 0.7255 | 1.3789 | 0.6854 | 0.3178 | 0.6354 |
| 7 | 0.6565 | 0.3190 | 0.7233 | 1.4291 | 0.6621 | 0.3158 | 0.6370 |
| 8 | 0.6102 | 0.3217 | 0.7265 | 1.4677 | 0.6127 | 0.3187 | 0.6413 |
| 9 | 0.5757 | 0.3169 | 0.7246 | 1.5631 | 0.5823 | 0.3126 | 0.6354 |
| 10 | 0.5475 | 0.3177 | 0.7281 | 1.5625 | 0.5725 | 0.3136 | 0.6364 |
| 11 | 0.5256 | 0.3220 | 0.7205 | 1.6288 | 0.5751 | 0.3169 | 0.6380 |
| 12 | 0.5080 | 0.3152 | 0.7245 | 1.6216 | 0.5665 | 0.3100 | 0.6392 |
| 13 | 0.4944 | 0.3183 | 0.7298 | 1.5312 | 0.5604 | 0.3137 | 0.6410 |
| 14 | 0.4800 | 0.3203 | 0.7276 | 1.5972 | 0.5613 | 0.3160 | 0.6416 |
| 15 | 0.4717 | 0.3177 | 0.7277 | 1.5831 | 0.5620 | 0.3135 | 0.6404 |
| 16 | 0.4678 | 0.3172 | 0.7266 | 1.5992 | 0.5608 | 0.3130 | 0.6403 |
| 17 | 0.4593 | 0.3181 | 0.7275 | 1.6071 | 0.5608 | 0.3136 | 0.6409 |
| 18 | 0.4568 | 0.3186 | 0.7267 | 1.6060 | 0.5608 | 0.3142 | 0.6408 |

⭐ = new best at that epoch.

## IoU per class (best epoch 6)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8386 | road | 0.2315 |
| sidewalk | 0.7951 | stairs | 0.1793 |
| building | 0.7496 | traffic sign | 0.1009 |
| pedestrian | 0.7493 | other walkable surface | 0.0603 |
| vehicle | 0.7023 | bike rack | 0.0599 |
| paved trail | 0.6290 | traffic light | 0.0589 |
| hand rail | 0.6036 | opening-door | 0.0425 |
| sky | 0.6023 | opening-gate | 0.0395 |
| wall/fence | 0.5352 | guard rail/road barrier | 0.0229 |
| vegetation | 0.4989 | animal | 0.0182 |
| pole | 0.3614 | rider | 0.0000 |
| curb | 0.3342 | water body | 0.0000 |
| obstacle | 0.2923 | inaccessible surface | 0.0000 |
| tree | 0.2634 | railway track | absent |
| terrain | 0.2530 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7623 |
| sanpo3_not_safe | 0.2730 |
| sanpo3_obstacle | 0.8707 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.2817 / 0.4766 |
| Gradient norm, median | 67.82 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.65 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
