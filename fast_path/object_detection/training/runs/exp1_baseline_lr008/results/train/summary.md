# exp1_baseline_lr008

**finished** · 25/25 epochs · launches run001 · updated 2026-11-18 16:37:02 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.2975** (epoch 16, step 8448) |
| best.pt re-validated | 0.2975 |
| Pixel accuracy (best epoch) | 0.7140 |
| val_human mIoU (best epoch, human-labeled frames) | 0.2938 |
| sanpo3 mIoU (best epoch) | 0.6316 |
| Last validation mIoU | 0.2927 (epoch 25) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps |
| validation | every epoch |
| epochs | 25 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.008 |
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
| Forward GFLOPs per image | 1.3205 at 352x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 3.675 GB |
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 1h13m54s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.28 at $1.04/h |
| Speed (median) | 107 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 528 | 2.5013 | 0.2548 | 0.6827 | 1.1583 | 0.8411 | 0.2523 | 0.6071 |
| 2 | 1056 | 1.4455 | 0.2077 | 0.5905 | 1.4759 | 0.8287 | 0.2039 | 0.5636 |
| 3 ⭐ | 1584 | 1.3458 | 0.2608 | 0.6375 | 1.3735 | 0.7463 | 0.2568 | 0.6068 |
| 4 | 2112 | 1.1633 | 0.2389 | 0.6639 | 1.4174 | 0.7459 | 0.2356 | 0.5986 |
| 5 | 2640 | 1.0553 | 0.2539 | 0.6473 | 1.5687 | 0.7300 | 0.2500 | 0.5995 |
| 6 | 3168 | 0.9539 | 0.2510 | 0.6511 | 1.4282 | 0.6822 | 0.2471 | 0.6226 |
| 7 ⭐ | 3696 | 0.8882 | 0.2705 | 0.6784 | 1.6650 | 0.6388 | 0.2676 | 0.6066 |
| 8 ⭐ | 4224 | 0.8385 | 0.2780 | 0.6753 | 1.4268 | 0.6397 | 0.2749 | 0.6204 |
| 9 ⭐ | 4752 | 0.7806 | 0.2860 | 0.6994 | 1.3391 | 0.6228 | 0.2834 | 0.6483 |
| 10 ⭐ | 5280 | 0.7384 | 0.2950 | 0.7059 | 1.3895 | 0.6183 | 0.2933 | 0.6416 |
| 11 | 5808 | 0.6386 | 0.2864 | 0.6992 | 1.6354 | 0.5802 | 0.2819 | 0.6288 |
| 12 | 6336 | 0.6004 | 0.2867 | 0.7032 | 1.5259 | 0.5763 | 0.2831 | 0.6320 |
| 13 | 6864 | 0.5664 | 0.2925 | 0.7123 | 1.5094 | 0.5757 | 0.2888 | 0.6444 |
| 14 | 7392 | 0.5477 | 0.2950 | 0.7062 | 1.5952 | 0.5702 | 0.2913 | 0.6355 |
| 15 | 7920 | 0.5177 | 0.2930 | 0.7119 | 1.6062 | 0.5629 | 0.2896 | 0.6291 |
| 16 ⭐ | 8448 | 0.4985 | 0.2975 | 0.7140 | 1.6411 | 0.5554 | 0.2938 | 0.6316 |
| 17 | 8976 | 0.4771 | 0.2913 | 0.7056 | 1.6988 | 0.5495 | 0.2876 | 0.6322 |
| 18 | 9504 | 0.4630 | 0.2959 | 0.7115 | 1.6626 | 0.5463 | 0.2919 | 0.6355 |
| 19 | 10032 | 0.4475 | 0.2956 | 0.7109 | 1.6711 | 0.5455 | 0.2921 | 0.6315 |
| 20 | 10560 | 0.4347 | 0.2969 | 0.7127 | 1.6563 | 0.5453 | 0.2933 | 0.6358 |
| 21 | 11088 | 0.4239 | 0.2956 | 0.7111 | 1.6769 | 0.5378 | 0.2924 | 0.6360 |
| 22 | 11616 | 0.4151 | 0.2948 | 0.7101 | 1.6979 | 0.5336 | 0.2917 | 0.6360 |
| 23 | 12144 | 0.4109 | 0.2940 | 0.7098 | 1.7009 | 0.5292 | 0.2908 | 0.6362 |
| 24 | 12672 | 0.4085 | 0.2928 | 0.7095 | 1.7159 | 0.5228 | 0.2894 | 0.6360 |
| 25 | 13200 | 0.4025 | 0.2927 | 0.7095 | 1.7184 | 0.5228 | 0.2893 | 0.6362 |

⭐ = new best at that epoch.

## IoU per class (best epoch 16, step 8448)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8618 | road | 0.2171 |
| pedestrian | 0.7348 | stairs | 0.0950 |
| building | 0.7327 | other walkable surface | 0.0604 |
| sidewalk | 0.7300 | bike rack | 0.0495 |
| vehicle | 0.6913 | traffic light | 0.0263 |
| paved trail | 0.6106 | traffic sign | 0.0223 |
| sky | 0.5564 | opening-door | 0.0154 |
| wall/fence | 0.5213 | water body | 0.0106 |
| vegetation | 0.4728 | guard rail/road barrier | 0.0047 |
| hand rail | 0.4016 | opening-gate | 0.0000 |
| pole | 0.3271 | rider | 0.0000 |
| curb | 0.3166 | animal | 0.0000 |
| obstacle | 0.3075 | inaccessible surface | 0.0000 |
| tree | 0.2839 | railway track | absent |
| terrain | 0.2807 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7652 |
| sanpo3_not_safe | 0.2570 |
| sanpo3_obstacle | 0.8727 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.9282 / 0.5161 |
| Gradient norm, median | 37.90 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.05 / -4.29 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (27 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
