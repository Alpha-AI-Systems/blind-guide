# exp1_baseline_lr004

**finished** · 25/25 epochs · launches run001 · updated 2026-11-18 16:36:42 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3052** (epoch 20, step 10560) |
| best.pt re-validated | 0.3051 |
| Pixel accuracy (best epoch) | 0.7177 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3008 |
| sanpo3 mIoU (best epoch) | 0.6364 |
| Last validation mIoU | 0.3034 (epoch 25) |

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
| lr0 | 0.004 |
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
| Training time | 1h13m43s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.27 at $1.04/h |
| Speed (median) | 107 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 528 | 2.7321 | 0.2666 | 0.6997 | 1.2897 | 0.8383 | 0.2610 | 0.6089 |
| 2 | 1056 | 1.4131 | 0.2540 | 0.6481 | 1.4619 | 0.8431 | 0.2506 | 0.5852 |
| 3 | 1584 | 1.2026 | 0.2616 | 0.6423 | 1.3933 | 0.7670 | 0.2586 | 0.6068 |
| 4 ⭐ | 2112 | 1.0186 | 0.2675 | 0.6955 | 1.4629 | 0.7249 | 0.2641 | 0.6117 |
| 5 ⭐ | 2640 | 0.9360 | 0.2825 | 0.6895 | 1.5602 | 0.7083 | 0.2788 | 0.6080 |
| 6 | 3168 | 0.8557 | 0.2800 | 0.6768 | 1.3220 | 0.6907 | 0.2793 | 0.6473 |
| 7 ⭐ | 3696 | 0.7932 | 0.2835 | 0.6967 | 1.5553 | 0.6401 | 0.2794 | 0.6162 |
| 8 ⭐ | 4224 | 0.7528 | 0.3007 | 0.7174 | 1.4233 | 0.6297 | 0.2976 | 0.6347 |
| 9 | 4752 | 0.7079 | 0.2897 | 0.6919 | 1.4324 | 0.6240 | 0.2850 | 0.6287 |
| 10 | 5280 | 0.6777 | 0.2989 | 0.6994 | 1.5204 | 0.6179 | 0.2936 | 0.6296 |
| 11 | 5808 | 0.5913 | 0.2966 | 0.7102 | 1.4501 | 0.5817 | 0.2927 | 0.6294 |
| 12 | 6336 | 0.5655 | 0.2950 | 0.7030 | 1.6634 | 0.5845 | 0.2913 | 0.6302 |
| 13 | 6864 | 0.5421 | 0.2984 | 0.7043 | 1.5402 | 0.5871 | 0.2935 | 0.6331 |
| 14 ⭐ | 7392 | 0.5221 | 0.3038 | 0.7148 | 1.5282 | 0.5738 | 0.2991 | 0.6333 |
| 15 ⭐ | 7920 | 0.5021 | 0.3041 | 0.7177 | 1.5108 | 0.5736 | 0.2995 | 0.6347 |
| 16 | 8448 | 0.4848 | 0.3014 | 0.7155 | 1.6034 | 0.5583 | 0.2969 | 0.6331 |
| 17 | 8976 | 0.4675 | 0.3037 | 0.7160 | 1.5425 | 0.5500 | 0.2988 | 0.6388 |
| 18 | 9504 | 0.4579 | 0.3035 | 0.7167 | 1.5895 | 0.5487 | 0.2993 | 0.6364 |
| 19 ⭐ | 10032 | 0.4451 | 0.3050 | 0.7168 | 1.5786 | 0.5482 | 0.3007 | 0.6372 |
| 20 ⭐ | 10560 | 0.4350 | 0.3052 | 0.7177 | 1.5878 | 0.5462 | 0.3008 | 0.6364 |
| 21 | 11088 | 0.4280 | 0.3050 | 0.7165 | 1.5914 | 0.5427 | 0.3012 | 0.6374 |
| 22 | 11616 | 0.4209 | 0.3039 | 0.7152 | 1.6180 | 0.5397 | 0.3000 | 0.6361 |
| 23 | 12144 | 0.4171 | 0.3036 | 0.7151 | 1.6208 | 0.5360 | 0.2994 | 0.6352 |
| 24 | 12672 | 0.4156 | 0.3033 | 0.7153 | 1.6195 | 0.5344 | 0.2990 | 0.6352 |
| 25 | 13200 | 0.4108 | 0.3034 | 0.7153 | 1.6216 | 0.5333 | 0.2991 | 0.6351 |

⭐ = new best at that epoch.

## IoU per class (best epoch 20, step 10560)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8580 | road | 0.2209 |
| pedestrian | 0.7548 | stairs | 0.1029 |
| building | 0.7354 | other walkable surface | 0.0445 |
| sidewalk | 0.7288 | bike rack | 0.0370 |
| vehicle | 0.6930 | traffic sign | 0.0324 |
| paved trail | 0.6246 | traffic light | 0.0273 |
| sky | 0.5695 | opening-door | 0.0201 |
| hand rail | 0.5421 | opening-gate | 0.0198 |
| wall/fence | 0.5127 | guard rail/road barrier | 0.0111 |
| vegetation | 0.4796 | animal | 0.0009 |
| pole | 0.3383 | rider | 0.0000 |
| obstacle | 0.3331 | water body | 0.0000 |
| curb | 0.3160 | inaccessible surface | 0.0000 |
| tree | 0.2854 | railway track | absent |
| terrain | 0.2568 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7700 |
| sanpo3_not_safe | 0.2619 |
| sanpo3_obstacle | 0.8772 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.9282 / 0.5140 |
| Gradient norm, median | 53.94 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 10 |
| Update/weight ratio, log10 median, first / last | -3.34 / -4.50 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (27 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
