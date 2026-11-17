# exp6_synthetic_then_real/stage4_real_chest

**finished** · 18/18 epochs · launches run001 · updated 2026-11-17 19:58:00 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3280** (epoch 12, step 6336) |
| best.pt re-validated | 0.3280 |
| Pixel accuracy (best epoch) | 0.7334 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3236 |
| sanpo3 mIoU (best epoch) | 0.6505 |
| Last validation mIoU | 0.3267 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps |
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
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 48 GB (modal-container) |
| Training time | 27m08s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 202 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 528 | 2.8118 | 0.2776 | 0.7122 | 1.0670 | 0.8132 | 0.2736 | 0.6370 |
| 2 ⭐ | 1056 | 1.2937 | 0.3106 | 0.7192 | 1.1452 | 0.7514 | 0.3058 | 0.6917 |
| 3 | 1584 | 0.9921 | 0.3076 | 0.7069 | 1.2324 | 0.7113 | 0.3039 | 0.6269 |
| 4 ⭐ | 2112 | 0.8201 | 0.3183 | 0.6947 | 1.3927 | 0.6944 | 0.3148 | 0.6542 |
| 5 | 2640 | 0.7372 | 0.3176 | 0.7187 | 1.3351 | 0.6561 | 0.3142 | 0.6507 |
| 6 ⭐ | 3168 | 0.6733 | 0.3223 | 0.7213 | 1.3903 | 0.6488 | 0.3171 | 0.6455 |
| 7 | 3696 | 0.6299 | 0.3161 | 0.7242 | 1.3500 | 0.6479 | 0.3132 | 0.6363 |
| 8 | 4224 | 0.5890 | 0.3211 | 0.7279 | 1.4356 | 0.6121 | 0.3162 | 0.6570 |
| 9 ⭐ | 4752 | 0.5585 | 0.3261 | 0.7336 | 1.4234 | 0.5903 | 0.3215 | 0.6445 |
| 10 | 5280 | 0.5330 | 0.3240 | 0.7313 | 1.4751 | 0.5786 | 0.3188 | 0.6524 |
| 11 | 5808 | 0.5077 | 0.3257 | 0.7365 | 1.4584 | 0.5726 | 0.3217 | 0.6513 |
| 12 ⭐ | 6336 | 0.4900 | 0.3280 | 0.7334 | 1.4999 | 0.5707 | 0.3236 | 0.6505 |
| 13 | 6864 | 0.4742 | 0.3268 | 0.7374 | 1.4861 | 0.5677 | 0.3225 | 0.6530 |
| 14 | 7392 | 0.4621 | 0.3266 | 0.7346 | 1.5275 | 0.5661 | 0.3226 | 0.6468 |
| 15 | 7920 | 0.4508 | 0.3260 | 0.7356 | 1.5527 | 0.5633 | 0.3211 | 0.6476 |
| 16 | 8448 | 0.4468 | 0.3270 | 0.7355 | 1.5404 | 0.5631 | 0.3230 | 0.6498 |
| 17 | 8976 | 0.4415 | 0.3266 | 0.7348 | 1.5433 | 0.5630 | 0.3224 | 0.6506 |
| 18 | 9504 | 0.4401 | 0.3267 | 0.7349 | 1.5421 | 0.5628 | 0.3226 | 0.6495 |

⭐ = new best at that epoch.

## IoU per class (best epoch 12, step 6336)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8626 | road | 0.2602 |
| sidewalk | 0.7805 | stairs | 0.1178 |
| pedestrian | 0.7763 | traffic light | 0.0897 |
| building | 0.7622 | other walkable surface | 0.0853 |
| vehicle | 0.7213 | bike rack | 0.0837 |
| paved trail | 0.6097 | traffic sign | 0.0670 |
| wall/fence | 0.5482 | animal | 0.0532 |
| sky | 0.5415 | opening-gate | 0.0521 |
| hand rail | 0.5200 | opening-door | 0.0453 |
| vegetation | 0.5007 | guard rail/road barrier | 0.0192 |
| pole | 0.3676 | rider | 0.0000 |
| curb | 0.3638 | water body | 0.0000 |
| obstacle | 0.3463 | inaccessible surface | 0.0000 |
| terrain | 0.3092 | railway track | absent |
| tree | 0.2998 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7757 |
| sanpo3_not_safe | 0.2956 |
| sanpo3_obstacle | 0.8801 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 14.0972 / 0.5398 |
| Gradient norm, median | 71.04 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.66 / -4.65 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
