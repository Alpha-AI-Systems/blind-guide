# exp7_synthetic_mixed/stage1_mix_head

**finished** · 25/25 epochs · launches run001 · updated 2026-11-17 20:10:12 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3739** (epoch 15, step 9480) |
| best.pt re-validated | 0.3739 |
| Pixel accuracy (best epoch) | 0.8218 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3684 |
| sanpo3 mIoU (best epoch) | 0.8749 |
| Last validation mIoU | 0.3653 (epoch 25) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps + train_head_5fps (16+16 per batch) |
| validation | every epoch |
| epochs | 25 |
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
| Training time | 41m41s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 210 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 632 | 1.7883 | 0.2948 | 0.7685 | 0.8612 | 0.8344 | 0.2897 | 0.8335 |
| 2 ⭐ | 1264 | 1.0144 | 0.3377 | 0.7883 | 0.8200 | 0.7738 | 0.3319 | 0.8349 |
| 3 ⭐ | 1896 | 0.7968 | 0.3520 | 0.8067 | 0.8836 | 0.6603 | 0.3485 | 0.8484 |
| 4 | 2528 | 0.6904 | 0.3476 | 0.7866 | 0.9752 | 0.6139 | 0.3420 | 0.8461 |
| 5 | 3160 | 0.6290 | 0.3506 | 0.8017 | 0.9538 | 0.5770 | 0.3501 | 0.8581 |
| 6 ⭐ | 3792 | 0.5960 | 0.3553 | 0.8031 | 1.0277 | 0.5565 | 0.3513 | 0.8545 |
| 7 ⭐ | 4424 | 0.5667 | 0.3584 | 0.8123 | 0.9261 | 0.5646 | 0.3526 | 0.8569 |
| 8 | 5056 | 0.5441 | 0.3533 | 0.8042 | 1.0071 | 0.5547 | 0.3478 | 0.8605 |
| 9 ⭐ | 5688 | 0.5250 | 0.3586 | 0.8161 | 0.9897 | 0.5449 | 0.3542 | 0.8740 |
| 10 ⭐ | 6320 | 0.5108 | 0.3675 | 0.8151 | 1.0871 | 0.5275 | 0.3624 | 0.8673 |
| 11 | 6952 | 0.4957 | 0.3639 | 0.8092 | 1.1750 | 0.5020 | 0.3579 | 0.8645 |
| 12 ⭐ | 7584 | 0.4829 | 0.3738 | 0.8207 | 1.1317 | 0.5084 | 0.3677 | 0.8772 |
| 13 | 8216 | 0.4698 | 0.3706 | 0.8206 | 1.1359 | 0.5241 | 0.3642 | 0.8761 |
| 14 | 8848 | 0.4544 | 0.3663 | 0.8184 | 1.1714 | 0.5150 | 0.3615 | 0.8721 |
| 15 ⭐ | 9480 | 0.4414 | 0.3739 | 0.8218 | 1.1993 | 0.5113 | 0.3684 | 0.8749 |
| 16 | 10112 | 0.4346 | 0.3678 | 0.8182 | 1.2164 | 0.5059 | 0.3625 | 0.8744 |
| 17 | 10744 | 0.4275 | 0.3681 | 0.8195 | 1.2249 | 0.5012 | 0.3617 | 0.8751 |
| 18 | 11376 | 0.4203 | 0.3683 | 0.8193 | 1.2483 | 0.5016 | 0.3607 | 0.8753 |
| 19 | 12008 | 0.4145 | 0.3672 | 0.8198 | 1.2396 | 0.5006 | 0.3596 | 0.8748 |
| 20 | 12640 | 0.4108 | 0.3675 | 0.8198 | 1.2450 | 0.4988 | 0.3599 | 0.8746 |
| 21 | 13272 | 0.4057 | 0.3659 | 0.8188 | 1.2521 | 0.4991 | 0.3582 | 0.8739 |
| 22 | 13904 | 0.4018 | 0.3648 | 0.8183 | 1.2595 | 0.5006 | 0.3572 | 0.8734 |
| 23 | 14536 | 0.4020 | 0.3652 | 0.8187 | 1.2598 | 0.5000 | 0.3577 | 0.8737 |
| 24 | 15168 | 0.3995 | 0.3652 | 0.8189 | 1.2608 | 0.4992 | 0.3577 | 0.8740 |
| 25 | 15800 | 0.3964 | 0.3653 | 0.8189 | 1.2617 | 0.4988 | 0.3578 | 0.8741 |

⭐ = new best at that epoch.

## IoU per class (best epoch 15, step 9480)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9566 | pole | 0.2309 |
| paved trail | 0.8663 | animal | 0.2014 |
| sidewalk | 0.8109 | traffic sign | 0.1808 |
| pedestrian | 0.7842 | traffic light | 0.1098 |
| crosswalk | 0.6929 | stairs | 0.0735 |
| wall/fence | 0.6915 | other walkable surface | 0.0238 |
| terrain | 0.6627 | bike rack | 0.0106 |
| vegetation | 0.6523 | opening-door | 0.0079 |
| tree | 0.6382 | rider | 0.0028 |
| building | 0.6282 | inaccessible surface | 0.0026 |
| water body | 0.5958 | guard rail/road barrier | 0.0000 |
| road | 0.5670 | hand rail | 0.0000 |
| vehicle | 0.5288 | opening-gate | 0.0000 |
| obstacle | 0.2948 | railway track | absent |
| curb | 0.2546 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8313 |
| sanpo3_not_safe | 0.8937 |
| sanpo3_obstacle | 0.8996 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 5.1549 / 0.4100 |
| Gradient norm, median | 45.92 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 11 |
| Update/weight ratio, log10 median, first / last | -3.73 / -4.76 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (32 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
