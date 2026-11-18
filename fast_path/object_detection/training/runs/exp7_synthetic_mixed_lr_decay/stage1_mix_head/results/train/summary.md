# exp7_synthetic_mixed_lr_decay/stage1_mix_head

**finished** · 25/25 epochs · launches run001 · updated 2026-11-18 17:32:28 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3773** (epoch 16, step 10112) |
| best.pt re-validated | 0.3773 |
| Pixel accuracy (best epoch) | 0.8196 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3733 |
| sanpo3 mIoU (best epoch) | 0.8690 |
| Last validation mIoU | 0.3692 (epoch 25) |

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
| Hardware | NVIDIA A100-SXM4-80GB, 8 CPU cores, 64 GB (modal-container) |
| Training time | 41m15s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 213 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 632 | 1.7883 | 0.2953 | 0.7686 | 0.8598 | 0.8343 | 0.2901 | 0.8339 |
| 2 ⭐ | 1264 | 1.0143 | 0.3401 | 0.7888 | 0.8065 | 0.7739 | 0.3343 | 0.8346 |
| 3 ⭐ | 1896 | 0.7982 | 0.3559 | 0.8031 | 0.8946 | 0.6667 | 0.3524 | 0.8492 |
| 4 | 2528 | 0.6902 | 0.3530 | 0.7960 | 0.9690 | 0.6072 | 0.3471 | 0.8567 |
| 5 | 3160 | 0.6296 | 0.3495 | 0.8000 | 0.9537 | 0.5705 | 0.3458 | 0.8444 |
| 6 | 3792 | 0.5951 | 0.3557 | 0.7982 | 1.0125 | 0.5589 | 0.3525 | 0.8565 |
| 7 | 4424 | 0.5670 | 0.3486 | 0.8076 | 0.9775 | 0.5632 | 0.3445 | 0.8560 |
| 8 ⭐ | 5056 | 0.5434 | 0.3622 | 0.8153 | 0.9306 | 0.5477 | 0.3558 | 0.8773 |
| 9 ⭐ | 5688 | 0.5250 | 0.3706 | 0.8145 | 0.9740 | 0.5452 | 0.3669 | 0.8624 |
| 10 ⭐ | 6320 | 0.5107 | 0.3715 | 0.8143 | 1.0950 | 0.5264 | 0.3672 | 0.8624 |
| 11 | 6952 | 0.4943 | 0.3679 | 0.8143 | 1.1631 | 0.5017 | 0.3658 | 0.8646 |
| 12 ⭐ | 7584 | 0.4821 | 0.3744 | 0.8193 | 1.1342 | 0.5010 | 0.3713 | 0.8714 |
| 13 | 8216 | 0.4698 | 0.3707 | 0.8198 | 1.1680 | 0.5049 | 0.3681 | 0.8672 |
| 14 | 8848 | 0.4576 | 0.3692 | 0.8200 | 1.1823 | 0.5121 | 0.3663 | 0.8685 |
| 15 ⭐ | 9480 | 0.4422 | 0.3745 | 0.8229 | 1.2098 | 0.5027 | 0.3706 | 0.8705 |
| 16 ⭐ | 10112 | 0.4350 | 0.3773 | 0.8196 | 1.2292 | 0.4983 | 0.3733 | 0.8690 |
| 17 | 10744 | 0.4278 | 0.3725 | 0.8211 | 1.2328 | 0.4944 | 0.3697 | 0.8718 |
| 18 | 11376 | 0.4199 | 0.3732 | 0.8206 | 1.2394 | 0.4925 | 0.3697 | 0.8710 |
| 19 | 12008 | 0.4149 | 0.3718 | 0.8208 | 1.2381 | 0.4919 | 0.3683 | 0.8704 |
| 20 | 12640 | 0.4107 | 0.3725 | 0.8203 | 1.2400 | 0.4913 | 0.3688 | 0.8699 |
| 21 | 13272 | 0.4054 | 0.3705 | 0.8197 | 1.2383 | 0.4893 | 0.3665 | 0.8703 |
| 22 | 13904 | 0.4023 | 0.3689 | 0.8190 | 1.2440 | 0.4904 | 0.3648 | 0.8697 |
| 23 | 14536 | 0.4020 | 0.3688 | 0.8191 | 1.2460 | 0.4912 | 0.3649 | 0.8697 |
| 24 | 15168 | 0.3992 | 0.3690 | 0.8192 | 1.2473 | 0.4911 | 0.3651 | 0.8696 |
| 25 | 15800 | 0.3963 | 0.3692 | 0.8192 | 1.2498 | 0.4910 | 0.3654 | 0.8695 |

⭐ = new best at that epoch.

## IoU per class (best epoch 16, step 10112)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9557 | traffic sign | 0.2479 |
| paved trail | 0.8702 | stairs | 0.2176 |
| sidewalk | 0.8194 | pole | 0.2062 |
| pedestrian | 0.7829 | traffic light | 0.1299 |
| crosswalk | 0.7070 | animal | 0.1028 |
| wall/fence | 0.6977 | opening-door | 0.0140 |
| terrain | 0.6463 | other walkable surface | 0.0089 |
| tree | 0.6366 | bike rack | 0.0065 |
| vegetation | 0.6364 | inaccessible surface | 0.0042 |
| building | 0.6262 | rider | 0.0018 |
| water body | 0.5899 | guard rail/road barrier | 0.0000 |
| road | 0.5598 | hand rail | 0.0000 |
| vehicle | 0.5355 | opening-gate | 0.0000 |
| obstacle | 0.2914 | railway track | absent |
| curb | 0.2690 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8229 |
| sanpo3_not_safe | 0.8899 |
| sanpo3_obstacle | 0.8943 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 5.1549 / 0.4128 |
| Gradient norm, median | 45.60 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 11 |
| Update/weight ratio, log10 median, first / last | -3.73 / -4.76 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (32 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
