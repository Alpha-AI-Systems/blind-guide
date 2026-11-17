# local_head_pretrain_ade20k

**finished** · 25/25 epochs · launches run001 · updated 2026-11-17 14:46:58 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3688** (epoch 12) |
| best.pt re-validated | 0.3688 |
| Pixel accuracy (best epoch) | 0.8046 |
| sanpo3 mIoU (best epoch) | 0.8641 |
| Last epoch mIoU | 0.3676 (epoch 25) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | C:\Projects\blind-guide\weights\yolo26n-sem-ade20k.pt |
| data | inputs |
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
| Peak GPU memory | 6.724 GB |
| Hardware | NVIDIA GeForce RTX 4060 Ti, 4 CPU cores, 32 GB (local) |
| Training time | 1h27m24s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 52 img/s, data wait 46% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss | sanpo3 mIoU |
|---|---|---|---|---|---|---|
| 1 ⭐ | 2.5395 | 0.2977 | 0.7701 | 0.8774 | 0.8359 | 0.8178 |
| 2 ⭐ | 1.1743 | 0.3364 | 0.7917 | 0.8187 | 0.8117 | 0.8469 |
| 3 | 0.9221 | 0.3335 | 0.7806 | 0.9551 | 0.7666 | 0.8551 |
| 4 ⭐ | 0.7767 | 0.3629 | 0.8025 | 0.8680 | 0.7512 | 0.8642 |
| 5 | 0.6972 | 0.3519 | 0.7992 | 0.9909 | 0.6880 | 0.8515 |
| 6 ⭐ | 0.6436 | 0.3664 | 0.8113 | 1.0229 | 0.6494 | 0.8683 |
| 7 | 0.6046 | 0.3584 | 0.8070 | 1.0320 | 0.5914 | 0.8655 |
| 8 | 0.5753 | 0.3626 | 0.8067 | 1.0875 | 0.5524 | 0.8619 |
| 9 ⭐ | 0.5554 | 0.3676 | 0.8062 | 1.0217 | 0.5452 | 0.8602 |
| 10 | 0.5341 | 0.3639 | 0.8043 | 1.0522 | 0.5466 | 0.8640 |
| 11 | 0.5078 | 0.3602 | 0.8051 | 1.0552 | 0.5320 | 0.8595 |
| 12 ⭐ | 0.4947 | 0.3688 | 0.8046 | 1.1035 | 0.5400 | 0.8641 |
| 13 | 0.4740 | 0.3687 | 0.8050 | 1.0929 | 0.5338 | 0.8623 |
| 14 | 0.4617 | 0.3687 | 0.8125 | 1.0388 | 0.5209 | 0.8696 |
| 15 | 0.4490 | 0.3631 | 0.8057 | 1.0838 | 0.5306 | 0.8678 |
| 16 | 0.4370 | 0.3640 | 0.8103 | 1.0884 | 0.5198 | 0.8655 |
| 17 | 0.4268 | 0.3686 | 0.8108 | 1.0919 | 0.5193 | 0.8661 |
| 18 | 0.4204 | 0.3662 | 0.8111 | 1.0730 | 0.5135 | 0.8707 |
| 19 | 0.4105 | 0.3644 | 0.8125 | 1.0913 | 0.5118 | 0.8674 |
| 20 | 0.4054 | 0.3688 | 0.8135 | 1.0850 | 0.5038 | 0.8675 |
| 21 | 0.4020 | 0.3665 | 0.8128 | 1.1007 | 0.5085 | 0.8689 |
| 22 | 0.3972 | 0.3670 | 0.8126 | 1.0965 | 0.5059 | 0.8679 |
| 23 | 0.3943 | 0.3656 | 0.8119 | 1.1082 | 0.5070 | 0.8674 |
| 24 | 0.3932 | 0.3671 | 0.8125 | 1.1053 | 0.5063 | 0.8684 |
| 25 | 0.3925 | 0.3676 | 0.8129 | 1.1037 | 0.5056 | 0.8683 |

⭐ = new best at that epoch.

## IoU per class (best epoch 12)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9567 | curb | 0.2493 |
| paved trail | 0.8540 | animal | 0.2302 |
| sidewalk | 0.7868 | pole | 0.2189 |
| pedestrian | 0.7758 | stairs | 0.1363 |
| wall/fence | 0.6680 | traffic light | 0.0482 |
| crosswalk | 0.6465 | opening-door | 0.0238 |
| tree | 0.6144 | other walkable surface | 0.0209 |
| building | 0.6103 | inaccessible surface | 0.0032 |
| terrain | 0.6078 | rider | 0.0000 |
| vegetation | 0.6020 | guard rail/road barrier | 0.0000 |
| road | 0.5580 | hand rail | 0.0000 |
| vehicle | 0.5488 | opening-gate | 0.0000 |
| water body | 0.5397 | bike rack | 0.0000 |
| traffic sign | 0.3231 | railway track | absent |
| obstacle | 0.3046 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8123 |
| sanpo3_not_safe | 0.8940 |
| sanpo3_obstacle | 0.8861 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 7.0211 / 0.3906 |
| Gradient norm, median | 51.02 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 7 |
| Update/weight ratio, log10 median, first / last | -3.42 / -4.75 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (16 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
