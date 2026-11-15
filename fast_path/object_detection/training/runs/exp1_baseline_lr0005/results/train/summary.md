# exp1_baseline_lr0005

**finished** · 25/25 epochs · launches run001 · updated 2026-11-14 21:59:21 UTC

> [!WARNING]
> **Corrected validation scores (added 2026-11-15).** This run validated at Ultralytics' default size, 352x198
> (frames sized by the long side), while it trained at 626x352 (short side): objects looked about 1.8x smaller in
> validation than in training. The logged val scores below are therefore too low, and best.pt was chosen with them.
> Training itself was not affected. Both checkpoints, re-validated on the same validation split at the training
> scale (626x352), as later runs do (data.validate_at_training_scale):
>
> | Checkpoint | val mIoU | Pixel acc | 3-class mIoU (sanpo3) |
> |---|---|---|---|
> | best.pt | **0.3052** | 0.7208 | 0.6374 |
> | last.pt | **0.3055** | 0.7208 | 0.6372 |

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.2896** (epoch 23) |
| best.pt re-validated | 0.2896 |
| Pixel accuracy (best epoch) | 0.6952 |
| Last epoch mIoU | 0.2894 (epoch 25) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_352 |
| epochs | 25 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.0005 |
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
| Peak GPU memory | 3.67 GB |
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 1h08m40s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.18 at $1.04/h |
| Speed (median) | 106 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss |
|---|---|---|---|---|---|
| 1 ⭐ | 3.4097 | 0.2000 | 0.6290 | 1.3806 | 0.8908 |
| 2 ⭐ | 1.8970 | 0.2187 | 0.6461 | 1.3181 | 0.8644 |
| 3 ⭐ | 1.4887 | 0.2486 | 0.6749 | 1.3464 | 0.8507 |
| 4 ⭐ | 1.2906 | 0.2517 | 0.6736 | 1.3393 | 0.8440 |
| 5 ⭐ | 1.1686 | 0.2661 | 0.6780 | 1.3799 | 0.8393 |
| 6 ⭐ | 1.0877 | 0.2798 | 0.6882 | 1.3722 | 0.8317 |
| 7 ⭐ | 1.0202 | 0.2805 | 0.6835 | 1.4965 | 0.8095 |
| 8 ⭐ | 0.9517 | 0.2812 | 0.6884 | 1.4494 | 0.7883 |
| 9 ⭐ | 0.9027 | 0.2880 | 0.6893 | 1.4942 | 0.7610 |
| 10 | 0.8589 | 0.2860 | 0.6843 | 1.4707 | 0.7508 |
| 11 | 0.7815 | 0.2841 | 0.6943 | 1.5196 | 0.7326 |
| 12 ⭐ | 0.7578 | 0.2880 | 0.6935 | 1.5495 | 0.7249 |
| 13 | 0.7373 | 0.2872 | 0.6925 | 1.5640 | 0.7229 |
| 14 ⭐ | 0.7232 | 0.2889 | 0.6949 | 1.5668 | 0.7198 |
| 15 | 0.7067 | 0.2834 | 0.6923 | 1.5667 | 0.7186 |
| 16 ⭐ | 0.6956 | 0.2894 | 0.6949 | 1.5539 | 0.7165 |
| 17 | 0.6867 | 0.2873 | 0.6938 | 1.5764 | 0.7160 |
| 18 ⭐ | 0.6800 | 0.2896 | 0.6953 | 1.5693 | 0.7140 |
| 19 | 0.6716 | 0.2878 | 0.6941 | 1.5850 | 0.7128 |
| 20 | 0.6654 | 0.2883 | 0.6936 | 1.5862 | 0.7122 |
| 21 | 0.6606 | 0.2874 | 0.6942 | 1.5798 | 0.7112 |
| 22 | 0.6590 | 0.2885 | 0.6949 | 1.5786 | 0.7104 |
| 23 ⭐ | 0.6563 | 0.2896 | 0.6952 | 1.5783 | 0.7098 |
| 24 | 0.6564 | 0.2894 | 0.6950 | 1.5810 | 0.7095 |
| 25 | 0.6545 | 0.2894 | 0.6952 | 1.5822 | 0.7093 |

⭐ = new best at that epoch.

## IoU per class (best epoch 23)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8124 | tree | 0.2144 |
| sidewalk | 0.7615 | opening-gate | 0.1378 |
| building | 0.7188 | stairs | 0.0908 |
| pedestrian | 0.6837 | traffic sign | 0.0701 |
| vehicle | 0.6477 | bike rack | 0.0474 |
| paved trail | 0.5901 | other walkable surface | 0.0364 |
| sky | 0.5423 | opening-door | 0.0351 |
| hand rail | 0.4515 | traffic light | 0.0327 |
| vegetation | 0.4504 | guard rail/road barrier | 0.0168 |
| wall/fence | 0.4367 | animal | 0.0057 |
| obstacle | 0.2913 | rider | 0.0000 |
| terrain | 0.2862 | water body | 0.0000 |
| pole | 0.2685 | inaccessible surface | 0.0000 |
| curb | 0.2636 | railway track | absent |
| road | 0.2181 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.9350 / 0.6921 |
| Gradient norm, median | 98.19 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 11 |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (27 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
