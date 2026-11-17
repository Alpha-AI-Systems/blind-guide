# exp6_synthetic_then_real/stage2_real_head

**finished** · 25/25 epochs · launches run001 · updated 2026-11-17 18:57:41 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3773** (epoch 11, step 3476) |
| best.pt re-validated | 0.3772 |
| Pixel accuracy (best epoch) | 0.8108 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3682 |
| sanpo3 mIoU (best epoch) | 0.8780 |
| Last validation mIoU | 0.3643 (epoch 25) |

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
| Training time | 22m32s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 202 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 316 | 2.6741 | 0.3086 | 0.7654 | 1.0145 | 0.8204 | 0.3027 | 0.8362 |
| 2 ⭐ | 632 | 1.2082 | 0.3274 | 0.7797 | 0.8721 | 0.8008 | 0.3211 | 0.8573 |
| 3 ⭐ | 948 | 0.9353 | 0.3353 | 0.7973 | 0.9306 | 0.7401 | 0.3306 | 0.8613 |
| 4 ⭐ | 1264 | 0.7685 | 0.3483 | 0.8025 | 0.9624 | 0.6997 | 0.3467 | 0.8818 |
| 5 ⭐ | 1580 | 0.6790 | 0.3586 | 0.8044 | 0.9883 | 0.6890 | 0.3501 | 0.8675 |
| 6 | 1896 | 0.6315 | 0.3457 | 0.8030 | 1.0432 | 0.6677 | 0.3432 | 0.8709 |
| 7 ⭐ | 2212 | 0.6012 | 0.3618 | 0.8003 | 1.0692 | 0.6106 | 0.3586 | 0.8717 |
| 8 ⭐ | 2528 | 0.5647 | 0.3709 | 0.8103 | 1.0546 | 0.5803 | 0.3643 | 0.8800 |
| 9 | 2844 | 0.5434 | 0.3592 | 0.8112 | 1.0265 | 0.5585 | 0.3547 | 0.8786 |
| 10 ⭐ | 3160 | 0.5249 | 0.3754 | 0.8131 | 1.0453 | 0.5482 | 0.3677 | 0.8771 |
| 11 ⭐ | 3476 | 0.5046 | 0.3773 | 0.8108 | 1.0336 | 0.5419 | 0.3682 | 0.8780 |
| 12 | 3792 | 0.4886 | 0.3759 | 0.8153 | 1.0322 | 0.5357 | 0.3686 | 0.8803 |
| 13 | 4108 | 0.4769 | 0.3597 | 0.8192 | 1.0300 | 0.5280 | 0.3539 | 0.8856 |
| 14 | 4424 | 0.4570 | 0.3552 | 0.8063 | 1.0736 | 0.5478 | 0.3492 | 0.8720 |
| 15 | 4740 | 0.4487 | 0.3595 | 0.8144 | 1.0327 | 0.5379 | 0.3528 | 0.8768 |
| 16 | 5056 | 0.4303 | 0.3720 | 0.8202 | 1.0429 | 0.5283 | 0.3647 | 0.8786 |
| 17 | 5372 | 0.4213 | 0.3662 | 0.8190 | 1.0471 | 0.5165 | 0.3595 | 0.8798 |
| 18 | 5688 | 0.4128 | 0.3589 | 0.8133 | 1.0867 | 0.5192 | 0.3534 | 0.8764 |
| 19 | 6004 | 0.4079 | 0.3608 | 0.8158 | 1.0759 | 0.5210 | 0.3559 | 0.8765 |
| 20 | 6320 | 0.4044 | 0.3663 | 0.8206 | 1.0463 | 0.5236 | 0.3608 | 0.8827 |
| 21 | 6636 | 0.3957 | 0.3618 | 0.8171 | 1.0730 | 0.5186 | 0.3569 | 0.8784 |
| 22 | 6952 | 0.3938 | 0.3634 | 0.8191 | 1.0819 | 0.5144 | 0.3585 | 0.8792 |
| 23 | 7268 | 0.3919 | 0.3644 | 0.8194 | 1.0835 | 0.5151 | 0.3593 | 0.8795 |
| 24 | 7584 | 0.3895 | 0.3640 | 0.8192 | 1.0838 | 0.5163 | 0.3583 | 0.8788 |
| 25 | 7900 | 0.3902 | 0.3643 | 0.8195 | 1.0762 | 0.5150 | 0.3588 | 0.8799 |

⭐ = new best at that epoch.

## IoU per class (best epoch 11, step 3476)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9538 | obstacle | 0.3117 |
| paved trail | 0.8265 | pole | 0.2243 |
| sidewalk | 0.7738 | animal | 0.1732 |
| pedestrian | 0.7708 | traffic light | 0.1645 |
| wall/fence | 0.6525 | other walkable surface | 0.0707 |
| vegetation | 0.6520 | stairs | 0.0641 |
| crosswalk | 0.6446 | opening-door | 0.0392 |
| terrain | 0.6379 | hand rail | 0.0017 |
| tree | 0.6248 | rider | 0.0009 |
| building | 0.6070 | guard rail/road barrier | 0.0000 |
| road | 0.5748 | opening-gate | 0.0000 |
| vehicle | 0.5154 | inaccessible surface | 0.0000 |
| water body | 0.5089 | bike rack | 0.0000 |
| traffic sign | 0.4216 | railway track | absent |
| curb | 0.3489 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8352 |
| sanpo3_not_safe | 0.8940 |
| sanpo3_obstacle | 0.9046 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 9.8479 / 0.3559 |
| Gradient norm, median | 52.89 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.43 / -4.74 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (16 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
