# exp1_baseline_lr001

**finished** · 25/25 epochs · launches run001 · updated 2026-11-14 22:00:26 UTC

> [!WARNING]
> **Corrected validation scores (added 2026-11-15).** This run validated at Ultralytics' default size, 352x198
> (frames sized by the long side), while it trained at 626x352 (short side): objects looked about 1.8x smaller in
> validation than in training. The logged val scores below are therefore too low, and best.pt was chosen with them.
> Training itself was not affected. Both checkpoints, re-validated on the same validation split at the training
> scale (626x352), as later runs do (data.validate_at_training_scale):
>
> | Checkpoint | val mIoU | Pixel acc | 3-class mIoU (sanpo3) |
> |---|---|---|---|
> | best.pt | **0.3065** | 0.7121 | 0.6421 |
> | last.pt | **0.3033** | 0.7230 | 0.6455 |

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.2913** (epoch 10) |
| best.pt re-validated | 0.2912 |
| Pixel accuracy (best epoch) | 0.6882 |
| Last epoch mIoU | 0.2887 (epoch 25) |

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
| lr0 | 0.001 |
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
| Training time | 1h08m35s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.18 at $1.04/h |
| Speed (median) | 106 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss |
|---|---|---|---|---|---|
| 1 ⭐ | 3.2132 | 0.2100 | 0.6426 | 1.4161 | 0.8772 |
| 2 ⭐ | 1.6813 | 0.2405 | 0.6472 | 1.4270 | 0.8561 |
| 3 ⭐ | 1.3081 | 0.2594 | 0.6650 | 1.3916 | 0.8428 |
| 4 ⭐ | 1.1185 | 0.2787 | 0.6811 | 1.3476 | 0.8293 |
| 5 ⭐ | 0.9928 | 0.2845 | 0.6885 | 1.3734 | 0.7727 |
| 6 ⭐ | 0.8941 | 0.2852 | 0.6819 | 1.5429 | 0.7402 |
| 7 ⭐ | 0.8365 | 0.2888 | 0.6910 | 1.5231 | 0.7219 |
| 8 ⭐ | 0.8057 | 0.2892 | 0.6946 | 1.4993 | 0.7193 |
| 9 | 0.7626 | 0.2873 | 0.6924 | 1.4150 | 0.7147 |
| 10 ⭐ | 0.7399 | 0.2913 | 0.6882 | 1.5841 | 0.7099 |
| 11 | 0.6666 | 0.2873 | 0.6929 | 1.6373 | 0.6791 |
| 12 | 0.6421 | 0.2865 | 0.6978 | 1.5351 | 0.6593 |
| 13 | 0.6133 | 0.2896 | 0.6924 | 1.5854 | 0.6222 |
| 14 | 0.5915 | 0.2891 | 0.6952 | 1.6349 | 0.6181 |
| 15 | 0.5740 | 0.2886 | 0.7004 | 1.6019 | 0.6103 |
| 16 | 0.5642 | 0.2880 | 0.6988 | 1.6212 | 0.5931 |
| 17 | 0.5516 | 0.2866 | 0.6973 | 1.6234 | 0.5907 |
| 18 | 0.5451 | 0.2893 | 0.6997 | 1.5782 | 0.5871 |
| 19 | 0.5359 | 0.2906 | 0.7009 | 1.5895 | 0.5866 |
| 20 | 0.5290 | 0.2885 | 0.7014 | 1.5978 | 0.5862 |
| 21 | 0.5249 | 0.2878 | 0.7005 | 1.5923 | 0.5858 |
| 22 | 0.5208 | 0.2888 | 0.7011 | 1.5911 | 0.5846 |
| 23 | 0.5194 | 0.2887 | 0.7012 | 1.5953 | 0.5837 |
| 24 | 0.5200 | 0.2886 | 0.7008 | 1.5970 | 0.5843 |
| 25 | 0.5166 | 0.2887 | 0.7006 | 1.5986 | 0.5849 |

⭐ = new best at that epoch.

## IoU per class (best epoch 10)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.7886 | tree | 0.1937 |
| sidewalk | 0.7184 | opening-gate | 0.1476 |
| building | 0.7178 | traffic sign | 0.0860 |
| pedestrian | 0.6882 | stairs | 0.0842 |
| vehicle | 0.6441 | bike rack | 0.0608 |
| paved trail | 0.5681 | other walkable surface | 0.0572 |
| hand rail | 0.5389 | opening-door | 0.0216 |
| sky | 0.5238 | animal | 0.0176 |
| wall/fence | 0.4744 | traffic light | 0.0167 |
| vegetation | 0.4731 | guard rail/road barrier | 0.0102 |
| obstacle | 0.2862 | rider | 0.0000 |
| pole | 0.2832 | water body | 0.0000 |
| curb | 0.2736 | inaccessible surface | 0.0000 |
| terrain | 0.2697 | railway track | absent |
| road | 0.2129 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.9282 / 0.6408 |
| Gradient norm, median | 85.92 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 10 |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (27 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
