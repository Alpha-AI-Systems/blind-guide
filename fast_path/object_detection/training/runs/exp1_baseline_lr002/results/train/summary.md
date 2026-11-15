# exp1_baseline_lr002

**finished** · 25/25 epochs · launches run002 · updated 2026-11-14 22:07:22 UTC

> [!WARNING]
> **Corrected validation scores (added 2026-11-15).** This run validated at Ultralytics' default size, 352x198
> (frames sized by the long side), while it trained at 626x352 (short side): objects looked about 1.8x smaller in
> validation than in training. The logged val scores below are therefore too low, and best.pt was chosen with them.
> Training itself was not affected. Both checkpoints, re-validated on the same validation split at the training
> scale (626x352), as later runs do (data.validate_at_training_scale):
>
> | Checkpoint | val mIoU | Pixel acc | 3-class mIoU (sanpo3) |
> |---|---|---|---|
> | best.pt | **0.3139** | 0.7264 | 0.6384 |
> | last.pt | **0.3125** | 0.7272 | 0.6378 |

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.2961** (epoch 19) |
| best.pt re-validated | 0.2963 |
| Pixel accuracy (best epoch) | 0.7038 |
| Last epoch mIoU | 0.2939 (epoch 25) |

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
| Peak GPU memory | 3.69 GB |
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 1h09m17s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.20 at $1.04/h |
| Speed (median) | 105 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss |
|---|---|---|---|---|---|
| 1 ⭐ | 2.9789 | 0.2150 | 0.6467 | 1.4368 | 0.8689 |
| 2 ⭐ | 1.5063 | 0.2273 | 0.6405 | 1.5771 | 0.8574 |
| 3 ⭐ | 1.2081 | 0.2644 | 0.6607 | 1.3539 | 0.8264 |
| 4 ⭐ | 0.9939 | 0.2739 | 0.6776 | 1.4089 | 0.7395 |
| 5 ⭐ | 0.8964 | 0.2946 | 0.7026 | 1.4125 | 0.7197 |
| 6 | 0.8284 | 0.2783 | 0.6882 | 1.4593 | 0.7152 |
| 7 | 0.7796 | 0.2804 | 0.6927 | 1.5455 | 0.7075 |
| 8 | 0.7483 | 0.2902 | 0.6989 | 1.5029 | 0.6413 |
| 9 | 0.6934 | 0.2877 | 0.6924 | 1.5072 | 0.6281 |
| 10 | 0.6690 | 0.2841 | 0.6899 | 1.6376 | 0.6259 |
| 11 | 0.5884 | 0.2803 | 0.6993 | 1.6466 | 0.5969 |
| 12 | 0.5665 | 0.2897 | 0.7003 | 1.5879 | 0.5886 |
| 13 | 0.5454 | 0.2898 | 0.6975 | 1.6205 | 0.5841 |
| 14 | 0.5310 | 0.2901 | 0.6964 | 1.6465 | 0.5796 |
| 15 | 0.5134 | 0.2941 | 0.7020 | 1.6131 | 0.5808 |
| 16 | 0.5006 | 0.2918 | 0.7033 | 1.6815 | 0.5750 |
| 17 | 0.4881 | 0.2930 | 0.7015 | 1.6882 | 0.5709 |
| 18 | 0.4802 | 0.2926 | 0.7038 | 1.6627 | 0.5672 |
| 19 ⭐ | 0.4706 | 0.2961 | 0.7038 | 1.6782 | 0.5643 |
| 20 | 0.4630 | 0.2956 | 0.7047 | 1.6769 | 0.5625 |
| 21 | 0.4572 | 0.2954 | 0.7046 | 1.6755 | 0.5595 |
| 22 | 0.4522 | 0.2955 | 0.7056 | 1.6774 | 0.5562 |
| 23 | 0.4498 | 0.2951 | 0.7057 | 1.6880 | 0.5539 |
| 24 | 0.4491 | 0.2943 | 0.7053 | 1.6916 | 0.5542 |
| 25 | 0.4446 | 0.2939 | 0.7051 | 1.6904 | 0.5541 |

⭐ = new best at that epoch.

## IoU per class (best epoch 19)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8455 | road | 0.2195 |
| sidewalk | 0.7462 | traffic sign | 0.0834 |
| building | 0.7275 | stairs | 0.0786 |
| pedestrian | 0.6987 | other walkable surface | 0.0437 |
| vehicle | 0.6589 | guard rail/road barrier | 0.0293 |
| paved trail | 0.5972 | bike rack | 0.0288 |
| hand rail | 0.5517 | opening-door | 0.0281 |
| sky | 0.5372 | opening-gate | 0.0253 |
| wall/fence | 0.4972 | traffic light | 0.0185 |
| vegetation | 0.4616 | animal | 0.0079 |
| terrain | 0.3010 | rider | 0.0000 |
| pole | 0.2984 | water body | 0.0000 |
| obstacle | 0.2875 | inaccessible surface | 0.0000 |
| curb | 0.2785 | railway track | absent |
| tree | 0.2414 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.9282 / 0.5500 |
| Gradient norm, median | 69.59 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 10 |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (27 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
