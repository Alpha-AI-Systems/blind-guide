# final_yolo26s_640x352/stage2_mix_chest

**finished** · 10/10 epochs · launches run001, run002 · updated 2026-11-19 19:51:11 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3636** (epoch 4.51, step 9504) |
| best.pt re-validated | 0.3637 |
| Pixel accuracy (best epoch) | 0.7516 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3581 |
| sanpo3 mIoU (best epoch) | 0.6848 |
| Last validation mIoU | 0.3591 (epoch 9.02) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26s-sem-ade20k.pt |
| data | semantic_v2_short352 |
| train frames | 10 fps + train_chest_5fps (16+16 per batch) |
| classes | data.yaml's classes |
| validation | every 1056 steps |
| epochs | 10 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 1.0019 |
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
| Parameters | 6,505,722 (6.51 M) |
| Forward GFLOPs per image | 9.4756 at 640x352 |
| best.pt size | 13.2 MB |
| Peak GPU memory | 8.188 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 96 GB (modal-container) |
| Training time | 57m53s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 179 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 1056 | 1.4359 | 0.3385 | 0.7297 | 1.2751 | 0.7534 | 0.3330 | 0.6589 |
| 1 ⭐ | 2112 | 0.9094 | 0.3488 | 0.7219 | 1.4511 | 0.7020 | 0.3445 | 0.6495 |
| 1.50 | 3168 | 0.7608 | 0.3379 | 0.7357 | 1.4407 | 0.6052 | 0.3335 | 0.6458 |
| 2 ⭐ | 4224 | 0.6501 | 0.3527 | 0.7361 | 1.5395 | 0.5921 | 0.3474 | 0.6661 |
| 2.50 ⭐ | 5280 | 0.5772 | 0.3580 | 0.7364 | 1.7417 | 0.5693 | 0.3498 | 0.6587 |
| 3.01 | 6336 | 0.5377 | 0.3519 | 0.7348 | 1.8167 | 0.5377 | 0.3469 | 0.6546 |
| 3.51 | 7392 | 0.4966 | 0.3519 | 0.7476 | 1.8555 | 0.5142 | 0.3451 | 0.6639 |
| 4.01 ⭐ | 8448 | 0.4762 | 0.3620 | 0.7501 | 1.9096 | 0.5125 | 0.3576 | 0.6643 |
| 4.51 ⭐ | 9504 | 0.4477 | 0.3636 | 0.7516 | 1.8963 | 0.5068 | 0.3581 | 0.6848 |
| 5.01 | 10560 | 0.4297 | 0.3597 | 0.7460 | 1.9689 | 0.5014 | 0.3536 | 0.6737 |
| 5.51 | 11616 | 0.4122 | 0.3607 | 0.7501 | 1.9663 | 0.4966 | 0.3542 | 0.6732 |
| 6.01 | 12672 | 0.3979 | 0.3619 | 0.7488 | 1.9910 | 0.4963 | 0.3563 | 0.6686 |
| 6.51 | 13728 | 0.3840 | 0.3592 | 0.7486 | 2.0203 | 0.4958 | 0.3537 | 0.6714 |
| 7.01 | 14784 | 0.3781 | 0.3610 | 0.7503 | 2.0231 | 0.4940 | 0.3556 | 0.6754 |
| 7.51 | 15840 | 0.3684 | 0.3607 | 0.7504 | 2.0376 | 0.4933 | 0.3552 | 0.6773 |
| 8.02 | 16896 | 0.3639 | 0.3604 | 0.7505 | 2.0460 | 0.4930 | 0.3546 | 0.6782 |
| 8.52 | 17952 | 0.3593 | 0.3598 | 0.7503 | 2.0548 | 0.4930 | 0.3540 | 0.6781 |
| 9.02 | 19008 | 0.3583 | 0.3591 | 0.7500 | 2.0581 | 0.4930 | 0.3533 | 0.6775 |

⭐ = new best at that epoch.

## IoU per class (best epoch 4.51, step 9504)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8698 | obstacle | 0.3329 |
| sidewalk | 0.8037 | traffic light | 0.2370 |
| pedestrian | 0.7989 | bike rack | 0.1688 |
| vehicle | 0.7741 | animal | 0.1436 |
| building | 0.7707 | traffic sign | 0.1352 |
| paved trail | 0.6638 | stairs | 0.1191 |
| sky | 0.6218 | other walkable surface | 0.0682 |
| hand rail | 0.6000 | opening-gate | 0.0580 |
| wall/fence | 0.5486 | guard rail/road barrier | 0.0532 |
| vegetation | 0.4989 | opening-door | 0.0444 |
| pole | 0.4336 | rider | 0.0023 |
| curb | 0.4066 | water body | 0.0000 |
| road | 0.3620 | inaccessible surface | 0.0000 |
| tree | 0.3337 | railway track | absent |
| terrain | 0.3332 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7852 |
| sanpo3_not_safe | 0.3929 |
| sanpo3_obstacle | 0.8765 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 3.4805 / 0.3623 |
| Gradient norm, median | 61.15 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 13 |
| Update/weight ratio, log10 median, first / last | -3.98 / -4.66 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (40 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
