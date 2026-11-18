# exp6_synthetic_then_real_lr_decay/stage2_real_head

**finished** · 25/25 epochs · launches run001 · updated 2026-11-18 17:45:41 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3819** (epoch 20, step 6320) |
| best.pt re-validated | 0.3820 |
| Pixel accuracy (best epoch) | 0.8289 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3737 |
| sanpo3 mIoU (best epoch) | 0.8884 |
| Last validation mIoU | 0.3787 (epoch 25) |

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
| lr0 | 0.0014142136 |
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
| Peak GPU memory | 6.713 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 48 GB (modal-container) |
| Training time | 23m54s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 191 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 316 | 2.7789 | 0.3028 | 0.7634 | 1.0136 | 0.8236 | 0.2978 | 0.8395 |
| 2 ⭐ | 632 | 1.2732 | 0.3332 | 0.7917 | 0.8360 | 0.8039 | 0.3272 | 0.8638 |
| 3 | 948 | 0.9890 | 0.3329 | 0.7816 | 0.9602 | 0.7642 | 0.3293 | 0.8506 |
| 4 ⭐ | 1264 | 0.8268 | 0.3562 | 0.8029 | 0.8654 | 0.7282 | 0.3523 | 0.8813 |
| 5 ⭐ | 1580 | 0.7169 | 0.3741 | 0.8149 | 0.9212 | 0.6879 | 0.3683 | 0.8778 |
| 6 | 1896 | 0.6584 | 0.3646 | 0.8173 | 0.9288 | 0.6750 | 0.3599 | 0.8773 |
| 7 | 2212 | 0.6252 | 0.3605 | 0.8122 | 0.9771 | 0.6766 | 0.3553 | 0.8747 |
| 8 ⭐ | 2528 | 0.5910 | 0.3782 | 0.8234 | 0.9313 | 0.6520 | 0.3738 | 0.8842 |
| 9 | 2844 | 0.5681 | 0.3760 | 0.8252 | 0.9477 | 0.5957 | 0.3691 | 0.8887 |
| 10 | 3160 | 0.5429 | 0.3761 | 0.8188 | 1.0007 | 0.5870 | 0.3687 | 0.8791 |
| 11 | 3476 | 0.5208 | 0.3771 | 0.8221 | 1.0041 | 0.5540 | 0.3714 | 0.8828 |
| 12 | 3792 | 0.5068 | 0.3746 | 0.8207 | 1.0134 | 0.5468 | 0.3682 | 0.8844 |
| 13 | 4108 | 0.4953 | 0.3771 | 0.8288 | 0.9658 | 0.5428 | 0.3702 | 0.8939 |
| 14 | 4424 | 0.4806 | 0.3655 | 0.8181 | 0.9905 | 0.5415 | 0.3594 | 0.8813 |
| 15 | 4740 | 0.4732 | 0.3723 | 0.8225 | 0.9919 | 0.5367 | 0.3669 | 0.8840 |
| 16 | 5056 | 0.4604 | 0.3739 | 0.8235 | 1.0203 | 0.5318 | 0.3681 | 0.8853 |
| 17 ⭐ | 5372 | 0.4561 | 0.3811 | 0.8278 | 0.9977 | 0.5235 | 0.3712 | 0.8854 |
| 18 | 5688 | 0.4484 | 0.3750 | 0.8279 | 1.0078 | 0.5303 | 0.3681 | 0.8869 |
| 19 | 6004 | 0.4430 | 0.3727 | 0.8234 | 1.0181 | 0.5285 | 0.3650 | 0.8848 |
| 20 ⭐ | 6320 | 0.4408 | 0.3819 | 0.8289 | 1.0025 | 0.5285 | 0.3737 | 0.8884 |
| 21 | 6636 | 0.4326 | 0.3781 | 0.8263 | 1.0057 | 0.5284 | 0.3695 | 0.8843 |
| 22 | 6952 | 0.4325 | 0.3772 | 0.8264 | 1.0132 | 0.5262 | 0.3693 | 0.8851 |
| 23 | 7268 | 0.4297 | 0.3760 | 0.8260 | 1.0217 | 0.5257 | 0.3686 | 0.8849 |
| 24 | 7584 | 0.4289 | 0.3768 | 0.8254 | 1.0243 | 0.5259 | 0.3690 | 0.8849 |
| 25 | 7900 | 0.4283 | 0.3787 | 0.8268 | 1.0130 | 0.5250 | 0.3705 | 0.8861 |

⭐ = new best at that epoch.

## IoU per class (best epoch 20, step 6320)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9550 | obstacle | 0.3061 |
| paved trail | 0.8851 | animal | 0.2278 |
| sidewalk | 0.8280 | pole | 0.2204 |
| pedestrian | 0.7870 | other walkable surface | 0.0615 |
| crosswalk | 0.7143 | traffic light | 0.0250 |
| terrain | 0.6988 | opening-door | 0.0232 |
| vegetation | 0.6777 | stairs | 0.0168 |
| wall/fence | 0.6771 | bike rack | 0.0002 |
| road | 0.6582 | guard rail/road barrier | 0.0000 |
| tree | 0.6329 | hand rail | 0.0000 |
| building | 0.6272 | opening-gate | 0.0000 |
| vehicle | 0.5363 | rider | 0.0000 |
| traffic sign | 0.4162 | inaccessible surface | 0.0000 |
| water body | 0.3965 | railway track | absent |
| curb | 0.3227 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8523 |
| sanpo3_not_safe | 0.9021 |
| sanpo3_obstacle | 0.9108 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 9.8580 / 0.4086 |
| Gradient norm, median | 59.47 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.57 / -4.89 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (16 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
