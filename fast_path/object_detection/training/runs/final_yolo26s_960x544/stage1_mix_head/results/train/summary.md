# final_yolo26s_960x544/stage1_mix_head

**finished** · 13/13 epochs · launches run001 · updated 2026-11-19 17:29:53 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.4117** (epoch 10.02, step 12640) |
| best.pt re-validated | 0.4118 |
| Pixel accuracy (best epoch) | 0.8455 |
| val_human mIoU (best epoch, human-labeled frames) | 0.4093 |
| sanpo3 mIoU (best epoch) | 0.8920 |
| Last validation mIoU | 0.4096 (epoch 12.53) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26s-sem-ade20k.pt |
| data | semantic_v2_short544 |
| train frames | 10 fps + train_head_5fps (16+16 per batch) |
| classes | data.yaml's classes |
| validation | every 632 steps |
| epochs | 13 |
| batch | 32 |
| imgsz | 544 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 1.0024 |
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
| Forward GFLOPs per image | 22.8227 at 992x544 |
| best.pt size | 13.2 MB |
| Peak GPU memory | 19.527 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 16 CPU cores, 64 GB (modal-container) |
| Training time | 1h41m56s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 86 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 632 | 2.0460 | 0.3783 | 0.7981 | 0.9399 | 0.7943 | 0.3725 | 0.8492 |
| 1 ⭐ | 1264 | 0.8634 | 0.3924 | 0.8270 | 1.0623 | 0.7590 | 0.3858 | 0.8805 |
| 1.50 | 1896 | 0.6366 | 0.3856 | 0.8227 | 1.3893 | 0.5822 | 0.3821 | 0.8744 |
| 2 | 2528 | 0.5302 | 0.3808 | 0.8237 | 1.4586 | 0.5585 | 0.3747 | 0.8706 |
| 2.51 | 3160 | 0.4770 | 0.3806 | 0.8194 | 1.5477 | 0.5570 | 0.3750 | 0.8671 |
| 3.01 ⭐ | 3792 | 0.4509 | 0.3994 | 0.8314 | 1.6197 | 0.5520 | 0.3954 | 0.8834 |
| 3.51 ⭐ | 4424 | 0.4233 | 0.4068 | 0.8322 | 1.4985 | 0.5330 | 0.4020 | 0.8815 |
| 4.01 | 5056 | 0.4033 | 0.3899 | 0.8276 | 1.7553 | 0.5413 | 0.3881 | 0.8767 |
| 4.51 | 5688 | 0.3868 | 0.3914 | 0.8388 | 1.9369 | 0.4958 | 0.3872 | 0.8847 |
| 5.01 | 6320 | 0.3731 | 0.4055 | 0.8417 | 2.0767 | 0.4914 | 0.4026 | 0.8882 |
| 5.51 | 6952 | 0.3574 | 0.4026 | 0.8324 | 2.0844 | 0.4927 | 0.3991 | 0.8828 |
| 6.01 | 7584 | 0.3506 | 0.4023 | 0.8424 | 2.1095 | 0.4770 | 0.3996 | 0.8841 |
| 6.52 | 8216 | 0.3367 | 0.4010 | 0.8397 | 2.0862 | 0.4769 | 0.3987 | 0.8838 |
| 7.02 ⭐ | 8848 | 0.3319 | 0.4089 | 0.8373 | 2.0931 | 0.4765 | 0.4076 | 0.8834 |
| 7.52 | 9480 | 0.3183 | 0.3980 | 0.8395 | 2.1067 | 0.4773 | 0.3944 | 0.8852 |
| 8.02 | 10112 | 0.3149 | 0.4053 | 0.8458 | 2.1696 | 0.4726 | 0.4022 | 0.8941 |
| 8.52 ⭐ | 10744 | 0.3085 | 0.4114 | 0.8452 | 2.1949 | 0.4673 | 0.4086 | 0.8942 |
| 9.02 | 11376 | 0.3036 | 0.4069 | 0.8460 | 2.1572 | 0.4701 | 0.4044 | 0.8906 |
| 9.52 | 12008 | 0.2964 | 0.4073 | 0.8446 | 2.1544 | 0.4641 | 0.4045 | 0.8892 |
| 10.02 ⭐ | 12640 | 0.2946 | 0.4117 | 0.8455 | 2.1665 | 0.4654 | 0.4093 | 0.8920 |
| 10.52 | 13272 | 0.2904 | 0.4104 | 0.8438 | 2.1860 | 0.4657 | 0.4080 | 0.8906 |
| 11.03 | 13904 | 0.2873 | 0.4109 | 0.8427 | 2.1975 | 0.4661 | 0.4086 | 0.8893 |
| 11.53 | 14536 | 0.2841 | 0.4098 | 0.8420 | 2.2148 | 0.4669 | 0.4075 | 0.8888 |
| 12.03 | 15168 | 0.2850 | 0.4099 | 0.8426 | 2.2115 | 0.4667 | 0.4076 | 0.8894 |
| 12.53 | 15800 | 0.2816 | 0.4096 | 0.8428 | 2.2081 | 0.4663 | 0.4073 | 0.8895 |

⭐ = new best at that epoch.

## IoU per class (best epoch 10.02, step 12640)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9611 | curb | 0.3568 |
| paved trail | 0.9112 | animal | 0.3450 |
| sidewalk | 0.8477 | pole | 0.2959 |
| pedestrian | 0.8303 | other walkable surface | 0.1233 |
| wall/fence | 0.7530 | stairs | 0.0780 |
| crosswalk | 0.7358 | traffic sign | 0.0437 |
| vegetation | 0.7001 | opening-door | 0.0155 |
| terrain | 0.6940 | rider | 0.0035 |
| tree | 0.6587 | bike rack | 0.0022 |
| building | 0.6515 | guard rail/road barrier | 0.0000 |
| road | 0.6359 | hand rail | 0.0000 |
| water body | 0.5880 | opening-gate | 0.0000 |
| vehicle | 0.5256 | inaccessible surface | 0.0000 |
| obstacle | 0.3962 | railway track | absent |
| traffic light | 0.3754 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8553 |
| sanpo3_not_safe | 0.9056 |
| sanpo3_obstacle | 0.9150 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 6.9749 / 0.2819 |
| Gradient norm, median | 41.91 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 11 |
| Update/weight ratio, log10 median, first / last | -3.75 / -4.76 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (32 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
