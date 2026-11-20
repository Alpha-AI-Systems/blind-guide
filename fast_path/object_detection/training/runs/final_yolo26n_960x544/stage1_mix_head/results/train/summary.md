# final_yolo26n_960x544/stage1_mix_head

**finished** · 13/13 epochs · launches run001 · updated 2026-11-19 17:18:34 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3983** (epoch 5.51, step 6952) |
| best.pt re-validated | 0.3983 |
| Pixel accuracy (best epoch) | 0.8279 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3948 |
| sanpo3 mIoU (best epoch) | 0.8784 |
| Last validation mIoU | 0.3906 (epoch 12.53) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
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
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 5.8335 at 992x544 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 15.954 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 16 CPU cores, 64 GB (modal-container) |
| Training time | 1h36m31s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 91 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 632 | 2.1897 | 0.3348 | 0.7728 | 1.0372 | 0.8203 | 0.3301 | 0.8227 |
| 1 ⭐ | 1264 | 1.0797 | 0.3654 | 0.7828 | 1.1327 | 0.7974 | 0.3614 | 0.8458 |
| 1.50 | 1896 | 0.8420 | 0.3646 | 0.7872 | 1.3526 | 0.7175 | 0.3628 | 0.8441 |
| 2 | 2528 | 0.7134 | 0.3631 | 0.8187 | 1.3039 | 0.6560 | 0.3603 | 0.8700 |
| 2.51 ⭐ | 3160 | 0.6310 | 0.3755 | 0.8145 | 1.4322 | 0.5720 | 0.3711 | 0.8621 |
| 3.01 | 3792 | 0.5884 | 0.3707 | 0.8119 | 1.4785 | 0.5777 | 0.3671 | 0.8717 |
| 3.51 ⭐ | 4424 | 0.5549 | 0.3836 | 0.8169 | 1.4881 | 0.5811 | 0.3817 | 0.8738 |
| 4.01 | 5056 | 0.5311 | 0.3755 | 0.8100 | 1.6471 | 0.5615 | 0.3704 | 0.8604 |
| 4.51 | 5688 | 0.5097 | 0.3763 | 0.8307 | 1.5111 | 0.5500 | 0.3733 | 0.8774 |
| 5.01 ⭐ | 6320 | 0.4887 | 0.3951 | 0.8254 | 1.5032 | 0.5527 | 0.3915 | 0.8826 |
| 5.51 ⭐ | 6952 | 0.4718 | 0.3983 | 0.8279 | 1.6252 | 0.5406 | 0.3948 | 0.8784 |
| 6.01 | 7584 | 0.4603 | 0.3970 | 0.8282 | 1.6060 | 0.5379 | 0.3924 | 0.8759 |
| 6.52 | 8216 | 0.4449 | 0.3918 | 0.8241 | 1.6731 | 0.5409 | 0.3866 | 0.8771 |
| 7.02 | 8848 | 0.4395 | 0.3938 | 0.8286 | 1.6909 | 0.5316 | 0.3898 | 0.8763 |
| 7.52 | 9480 | 0.4241 | 0.3952 | 0.8325 | 1.7655 | 0.5247 | 0.3928 | 0.8793 |
| 8.02 | 10112 | 0.4170 | 0.3893 | 0.8310 | 1.8577 | 0.5103 | 0.3875 | 0.8799 |
| 8.52 | 10744 | 0.4090 | 0.3877 | 0.8301 | 1.9940 | 0.5012 | 0.3856 | 0.8785 |
| 9.02 | 11376 | 0.4015 | 0.3921 | 0.8317 | 2.0335 | 0.4949 | 0.3896 | 0.8794 |
| 9.52 | 12008 | 0.3930 | 0.3938 | 0.8304 | 2.0510 | 0.4941 | 0.3911 | 0.8788 |
| 10.02 | 12640 | 0.3904 | 0.3934 | 0.8311 | 2.0595 | 0.4933 | 0.3907 | 0.8791 |
| 10.52 | 13272 | 0.3845 | 0.3906 | 0.8310 | 2.0683 | 0.4930 | 0.3876 | 0.8787 |
| 11.03 | 13904 | 0.3799 | 0.3911 | 0.8310 | 2.0834 | 0.4925 | 0.3876 | 0.8794 |
| 11.53 | 14536 | 0.3784 | 0.3908 | 0.8306 | 2.0916 | 0.4919 | 0.3873 | 0.8792 |
| 12.03 | 15168 | 0.3778 | 0.3906 | 0.8304 | 2.0896 | 0.4915 | 0.3870 | 0.8789 |
| 12.53 | 15800 | 0.3748 | 0.3906 | 0.8305 | 2.0916 | 0.4911 | 0.3870 | 0.8790 |

⭐ = new best at that epoch.

## IoU per class (best epoch 5.51, step 6952)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9554 | curb | 0.2994 |
| paved trail | 0.8654 | pole | 0.2729 |
| pedestrian | 0.7934 | animal | 0.2722 |
| sidewalk | 0.7928 | traffic sign | 0.2282 |
| crosswalk | 0.7180 | traffic light | 0.2076 |
| wall/fence | 0.7134 | other walkable surface | 0.0332 |
| vegetation | 0.6762 | opening-door | 0.0229 |
| terrain | 0.6706 | bike rack | 0.0012 |
| tree | 0.6537 | rider | 0.0008 |
| building | 0.6428 | guard rail/road barrier | 0.0000 |
| water body | 0.5793 | hand rail | 0.0000 |
| road | 0.5358 | opening-gate | 0.0000 |
| vehicle | 0.5057 | inaccessible surface | 0.0000 |
| stairs | 0.3740 | railway track | absent |
| obstacle | 0.3369 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8394 |
| sanpo3_not_safe | 0.8889 |
| sanpo3_obstacle | 0.9068 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 7.4425 / 0.3607 |
| Gradient norm, median | 53.96 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 11 |
| Update/weight ratio, log10 median, first / last | -3.71 / -4.75 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (32 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
