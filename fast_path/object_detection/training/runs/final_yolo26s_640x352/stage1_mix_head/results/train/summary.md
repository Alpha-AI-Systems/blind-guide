# final_yolo26s_640x352/stage1_mix_head

**finished** · 13/13 epochs · launches run001 · updated 2026-11-19 16:18:10 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3989** (epoch 1, step 1264) |
| best.pt re-validated | 0.3990 |
| Pixel accuracy (best epoch) | 0.8118 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3941 |
| sanpo3 mIoU (best epoch) | 0.8706 |
| Last validation mIoU | 0.3815 (epoch 12.53) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26s-sem-ade20k.pt |
| data | semantic_v2_short352 |
| train frames | 10 fps + train_head_5fps (16+16 per batch) |
| classes | data.yaml's classes |
| validation | every 632 steps |
| epochs | 13 |
| batch | 32 |
| imgsz | 352 |
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
| Forward GFLOPs per image | 9.4756 at 640x352 |
| best.pt size | 13.2 MB |
| Peak GPU memory | 8.19 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 96 GB (modal-container) |
| Training time | 43m42s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 200 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 632 | 2.2022 | 0.3752 | 0.7985 | 0.9786 | 0.8042 | 0.3674 | 0.8493 |
| 1 ⭐ | 1264 | 0.9751 | 0.3989 | 0.8118 | 0.9739 | 0.7502 | 0.3941 | 0.8706 |
| 1.50 | 1896 | 0.7416 | 0.3693 | 0.8008 | 1.4342 | 0.6048 | 0.3676 | 0.8646 |
| 2 | 2528 | 0.6271 | 0.3946 | 0.8299 | 1.4221 | 0.5661 | 0.3883 | 0.8892 |
| 2.51 | 3160 | 0.5641 | 0.3747 | 0.8112 | 1.5394 | 0.5619 | 0.3695 | 0.8722 |
| 3.01 | 3792 | 0.5351 | 0.3742 | 0.8084 | 1.5866 | 0.5800 | 0.3705 | 0.8710 |
| 3.51 | 4424 | 0.5044 | 0.3871 | 0.8263 | 1.6542 | 0.5447 | 0.3826 | 0.8745 |
| 4.01 | 5056 | 0.4815 | 0.3822 | 0.8187 | 1.9601 | 0.5126 | 0.3778 | 0.8774 |
| 4.51 | 5688 | 0.4627 | 0.3871 | 0.8278 | 1.9923 | 0.5020 | 0.3830 | 0.8786 |
| 5.01 | 6320 | 0.4469 | 0.3726 | 0.8250 | 2.0862 | 0.4998 | 0.3702 | 0.8838 |
| 5.51 | 6952 | 0.4303 | 0.3761 | 0.8193 | 2.2741 | 0.5021 | 0.3723 | 0.8756 |
| 6.01 | 7584 | 0.4220 | 0.3803 | 0.8266 | 2.2274 | 0.4925 | 0.3757 | 0.8780 |
| 6.52 | 8216 | 0.4096 | 0.3926 | 0.8310 | 2.1727 | 0.4984 | 0.3887 | 0.8805 |
| 7.02 | 8848 | 0.4030 | 0.3924 | 0.8332 | 2.1368 | 0.4939 | 0.3873 | 0.8824 |
| 7.52 | 9480 | 0.3882 | 0.3792 | 0.8312 | 2.2237 | 0.4893 | 0.3752 | 0.8831 |
| 8.02 | 10112 | 0.3850 | 0.3894 | 0.8301 | 2.2078 | 0.4901 | 0.3843 | 0.8852 |
| 8.52 | 10744 | 0.3771 | 0.3874 | 0.8338 | 2.2360 | 0.4906 | 0.3822 | 0.8881 |
| 9.02 | 11376 | 0.3714 | 0.3896 | 0.8311 | 2.2330 | 0.4877 | 0.3847 | 0.8855 |
| 9.52 | 12008 | 0.3641 | 0.3859 | 0.8303 | 2.2693 | 0.4865 | 0.3814 | 0.8849 |
| 10.02 | 12640 | 0.3613 | 0.3834 | 0.8317 | 2.2771 | 0.4862 | 0.3786 | 0.8865 |
| 10.52 | 13272 | 0.3565 | 0.3841 | 0.8321 | 2.2855 | 0.4865 | 0.3795 | 0.8857 |
| 11.03 | 13904 | 0.3537 | 0.3827 | 0.8322 | 2.2948 | 0.4851 | 0.3785 | 0.8862 |
| 11.53 | 14536 | 0.3505 | 0.3820 | 0.8320 | 2.3101 | 0.4834 | 0.3778 | 0.8866 |
| 12.03 | 15168 | 0.3510 | 0.3815 | 0.8317 | 2.3215 | 0.4834 | 0.3774 | 0.8863 |
| 12.53 | 15800 | 0.3483 | 0.3815 | 0.8315 | 2.3237 | 0.4828 | 0.3774 | 0.8862 |

⭐ = new best at that epoch.

## IoU per class (best epoch 1, step 1264)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9545 | obstacle | 0.3375 |
| paved trail | 0.8643 | other walkable surface | 0.3225 |
| sidewalk | 0.8256 | curb | 0.3163 |
| pedestrian | 0.7779 | traffic sign | 0.2827 |
| wall/fence | 0.6732 | animal | 0.2420 |
| building | 0.6548 | pole | 0.2342 |
| terrain | 0.6452 | stairs | 0.0848 |
| crosswalk | 0.6198 | inaccessible surface | 0.0107 |
| vegetation | 0.5976 | bike rack | 0.0037 |
| tree | 0.5649 | hand rail | 0.0024 |
| vehicle | 0.5058 | rider | 0.0010 |
| road | 0.4831 | guard rail/road barrier | 0.0000 |
| water body | 0.4634 | opening-gate | 0.0000 |
| opening-door | 0.3622 | railway track | absent |
| traffic light | 0.3394 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8277 |
| sanpo3_not_safe | 0.8801 |
| sanpo3_obstacle | 0.9039 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 7.0655 / 0.3578 |
| Gradient norm, median | 44.92 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 12 |
| Update/weight ratio, log10 median, first / last | -3.73 / -4.77 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (32 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
