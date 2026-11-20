# final_yolo26s_960x544/stage2_mix_chest

**finished** · 10/10 epochs · launches run001, run002 · updated 2026-11-20 12:47:33 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3770** (epoch 5.51, step 11616) |
| best.pt re-validated | 0.3770 |
| Pixel accuracy (best epoch) | 0.7633 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3706 |
| sanpo3 mIoU (best epoch) | 0.6561 |
| Last validation mIoU | 0.3714 (epoch 9.02) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26s-sem-ade20k.pt |
| data | semantic_v2_short544 |
| train frames | 10 fps + train_chest_5fps (16+16 per batch) |
| classes | data.yaml's classes |
| validation | every 1056 steps |
| epochs | 10 |
| batch | 32 |
| imgsz | 544 |
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
| Forward GFLOPs per image | 22.8227 at 992x544 |
| best.pt size | 13.2 MB |
| Peak GPU memory | 19.525 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 16 CPU cores, 64 GB (modal-container) |
| Training time | 2h04m34s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 86 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 1056 | 1.3151 | 0.3680 | 0.7513 | 1.2064 | 0.6258 | 0.3615 | 0.6890 |
| 1 | 2112 | 0.6782 | 0.3577 | 0.7248 | 1.3909 | 0.5982 | 0.3523 | 0.6714 |
| 1.50 | 3168 | 0.5862 | 0.3605 | 0.7280 | 1.5592 | 0.5680 | 0.3537 | 0.6471 |
| 2 | 4224 | 0.5155 | 0.3640 | 0.7539 | 1.7208 | 0.5320 | 0.3557 | 0.6536 |
| 2.50 ⭐ | 5280 | 0.4617 | 0.3725 | 0.7521 | 1.7748 | 0.5155 | 0.3657 | 0.6548 |
| 3.01 | 6336 | 0.4281 | 0.3667 | 0.7620 | 1.7906 | 0.5107 | 0.3611 | 0.6628 |
| 3.51 ⭐ | 7392 | 0.3933 | 0.3750 | 0.7626 | 1.8715 | 0.5003 | 0.3678 | 0.6528 |
| 4.01 ⭐ | 8448 | 0.3775 | 0.3755 | 0.7637 | 1.9415 | 0.4987 | 0.3674 | 0.6630 |
| 4.51 | 9504 | 0.3577 | 0.3714 | 0.7619 | 1.9572 | 0.4956 | 0.3656 | 0.6584 |
| 5.01 | 10560 | 0.3407 | 0.3712 | 0.7614 | 2.0218 | 0.4926 | 0.3644 | 0.6574 |
| 5.51 ⭐ | 11616 | 0.3271 | 0.3770 | 0.7633 | 2.0075 | 0.4839 | 0.3706 | 0.6561 |
| 6.01 | 12672 | 0.3162 | 0.3753 | 0.7640 | 2.0319 | 0.4796 | 0.3691 | 0.6560 |
| 6.51 | 13728 | 0.3047 | 0.3711 | 0.7626 | 2.0530 | 0.4811 | 0.3645 | 0.6574 |
| 7.01 | 14784 | 0.2984 | 0.3731 | 0.7639 | 2.0655 | 0.4807 | 0.3663 | 0.6584 |
| 7.51 | 15840 | 0.2922 | 0.3721 | 0.7643 | 2.0709 | 0.4790 | 0.3657 | 0.6591 |
| 8.02 | 16896 | 0.2879 | 0.3718 | 0.7644 | 2.0750 | 0.4783 | 0.3656 | 0.6596 |
| 8.52 | 17952 | 0.2832 | 0.3716 | 0.7641 | 2.0857 | 0.4783 | 0.3653 | 0.6591 |
| 9.02 | 19008 | 0.2831 | 0.3714 | 0.7641 | 2.0893 | 0.4782 | 0.3651 | 0.6590 |

⭐ = new best at that epoch.

## IoU per class (best epoch 5.51, step 11616)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8703 | animal | 0.3012 |
| sidewalk | 0.8392 | road | 0.2615 |
| pedestrian | 0.8227 | stairs | 0.1866 |
| vehicle | 0.7989 | traffic light | 0.1770 |
| building | 0.7876 | traffic sign | 0.1522 |
| hand rail | 0.6495 | bike rack | 0.1206 |
| sky | 0.6417 | other walkable surface | 0.1023 |
| paved trail | 0.6380 | opening-door | 0.0513 |
| wall/fence | 0.6319 | rider | 0.0123 |
| vegetation | 0.5421 | guard rail/road barrier | 0.0099 |
| pole | 0.4647 | water body | 0.0044 |
| curb | 0.4180 | opening-gate | 0.0000 |
| tree | 0.3853 | inaccessible surface | 0.0000 |
| obstacle | 0.3739 | railway track | absent |
| terrain | 0.3128 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7770 |
| sanpo3_not_safe | 0.3104 |
| sanpo3_obstacle | 0.8809 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 4.8801 / 0.2848 |
| Gradient norm, median | 55.58 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 13 |
| Update/weight ratio, log10 median, first / last | -3.99 / -4.64 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (39 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
