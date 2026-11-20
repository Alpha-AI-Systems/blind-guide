# final_yolo26n_640x352/stage1_mix_head

**finished** · 13/13 epochs · launches run001 · updated 2026-11-19 16:13:41 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3733** (epoch 4.01, step 5056) |
| best.pt re-validated | 0.3733 |
| Pixel accuracy (best epoch) | 0.8151 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3676 |
| sanpo3 mIoU (best epoch) | 0.8626 |
| Last validation mIoU | 0.3631 (epoch 12.53) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
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
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 2.4093 at 640x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 6.669 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 96 GB (modal-container) |
| Training time | 42m58s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 204 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 632 | 2.3665 | 0.2936 | 0.7525 | 1.1365 | 0.8367 | 0.2894 | 0.8038 |
| 1 ⭐ | 1264 | 1.2069 | 0.3409 | 0.7580 | 1.2320 | 0.7825 | 0.3365 | 0.8290 |
| 1.50 ⭐ | 1896 | 0.9570 | 0.3470 | 0.7756 | 1.3478 | 0.7449 | 0.3419 | 0.8294 |
| 2 ⭐ | 2528 | 0.8270 | 0.3483 | 0.8002 | 1.4895 | 0.6304 | 0.3450 | 0.8568 |
| 2.51 | 3160 | 0.7324 | 0.3462 | 0.7989 | 1.4549 | 0.5859 | 0.3420 | 0.8498 |
| 3.01 | 3792 | 0.6931 | 0.3418 | 0.7894 | 1.5203 | 0.5911 | 0.3373 | 0.8521 |
| 3.51 ⭐ | 4424 | 0.6543 | 0.3641 | 0.8105 | 1.4750 | 0.5830 | 0.3578 | 0.8603 |
| 4.01 ⭐ | 5056 | 0.6230 | 0.3733 | 0.8151 | 1.4988 | 0.5715 | 0.3676 | 0.8626 |
| 4.51 | 5688 | 0.6023 | 0.3586 | 0.8086 | 1.5705 | 0.5670 | 0.3531 | 0.8618 |
| 5.01 | 6320 | 0.5825 | 0.3620 | 0.8121 | 1.6724 | 0.5536 | 0.3568 | 0.8632 |
| 5.51 | 6952 | 0.5626 | 0.3629 | 0.8074 | 1.7562 | 0.5607 | 0.3557 | 0.8555 |
| 6.01 | 7584 | 0.5507 | 0.3594 | 0.8087 | 1.7045 | 0.5482 | 0.3536 | 0.8631 |
| 6.52 | 8216 | 0.5384 | 0.3657 | 0.8131 | 1.8838 | 0.5361 | 0.3591 | 0.8653 |
| 7.02 | 8848 | 0.5256 | 0.3608 | 0.8099 | 1.9661 | 0.5282 | 0.3567 | 0.8662 |
| 7.52 | 9480 | 0.5088 | 0.3643 | 0.8151 | 2.0578 | 0.5242 | 0.3590 | 0.8671 |
| 8.02 | 10112 | 0.5034 | 0.3658 | 0.8153 | 2.0564 | 0.5194 | 0.3621 | 0.8672 |
| 8.52 | 10744 | 0.4942 | 0.3660 | 0.8144 | 2.1280 | 0.5205 | 0.3624 | 0.8667 |
| 9.02 | 11376 | 0.4866 | 0.3671 | 0.8137 | 2.1258 | 0.5176 | 0.3635 | 0.8674 |
| 9.52 | 12008 | 0.4772 | 0.3660 | 0.8132 | 2.1555 | 0.5123 | 0.3614 | 0.8661 |
| 10.02 | 12640 | 0.4747 | 0.3674 | 0.8137 | 2.1730 | 0.5121 | 0.3623 | 0.8663 |
| 10.52 | 13272 | 0.4686 | 0.3681 | 0.8140 | 2.1842 | 0.5105 | 0.3626 | 0.8670 |
| 11.03 | 13904 | 0.4649 | 0.3651 | 0.8134 | 2.2021 | 0.5104 | 0.3601 | 0.8668 |
| 11.53 | 14536 | 0.4620 | 0.3635 | 0.8129 | 2.2161 | 0.5094 | 0.3588 | 0.8665 |
| 12.03 | 15168 | 0.4622 | 0.3633 | 0.8129 | 2.2206 | 0.5091 | 0.3588 | 0.8664 |
| 12.53 | 15800 | 0.4594 | 0.3631 | 0.8128 | 2.2214 | 0.5087 | 0.3587 | 0.8664 |

⭐ = new best at that epoch.

## IoU per class (best epoch 4.01, step 5056)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sky | 0.9516 | traffic sign | 0.2682 |
| paved trail | 0.8631 | pole | 0.2324 |
| sidewalk | 0.8291 | traffic light | 0.2087 |
| pedestrian | 0.7377 | stairs | 0.1490 |
| wall/fence | 0.6835 | bike rack | 0.0948 |
| terrain | 0.6675 | animal | 0.0543 |
| crosswalk | 0.6668 | other walkable surface | 0.0381 |
| vegetation | 0.6472 | opening-door | 0.0308 |
| building | 0.6156 | inaccessible surface | 0.0015 |
| tree | 0.6108 | rider | 0.0010 |
| vehicle | 0.5359 | guard rail/road barrier | 0.0000 |
| water body | 0.5090 | hand rail | 0.0000 |
| road | 0.4428 | opening-gate | 0.0000 |
| obstacle | 0.3293 | railway track | absent |
| curb | 0.2833 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.8197 |
| sanpo3_not_safe | 0.8655 |
| sanpo3_obstacle | 0.9026 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 7.4000 / 0.4683 |
| Gradient norm, median | 55.28 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 11 |
| Update/weight ratio, log10 median, first / last | -3.73 / -4.75 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (32 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
