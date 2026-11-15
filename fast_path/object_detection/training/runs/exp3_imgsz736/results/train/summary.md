# exp3_imgsz736

**finished** · 18/18 epochs · launches run001 · updated 2026-11-15 19:25:39 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3617** (epoch 11) |
| best.pt re-validated | 0.3616 |
| Pixel accuracy (best epoch) | 0.7494 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3552 |
| sanpo3 mIoU (best epoch) | 0.6495 |
| Last epoch mIoU | 0.3481 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short736 |
| epochs | 18 |
| batch | 32 |
| imgsz | 736 |
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
| Forward GFLOPs per image | 10.589 at 1312x736 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 28.672 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 6 CPU cores, 32 GB (modal-container) |
| Training time | 1h40m38s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $4.43 at $2.64/h |
| Speed (median) | 54 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 1 ⭐ | 2.6305 | 0.2827 | 0.7109 | 1.1569 | 0.8259 | 0.2784 | 0.6187 |
| 2 ⭐ | 1.2568 | 0.3242 | 0.7214 | 1.0582 | 0.7987 | 0.3202 | 0.6345 |
| 3 ⭐ | 0.9581 | 0.3321 | 0.7244 | 1.2754 | 0.7901 | 0.3265 | 0.6241 |
| 4 ⭐ | 0.7951 | 0.3384 | 0.7405 | 1.4114 | 0.7783 | 0.3346 | 0.6329 |
| 5 ⭐ | 0.6863 | 0.3500 | 0.7324 | 1.3741 | 0.7410 | 0.3474 | 0.6562 |
| 6 | 0.6222 | 0.3453 | 0.7408 | 1.3505 | 0.7115 | 0.3409 | 0.6436 |
| 7 | 0.5829 | 0.3364 | 0.7438 | 1.3212 | 0.7056 | 0.3315 | 0.6490 |
| 8 | 0.5411 | 0.3442 | 0.7507 | 1.3651 | 0.6452 | 0.3377 | 0.6499 |
| 9 | 0.5016 | 0.3483 | 0.7492 | 1.3781 | 0.5938 | 0.3436 | 0.6507 |
| 10 | 0.4641 | 0.3499 | 0.7457 | 1.3853 | 0.5847 | 0.3463 | 0.6526 |
| 11 ⭐ | 0.4426 | 0.3617 | 0.7494 | 1.3948 | 0.5795 | 0.3552 | 0.6495 |
| 12 | 0.4250 | 0.3527 | 0.7470 | 1.4545 | 0.5608 | 0.3454 | 0.6494 |
| 13 | 0.4078 | 0.3451 | 0.7478 | 1.4294 | 0.5576 | 0.3393 | 0.6486 |
| 14 | 0.3961 | 0.3484 | 0.7474 | 1.4307 | 0.5527 | 0.3423 | 0.6493 |
| 15 | 0.3863 | 0.3500 | 0.7481 | 1.4566 | 0.5488 | 0.3442 | 0.6485 |
| 16 | 0.3791 | 0.3501 | 0.7492 | 1.4292 | 0.5509 | 0.3442 | 0.6508 |
| 17 | 0.3768 | 0.3512 | 0.7497 | 1.4480 | 0.5509 | 0.3455 | 0.6501 |
| 18 | 0.3740 | 0.3481 | 0.7495 | 1.4386 | 0.5497 | 0.3420 | 0.6499 |

⭐ = new best at that epoch.

## IoU per class (best epoch 11)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8770 | terrain | 0.2988 |
| sidewalk | 0.8083 | road | 0.2392 |
| pedestrian | 0.7916 | animal | 0.2175 |
| building | 0.7647 | stairs | 0.1607 |
| vehicle | 0.7456 | traffic sign | 0.1087 |
| paved trail | 0.6358 | bike rack | 0.0871 |
| sky | 0.6315 | traffic light | 0.0736 |
| wall/fence | 0.6023 | other walkable surface | 0.0727 |
| hand rail | 0.5142 | opening-door | 0.0617 |
| vegetation | 0.5109 | guard rail/road barrier | 0.0184 |
| pole | 0.4285 | water body | 0.0000 |
| curb | 0.4001 | rider | 0.0000 |
| obstacle | 0.3749 | inaccessible surface | 0.0000 |
| tree | 0.3643 | railway track | absent |
| opening-gate | 0.3387 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7754 |
| sanpo3_not_safe | 0.2891 |
| sanpo3_obstacle | 0.8840 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.8550 / 0.4576 |
| Gradient norm, median | 69.81 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 8 |
| Update/weight ratio, log10 median, first / last | -3.66 / -4.63 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
