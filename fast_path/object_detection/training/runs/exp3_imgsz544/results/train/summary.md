# exp3_imgsz544

**finished** · 18/18 epochs · launches run001 · updated 2026-11-15 19:17:49 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3430** (epoch 11) |
| best.pt re-validated | 0.3430 |
| Pixel accuracy (best epoch) | 0.7369 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3382 |
| sanpo3 mIoU (best epoch) | 0.6459 |
| Last epoch mIoU | 0.3391 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short544 |
| epochs | 18 |
| batch | 32 |
| imgsz | 544 |
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
| Forward GFLOPs per image | 5.8335 at 992x544 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 15.953 GB |
| Hardware | NVIDIA A10, 6 CPU cores, 48 GB (modal-container) |
| Training time | 1h37m39s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $2.88 at $1.77/h |
| Speed (median) | 56 img/s, data wait 0% |

## Validation per epoch

| Epoch | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 1 ⭐ | 2.6593 | 0.2762 | 0.7190 | 1.1505 | 0.8269 | 0.2717 | 0.6165 |
| 2 ⭐ | 1.2730 | 0.3014 | 0.7091 | 1.2872 | 0.8064 | 0.2975 | 0.6315 |
| 3 ⭐ | 0.9792 | 0.3237 | 0.7262 | 1.2205 | 0.7971 | 0.3170 | 0.6286 |
| 4 ⭐ | 0.8015 | 0.3375 | 0.7328 | 1.3207 | 0.7473 | 0.3332 | 0.6383 |
| 5 | 0.7066 | 0.3356 | 0.7127 | 1.4533 | 0.7369 | 0.3309 | 0.6330 |
| 6 ⭐ | 0.6414 | 0.3409 | 0.7392 | 1.3479 | 0.7114 | 0.3345 | 0.6429 |
| 7 | 0.5967 | 0.3312 | 0.7333 | 1.4245 | 0.6600 | 0.3260 | 0.6418 |
| 8 | 0.5511 | 0.3282 | 0.7326 | 1.4030 | 0.6361 | 0.3227 | 0.6437 |
| 9 | 0.5153 | 0.3380 | 0.7370 | 1.4276 | 0.5940 | 0.3333 | 0.6467 |
| 10 | 0.4836 | 0.3367 | 0.7375 | 1.4142 | 0.5678 | 0.3322 | 0.6479 |
| 11 ⭐ | 0.4585 | 0.3430 | 0.7369 | 1.4683 | 0.5614 | 0.3382 | 0.6459 |
| 12 | 0.4426 | 0.3428 | 0.7397 | 1.4712 | 0.5601 | 0.3363 | 0.6454 |
| 13 | 0.4270 | 0.3414 | 0.7367 | 1.4792 | 0.5584 | 0.3365 | 0.6442 |
| 14 | 0.4171 | 0.3418 | 0.7347 | 1.4843 | 0.5563 | 0.3370 | 0.6457 |
| 15 | 0.4069 | 0.3403 | 0.7337 | 1.5053 | 0.5571 | 0.3353 | 0.6437 |
| 16 | 0.3994 | 0.3399 | 0.7356 | 1.4889 | 0.5547 | 0.3353 | 0.6452 |
| 17 | 0.3965 | 0.3403 | 0.7343 | 1.4991 | 0.5550 | 0.3357 | 0.6460 |
| 18 | 0.3946 | 0.3391 | 0.7351 | 1.4958 | 0.5537 | 0.3347 | 0.6453 |

⭐ = new best at that epoch.

## IoU per class (best epoch 11)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8705 | road | 0.2346 |
| pedestrian | 0.7782 | opening-gate | 0.1613 |
| building | 0.7643 | stairs | 0.1461 |
| sidewalk | 0.7628 | animal | 0.1418 |
| vehicle | 0.7275 | bike rack | 0.0870 |
| sky | 0.6136 | traffic sign | 0.0788 |
| paved trail | 0.6114 | other walkable surface | 0.0611 |
| wall/fence | 0.5871 | opening-door | 0.0512 |
| vegetation | 0.5259 | traffic light | 0.0432 |
| hand rail | 0.5064 | guard rail/road barrier | 0.0110 |
| pole | 0.4267 | rider | 0.0000 |
| curb | 0.4057 | water body | 0.0000 |
| tree | 0.3518 | inaccessible surface | 0.0000 |
| obstacle | 0.3437 | railway track | absent |
| terrain | 0.3112 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7724 |
| sanpo3_not_safe | 0.2842 |
| sanpo3_obstacle | 0.8809 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.7481 / 0.4804 |
| Gradient norm, median | 68.11 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 9 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.65 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
