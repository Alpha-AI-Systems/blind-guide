# exp6_synthetic_then_real/stage3_synthetic_chest

**finished** · 9/9 epochs · launches run001 · updated 2026-11-17 19:27:58 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.7448** (epoch 8.94, step 9504) |
| best.pt re-validated | 0.7448 |
| Pixel accuracy (best epoch) | 0.9552 |
| sanpo3 mIoU (best epoch) | 0.9532 |
| Last validation mIoU | 0.7448 (epoch 8.94) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | train_chest_5fps.txt |
| validation | every 528 steps |
| epochs | 9 |
| batch | 32 |
| imgsz | 352 |
| optimizer | MuSGD |
| lr0 | 0.002 |
| lrf | 0.01 |
| warmup_epochs | 0.9934 |
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
| Peak GPU memory | 6.67 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 48 GB (modal-container) |
| Training time | 27m09s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 204 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 528 | 0.7262 | 0.6644 | 0.9392 | 0.1885 | 0.2875 | 0.9368 |
| 0.99 ⭐ | 1056 | 0.4153 | 0.6953 | 0.9449 | 0.1671 | 0.2716 | 0.9438 |
| 1.49 ⭐ | 1584 | 0.3823 | 0.7076 | 0.9481 | 0.1563 | 0.2657 | 0.9444 |
| 1.99 ⭐ | 2112 | 0.3636 | 0.7139 | 0.9489 | 0.1520 | 0.2688 | 0.9476 |
| 2.48 ⭐ | 2640 | 0.3486 | 0.7228 | 0.9514 | 0.1432 | 0.2597 | 0.9490 |
| 2.98 ⭐ | 3168 | 0.3370 | 0.7259 | 0.9516 | 0.1402 | 0.2606 | 0.9497 |
| 3.48 ⭐ | 3696 | 0.3304 | 0.7287 | 0.9526 | 0.1394 | 0.2573 | 0.9496 |
| 3.97 ⭐ | 4224 | 0.3238 | 0.7326 | 0.9534 | 0.1372 | 0.2563 | 0.9511 |
| 4.47 | 4752 | 0.3171 | 0.7288 | 0.9516 | 0.1439 | 0.2568 | 0.9475 |
| 4.97 ⭐ | 5280 | 0.3130 | 0.7375 | 0.9539 | 0.1350 | 0.2534 | 0.9516 |
| 5.46 ⭐ | 5808 | 0.3088 | 0.7388 | 0.9542 | 0.1351 | 0.2525 | 0.9521 |
| 5.96 ⭐ | 6336 | 0.3044 | 0.7418 | 0.9543 | 0.1341 | 0.2514 | 0.9523 |
| 6.46 ⭐ | 6864 | 0.3003 | 0.7429 | 0.9549 | 0.1320 | 0.2496 | 0.9528 |
| 6.95 ⭐ | 7392 | 0.2989 | 0.7434 | 0.9549 | 0.1319 | 0.2495 | 0.9529 |
| 7.45 | 7920 | 0.2961 | 0.7429 | 0.9548 | 0.1336 | 0.2500 | 0.9529 |
| 7.95 ⭐ | 8448 | 0.2960 | 0.7447 | 0.9551 | 0.1321 | 0.2490 | 0.9532 |
| 8.44 | 8976 | 0.2941 | 0.7443 | 0.9551 | 0.1321 | 0.2486 | 0.9530 |
| 8.94 ⭐ | 9504 | 0.2924 | 0.7448 | 0.9552 | 0.1315 | 0.2483 | 0.9532 |

⭐ = new best at that epoch.

## IoU per class (best epoch 8.94, step 9504)

| Class | IoU | Class | IoU |
|---|---|---|---|
| sidewalk | 0.9627 | traffic light | 0.5432 |
| pedestrian | 0.9479 | bike rack | 0.4500 |
| building | 0.9416 | rider | 0.4000 |
| sky | 0.9375 | guard rail/road barrier | absent |
| road | 0.9315 | paved trail | absent |
| vehicle | 0.8838 | hand rail | absent |
| terrain | 0.8657 | opening-door | absent |
| vegetation | 0.8535 | opening-gate | absent |
| crosswalk | 0.7703 | animal | absent |
| tree | 0.7693 | stairs | absent |
| obstacle | 0.6978 | water body | absent |
| curb | 0.6617 | other walkable surface | absent |
| wall/fence | 0.6019 | inaccessible surface | absent |
| traffic sign | 0.6010 | railway track | absent |
| pole | 0.5862 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.9538 |
| sanpo3_not_safe | 0.9365 |
| sanpo3_obstacle | 0.9693 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 2.6694 / 0.2795 |
| Gradient norm, median | 16.51 (clipped at 10; 90% of steps above) |
| Mixed-precision overflow steps | 7 |
| Update/weight ratio, log10 median, first / last | -3.65 / -4.63 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
