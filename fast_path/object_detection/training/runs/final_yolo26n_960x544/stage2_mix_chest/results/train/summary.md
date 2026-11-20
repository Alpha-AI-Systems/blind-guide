# final_yolo26n_960x544/stage2_mix_chest

**finished** · 10/10 epochs · launches run001, run002 · updated 2026-11-20 12:51:38 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3521** (epoch 8.52, step 17952) |
| best.pt re-validated | 0.3521 |
| Pixel accuracy (best epoch) | 0.7470 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3459 |
| sanpo3 mIoU (best epoch) | 0.6610 |
| Last validation mIoU | 0.3520 (epoch 9.02) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
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
| Parameters | 1,634,202 (1.63 M) |
| Forward GFLOPs per image | 5.8335 at 992x544 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 15.953 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 16 CPU cores, 64 GB (modal-container) |
| Training time | 2h01m18s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 92 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 1056 | 1.5197 | 0.3340 | 0.7175 | 1.1824 | 0.6729 | 0.3287 | 0.6619 |
| 1 | 2112 | 0.9368 | 0.3303 | 0.7002 | 1.3983 | 0.6489 | 0.3241 | 0.6524 |
| 1.50 ⭐ | 3168 | 0.7999 | 0.3399 | 0.7426 | 1.3844 | 0.6159 | 0.3337 | 0.6779 |
| 2 | 4224 | 0.7037 | 0.3278 | 0.7197 | 1.5958 | 0.6078 | 0.3220 | 0.6545 |
| 2.50 ⭐ | 5280 | 0.6350 | 0.3469 | 0.7358 | 1.5791 | 0.5949 | 0.3419 | 0.6730 |
| 3.01 | 6336 | 0.5994 | 0.3388 | 0.7410 | 1.7030 | 0.5937 | 0.3343 | 0.6538 |
| 3.51 ⭐ | 7392 | 0.5564 | 0.3476 | 0.7437 | 1.7067 | 0.5684 | 0.3425 | 0.6589 |
| 4.01 ⭐ | 8448 | 0.5363 | 0.3484 | 0.7453 | 1.7636 | 0.5622 | 0.3411 | 0.6634 |
| 4.51 | 9504 | 0.5079 | 0.3475 | 0.7466 | 1.7938 | 0.5638 | 0.3413 | 0.6558 |
| 5.01 ⭐ | 10560 | 0.4851 | 0.3489 | 0.7469 | 1.7574 | 0.5631 | 0.3431 | 0.6642 |
| 5.51 ⭐ | 11616 | 0.4646 | 0.3514 | 0.7450 | 1.8248 | 0.5442 | 0.3459 | 0.6566 |
| 6.01 | 12672 | 0.4499 | 0.3505 | 0.7421 | 1.8609 | 0.5352 | 0.3440 | 0.6584 |
| 6.51 | 13728 | 0.4324 | 0.3511 | 0.7452 | 1.8867 | 0.5317 | 0.3443 | 0.6612 |
| 7.01 ⭐ | 14784 | 0.4233 | 0.3518 | 0.7458 | 1.9402 | 0.5250 | 0.3454 | 0.6616 |
| 7.51 | 15840 | 0.4117 | 0.3515 | 0.7461 | 1.9687 | 0.5219 | 0.3452 | 0.6620 |
| 8.02 ⭐ | 16896 | 0.4058 | 0.3519 | 0.7464 | 1.9811 | 0.5204 | 0.3457 | 0.6615 |
| 8.52 ⭐ | 17952 | 0.4001 | 0.3521 | 0.7470 | 1.9919 | 0.5194 | 0.3459 | 0.6610 |
| 9.02 | 19008 | 0.3982 | 0.3520 | 0.7474 | 1.9964 | 0.5191 | 0.3459 | 0.6609 |

⭐ = new best at that epoch.

## IoU per class (best epoch 8.52, step 17952)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8728 | road | 0.2792 |
| pedestrian | 0.7973 | traffic light | 0.1539 |
| building | 0.7793 | animal | 0.1394 |
| vehicle | 0.7720 | stairs | 0.0989 |
| sidewalk | 0.7579 | bike rack | 0.0959 |
| hand rail | 0.6417 | traffic sign | 0.0951 |
| paved trail | 0.6377 | other walkable surface | 0.0836 |
| wall/fence | 0.6065 | opening-door | 0.0627 |
| sky | 0.5893 | opening-gate | 0.0082 |
| vegetation | 0.5313 | rider | 0.0071 |
| pole | 0.4210 | guard rail/road barrier | 0.0024 |
| curb | 0.4177 | water body | 0.0000 |
| obstacle | 0.3721 | inaccessible surface | 0.0000 |
| tree | 0.3189 | railway track | absent |
| terrain | 0.3167 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7800 |
| sanpo3_not_safe | 0.3236 |
| sanpo3_obstacle | 0.8795 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 5.1946 / 0.3986 |
| Gradient norm, median | 70.66 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 14 |
| Update/weight ratio, log10 median, first / last | -3.98 / -4.64 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (41 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
