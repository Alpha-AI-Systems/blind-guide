# exp9_class_weights

**finished** · 18/18 epochs · launches run001, run002 · updated 2026-11-19 08:54:38 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3257** (epoch 13, step 6864) |
| best.pt re-validated | 0.3257 |
| Pixel accuracy (best epoch) | 0.7241 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3195 |
| sanpo3 mIoU (best epoch) | 0.6470 |
| Last validation mIoU | 0.3235 (epoch 18) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v1_short352 |
| train frames | 5 fps |
| classes | data.yaml's classes |
| validation | every epoch |
| epochs | 18 |
| batch | 32 |
| imgsz | 352 |
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
| Forward GFLOPs per image | 2.4093 at 640x352 |
| best.pt size | 3.44 MB |
| Peak GPU memory | 6.669 GB |
| Hardware | Tesla T4, 4 CPU cores, 32 GB (modal-container) |
| Training time | 1h28m56s (training loop, all launches; start-up of a few minutes per launch not included) |
| Estimated cost | $1.53 at $1.04/h |
| Speed (median) | 62 img/s, data wait 0% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 1 ⭐ | 528 | 3.2277 | 0.2764 | 0.6688 | 1.4424 | 0.8401 | 0.2698 | 0.6159 |
| 2 ⭐ | 1056 | 1.5318 | 0.2949 | 0.6907 | 1.5423 | 0.8192 | 0.2918 | 0.6209 |
| 3 ⭐ | 1584 | 1.1894 | 0.3068 | 0.6934 | 1.4924 | 0.8078 | 0.3037 | 0.6337 |
| 4 | 2112 | 1.0158 | 0.3040 | 0.6856 | 1.7517 | 0.7717 | 0.3007 | 0.6301 |
| 5 ⭐ | 2640 | 0.8769 | 0.3128 | 0.7090 | 1.7710 | 0.7573 | 0.3079 | 0.6403 |
| 6 ⭐ | 3168 | 0.8071 | 0.3191 | 0.7133 | 1.6576 | 0.7353 | 0.3148 | 0.6455 |
| 7 | 3696 | 0.7530 | 0.3112 | 0.7109 | 1.6660 | 0.6971 | 0.3063 | 0.6553 |
| 8 | 4224 | 0.6899 | 0.3169 | 0.7159 | 1.8024 | 0.6364 | 0.3108 | 0.6395 |
| 9 ⭐ | 4752 | 0.6514 | 0.3232 | 0.7136 | 1.9006 | 0.6272 | 0.3185 | 0.6422 |
| 10 ⭐ | 5280 | 0.6176 | 0.3241 | 0.7212 | 1.8136 | 0.6279 | 0.3182 | 0.6430 |
| 11 | 5808 | 0.5931 | 0.3237 | 0.7260 | 1.8216 | 0.6076 | 0.3185 | 0.6440 |
| 12 | 6336 | 0.5721 | 0.3227 | 0.7213 | 1.8347 | 0.5935 | 0.3173 | 0.6464 |
| 13 ⭐ | 6864 | 0.5500 | 0.3257 | 0.7241 | 1.8323 | 0.5808 | 0.3195 | 0.6470 |
| 14 | 7392 | 0.5374 | 0.3216 | 0.7200 | 1.9217 | 0.5813 | 0.3157 | 0.6427 |
| 15 | 7920 | 0.5268 | 0.3233 | 0.7240 | 1.8964 | 0.5785 | 0.3178 | 0.6455 |
| 16 | 8448 | 0.5170 | 0.3246 | 0.7254 | 1.8779 | 0.5784 | 0.3194 | 0.6487 |
| 17 | 8976 | 0.5114 | 0.3242 | 0.7248 | 1.8935 | 0.5775 | 0.3185 | 0.6483 |
| 18 | 9504 | 0.5067 | 0.3235 | 0.7251 | 1.8927 | 0.5765 | 0.3180 | 0.6469 |

⭐ = new best at that epoch.

## IoU per class (best epoch 13, step 6864)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8541 | road | 0.2535 |
| sidewalk | 0.7595 | stairs | 0.1463 |
| building | 0.7528 | animal | 0.1002 |
| pedestrian | 0.7474 | opening-door | 0.0981 |
| vehicle | 0.7051 | traffic sign | 0.0892 |
| paved trail | 0.6071 | other walkable surface | 0.0676 |
| sky | 0.5527 | bike rack | 0.0536 |
| wall/fence | 0.5312 | guard rail/road barrier | 0.0294 |
| hand rail | 0.5191 | traffic light | 0.0255 |
| vegetation | 0.4924 | opening-gate | 0.0134 |
| curb | 0.3906 | rider | 0.0000 |
| pole | 0.3647 | water body | 0.0000 |
| obstacle | 0.3326 | inaccessible surface | 0.0000 |
| terrain | 0.3306 | railway track | absent |
| tree | 0.3034 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7689 |
| sanpo3_not_safe | 0.2990 |
| sanpo3_obstacle | 0.8731 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 8.8504 / 0.5225 |
| Gradient norm, median | 80.89 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 10 |
| Update/weight ratio, log10 median, first / last | -3.64 / -4.63 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (20 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
