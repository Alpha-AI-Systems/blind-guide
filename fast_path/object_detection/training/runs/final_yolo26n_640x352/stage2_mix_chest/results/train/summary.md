# final_yolo26n_640x352/stage2_mix_chest

**finished** · 10/10 epochs · launches run001, run002 · updated 2026-11-19 19:36:41 UTC

## Result

| Score | Value |
|---|---|
| **Best val mIoU** | **0.3381** (epoch 2.50, step 5280) |
| Pixel accuracy (best epoch) | 0.7355 |
| val_human mIoU (best epoch, human-labeled frames) | 0.3337 |
| sanpo3 mIoU (best epoch) | 0.6552 |
| Last validation mIoU | 0.3277 (epoch 9.02) |

> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes from the separate test evaluation (human-labeled frames, full resolution).

## Setup

| Setting | Value |
|---|---|
| weights | yolo26n-sem-ade20k.pt |
| data | semantic_v2_short352 |
| train frames | 10 fps + train_chest_5fps (16+16 per batch) |
| classes | data.yaml's classes |
| validation | every 1056 steps |
| epochs | 10 |
| batch | 32 |
| imgsz | 352 |
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
| Forward GFLOPs per image | 2.4093 at 640x352 |
| Peak GPU memory | 6.669 GB |
| Hardware | NVIDIA A100-SXM4-40GB, 8 CPU cores, 96 GB (modal-container) |
| Training time | 55m51s (training loop, all launches; start-up of a few minutes per launch not included) |
| Speed (median) | 203 img/s, data wait 1% |

## Validation during training

| Epoch | Step | Train loss | mIoU | Pixel acc | CE loss | Dice loss | val_human mIoU | sanpo3 mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.50 ⭐ | 1056 | 1.6424 | 0.2973 | 0.6944 | 1.2886 | 0.6907 | 0.2908 | 0.6635 |
| 1 ⭐ | 2112 | 1.0491 | 0.3044 | 0.7002 | 1.4889 | 0.6687 | 0.2985 | 0.6400 |
| 1.50 ⭐ | 3168 | 0.9085 | 0.3166 | 0.7247 | 1.5347 | 0.6221 | 0.3096 | 0.6409 |
| 2 ⭐ | 4224 | 0.8092 | 0.3354 | 0.7164 | 1.5225 | 0.6222 | 0.3310 | 0.6432 |
| 2.50 ⭐ | 5280 | 0.7355 | 0.3381 | 0.7355 | 1.6267 | 0.6073 | 0.3337 | 0.6552 |
| 3.01 | 6336 | 0.6931 | 0.3234 | 0.7188 | 1.7014 | 0.6009 | 0.3173 | 0.6423 |
| 3.51 | 7392 | 0.6536 | 0.3275 | 0.7345 | 1.7266 | 0.5796 | 0.3214 | 0.6500 |
| 4.01 | 8448 | 0.6319 | 0.3246 | 0.7308 | 1.8397 | 0.5820 | 0.3176 | 0.6498 |
| 4.51 | 9504 | 0.5986 | 0.3239 | 0.7310 | 1.8189 | 0.5787 | 0.3178 | 0.6458 |
| 5.01 | 10560 | 0.5726 | 0.3269 | 0.7303 | 1.8594 | 0.5708 | 0.3205 | 0.6499 |
| 5.51 | 11616 | 0.5544 | 0.3277 | 0.7335 | 1.8877 | 0.5512 | 0.3215 | 0.6471 |
| 6.01 | 12672 | 0.5331 | 0.3297 | 0.7350 | 1.9635 | 0.5378 | 0.3238 | 0.6483 |
| 6.51 | 13728 | 0.5173 | 0.3303 | 0.7327 | 1.9647 | 0.5357 | 0.3246 | 0.6473 |
| 7.01 | 14784 | 0.5075 | 0.3299 | 0.7321 | 1.9763 | 0.5351 | 0.3244 | 0.6489 |
| 7.51 | 15840 | 0.4966 | 0.3289 | 0.7315 | 1.9951 | 0.5340 | 0.3237 | 0.6478 |
| 8.02 | 16896 | 0.4915 | 0.3283 | 0.7313 | 2.0067 | 0.5333 | 0.3233 | 0.6476 |
| 8.52 | 17952 | 0.4844 | 0.3277 | 0.7310 | 2.0220 | 0.5329 | 0.3228 | 0.6473 |
| 9.02 | 19008 | 0.4849 | 0.3277 | 0.7310 | 2.0289 | 0.5325 | 0.3227 | 0.6470 |

⭐ = new best at that epoch.

## IoU per class (best epoch 2.50, step 5280)

| Class | IoU | Class | IoU |
|---|---|---|---|
| crosswalk | 0.8611 | tree | 0.2742 |
| sidewalk | 0.8184 | opening-gate | 0.1856 |
| pedestrian | 0.7550 | bike rack | 0.1679 |
| building | 0.7362 | stairs | 0.1447 |
| vehicle | 0.7100 | traffic light | 0.1332 |
| paved trail | 0.6381 | traffic sign | 0.1215 |
| hand rail | 0.5470 | other walkable surface | 0.0810 |
| wall/fence | 0.5450 | guard rail/road barrier | 0.0499 |
| sky | 0.5304 | opening-door | 0.0451 |
| vegetation | 0.5150 | rider | 0.0037 |
| curb | 0.3478 | animal | 0.0000 |
| pole | 0.3435 | water body | 0.0000 |
| obstacle | 0.3367 | inaccessible surface | 0.0000 |
| terrain | 0.2885 | railway track | absent |
| road | 0.2877 |  |  |

absent = no pixels of that class in the validation set (left out of the mIoU).

## Class groups (best epoch)

| Group | IoU |
|---|---|
| sanpo3_safe | 0.7776 |
| sanpo3_not_safe | 0.3195 |
| sanpo3_obstacle | 0.8686 |

## Training health

| Item | Value |
|---|---|
| Train loss, first / last logged | 5.3644 / 0.4987 |
| Gradient norm, median | 72.11 (clipped at 10; 100% of steps above) |
| Mixed-precision overflow steps | 13 |
| Update/weight ratio, log10 median, first / last | -3.98 / -4.63 (about -3 is healthy) |

## Files

- Plots: [best_epoch_iou_per_class](plots/best_epoch_iou_per_class.png), [learning_rate](plots/learning_rate.png), [loss_curves](plots/loss_curves.png), [training_diagnostics](plots/training_diagnostics.png), [ud_ratio](plots/ud_ratio.png), [val_aux_loss](plots/val_aux_loss.png), [val_ce_loss](plots/val_ce_loss.png), [val_dice_loss](plots/val_dice_loss.png), [val_human_aux_loss](plots/val_human_aux_loss.png), [val_human_ce_loss](plots/val_human_ce_loss.png), [val_human_dice_loss](plots/val_human_dice_loss.png), [val_iou_per_class](plots/val_iou_per_class.png), [val_mIoU](plots/val_mIoU.png), [val_pixel_acc](plots/val_pixel_acc.png)
- Debug images (exact model input): `debug/` (41 saved)
- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`
- Ultralytics: `ultralytics/` (weights/best.pt, last.pt, results.csv)
