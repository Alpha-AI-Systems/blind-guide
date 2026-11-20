# Leaderboard

Sorted by test mIoU on the human-labeled test frames (SANPO paper's protocol). Written by runs/<run>/test.py, 2026-11-20 16:10 UTC.

| # | Run | Ckpt | **Test mIoU** (human) | Test mIoU (all) | Test 3-class | Test head (human) | Test paper: chest + head (human) | Val mIoU | Val mIoU (human) | imgsz | Train input | Train frames | lr0 | Start | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | final_yolo26s_960x544 | best | **0.4118** | 0.4164 | 0.7623 | 0.4159 | 0.4276 | 0.3755 | 0.3674 | 544 | full frame | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best | 6.51 M |
| 2 | final_yolo26n_960x544 | best | **0.4089** | 0.4134 | 0.7769 | 0.3927 | 0.4256 | 0.3504 | 0.3440 | 544 | full frame | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best | 1.63 M |
| 3 | final_yolo26s_640x352 | best | **0.3994** | 0.4038 | 0.7571 | 0.3768 | 0.4129 | 0.3621 | 0.3573 | 352 | full frame | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best | 6.51 M |
| 4 | exp3_imgsz736 | best | **0.3926** | 0.3982 | 0.7690 | 0.3588 | 0.3988 | 0.3617 | 0.3552 | 736 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 5 | exp4_frames_10fps_yolo26s | best | **0.3897** | 0.3957 | 0.7593 | 0.3460 | 0.3946 | 0.3507 | 0.3435 | 352 | full frame | 10 fps | 0.002 | yolo26s-sem-ade20k | 6.51 M |
| 6 | exp7_synthetic_mixed | best | **0.3884** | 0.3926 | 0.7626 | 0.3625 | 0.3973 | 0.3312 | 0.3270 | 352 | full frame | 5 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best | 1.63 M |
| 7 | exp5_real_head_then_chest | best | **0.3815** | 0.3872 | 0.7614 | 0.3237 | 0.3809 | 0.3310 | 0.3239 | 352 | full frame | 5 fps | 0.002 | stage1_real_head_best | 1.63 M |
| 8 | exp6_synthetic_then_real_lr_decay | best | **0.3812** | 0.3856 | 0.7527 | 0.3382 | 0.3749 | 0.3297 | 0.3239 | 352 | full frame | 5 fps | 0.00070710678 | stage3_synthetic_chest_best | 1.63 M |
| 9 | exp3_imgsz544 | best | **0.3780** | 0.3852 | 0.7635 | 0.3471 | 0.3864 | 0.3430 | 0.3382 | 544 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 10 | final_yolo26n_640x352 | best | **0.3705** | 0.3770 | 0.7480 | 0.3368 | 0.3772 | 0.3380 | 0.3336 | 352 | full frame | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best | 1.63 M |
| 11 | exp4_frames_15fps | best | **0.3696** | 0.3757 | 0.7480 | 0.3281 | 0.3712 | 0.3236 | 0.3190 | 352 | full frame | 15 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 12 | exp4_frames_10fps | best | **0.3646** | 0.3716 | 0.7546 | 0.3112 | 0.3631 | 0.3259 | 0.3234 | 352 | full frame | 10 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 13 | exp7_synthetic_mixed_lr_decay | best | **0.3643** | 0.3719 | 0.7552 | 0.3769 | 0.3809 | 0.3332 | 0.3274 | 352 | full frame | 5 fps + train_chest_5fps (16+16 per batch) | 0.001 | stage1_mix_head_best | 1.63 M |
| 14 | exp9_class_weights | best | **0.3641** | 0.3714 | 0.7423 | 0.3192 | 0.3652 | 0.3257 | 0.3195 | 352 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 15 | exp6_synthetic_then_real | best | **0.3638** | 0.3689 | 0.7545 | 0.3287 | 0.3644 | 0.3280 | 0.3236 | 352 | full frame | 5 fps | 0.002 | stage3_synthetic_chest_best | 1.63 M |
| 16 | exp4_frames_human_only | best | **0.3616** | 0.3663 | 0.7566 | 0.3087 | 0.3566 | 0.3185 | 0.3150 | 352 | full frame | human (5 fps) | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 17 | exp2_start_cityscapes | best | **0.3518** | 0.3585 | 0.7432 | 0.3090 | 0.3517 | 0.3209 | 0.3189 | 352 | full frame | 5 fps | 0.002 | yolo26n-sem | 1.63 M |
| 18 | exp2_start_ade20k | best | **0.3483** | 0.3556 | 0.7407 | 0.3159 | 0.3491 | 0.3222 | 0.3178 | 352 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 19 | exp1_baseline_lr002 | best | **0.3276** | 0.3338 | 0.7487 | 0.2924 | 0.3237 | 0.3138 | 0.3091 | 352 | center square | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |

Test columns: **Test mIoU** = chest camera, human-labeled frames of SANPO's official test walks (our main score); **Test head** = the head camera's official test walks; **Test paper** = both cameras together, the SANPO paper's test set (human-labeled frames of SANPO-Real test).

Train input: **center square** = Ultralytics' default, training saw a centered imgsz x imgsz square of each frame; **full frame** = the whole 16:9 frame (ultralytics.rect).

Train frames: the training walks' frames used: **5 fps** (train.txt), **10 fps** / **15 fps** (data.train_fps, picked from train_15fps.txt), **human** (only human-labeled frames, train_human.txt).

## 3 classes: safe / not safe / obstacle (SANPO paper, Appendix A.6.1)

Sorted by the chest test. **Trained on 3** = the model's outputs are the 3 groups (data.class_merge); **29, merged** = a 29-class model whose prediction and label are merged into the groups afterwards (logging.class_groups, from the confusion matrix).

| # | Run | Classes | **Test chest** (human) | Test head (human) | Test paper: chest + head (human) | Val | imgsz | Train frames | lr0 | Start |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | final_yolo26n_960x544 | 29, merged | **0.7769** | 0.7843 | 0.7890 | 0.6602 | 544 | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best |
| 2 | exp8_three_classes | **trained on 3** | **0.7691** | 0.6932 | 0.7444 | 0.6416 | 352 | 5 fps | 0.002 | yolo26n-sem-ade20k |
| 3 | exp3_imgsz736 | 29, merged | **0.7690** | 0.7369 | 0.7629 | 0.6495 | 736 | 5 fps | 0.002 | yolo26n-sem-ade20k |
| 4 | exp3_imgsz544 | 29, merged | **0.7635** | 0.7665 | 0.7722 | 0.6459 | 544 | 5 fps | 0.002 | yolo26n-sem-ade20k |
| 5 | exp7_synthetic_mixed | 29, merged | **0.7626** | 0.7772 | 0.7777 | 0.6595 | 352 | 5 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best |
| 6 | final_yolo26s_960x544 | 29, merged | **0.7623** | 0.8017 | 0.7868 | 0.6629 | 544 | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best |
| 7 | exp5_real_head_then_chest | 29, merged | **0.7614** | 0.7155 | 0.7509 | 0.6515 | 352 | 5 fps | 0.002 | stage1_real_head_best |
| 8 | exp4_frames_10fps_yolo26s | 29, merged | **0.7593** | 0.7338 | 0.7577 | 0.6484 | 352 | 10 fps | 0.002 | yolo26s-sem-ade20k |
| 9 | final_yolo26s_640x352 | 29, merged | **0.7571** | 0.7759 | 0.7740 | 0.6656 | 352 | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best |
| 10 | exp4_frames_human_only | 29, merged | **0.7566** | 0.7122 | 0.7461 | 0.6426 | 352 | human (5 fps) | 0.002 | yolo26n-sem-ade20k |
| 11 | exp7_synthetic_mixed_lr_decay | 29, merged | **0.7552** | 0.7814 | 0.7752 | 0.6392 | 352 | 5 fps + train_chest_5fps (16+16 per batch) | 0.001 | stage1_mix_head_best |
| 12 | exp4_frames_10fps | 29, merged | **0.7546** | 0.7432 | 0.7593 | 0.6445 | 352 | 10 fps | 0.002 | yolo26n-sem-ade20k |
| 13 | exp6_synthetic_then_real | 29, merged | **0.7545** | 0.7147 | 0.7457 | 0.6504 | 352 | 5 fps | 0.002 | stage3_synthetic_chest_best |
| 14 | exp6_synthetic_then_real_lr_decay | 29, merged | **0.7527** | 0.7209 | 0.7488 | 0.6606 | 352 | 5 fps | 0.00070710678 | stage3_synthetic_chest_best |
| 15 | exp1_baseline_lr002 | 29, merged | **0.7487** | 0.6887 | 0.7303 | 0.6384 | 352 | 5 fps | 0.002 | yolo26n-sem-ade20k |
| 16 | exp4_frames_15fps | 29, merged | **0.7480** | 0.7297 | 0.7490 | 0.6385 | 352 | 15 fps | 0.002 | yolo26n-sem-ade20k |
| 17 | final_yolo26n_640x352 | 29, merged | **0.7480** | 0.7612 | 0.7633 | 0.6552 | 352 | 10 fps + train_chest_5fps (16+16 per batch) | 0.002 | stage1_mix_head_best |
| 18 | exp2_start_cityscapes | 29, merged | **0.7432** | 0.7394 | 0.7503 | 0.6511 | 352 | 5 fps | 0.002 | yolo26n-sem |
| 19 | exp9_class_weights | 29, merged | **0.7423** | 0.7321 | 0.7460 | 0.6470 | 352 | 5 fps | 0.002 | yolo26n-sem-ade20k |
| 20 | exp2_start_ade20k | 29, merged | **0.7407** | 0.7225 | 0.7414 | 0.6353 | 352 | 5 fps | 0.002 | yolo26n-sem-ade20k |

Left out: exp1_baseline_lr0005 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was)); exp1_baseline_lr001 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was)); exp1_baseline_lr004 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was)); exp1_baseline_lr008 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was))

Choose between runs with the **Val** column; the test columns are the reported scores.
