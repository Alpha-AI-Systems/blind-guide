# Leaderboard

Sorted by test mIoU on the human-labeled test frames (SANPO paper's protocol). Written by runs/<run>/test.py, 2026-11-17 07:23 UTC.

| # | Run | Ckpt | **Test mIoU** (human) | Test mIoU (all) | Test 3-class | Val mIoU | Val mIoU (human) | imgsz | Train input | Train frames | lr0 | Start | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exp3_imgsz736 | best | **0.3926** | 0.3982 | 0.7690 | 0.3617 | 0.3552 | 736 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 2 | exp3_imgsz544 | best | **0.3780** | 0.3852 | 0.7635 | 0.3430 | 0.3382 | 544 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 3 | exp4_frames_15fps | best | **0.3696** | 0.3757 | 0.7480 | 0.3236 | 0.3190 | 352 | full frame | 15 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 4 | exp4_frames_10fps | best | **0.3646** | 0.3716 | 0.7546 | 0.3259 | 0.3234 | 352 | full frame | 10 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 5 | exp4_frames_human_only | best | **0.3616** | 0.3663 | 0.7566 | 0.3185 | 0.3150 | 352 | full frame | human (5 fps) | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 6 | exp2_start_cityscapes | best | **0.3518** | 0.3585 | 0.7432 | 0.3209 | 0.3189 | 352 | full frame | 5 fps | 0.002 | yolo26n-sem | 1.63 M |
| 7 | exp2_start_ade20k | best | **0.3483** | 0.3556 | 0.7407 | 0.3222 | 0.3178 | 352 | full frame | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |
| 8 | exp1_baseline_lr002 | best | **0.3276** | 0.3338 | 0.7487 | 0.3138 | 0.3091 | 352 | center square | 5 fps | 0.002 | yolo26n-sem-ade20k | 1.63 M |

Train input: **center square** = Ultralytics' default, training saw a centered imgsz x imgsz square of each frame; **full frame** = the whole 16:9 frame (ultralytics.rect).

Train frames: the training walks' frames used: **5 fps** (train.txt), **10 fps** / **15 fps** (data.train_fps, picked from train_15fps.txt), **human** (only human-labeled frames, train_human.txt).

Left out: exp1_baseline_lr0005 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was)); exp1_baseline_lr001 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was))

Choose between runs with the **Val** column; the test columns are the reported scores.
