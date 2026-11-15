# Leaderboard

Sorted by test mIoU on the human-labeled test frames (SANPO paper's protocol). Written by runs/<run>/test.py, 2026-11-15 09:00 UTC.

| # | Run | Ckpt | **Test mIoU** (human) | Test mIoU (all) | Test 3-class | Val mIoU | imgsz | lr0 | Start | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exp1_baseline_lr002 | best | **0.3276** | 0.3338 | 0.7487 | 0.3139 | 352 | 0.002 | yolo26n-sem-ade20k |  |

Left out: exp1_baseline_lr0005 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was)); exp1_baseline_lr001 (learning-rate sweep, not chosen on val (exp1_baseline_lr002 was))

Choose between runs with the **Val** column; the test columns are the reported scores.
