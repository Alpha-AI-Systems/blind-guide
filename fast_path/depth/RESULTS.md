# Depth results on SANPO (chest camera)

All numbers are on SANPO-Real **official test** walks, chest camera, against SANPO's dense measured
depth (stereo), 0.1–30 m. Values are the **per-frame median** (the typical frame); a few failed frames
can dominate a mean.

- **AbsRel**: mean |pred − true| / true (lower is better)
- **δ1**: share of pixels within 25 % of the true distance (higher is better)
- **raw**: the model's meters as the phone would get them
- **scaled**: each frame rescaled by its median ratio first, so only the shape counts
- **<3 m**: obstacle pixels (SANPO segmentation: pedestrian, vehicle, pole, ...) closer than 3 m

## Test set A: 300 frames from 30 test walks (10 per walk, 1 per second)

| model | params | raw AbsRel | raw δ1 | raw AbsRel <3 m | raw δ1 <3 m | scaled AbsRel | scaled AbsRel <3 m |
|---|---|---|---|---|---|---|---|
| DAv2-Small metric outdoor (pretrained) | 24.8M | 0.838 | 0.049 | 1.374 | 0.000 | 0.197 | 0.176 |
| Depth Pro (pretrained, teacher) | 952M | 0.406 | 0.145 | 0.488 | 0.012 | 0.182 | 0.111 |
| student, gt-only, 60 walks, 1.5k steps | 24.8M | 0.162 | 0.792 | 0.146 | 0.870 | 0.119 | 0.128 |
| student, distill, 60 walks, 1.5k steps | 24.8M | 0.194 | 0.728 | 0.144 | 0.906 | 0.138 | 0.154 |
| student, distill-u, 300 walks, 6k steps | 24.8M | 0.156 | 0.817 | 0.150 | 0.862 | 0.121 | 0.146 |
| **student, gt-only, 300 walks, 6k steps** | 24.8M | **0.139** | **0.851** | **0.126** | **0.917** | **0.102** | **0.113** |

## Test set B: 43 frames with an obstacle within 3 m (10 test walks)

| model | raw AbsRel | raw δ1 | raw AbsRel <3 m | raw δ1 <3 m | scaled AbsRel |
|---|---|---|---|---|---|
| DAv2-Small metric outdoor (pretrained) | 1.089 | 0.044 | 1.376 | 0.000 | 0.164 |
| Depth Pro (pretrained) | 0.559 | 0.014 | 0.526 | 0.003 | 0.146 |
| student, gt-only, 60 walks | 0.126 | 0.879 | 0.108 | 0.938 | 0.103 |
| student, distill-u, 300 walks | 0.120 | 0.892 | 0.113 | 0.934 | 0.101 |
| **student, gt-only, 300 walks** | **0.106** | **0.905** | **0.107** | **0.946** | **0.087** |

## What the runs were

Student = DAv2-Small metric outdoor, fine-tuned (encoder lr 5e-6, decoder 5e-5, AdamW, batch 8,
random 518x518 crops of the 518x924 input). Training walks come only from SANPO's official **train**
split; checkpoints are picked on held-out train walks (8 or 20), never on test walks.

| run | loss |
|---|---|
| gt-only | SiLog (λ 0.5) to SANPO measured depth |
| distill | gt-only + Depth Pro shape loss (SiLog λ 1 = scale-invariant), weight 1, on the gt walks and on 60 walks whose measured depth is hidden |
| distill-u | gt-only + Depth Pro shape loss, weight 0.25, **only** on 120 walks whose measured depth is hidden (stand-in for phone video with no depth); Depth Pro labels that disagree with the first gt-only student are dropped |

## Findings

1. **Fine-tuning on measured depth fixes the meters.** Off-the-shelf models are unusable for warnings:
   for an obstacle within 3 m, almost no pixel is within 25 % (δ1 ≈ 0). After fine-tuning, 92–95 % are.
   The chest camera is always at about the same height, so the scale is learnable.
2. **More measured depth helps:** 60 → 300 walks improved every number.
3. **Depth Pro as a teacher did not help** on this camera, in either setup. The student's shape becomes
   better than the teacher's (scaled AbsRel 0.10 vs 0.18), so copying the teacher pulls it down.
4. Depth Pro fails badly on ~2 % of frames (raw AbsRel 2–5x). A filter against an already fine-tuned
   student catches them (dropped labels: true error 0.30 vs 0.14 kept); a filter against the pretrained
   student does not.
5. Open: all tests use SANPO's camera. How the student does on our own phone camera is not measured yet;
   that is where a teacher could still matter.

## Reproduce (Modal, from the repo root)

```bash
# zero-shot evaluation (frames are downloaded inside the container and cached on the Volume)
python -m modal run fast_path/depth/scripts/eval_depth_modal.py --official-split test --sessions 30
python -m modal run fast_path/depth/scripts/eval_depth_modal.py --selection near --official-split test --sessions 10 --max-frames 5
# training (defaults = the 300-walk run; prints the comparison table)
python -m modal run fast_path/depth/scripts/train_distill_modal.py
```

The best checkpoint (gt-only, 300 walks) is on the Modal Volume `blind-guide-depth` at
`/vol/distill/runs/walks_v8-g60-u60-g240-u60-v12_f10_seed0_steps6000_b8_wt0.25_f0.35/gt-only/best`.
Depth Pro weights are research-only (Apple sample-code license).
