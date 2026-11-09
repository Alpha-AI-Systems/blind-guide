# SANPO Depth Evaluation — Initial Baselines

## Setup
- Dataset: 50 SANPO-Real chest-camera frames, 5 sessions
- Device: NVIDIA RTX 4060 Ti
- Input size: 518, evaluated model input 518 x 924

## Pretrained Models

| Model | AbsRel | RMSE (m) | Delta1 | Mean latency (ms) |
|---|---:|---:|---:|---:|
| DA-V2 Small Metric Outdoor, raw | 0.7829 | 7.4077 | 0.1446 | 64.0 |
| DA-V2 Base Metric Outdoor, raw | 0.8982 | 8.8817 | 0.1037 | 90.8 |
| DA-V2 Small Metric Outdoor, ZED GT | 0.7786 | 7.0546 | 0.1470 | 70.0 |
| DA-V2 Small Relative, SSI-aligned | 0.142 | 2.459 | 0.820 | 56.7 |

## Investigations
- Official Hugging Face preprocessing and custom preprocessing produced similar results.
- Metric distance bias varied substantially by recording session.
- Leave-one-session-out fixed-scale correction reduced mean AbsRel from 0.7829 to 0.4800.
- Fixed-scale correction worsened some sessions.
- Near-distance accuracy remained poor for the raw metric models.

## Conclusions
- Base Metric Outdoor was worse than Small Metric Outdoor on this sample.
- Raw metric predictions are not sufficiently reliable for distance-based navigation warnings.
- SSI-aligned relative results use ground truth and cannot be treated as deployment-time meter accuracy.
- Further mobile-focused model evaluation and per-object depth testing are required.

## Limitations
This is a small preliminary dataset. The results do not establish navigation safety.
