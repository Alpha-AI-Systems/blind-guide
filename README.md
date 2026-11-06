# blind-guide

Chest-camera assistant for blind pedestrians. This repo currently holds the **depth** part of the
fast path (Option A: YOLO26-N + **depth model** + tracker + rules).

## Setup

```bash
pip install -r requirements.txt
```

No GPU is required for testing (CPU works, just slowly). Model weights download from Hugging Face on first use.

## Layout

```
depth/models.py      DepthEstimator (Depth Anything V2 via transformers), box_distance(), colorize_depth()
depth/sanpo.py       SANPO file formats, remote listing/download, local index, SanpoDepthDataset
depth/metrics.py     AbsRel / RMSE / delta metrics, per-distance-band and obstacle-pixel metrics, alignment
scripts/download_sanpo.py   fetch a subset of SANPO
scripts/eval_sanpo.py       evaluate a model on the downloaded frames
scripts/run_depth.py        run a model on an image / folder / video / webcam
```

## 1. Get SANPO data

SANPO (Google Research) is in the public bucket `gs://gresearch/sanpo_dataset/v0` and is read over plain
HTTPS, so no `gsutil` is needed. The full set is several TB, so the script samples a subset:

```bash
# 5 real sessions, chest camera, every 15th frame (1 fps), 10 frames each  (~200 MB)
python scripts/download_sanpo.py --split sanpo-real --sessions 5

# add sparse ZED stereo depth and semantic masks (masks only exist on some frames)
python scripts/download_sanpo.py --split sanpo-real --sessions 5 --gt depth_maps zed_depth_maps segmentation_masks

# synthetic: perfect dense GT plus masks on every frame (head camera only, 5 fps)
python scripts/download_sanpo.py --split sanpo-synthetic --camera head --sessions 5 --stride 5 --gt depth_maps segmentation_masks
```

Bucket contents: 702 real sessions and 1962 synthetic ones. Data is saved under `data/sanpo/`, which is git-ignored.

| file | real | synthetic |
|---|---|---|
| `video_frames/*.png` | 2208×1242, 15 fps, ZED 2i chest + ZED-M head | 2208×1242, 5 fps |
| `depth_maps/*.float16.gz` | dense, 1280×720, meters, 0 = invalid (sky) | dense, full res; sky ≈ 1e4 m |
| `zed_depth_maps/*.float16.gz` | sparse ZED SDK depth, full res, 0 = invalid | – |
| `segmentation_masks/*.png` | R = class id, B = instance id, some frames only | every frame |

Depth format: a gzip of float16 values. The first two values are (H, W). `sanpo.read_depth()` handles this.

## 2. Evaluate

```bash
python scripts/eval_sanpo.py --model da2-small-metric-outdoor   # metric: reports raw + median-scaled
python scripts/eval_sanpo.py --model da2-small                  # relative: scale/shift aligned
python scripts/eval_sanpo.py --model da2-small --split sanpo-synthetic
```

Each run writes `outputs/eval_<model>_<split>/` with `summary.json`, `per_frame.csv` and `vis/` images
(RGB | GT | prediction | error). Besides the standard metrics, it reports AbsRel / δ1 per distance band
(0–2, 2–5, 5–10, 10+ m) and on **obstacle pixels** (pedestrian, vehicle, pole, sign, …, taken from the SANPO masks).
These are the numbers that matter for warnings. `frames_per_metric` in the summary shows how many frames
back each number, because bands and obstacle pixels can be sparse.

Model short names are listed in `depth/models.py` (`MODELS`). Any HF depth-estimation repo id also works.
`--input-size` sets the network input's short side (518 by default; smaller is faster, e.g. 364 or 266).

## 3. Run on your own images / video

```bash
python scripts/run_depth.py --source street.jpg
python scripts/run_depth.py --source walk.mp4 --input-size 364
python scripts/run_depth.py --source 0 --show        # webcam
```

For metric models the overlay shows the distance to the nearest thing in the walking corridor
(the bottom-centre box). To get a distance for a YOLO detection, use `depth.box_distance(depth, box_xyxy)`.

## First results (smoke test, CPU, 518×924 input; small sample, not a benchmark)

| model | data | align | AbsRel ↓ | δ1 ↑ |
|---|---|---|---|---|
| DA-V2 Small Metric-Outdoor | real, 6 frames | none (raw meters) | 0.83 | 0.11 |
| DA-V2 Small Metric-Outdoor | real, 6 frames | median scale | 0.32 | 0.55 |
| DA-V2 Small (relative) | real, 6 frames | scale+shift | 0.20 | 0.73 |
| DA-V2 Small (relative) | synthetic, 3 frames | scale+shift | 0.06 | 0.97 |

Takeaway so far: the off-the-shelf metric checkpoint, trained on Virtual KITTI driving scenes, gets the
**scale** wrong on chest-height pedestrian video. On a city sidewalk it reads about 2.4× too far, and on a
forest trail it collapses to a flat ~4.5 m. The relative model's structure is much better. This is the
case for fine-tuning a metric head on SANPO (`SanpoDepthDataset` is ready for that), or for recovering
scale at runtime (known camera height + ground plane). Rerun on more sessions before quoting numbers.
