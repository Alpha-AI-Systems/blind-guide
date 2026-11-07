# Object Detection (YOLO26n): Blind Guide Fast Path

## Scope

This folder only performs object detection. **No depth, tracking, risk scoring,
or spoken warnings.** It is designed to be merged with the depth module
(`fast_path/depth/`) and a tracker later.

## Setup

All commands run from the repository root.

```powershell
python -m pip install -r requirements.txt
```

The single `requirements.txt` at the repo root covers the whole fast path
(detection and depth). For GPU, install the CUDA build of PyTorch first
(see pytorch.org).

Ultralytics downloads `yolo26n.pt` automatically the first time it is used
(internet required). A bare weight name like `yolo26n.pt` is stored in
`<repo>/weights/`, whatever folder you run from; a path with a folder
(e.g. `runs/train/best.pt`) is used as given. `weights/` and `*.pt` are git-ignored.

## Run the demo

```powershell
# Webcam
python -m fast_path.object_detection.run_detection_demo --source 0

# Video, saving an annotated copy
python -m fast_path.object_detection.run_detection_demo --source path/to/walk.mp4 --save-video outputs/yolo/walk_annotated.mp4

# Image, saving an annotated copy
python -m fast_path.object_detection.run_detection_demo --source path/to/test.jpg --save-image outputs/yolo/test_annotated.jpg

# Log detections (not images) to JSONL, stop after 120 frames
python -m fast_path.object_detection.run_detection_demo --source 0 --max-frames 120 --jsonl outputs/yolo/detections.jsonl
```

Press **Q** to stop a webcam or video window. Add `--no-display` to run without a window.

| Option | Meaning |
|---|---|
| `--source` | Webcam index (`0`), image path, or video path |
| `--weights` | Weights file or Ultralytics model name (default `yolo26n.pt`) |
| `--confidence` | Minimum confidence to keep a box (default 0.35) |
| `--image-size` | Model input size, long side in pixels (default 640) |
| `--device` | `cpu` or GPU index `0` (default: automatic) |
| `--warmup` | Untimed warm-up runs before timing an image (default 3) |
| `--half` / `--no-half` | FP16 on GPU (default on; ignored on CPU) or FP32 |
| `--classes` | Only keep these classes, by name or id, e.g. `--classes person car bicycle` (default: all) |
| `--save-image` | Save the annotated image (image source) |
| `--save-video` | Save the annotated video as `.mp4` (video or webcam source) |
| `--jsonl` | Save per-frame detections as JSON lines |
| `--no-display` | Do not open a window |
| `--max-frames` | Stop after N frames |

## Tests

```powershell
python -m unittest discover -s fast_path/object_detection/tests -t . -v
```

The tests use a fake model, so no weights or GPU are needed.

## Python API (for the depth and tracking code)

```python
from fast_path.object_detection import YOLODetector

detector = YOLODetector(weights="yolo26n.pt", confidence=0.35)
prediction = detector.predict(frame)  # frame: BGR NumPy array (H, W, 3), e.g. from OpenCV
for obj in prediction.detections:
    print(obj.class_name, obj.confidence, obj.bbox_xyxy, obj.center_xy)
```

- `bbox_xyxy`: `(x1, y1, x2, y2)` in original-frame pixels; `(x1, y1)` is the top-left corner, `(x2, y2)` the bottom-right.
- `center_xy`: center of the box (not of the object, and not a distance).
- Timing, in milliseconds:
  - `total_ms`: wall-clock time of the whole `predict()` call.
  - `preprocess_ms`, `model_ms`, `postprocess_ms`: the Ultralytics per-stage breakdown.
  - None of these is the full end-to-end warning latency on the phone.
- `prediction.to_dict()`: JSON-ready version of the result.

## Team integration notes

- Give YOLO and depth the **same original frame**, and match boxes to the depth map in a separate integration module.
- YOLO's confidence is not a collision probability.
- A box center is not a distance in meters.

## Research note

YOLO26n is pretrained on COCO (80 classes). It returns boxes, classes and scores.
It does not give depth or collision risk, and COCO has no classes for stairs,
curbs, potholes or other drop-offs. Dataset options for evaluating and fine-tuning
on pedestrian hazards are being collected separately.
