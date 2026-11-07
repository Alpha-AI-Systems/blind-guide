# YOLO26n Detection Module — Blind Guide

## Scope

This directory only performs object detection. **No depth, tracking, risk
scoring, or spoken warnings.** It is designed to integrate with the team's
separate depth and tracking modules after the first working baseline.

## Setup on Windows (from the cloned repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r fast_path/object_detection/requirements.txt
```

If the repository already has a `.venv` and dependency management, use its
existing environment instead; do not overwrite the team's files.

## Run

```powershell
python -m fast_path.object_detection.run --source 0
python -m fast_path.object_detection.run --source "examples/walk.mp4"
python -m fast_path.object_detection.run --source "examples/test.jpg" --save-image "outputs/yolo/test_annotated.jpg"
python -m fast_path.object_detection.run --source 0 --max-frames 120 --jsonl "outputs/yolo/detections.jsonl"
python -m unittest discover -s fast_path/object_detection/tests -t . -v
```

Press **Q** to stop a webcam/video window. Ultralytics automatically downloads
`yolo26n.pt` the first time it's used (internet required). The filename
`yolo26n.pt` is from the official released YOLO26 family.

## Python API for Depth and Tracking teammates

```python
from fast_path.object_detection import YOLODetector

detector = YOLODetector(weights="yolo26n.pt", confidence=0.35)
# frame = cv2.VideoCapture(...).read()[1]  # BGR NumPy array
prediction = detector.predict(frame)
for obj in prediction.detections:
    print(obj.class_name, obj.confidence, obj.bbox_xyxy, obj.center_xy)
```

`bbox_xyxy`: `(x1, y1, x2, y2)` in original-frame pixels, where `(x1,y1)` is
top-left and `(x2,y2)` bottom-right. `center_xy` is a convenient center point,
**not** an estimated physical distance. `prediction.inference_ms` measures one
prediction call on the current machine; it is **not** full end-to-end warning
latency or a validated wearable FPS measurement.

`prediction.to_dict()` returns JSON-compatible metadata containing frame width,
height, inference time, and detections. The optional `--jsonl` flag records
metadata, not raw camera images.

## Team integration

Use the **same original frame** as input for YOLO and Depth; match detections to
a depth map in a separate integration module. Do not equate YOLO's confidence
with a collision probability, or a bounding box center with an accurate meter
distance. Tracking IDs and risk rules are intentionally left for later work.

## Repository hygiene

Do not commit model weights, `.venv`, images/videos, or generated output data.
Append entries to the repository's existing `.gitignore` if needed:

```
.venv/
__pycache__/
*.pyc
*.pt
outputs/
```

Do not replace or overwrite an existing `.gitignore`.

## PR checklist

- [ ] Branch: `feat/yolo26n-detector` (not `main`)
- [ ] Code kept inside `fast_path/object_detection/`
- [ ] `python -m unittest discover -s fast_path/object_detection/tests -t . -v` passes
- [ ] Tested image or webcam inference locally
- [ ] No model weights, personal footage, or `outputs/` in commit
- [ ] Note local inference time and any observed issues in the PR

## Research note

YOLO26n is a pretrained COCO object detector. It returns boxes/classes/scores,
not a depth map, collision probability, or reliable stair/drop-off detection.
Future comparison with text-promptable YOLOE-26n should use comparable data and
the same risk-evaluation pipeline after depth/tracking integration.
