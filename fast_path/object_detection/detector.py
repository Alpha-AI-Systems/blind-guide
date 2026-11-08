"""Reusable object detection with YOLO26n (fixed COCO classes) or YOLOE-26
(open vocabulary, classes given as text prompts).

This module performs detection ONLY. Depth, tracking and risk assessment belong
in separate components so they can be tested and replaced independently.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

# =============================================================================
# DEFAULT SETTINGS (edit here). Used when YOLODetector(...) gets no value.
# =============================================================================

# ---- Shared by both detectors ----------------------------------------------
DEFAULT_CONFIDENCE = 0.35  # keep boxes with confidence >= this (0 to 1)
DEFAULT_IMAGE_SIZE = 640  # model input size, long side in pixels (e.g. 1920x1080 -> 640x384)
DEFAULT_DEVICE = None  # None = automatic (GPU if available), "cpu", or "0"
DEFAULT_HALF = True  # FP16 on GPU (ignored on CPU)
DEFAULT_CLASSES = None  # None = all classes, e.g. ["person", "car"]

# ---- YOLO26 (fixed 80 COCO classes) ----------------------------------------
DEFAULT_YOLO_WEIGHTS = "yolo26n.pt"

# ---- YOLOE-26 (open vocabulary, used when text prompts are given) ----------
DEFAULT_YOLOE_WEIGHTS = "yoloe-26n-seg.pt"  # all released YOLOE checkpoints are *-seg
DEFAULT_TEXT_PROMPTS = None  # None = use YOLO26; e.g. ["pothole", "stairs"]

# ---- Paths -----------------------------------------------------------------
# Bare weight names (e.g. "yolo26n.pt") are stored/downloaded here, not in the
# current working directory. Path: <repo root>/weights/
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"

# =============================================================================


def resolve_weights(weights: str) -> str:
    """Map a bare filename like 'yolo26n.pt' to <repo>/weights/yolo26n.pt.

    Paths that include a folder (e.g. 'runs/train/best.pt') are used as given.
    """
    path = Path(weights)
    if path.parent != Path(".") or path.is_absolute():
        return weights
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / path.name)


def load_yoloe_detection_only(weights: str) -> Any:
    """Load YOLOE as a box-only model (no mask branch).

    All released YOLOE checkpoints are segmentation models (*-seg.pt). Following
    the official Ultralytics docs, we build the matching detection architecture
    from its YAML (e.g. yoloe-26n.yaml) and load the seg weights into it:
        YOLOE("yoloe-26n.yaml").load("yoloe-26n-seg.pt")
    This removes the mask branch: we only use boxes, it is lighter, and it
    avoids an FP16 crash in the Ultralytics 8.4.11 mask step.
    A path that is not a "*-seg.pt" checkpoint (e.g. a fine-tuned detection
    model) is loaded as given.
    """
    from ultralytics import YOLOE

    resolved = resolve_weights(weights)
    name = Path(resolved).name
    if not name.endswith("-seg.pt"):
        return YOLOE(resolved)
    architecture = name.replace("-seg.pt", ".yaml")  # yoloe-26n-seg.pt -> yoloe-26n.yaml
    return YOLOE(architecture).load(resolved)


@dataclass(frozen=True)
class Detection:
    """One object's location in ORIGINAL frame pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]  # left, top, right, bottom
    center_xy: tuple[int, int]


@dataclass(frozen=True)
class FrameDetections:
    """All detections produced from one camera frame."""

    width: int
    height: int
    total_ms: float  # wall-clock time of the whole predict() call
    preprocess_ms: float | None  # resize/letterbox + tensor conversion (Ultralytics)
    model_ms: float | None  # neural network forward pass only (Ultralytics)
    postprocess_ms: float | None  # raw outputs to boxes in frame pixels (Ultralytics)
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-compatible objects for later integration/logging."""
        return asdict(self)


class YOLODetector:
    """Load the model once, then reuse it for each image or video frame.

    Parameters:
        weights: Local weights path or an Ultralytics model filename. None picks
            yolo26n.pt, or yoloe-26n-seg.pt when text_prompts is given.
        text_prompts: Optional class names for YOLOE (e.g. ["pothole", "stairs"]).
            Applied once here; predict() is unchanged. None = plain YOLO26.
        confidence: Minimum model confidence score accepted (0.0 to 1.0).
        image_size: Model input size (pixels); e.g. 640, the image longer number which coulc be either the H or the W is shrunk down to this number. for example: 1920*1080 -> 640*384.
        device: None=Ultralytics automatic, 'cpu', or GPU index '0'.
        classes: Optional classes to keep, as names ("person") or ids (0).
            None keeps all classes. Filtering is done inside Ultralytics.
        half: Run in FP16 on GPU. Ultralytics ignores it on CPU.
        Defaults for all of these are the DEFAULT_* variables at the top of this file.
        model: Optional model instance for isolated unit testing.
    """

    def __init__(
        self,
        weights: str | None = None,
        text_prompts: Sequence[str] | None = DEFAULT_TEXT_PROMPTS,
        confidence: float = DEFAULT_CONFIDENCE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: str | None = DEFAULT_DEVICE,
        classes: Sequence[str | int] | None = DEFAULT_CLASSES,
        half: bool = DEFAULT_HALF,
        model: Any | None = None,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.half = half

        prompts = None
        if text_prompts is not None:
            prompts = [str(p).strip() for p in text_prompts if str(p).strip()]
            if not prompts:
                raise ValueError("text_prompts must contain at least one name")
        self.text_prompts = prompts

        if model is None:
            # Import lazily so the detection schema/tests are usable without
            # downloading YOLO weights or importing its heavy dependencies.
            chosen = weights or (DEFAULT_YOLO_WEIGHTS if prompts is None else DEFAULT_YOLOE_WEIGHTS)
            self.weights_path = resolve_weights(chosen)
            if prompts is None:
                from ultralytics import YOLO

                self.model = YOLO(resolve_weights(weights or DEFAULT_YOLO_WEIGHTS)) # core YOLO class from ulatralysic
            else:
                self.model = load_yoloe_detection_only(weights or DEFAULT_YOLOE_WEIGHTS)
        else:
            self.model = model
            self.weights_path = None

        if prompts is not None:
            self._apply_text_prompts(prompts)

        self.class_ids = self._resolve_classes(classes)

    def _apply_text_prompts(self, prompts: list[str]) -> None:
        """Set YOLOE's vocabulary to the prompt names (runs the text encoder once).

        The first call downloads the MobileCLIP text encoder (~254 MB) into the
        current directory, so we run it from WEIGHTS_DIR to keep it there.
        """
        if not hasattr(self.model, "set_classes"):
            raise ValueError("text_prompts need a YOLOE model (e.g. yoloe-26n-seg.pt)")
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        previous_dir = os.getcwd()
        try:
            os.chdir(WEIGHTS_DIR)
            self.model.set_classes(prompts)
        finally:
            os.chdir(previous_dir)

    def _resolve_classes(self, classes: Sequence[str | int] | None) -> list[int] | None:
        """Convert class names/ids to a sorted list of valid class ids."""
        if classes is None:
            return None
        names = getattr(self.model, "names", None) or {}
        if not isinstance(names, dict):
            names = dict(enumerate(names))
        name_to_id = {str(name).lower(): idx for idx, name in names.items()}

        ids: set[int] = set()
        for item in classes:
            text = str(item).strip()
            if text.isdecimal():
                class_id = int(text)
                if names and class_id not in names:
                    raise ValueError(f"Unknown class id: {class_id}")
            elif text.lower() in name_to_id:
                class_id = name_to_id[text.lower()]
            else:
                raise ValueError(f"Unknown class name: {text!r}")
            ids.add(class_id)
        if not ids:
            raise ValueError("classes must not be empty (use None for all classes)")
        return sorted(ids)

    def predict(self, frame: Any) -> FrameDetections:
        """Run detection on a BGR OpenCV/NumPy frame (H, W, 3)."""
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) != 3:
            raise ValueError("frame must be a BGR image with shape (H, W, 3)")

        height, width, channels = frame.shape
        if width <= 0 or height <= 0 or channels != 3:
            raise ValueError("frame must be a nonempty 3-channel BGR image")

        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "half": self.half,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.class_ids is not None:
            kwargs["classes"] = self.class_ids

        started = perf_counter()
        results = self.model.predict(**kwargs)
        result = results[0]

        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            # Ultralytics exposes torch tensors; move values to CPU only for
            # the small amount of structured detection metadata we need.
            coordinates = boxes.xyxy.cpu().tolist()
            scores = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            names = result.names

            for coords, score, cls in zip(coordinates, scores, classes):
                class_id = int(cls)
                class_name = str(
                    names.get(class_id, f"class_{class_id}")
                    if isinstance(names, dict)
                    else names[class_id]
                )
                x1 = max(0, min(width, round(float(coords[0]))))
                y1 = max(0, min(height, round(float(coords[1]))))
                x2 = max(0, min(width, round(float(coords[2]))))
                y2 = max(0, min(height, round(float(coords[3]))))

                # Ignore degenerate boxes rather than passing invalid regions
                # to the future depth estimator or tracker.
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=round(float(score), 4),
                        bbox_xyxy=(x1, y1, x2, y2),
                        center_xy=((x1 + x2) // 2, (y1 + y2) // 2),
                    )
                )

        elapsed_ms = (perf_counter() - started) * 1000.0

        # Ultralytics reports per-stage times in result.speed (milliseconds).
        speed = getattr(result, "speed", None) or {}

        def stage(key: str) -> float | None:
            value = speed.get(key)
            return None if value is None else round(float(value), 2)

        return FrameDetections(
            width=width,
            height=height,
            total_ms=round(elapsed_ms, 2),
            preprocess_ms=stage("preprocess"),
            model_ms=stage("inference"),
            postprocess_ms=stage("postprocess"),
            detections=tuple(detections),
        )
