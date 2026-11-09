"""Run YOLO26n (or YOLOE-26 with --prompts) on a webcam, video, or image.

Run from the repository root:
    python -m fast_path.object_detection.run_detection_demo --source 0
    python -m fast_path.object_detection.run_detection_demo --source path/to/video.mp4
    python -m fast_path.object_detection.run_detection_demo --source path/to/image.jpg --save-image out.jpg
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

if __package__ in (None, ""):
    # Run directly as a file (python fast_path/object_detection/run_detection_demo.py):
    # add the repo root to the import path so the package import below works.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fast_path.object_detection.detector import FrameDetections, YOLODetector  # noqa: E402

# =============================================================================
# DEMO SETTINGS (edit here). Command-line options override these.
# =============================================================================

# ---- Input -----------------------------------------------------------------
SOURCE = r"C:\Users\mamou\Downloads\potholes.jpg"  # webcam index ("0"), or an image/video path
MAX_FRAMES = None  # stop a video/webcam after N frames; None = run to the end

# ---- Detector (passed to YOLODetector) -------------------------------------
WEIGHTS = None  # None = yolo26n.pt, or yoloe-26n-seg.pt when PROMPTS is set
PROMPTS = ["pothole", "hole with water"]  # YOLOE text prompts, e.g. ["pothole", "stairs"]; None = YOLO26
CONFIDENCE = 0.10  # keep boxes with confidence >= this (0 to 1)
IMAGE_SIZE = 640  # model input size, long side in pixels
DEVICE = None  # None = automatic, "cpu", or "0"
HALF = True  # FP16 on GPU
CLASSES = None  # e.g. ["person", "car"]; None = all classes

# ---- Timing ----------------------------------------------------------------
WARMUP_RUNS = 3  # untimed runs before timing an image (excludes GPU start-up)
PRINT_EVERY_N_FRAMES = 30  # status line frequency for videos

# ---- Outputs ---------------------------------------------------------------
SAVE_IMAGE = None  # path for the annotated image, or None
SAVE_VIDEO = None  # path for the annotated .mp4, or None
JSONL = None  # path for per-frame detections, or None
DISPLAY = True  # show a window
FALLBACK_FPS = 30.0  # used when a webcam reports 0 FPS

# ---- Drawing (colors are BGR) ----------------------------------------------
BOX_COLOR = (0, 200, 0)
INFO_COLOR = (0, 220, 255)
WINDOW_MAX_WIDTH = 1600  # before/after window is scaled to fit this width

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================================================


def draw_detections(image, prediction: FrameDetections):
    """Draw detections on a COPY to keep the original frame unchanged."""
    annotated = image.copy()
    for item in prediction.detections:
        x1, y1, x2, y2 = item.bbox_xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        caption = f"{item.class_name}: {item.confidence:.2f}"
        cv2.putText(
            annotated,
            caption,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
        )
    cv2.putText(
        annotated,
        f"{len(prediction.detections)} objects | {prediction.total_ms:.0f} ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        INFO_COLOR,
        2,
    )
    return annotated


def write_jsonl(stream: TextIO | None, frame_id: int, result: FrameDetections) -> None:
    """Store detection metadata, not raw images, in newline-delimited JSON."""
    if stream is None:
        return
    record = {
        "frame_id": frame_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **result.to_dict(),
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(detector: YOLODetector, args: argparse.Namespace, result: FrameDetections) -> None:
    """Print settings, latency breakdown and detections in readable form."""
    model = detector.model
    try:
        parameter = next(model.model.parameters())
        device, dtype = str(parameter.device), str(parameter.dtype).replace("torch.", "")
    except (AttributeError, StopIteration, TypeError):
        device, dtype = "unknown", "unknown"

    print("=== Settings ===")
    print(f"weights      : {detector.weights_path}")
    print(f"text prompts : {detector.text_prompts or 'none (fixed COCO classes)'}")
    print(f"device       : {device}")
    print(f"precision    : {dtype} (half={detector.half})")
    print(f"image size   : {args.image_size}")
    print(f"confidence   : {args.confidence}")
    print(f"classes      : {detector.class_ids if detector.class_ids is not None else 'all'}")
    print(f"warm-up runs : {args.warmup}")

    print("\n=== Latency (ms) ===")
    print(f"preprocess   : {result.preprocess_ms}")
    print(f"model        : {result.model_ms}")
    print(f"postprocess  : {result.postprocess_ms}")
    print(f"total call   : {result.total_ms}")

    print(f"\n=== Detections: {len(result.detections)} (frame {result.width}x{result.height}) ===")
    for i, d in enumerate(result.detections, 1):
        print(
            f"{i:>2}. {d.class_name:<15} id={d.class_id:<3} conf={d.confidence:.3f} "
            f"box={d.bbox_xyxy} center={d.center_xy}"
        )


def window_is_open(window: str) -> bool:
    """True while the OpenCV window exists (False after the user clicks X)."""
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def wait_until_closed(window: str) -> None:
    """Wait for any key or for the window to be closed.

    cv2.waitKey(0) alone blocks forever if the window is closed with X, and
    Ctrl+C in the terminal cannot interrupt it. Polling every 50 ms fixes both.
    """
    try:
        while window_is_open(window):
            if cv2.waitKey(50) != -1:  # any key pressed
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind Guide: YOLO26n / YOLOE-26 detection")
    parser.add_argument("--source", default=SOURCE, help="Camera index, image, or video path")
    parser.add_argument(
        "--weights", default=WEIGHTS,
        help="Weights file/model name (default yolo26n.pt, or yoloe-26n-seg.pt with --prompts)",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=PROMPTS,
        help='YOLOE text prompts, e.g. --prompts pothole puddle "traffic cone". Default: plain YOLO26',
    )
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default=DEVICE, help="cpu or GPU index 0; default automatic")
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help="Untimed warm-up runs before measuring an image (default 3)",
    )
    parser.add_argument(
        "--half", action=argparse.BooleanOptionalAction, default=HALF,
        help="FP16 on GPU (default on); use --no-half for FP32",
    )
    parser.add_argument(
        "--classes", nargs="+", default=CLASSES,
        help="Only keep these classes, by name or id (e.g. person car 9). Default: all",
    )
    parser.add_argument("--jsonl", default=JSONL, help="Optional JSONL detections file")
    parser.add_argument("--save-image", default=SAVE_IMAGE, help="Save annotated output for image source")
    parser.add_argument("--save-video", default=SAVE_VIDEO, help="Save annotated output video (.mp4) for video/webcam source")
    parser.add_argument("--no-display", action="store_true", default=not DISPLAY, help="Run without an OpenCV window")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Limit video frames for testing")
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    detector = YOLODetector(
        weights=args.weights,
        text_prompts=args.prompts,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        classes=args.classes,
        half=args.half,
    )

    output_file = None
    try:
        if args.jsonl:
            path = Path(args.jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            output_file = path.open("w", encoding="utf-8")

        source = args.source
        if Path(source).is_file() and Path(source).suffix.lower() in IMAGE_EXTENSIONS:
            image = cv2.imread(source)
            if image is None:
                raise RuntimeError(f"Could not read image: {source}")
            # Untimed warm-up runs, so the reported latency excludes one-off
            # GPU/CUDA initialisation.
            for _ in range(args.warmup):
                detector.predict(image)
            result = detector.predict(image)
            write_jsonl(output_file, 0, result)
            print_summary(detector, args, result)
            print("\nFull result (JSON):")
            print(json.dumps(result.to_dict(), indent=2))
            annotated = draw_detections(image, result)
            if args.save_image:
                destination = Path(args.save_image)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), annotated):
                    raise RuntimeError(f"Could not save image: {destination}")
                print(f"Annotated image saved to {destination}")
            if not args.no_display:
                # Before (left) and after (right), in one resizable window.
                before = image.copy()
                cv2.putText(before, "Before", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)
                separator = np.full((image.shape[0], 8, 3), 255, dtype=image.dtype)
                comparison = np.hstack([before, separator, annotated])
                window = "Blind Guide | before (left) / after (right), any key to close"
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, min(comparison.shape[1], WINDOW_MAX_WIDTH), int(min(comparison.shape[1], WINDOW_MAX_WIDTH) * comparison.shape[0] / comparison.shape[1]))
                cv2.imshow(window, comparison)
                wait_until_closed(window)
            return

        capture_source = int(source) if source.isdecimal() else source
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera/video: {source}")

        writer = None
        if args.save_video:
            video_path = Path(args.save_video)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = FALLBACK_FPS  # webcams often report 0
            size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video file: {video_path}")

        frame_id = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = detector.predict(frame)
                write_jsonl(output_file, frame_id, result)

                if frame_id % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"frame={frame_id}, detections={len(result.detections)}, "
                        f"total={result.total_ms:.1f} ms "
                        f"(pre={result.preprocess_ms} model={result.model_ms} "
                        f"post={result.postprocess_ms} ms)"
                    )
                if writer is not None or not args.no_display:
                    annotated = draw_detections(frame, result)
                    if writer is not None:
                        writer.write(annotated)
                    if not args.no_display:
                        video_window = "Blind Guide | detection (press Q to quit)"
                        cv2.imshow(video_window, annotated)
                        key = cv2.waitKey(1) & 0xFF
                        # Stop on Q, Esc, or when the window is closed with X.
                        if key in (ord("q"), 27) or not window_is_open(video_window):
                            break
                frame_id += 1
                if args.max_frames is not None and frame_id >= args.max_frames:
                    break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
                print(f"Annotated video saved to {args.save_video}")
            if not args.no_display:
                cv2.destroyAllWindows()
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
