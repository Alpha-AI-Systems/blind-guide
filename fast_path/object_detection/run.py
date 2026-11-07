"""Run YOLO26n on a webcam, video, or image.

Run from the repository root:
    python -m fast_path.object_detection.run --source 0
    python -m fast_path.object_detection.run --source path/to/video.mp4
    python -m fast_path.object_detection.run --source path/to/image.jpg --save-image out.jpg
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import cv2

from .detector import FrameDetections, YOLODetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_detections(image, prediction: FrameDetections):
    """Draw detections on a COPY to keep the original frame unchanged."""
    annotated = image.copy()
    for item in prediction.detections:
        x1, y1, x2, y2 = item.bbox_xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        caption = f"{item.class_name}: {item.confidence:.2f}"
        cv2.putText(
            annotated,
            caption,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
        )
    cv2.putText(
        annotated,
        f"YOLO26n | {len(prediction.detections)} objects | {prediction.inference_ms:.0f} ms",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 220, 255),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind Guide — YOLO26n detection")
    parser.add_argument("--source", default="0", help="Camera index, image, or video path")
    parser.add_argument("--weights", default="yolo26n.pt", help="Weights file/model name")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default=None, help="cpu or GPU index 0; default automatic")
    parser.add_argument("--jsonl", default=None, help="Optional JSONL detections file")
    parser.add_argument("--save-image", default=None, help="Save annotated output for image source")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit video frames for testing")
    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    detector = YOLODetector(
        weights=args.weights,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
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
            result = detector.predict(image)
            write_jsonl(output_file, 0, result)
            print(json.dumps(result.to_dict(), indent=2))
            annotated = draw_detections(image, result)
            if args.save_image:
                destination = Path(args.save_image)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), annotated):
                    raise RuntimeError(f"Could not save image: {destination}")
                print(f"Annotated image saved to {destination}")
            if not args.no_display:
                cv2.imshow("Blind Guide | YOLO26n", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return

        capture_source = int(source) if source.isdecimal() else source
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera/video: {source}")

        frame_id = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = detector.predict(frame)
                write_jsonl(output_file, frame_id, result)

                if frame_id % 30 == 0:
                    print(
                        f"frame={frame_id}, detections={len(result.detections)}, "
                        f"inference={result.inference_ms:.1f} ms"
                    )
                if not args.no_display:
                    annotated = draw_detections(frame, result)
                    cv2.imshow("Blind Guide | YOLO26n (press Q to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_id += 1
                if args.max_frames is not None and frame_id >= args.max_frames:
                    break
        finally:
            capture.release()
            if not args.no_display:
                cv2.destroyAllWindows()
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
