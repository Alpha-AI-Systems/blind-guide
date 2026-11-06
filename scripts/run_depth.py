"""Run a depth model on an image, a folder of images, a video file, or a webcam.

Examples:
    python scripts/run_depth.py --source path/to/street.jpg
    python scripts/run_depth.py --source path/to/frames_dir
    python scripts/run_depth.py --source walk.mp4 --input-size 364
    python scripts/run_depth.py --source 0 --show                 # webcam 0, live window (q to quit)

Each output is RGB | colorized depth, with the latency and (for metric models) the
distance to the closest thing in the walking corridor -- the bottom-centre region in
front of the user, which is what the warning rules will look at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from depth.models import DepthEstimator, colorize_depth  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def corridor_distance(depth_m: np.ndarray, width_frac: float = 0.4, top_frac: float = 0.35) -> float:
    """5th-percentile depth in the bottom-centre box (the path straight ahead)."""
    h, w = depth_m.shape
    x0, x1 = int(w * (0.5 - width_frac / 2)), int(w * (0.5 + width_frac / 2))
    region = depth_m[int(h * top_frac):, x0:x1]
    return float(np.percentile(region, 5))


def render(est: DepthEstimator, bgr: np.ndarray, vmax: float) -> np.ndarray:
    depth = est.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    vis = colorize_depth(depth, est.is_metric, vmax)
    text = f"{est.last_latency_ms:.0f} ms"
    if est.is_metric:
        h, w = depth.shape
        cv2.rectangle(vis, (int(w * 0.3), int(h * 0.35)), (int(w * 0.7), h - 1), (255, 255, 255), 2)
        text += f" | corridor: {corridor_distance(depth):.1f} m"
    cv2.putText(vis, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([bgr, vis])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image, folder, video file, or webcam index")
    ap.add_argument("--model", default="da2-small-metric-outdoor")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--vmax", type=float, default=10.0, help="color scale max (m) for metric models")
    ap.add_argument("--max-width", type=int, default=1280, help="downscale big inputs before inference")
    ap.add_argument("--out", default="outputs/run_depth")
    ap.add_argument("--show", action="store_true", help="open a live window")
    args = ap.parse_args()

    est = DepthEstimator(args.model, args.device, args.input_size, args.fp16)
    print(f"{est.repo_id}: {est.num_params / 1e6:.1f}M params, "
          f"{'metric (m)' if est.is_metric else 'relative'}, device={est.device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def shrink(bgr: np.ndarray) -> np.ndarray:
        if bgr.shape[1] <= args.max_width:
            return bgr
        s = args.max_width / bgr.shape[1]
        return cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    src = Path(args.source)
    if src.is_dir() or (src.is_file() and src.suffix.lower() in IMG_EXT):
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
        lat = []
        for p in files:
            frame = render(est, shrink(cv2.imread(str(p))), args.vmax)
            lat.append(est.last_latency_ms)
            cv2.imwrite(str(out / f"{p.stem}_depth.jpg"), frame)
            print(f"{p.name}: {est.last_latency_ms:.0f} ms")
            if args.show:
                cv2.imshow("depth", frame)
                cv2.waitKey(0)
        if len(lat) > 1:
            print(f"mean latency (excluding first): {np.mean(lat[1:]):.0f} ms")
    else:
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        if not cap.isOpened():
            sys.exit(f"Cannot open {args.source}")
        writer, n, lat = None, 0, []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame = render(est, shrink(bgr), args.vmax)
            lat.append(est.last_latency_ms)
            n += 1
            if writer is None:
                name = "webcam" if args.source.isdigit() else Path(args.source).stem
                fps = cap.get(cv2.CAP_PROP_FPS) or 15
                writer = cv2.VideoWriter(str(out / f"{name}_depth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            if args.show:
                cv2.imshow("depth", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if writer:
            writer.release()
        if lat:
            print(f"{n} frames, mean latency {np.mean(lat[1:] or lat):.0f} ms")
    if args.show:
        cv2.destroyAllWindows()
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
