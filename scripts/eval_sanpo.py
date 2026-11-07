"""Evaluate a pretrained depth model on the downloaded SANPO frames.

Examples:
    python scripts/eval_sanpo.py --model da2-small-metric-outdoor
    python scripts/eval_sanpo.py --model da2-small                     # relative -> scale/shift aligned
    python scripts/eval_sanpo.py --model da2-small-metric-outdoor --input-size 364 --limit 20

Outputs (in --out, default outputs/eval_<model>_<split>):
    per_frame.csv   one row per frame and alignment mode
    summary.json    mean of every metric + latency + config
    vis/*.jpg       RGB | GT | prediction | abs-rel error for the first --save-vis frames

Alignment modes:
    none    raw metric prediction (what the phone actually gets) -- metric models only
    median  per-frame median scaling (isolates structure error from global scale error)
    ssi     scale+shift fit in inverse-depth space -- the standard protocol for relative models
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from depth import sanpo  # noqa: E402
from depth.metrics import align_disparity_ssi, align_median, frame_report  # noqa: E402
from depth.models import DepthEstimator, colorize_depth  # noqa: E402


def save_vis(path: Path, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, vmax: float) -> None:
    h, w = gt.shape
    rgb_s = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    gt_v = colorize_depth(np.where(mask, gt, 0), True, vmax)
    pr_v = colorize_depth(pred, True, vmax)
    err = np.where(mask, np.abs(pred - gt) / np.maximum(gt, 1e-6), 0)
    err_v = cv2.applyColorMap((np.clip(err / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    err_v[~mask] = 0
    tiles = [rgb_s, gt_v, pr_v, err_v]
    for t, label in zip(tiles, ["RGB", f"GT (0-{vmax:g} m)", "prediction", "abs-rel error (0-50%)"]):
        cv2.putText(t, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(path), cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/sanpo")
    ap.add_argument("--split", default="sanpo-real", choices=sanpo.SPLITS)
    ap.add_argument("--camera", default=None, help="chest / head (default: any)")
    ap.add_argument("--model", default="da2-small-metric-outdoor", help="short name from depth.MODELS or HF id")
    ap.add_argument("--input-size", type=int, default=518, help="short side of the network input")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gt", default="depth_maps", choices=sanpo.GT_KINDS)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=30.0, help="ignore GT beyond this (m)")
    ap.add_argument("--align", nargs="+", choices=["none", "median", "ssi"],
                    help="default: none+median for metric models, ssi for relative")
    ap.add_argument("--limit", type=int, default=0, help="evaluate at most N frames (0 = all)")
    ap.add_argument("--save-vis", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames = sanpo.index_local(args.root, args.split, args.camera, args.gt)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        sys.exit(f"No frames with {args.gt} under {args.root}/{args.split}. Run scripts/download_sanpo.py first.")

    est = DepthEstimator(args.model, args.device, args.input_size, args.fp16)
    modes = args.align or (["none", "median"] if est.is_metric else ["ssi"])
    if "none" in modes and not est.is_metric:
        sys.exit("--align none needs a metric model (relative output has no unit)")
    print(f"model {est.repo_id} ({est.num_params / 1e6:.1f}M params, "
          f"{'metric' if est.is_metric else 'relative'}) on {est.device}; "
          f"{len(frames)} frames; align={modes}")

    out = Path(args.out or f"outputs/eval_{args.model.replace('/', '_')}_{args.split}")
    (out / "vis").mkdir(parents=True, exist_ok=True)

    rows, latencies = [], []
    for i, fr in enumerate(tqdm(frames, desc="eval", unit="frame")):
        rgb = sanpo.read_rgb(fr.rgb)
        gt = sanpo.read_depth(fr.gt_path(args.gt))
        mask = np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
        if mask.sum() < 100:
            continue
        raw = est.predict(rgb, out_size=gt.shape)
        if i > 0:  # first call includes warm-up
            latencies.append(est.last_latency_ms)

        obstacles = None
        if fr.seg is not None:
            sem = sanpo.read_semantic(fr.seg)
            if sem.shape != gt.shape:
                sem = cv2.resize(sem, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            obstacles = np.isin(sem, sanpo.OBSTACLE_IDS)

        for mode in modes:
            if mode == "none":
                pred = raw
            elif mode == "median":
                pred = align_median(raw if est.is_metric else 1.0 / np.maximum(raw, 1e-6), gt, mask)
            else:
                disp = 1.0 / np.maximum(raw, 1e-3) if est.is_metric else raw
                pred = align_disparity_ssi(disp, gt, mask, args.min_depth, args.max_depth)
            rows.append({"key": fr.key, "align": mode} | frame_report(pred, gt, mask, obstacles))

        if i < args.save_vis:
            vis_pred = raw if est.is_metric else pred
            save_vis(out / "vis" / f"{i:03d}_{fr.session[:8]}_{fr.frame_id}.jpg", rgb, gt, vis_pred, mask, 10.0)

    if not rows:
        sys.exit("No frame had enough valid GT pixels.")

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "model": est.repo_id, "params_M": round(est.num_params / 1e6, 2), "metric": est.is_metric,
        "split": args.split, "gt": args.gt, "depth_range_m": [args.min_depth, args.max_depth],
        "input_size": args.input_size, "input_shape_for_sanpo": est.input_shape(1242, 2208),
        "device": str(est.device), "n_frames": len({r["key"] for r in rows}),
        "latency_ms_mean": round(float(np.mean(latencies)), 1) if latencies else None,
        "results": {},
    }
    for mode in modes:
        sub = [r for r in rows if r["align"] == mode]
        keys = [k for k in fields if k not in ("key", "align", "n_px")
                and any(np.isfinite(r.get(k, np.nan)) for r in sub)]
        summary["results"][mode] = {k: round(float(np.nanmean([r.get(k, np.nan) for r in sub])), 4) for k in keys}
        # how many frames actually had pixels for each metric (bands / obstacles can be sparse)
        summary.setdefault("frames_per_metric", {k: int(np.sum(np.isfinite([r.get(k, np.nan) for r in sub])))
                                                 for k in keys})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{summary['n_frames']} frames, mean latency {summary['latency_ms_mean']} ms "
          f"(input {summary['input_shape_for_sanpo']}, {summary['device']})")
    for mode, res in summary["results"].items():
        print(f"[{mode}] " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
