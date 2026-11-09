"""Leave-one-SANPO-session-out test of a fixed metric-depth scale correction.

This is a diagnostic for model calibration, NOT a deployable safety system.
For each held-out SANPO session, derive ONE constant scale from all OTHER
sessions' reference depth. Never use held-out ground truth to calculate scale.

Run from the blind-guide repository root:
    python fast_path/depth/scripts/evaluate_fixed_scale_cv.py --device cuda

Requires the existing fast_path package and already downloaded SANPO frames.
Prints session-wise raw-vs-calibrated metrics; saves a per-frame CSV under outputs/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from fast_path.depth import sanpo
from fast_path.depth.metrics import depth_metrics
from fast_path.depth.models import DepthEstimator


def mean_finite(values: list[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")


def median_finite(values: list[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def near_false_far(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, cutoff: float) -> tuple[float, int]:
    """Among GT pixels nearer than cutoff, fraction predicted >= cutoff.

    This is a PIXEL-LEVEL threshold statistic, not an obstacle miss rate.
    """
    near = mask & (gt < cutoff)
    count = int(near.sum())
    return (float(np.mean(pred[near] >= cutoff)) if count else float("nan"), count)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/sanpo")
    ap.add_argument("--split", default="sanpo-real")
    ap.add_argument("--camera", default="chest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=30.0)
    ap.add_argument("--near-cutoff", type=float, default=2.0)
    ap.add_argument("--out", default="outputs/depth_sanpo_metric_diagnostics/fixed_scale_cv.csv")
    args = ap.parse_args()
    if not (0 < args.min_depth < args.near_cutoff < args.max_depth):
        ap.error("Require 0 < min-depth < near-cutoff < max-depth")

    frames = sanpo.index_local(args.root, split=args.split, camera=args.camera, gt="depth_maps")
    sessions = sorted({f.session for f in frames})
    if len(sessions) < 3:
        sys.exit("Need at least THREE distinct sessions to run session-held-out calibration.")
    print(f"Found {len(frames)} frames across {len(sessions)} distinct sessions.")
    print("Pass 1/2: Estimate per-frame calibration scales from reference depth.")

    estimator = DepthEstimator("da2-small-metric-outdoor", device=args.device,
                               input_size=args.input_size)
    if not estimator.is_metric:
        sys.exit("Expected metric checkpoint")

    scales_by_session: dict[str, list[float]] = defaultdict(list)
    usable_frames = []
    for i, fr in enumerate(frames, 1):
        rgb = sanpo.read_rgb(fr.rgb)
        gt = sanpo.read_depth(fr.gt_path("depth_maps"))
        pred = estimator.predict(rgb, out_size=gt.shape)
        valid = (np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
                 & np.isfinite(pred) & (pred > 0))
        if valid.sum() < 100:
            continue
        factor = float(np.median(gt[valid]) / np.median(pred[valid]))
        if not np.isfinite(factor) or factor <= 0:
            continue
        scales_by_session[fr.session].append(factor)
        usable_frames.append(fr)
        if i % 10 == 0:
            print(f"  collected scales: {i}/{len(frames)}")

    if len(scales_by_session) < 3:
        sys.exit("Fewer than three usable sessions had sufficient valid depth.")

    # Session balance: one median scale per training session, then median across
    # the OTHER sessions. The held-out session contributes no GT to its factor.
    session_scales = {s: median_finite(v) for s, v in scales_by_session.items()}
    heldout_scale = {
        s: median_finite([v for other, v in session_scales.items() if other != s])
        for s in session_scales
    }
    print("\nCalibrations (derived EXCLUSIVELY from other sessions):")
    for s in sorted(session_scales):
        print(f"  held-out {s[:12]}: factor={heldout_scale[s]:.4f}  "
              f"(for context, held-out own median factor={session_scales[s]:.4f}; "
              "own factor is NOT used)")

    print("\nPass 2/2: Compare RAW vs CALIBRATED predictions on held-out sessions.")
    rows = []
    for i, fr in enumerate(usable_frames, 1):
        rgb = sanpo.read_rgb(fr.rgb)
        gt = sanpo.read_depth(fr.gt_path("depth_maps"))
        raw = estimator.predict(rgb, out_size=gt.shape)
        valid = (np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
                 & np.isfinite(raw) & (raw > 0))
        if valid.sum() < 100:
            continue
        calibrated = raw * heldout_scale[fr.session]
        baseline = depth_metrics(raw, gt, valid)
        adjusted = depth_metrics(calibrated, gt, valid)
        raw_miss, near_count = near_false_far(raw, gt, valid, args.near_cutoff)
        cal_miss, _ = near_false_far(calibrated, gt, valid, args.near_cutoff)
        rows.append({
            "key": fr.key,
            "session": fr.session,
            "calibration_scale_from_other_sessions": round(heldout_scale[fr.session], 6),
            "raw_abs_rel": baseline["abs_rel"],
            "calibrated_abs_rel": adjusted["abs_rel"],
            "raw_rmse_m": baseline["rmse"],
            "calibrated_rmse_m": adjusted["rmse"],
            "raw_d1": baseline["d1"],
            "calibrated_d1": adjusted["d1"],
            "gt_near_pixel_count": near_count,
            "raw_near_false_far_fraction": raw_miss,
            "calibrated_near_false_far_fraction": cal_miss,
        })
        if i % 10 == 0:
            print(f"  evaluated: {i}/{len(usable_frames)}")

    if not rows:
        sys.exit("No valid evaluation frames")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPER-SESSION RESULTS (mean of frame metrics; lower AbsRel is better):")
    print(f"{'Session':13s} {'RawAbsRel':>10s} {'CalAbsRel':>10s} {'RawD1':>8s} {'CalD1':>8s} {'NearRaw':>9s} {'NearCal':>9s}")
    for session in sorted(session_scales):
        rr = [r for r in rows if r["session"] == session]
        def mean(key: str) -> float:
            return mean_finite([r[key] for r in rr])
        print(f"{session[:12]:13s} {mean('raw_abs_rel'):10.3f} {mean('calibrated_abs_rel'):10.3f} "
              f"{mean('raw_d1'):8.3f} {mean('calibrated_d1'):8.3f} "
              f"{mean('raw_near_false_far_fraction'):9.3f} "
              f"{mean('calibrated_near_false_far_fraction'):9.3f}")
    print("\nMEAN ACROSS ALL HELD-OUT FRAMES:")
    for before, after in (("raw_abs_rel", "calibrated_abs_rel"),
                          ("raw_rmse_m", "calibrated_rmse_m"),
                          ("raw_d1", "calibrated_d1"),
                          ("raw_near_false_far_fraction", "calibrated_near_false_far_fraction")):
        print(f"  {before}: {mean_finite([r[before] for r in rows]):.4f} -> "
              f"{after}: {mean_finite([r[after] for r in rows]):.4f}")
    print(f"\nSaved {out.resolve()}")
    print("WARNING: This tests only fixed-scale correction and raw meter prediction.")
    print("It is NOT a validation of reliable obstacle warnings or navigation safety.")
    print("Further independent-session evaluation is needed before model selection.")


if __name__ == "__main__":
    main()
