"""Inspect raw metric-depth distance bias on selected downloaded SANPO frames.

Place in fast_path/depth/scripts/ and run from the project root:
    python fast_path/depth/scripts/diagnose_depth_bias.py --device cuda

Reports median predicted/reference ratio, signed median error, and a pixel-level
near-distance miss indicator. These diagnostics are NOT object detection metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# This script lives in <repo>/fast_path/depth/scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from fast_path.depth import sanpo  # noqa: E402
from fast_path.depth.models import DepthEstimator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/sanpo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="da2-small-metric-outdoor")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--frames", nargs="+",
        default=["DVKpZk7Q:000015", "DVKpZk7Q:000075", "wweXnLBK:000105"],
        help="Session prefix and frame ID, e.g. DVKpZk7Q:000075",
    )
    parser.add_argument("--out", default="outputs/depth_sanpo_metric_diagnostics/selected_bias.csv")
    args = parser.parse_args()

    requested = []
    for selector in args.frames:
        if ":" not in selector:
            parser.error(f"Invalid selector {selector!r}; use SESSION_PREFIX:FRAME_ID")
        requested.append(tuple(selector.split(":", 1)))

    frames = sanpo.index_local(args.root, split="sanpo-real", camera="chest", gt="depth_maps")
    selected = [
        frame for frame in frames
        if any(frame.session.startswith(prefix) and frame.frame_id == fid for prefix, fid in requested)
    ]
    if len(selected) != len(requested):
        found = [f"{f.session[:8]}:{f.frame_id}" for f in selected]
        sys.exit(f"Expected {len(requested)} frames; found {len(selected)}: {found}. Check --root/--frames.")

    estimator = DepthEstimator(model=args.model, device=args.device, input_size=args.input_size)
    if not estimator.is_metric:
        sys.exit("This diagnostic expects raw metric predictions in meters, not relative depth.")

    output_rows = []
    for frame in selected:
        gt = sanpo.read_depth(frame.gt_path("depth_maps"))
        rgb = sanpo.read_rgb(frame.rgb)
        predicted = estimator.predict(rgb, out_size=gt.shape)
        valid = (np.isfinite(gt) & (gt > 0.1) & (gt < 30.0)
                 & np.isfinite(predicted) & (predicted > 0))
        if not np.any(valid):
            continue

        gt_v = gt[valid]
        pred_v = predicted[valid]
        near = valid & (gt < 2.0)
        row = {
            "frame": frame.key,
            "valid_pixels": int(valid.sum()),
            "gt_median_m": round(float(np.median(gt_v)), 3),
            "pred_median_m": round(float(np.median(pred_v)), 3),
            "median_pred_gt_ratio": round(float(np.median(pred_v / gt_v)), 3),
            "median_signed_error_m": round(float(np.median(pred_v - gt_v)), 3),
            "near_gt_pixels": int(near.sum()),
            "near_gt_lt2m_pred_ge2m_pct": (
                round(float(100 * np.mean(predicted[near] >= 2.0)), 1)
                if np.any(near) else "N/A"
            ),
        }
        output_rows.append(row)
        print(f"\n{row['frame']}")
        print(f"  Ground-truth median distance: {row['gt_median_m']} m")
        print(f"  Predicted median distance:    {row['pred_median_m']} m")
        print(f"  Median prediction/GT ratio:  {row['median_pred_gt_ratio']} (1.0 is ideal)")
        print(f"  Median signed error:         {row['median_signed_error_m']} m (positive = predicts too far)")
        print(f"  GT pixels closer than 2m:    {row['near_gt_pixels']}")
        print(f"  Near pixels predicted >=2m:  {row['near_gt_lt2m_pred_ge2m_pct']}%")

    if not output_rows:
        sys.exit("No valid reference-depth pixels in the selected frames.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main()
