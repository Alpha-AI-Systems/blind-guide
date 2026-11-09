"""Compare our DepthEstimator preprocessing with the checkpoint's official HF processor.

Usage, from the blind-guide repository root:
    python fast_path/depth/scripts/verify_depth_preprocessing.py --device cuda
    python fast_path/depth/scripts/verify_depth_preprocessing.py --device cuda --all

Only uses existing SANPO files and the cached pretrained model. Does NOT train.

The current project wrapper applies custom bicubic resize and bilinear depth
upsampling. The official path uses AutoImageProcessor and bicubic depth
upsampling per the model card. This checks if that difference materially changes
raw depth errors before considering scale calibration or fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor

from fast_path.depth import sanpo
from fast_path.depth.metrics import depth_metrics
from fast_path.depth.models import DepthEstimator


DEFAULT_FRAMES = (
    ("DVKpZk7Q", "000015"),
    ("DVKpZk7Q", "000075"),
    ("wweXnLBK", "000105"),
)


def raw_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Raw metric-depth metrics on valid GT depth values (no GT alignment)."""
    valid = (
        np.isfinite(gt) & (gt > 0.1) & (gt < 30)
        & np.isfinite(pred) & (pred > 0)
    )
    if valid.sum() < 100:
        raise ValueError("Too few pixels with valid GT and positive prediction")
    m = depth_metrics(pred, gt, valid)
    gt_v, pr_v = gt[valid], pred[valid]
    near = valid & (gt < 2.0)
    return {
        "abs_rel": m["abs_rel"],
        "rmse_m": m["rmse"],
        "d1": m["d1"],
        "median_pred_gt_ratio": float(np.median(pr_v / gt_v)),
        "median_signed_error_m": float(np.median(pr_v - gt_v)),
        "near_gt_px": int(near.sum()),
        "near_pred_ge_2m_pct": float(100 * np.mean(pred[near] >= 2.0)) if near.any() else float("nan"),
        "valid_px": int(valid.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/sanpo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--all", action="store_true", help="Evaluate all downloaded chest frames, not just three")
    ap.add_argument("--out", default="outputs/depth_sanpo_metric_diagnostics/preprocessing_check.csv")
    args = ap.parse_args()

    frames = sanpo.index_local(args.root, split="sanpo-real", camera="chest", gt="depth_maps")
    if not args.all:
        frames = [f for f in frames if any(
            f.session.startswith(prefix) and f.frame_id == fid for prefix, fid in DEFAULT_FRAMES
        )]
        if len(frames) != len(DEFAULT_FRAMES):
            sys.exit(f"Expected {len(DEFAULT_FRAMES)} selected frames, found {len(frames)}. Check {args.root!r}.")
    if not frames:
        sys.exit(f"No valid SANPO frames found under {args.root!r}.")

    estimator = DepthEstimator("da2-small-metric-outdoor", device=args.device,
                               input_size=args.input_size)
    if not estimator.is_metric:
        sys.exit("Expected the outdoor metric-depth model.")
    processor = AutoImageProcessor.from_pretrained(estimator.repo_id)
    print(f"Comparing custom wrapper vs official HF processor for {len(frames)} frames.")
    print(f"HF processor: {processor.__class__.__name__}")
    rows = []

    for fr in frames:
        rgb = sanpo.read_rgb(fr.rgb)
        gt = sanpo.read_depth(fr.gt_path("depth_maps"))

        # Existing team's custom preprocessing + bilinear depth upsampling.
        custom = estimator.predict(rgb, out_size=gt.shape)

        # Official Hugging Face preprocessing + model-card bicubic depth upsampling.
        # Keep the same model weights and output resolution for a fair comparison.
        input_kwargs = {"images": Image.fromarray(rgb), "return_tensors": "pt"}
        if args.input_size != 518:
            input_kwargs["size"] = {"height": args.input_size, "width": args.input_size}
        official_input = processor(**input_kwargs)
        tensor_inputs = {k: v.to(estimator.device) for k, v in official_input.items()}
        with torch.inference_mode():
            output = estimator.model(**tensor_inputs).predicted_depth
            official = F.interpolate(output.unsqueeze(1).float(), size=gt.shape,
                                     mode="bicubic", align_corners=False)
            official = official[0, 0].cpu().numpy()

        for name, pred in (("custom", custom), ("official", official)):
            metrics = raw_metrics(pred, gt)
            rows.append({"key": fr.key, "pipeline": name, **metrics})
            print(f"{fr.session[:8]}/{fr.frame_id:>6s} {name:8s} "
                  f"AbsRel={metrics['abs_rel']:.3f} "
                  f"d1={metrics['d1']:.3f} "
                  f"ratio={metrics['median_pred_gt_ratio']:.3f} "
                  f"near_miss={metrics['near_pred_ge_2m_pct']:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nMean of per-frame scores (not pooled pixel scores):")
    for name in ("custom", "official"):
        subset = [r for r in rows if r["pipeline"] == name]
        print(f"  {name:8s} AbsRel={np.mean([r['abs_rel'] for r in subset]):.3f} "
              f"d1={np.mean([r['d1'] for r in subset]):.3f} "
              f"median-ratio={np.median([r['median_pred_gt_ratio'] for r in subset]):.3f}")
    print(f"\nSaved {out.resolve()}")
    print("Reminder: This is a raw-metric diagnostic, not a navigation-safety evaluation.")


if __name__ == "__main__":
    main()
