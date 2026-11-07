"""Standard monocular-depth metrics + the alignment needed for relative models.

All functions take float arrays in meters and a boolean validity mask.
"""

from __future__ import annotations

import numpy as np

# Distance bands that matter for a walking user: immediate / near / mid / far.
RANGE_BINS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, float("inf"))]


def align_median(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Scale-only alignment (median ratio). Removes global scale error of a metric model."""
    s = np.median(gt[mask]) / max(np.median(pred[mask]), 1e-6)
    return pred * s


def align_disparity_ssi(pred_disp: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                        min_depth: float, max_depth: float) -> np.ndarray:
    """Relative models output affine-invariant inverse depth: fit s, t so that s*pred + t ~ 1/gt
    (least squares over valid pixels), then invert to get depth in meters."""
    x = pred_disp[mask].astype(np.float64)
    y = 1.0 / gt[mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    disp = np.clip(s * pred_disp + t, 1.0 / max_depth, 1.0 / min_depth)
    return (1.0 / disp).astype(np.float32)


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """AbsRel, SqRel, RMSE, RMSE_log, delta<1.25^k over masked pixels (nan if mask is empty)."""
    keys = ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")
    if mask.sum() == 0:
        return {k: float("nan") for k in keys} | {"n_px": 0}
    p = np.maximum(pred[mask].astype(np.float64), 1e-6)
    g = gt[mask].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
        "d2": float(np.mean(ratio < 1.25**2)),
        "d3": float(np.mean(ratio < 1.25**3)),
        "n_px": int(mask.sum()),
    }


def frame_report(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                 obstacle_mask: np.ndarray | None = None) -> dict[str, float]:
    """Overall metrics, plus AbsRel / d1 per distance band and on obstacle pixels."""
    out = depth_metrics(pred, gt, mask)
    for lo, hi in RANGE_BINS:
        tag = f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}m+"
        m = depth_metrics(pred, gt, mask & (gt >= lo) & (gt < hi))
        out[f"abs_rel@{tag}"], out[f"d1@{tag}"] = m["abs_rel"], m["d1"]
    if obstacle_mask is not None:
        m = depth_metrics(pred, gt, mask & obstacle_mask)
        out["abs_rel@obstacles"], out["d1@obstacles"] = m["abs_rel"], m["d1"]
    return out
