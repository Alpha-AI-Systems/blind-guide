"""Pretrained monocular depth models (Depth Anything V2 via Hugging Face transformers).

Two kinds of model are supported:
  * metric   -> output is depth in meters (what the rules / tracker need on the phone)
  * relative -> output is affine-invariant inverse depth ("disparity"); only the
                ordering / shape is meaningful, it must be aligned to GT to get meters.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Short name -> Hugging Face repo id. Add new checkpoints here.
MODELS = {
    "da2-small": "depth-anything/Depth-Anything-V2-Small-hf",  # 24.8M, relative
    "da2-small-metric-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",  # meters, max 80 m
    "da2-small-metric-indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",  # meters, max 20 m
    "da2-base": "depth-anything/Depth-Anything-V2-Base-hf",  # 97.5M, relative
    "da2-base-metric-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14  # ViT patch size; input sides must be multiples of it


def _round_to_patch(x: float) -> int:
    return max(PATCH, int(round(x / PATCH)) * PATCH)


class DepthEstimator:
    """Thin wrapper: RGB uint8 image in, float32 depth map out.

    Preprocessing is done here (not with the HF image processor) so the exact
    input shape is under our control -- the same shape we will export for the phone.
    """

    def __init__(
        self,
        model: str = "da2-small-metric-outdoor",
        device: str | None = None,
        input_size: int = 518,
        fp16: bool = False,
    ):
        from transformers import AutoModelForDepthEstimation

        self.repo_id = MODELS.get(model, model)  # allow raw HF ids too
        self.name = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = input_size
        self.dtype = torch.float16 if (fp16 and self.device.type == "cuda") else torch.float32

        self.model = AutoModelForDepthEstimation.from_pretrained(self.repo_id)
        self.model.to(self.device, dtype=self.dtype).eval()

        cfg = self.model.config
        self.is_metric = getattr(cfg, "depth_estimation_type", "relative") == "metric"
        self.max_depth = float(getattr(cfg, "max_depth", 0) or 0) if self.is_metric else None
        self.num_params = sum(p.numel() for p in self.model.parameters())

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.last_latency_ms = 0.0

    def input_shape(self, h: int, w: int) -> tuple[int, int]:
        """Network input (H, W): short side = input_size, both sides multiples of 14."""
        s = self.input_size / min(h, w)
        return _round_to_patch(h * s), _round_to_patch(w * s)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray:
        """rgb: HxWx3 uint8 (RGB order). Returns float32 HxW at out_size (default: input size).

        Metric models -> meters. Relative models -> relative inverse depth (bigger = closer).
        """
        h, w = rgb.shape[:2]
        out_size = out_size or (h, w)
        ih, iw = self.input_shape(h, w)

        x = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = F.interpolate(x, size=(ih, iw), mode="bicubic", align_corners=False).clamp_(0, 1)
        x = ((x - self._mean) / self._std).to(self.dtype)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = self.model(pixel_values=x).predicted_depth  # (1, ih, iw)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000

        pred = F.interpolate(pred.unsqueeze(1).float(), size=out_size, mode="bilinear", align_corners=False)
        return pred[0, 0].cpu().numpy()


def box_distance(depth_m: np.ndarray, box_xyxy, percentile: float = 20.0, shrink: float = 0.2) -> float:
    """Distance (m) to an object given a detector box -- the depth<->YOLO hand-off.

    Uses a low percentile inside the box (shrunk to skip background at the edges),
    so the estimate leans toward the nearest part of the object. Returns nan if empty.
    """
    h, w = depth_m.shape
    x1, y1, x2, y2 = box_xyxy
    dx, dy = (x2 - x1) * shrink / 2, (y2 - y1) * shrink / 2
    x1, x2 = int(max(0, x1 + dx)), int(min(w, x2 - dx))
    y1, y2 = int(max(0, y1 + dy)), int(min(h, y2 - dy))
    patch = depth_m[y1:y2, x1:x2]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.percentile(patch, percentile)) if patch.size else float("nan")


def colorize_depth(depth: np.ndarray, is_metric: bool = True, vmax: float | None = 10.0) -> np.ndarray:
    """BGR visualization, near = warm. Metric depth uses a fixed [0, vmax] m scale so frames are comparable."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if is_metric:
        if not vmax:
            vmax = float(np.percentile(d[valid], 95)) if valid.any() else 1.0
        near = 1.0 - np.clip(d / vmax, 0, 1)
    else:  # relative inverse depth: bigger = closer already
        lo, hi = (np.percentile(d[valid], [2, 98]) if valid.any() else (0, 1))
        near = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((near * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img
