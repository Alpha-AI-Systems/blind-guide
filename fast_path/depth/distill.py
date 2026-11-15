"""Distill Depth Pro (teacher) into Depth Anything V2 Small (student) on SANPO-Real.

The teacher stays frozen: it labels every training frame once, and the labels are cached.
The student (DAv2-Small, metric outdoor weights) is trained with

    loss = metric(student, SANPO measured depth)          on frames whose measured depth we use -> meters
         + W_TEACHER * shape(student, Depth Pro depth)    on frames with a usable teacher label -> shape

metric = SiLog loss with lambda 0.5 (the loss Depth Anything V2 used for its metric models).
shape  = SiLog loss with lambda 1 = standard deviation of log(student / teacher) over the image.
         It is zero for student = c * teacher for any c, so the teacher's scale error (which changes
         from walk to walk) costs nothing: only its shape is copied.

The teacher is used only on frames WITHOUT measured depth (where the student has nothing else to learn
from), at a low weight. Teacher labels that disagree strongly with a student already fine-tuned on
measured depth are dropped (check_teacher; Depth Pro fails badly on ~2% of frames). The check needs no
ground truth, so it works the same on unlabeled phone video later.

Every frame is cached at the student's input size for a 2208x1242 SANPO frame (518x924), resized exactly
as DepthEstimator.predict does at evaluation time.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .models import IMAGENET_MEAN, IMAGENET_STD

STUDENT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
TRAIN_HW = (518, 924)  # student input for 1242x2208 frames: short side 518, sides multiples of 14 (as in eval)
GT_RANGE = (0.1, 30.0)  # same range as eval_sanpo.py
TEACHER_RANGE = (0.1, 80.0)  # 80 m = the student's max_depth


# --------------------------------------------------------------------------- preprocessing (label step)

def resize_rgb(rgb: np.ndarray, hw: tuple[int, int] = TRAIN_HW) -> np.ndarray:
    """Bicubic resize exactly like DepthEstimator.predict, stored as uint8."""
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False).clamp_(0, 1)
    return (x[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)


def resize_depth_nearest(depth: np.ndarray, hw: tuple[int, int] = TRAIN_HW) -> np.ndarray:
    return cv2.resize(depth, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def valid_depth(d: np.ndarray, lo_hi: tuple[float, float]) -> np.ndarray:
    return np.isfinite(d) & (d > lo_hi[0]) & (d < lo_hi[1])


def median_scaled_abs_rel(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> float:
    """AbsRel of pred against ref after scaling pred by the ratio of medians (nan if < 100 pixels)."""
    if valid.sum() < 100:
        return float("nan")
    p, r = pred[valid].astype(np.float64), ref[valid].astype(np.float64)
    p = p * np.median(r) / max(np.median(p), 1e-6)
    return float(np.mean(np.abs(p - r) / r))


def raw_abs_rel(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() < 100:
        return float("nan")
    p, r = pred[valid].astype(np.float64), ref[valid].astype(np.float64)
    return float(np.mean(np.abs(p - r) / r))


# --------------------------------------------------------------------------- cached samples

@dataclass
class Sample:
    key: str
    role: str  # "gt" (measured depth used), "unlabeled" (measured depth NOT used), "val"
    rgb: np.ndarray  # 518x924x3 uint8
    gt: np.ndarray  # 518x924 float16 meters (nan/0 = invalid)
    teacher: np.ndarray | None  # 518x924 float16 meters
    teacher_ok: bool  # teacher label present and passed the agreement check


def load_samples(items: list[tuple[Path, str]], teacher_roles: tuple[str, ...] = ("unlabeled",)) -> list[Sample]:
    """items: (cached .npz, role). The teacher label is loaded only for frames whose role is in teacher_roles."""
    out = []
    for path, role in items:
        with np.load(path, allow_pickle=False) as z:
            teacher = z["teacher"] if role in teacher_roles and "teacher" in z.files else None
            out.append(Sample(str(z["key"]), role, z["rgb"], z["gt"], teacher, teacher is not None))
    return out


@torch.no_grad()
def check_teacher(samples: list[Sample], reference: str, thr: float, device=None) -> dict:
    """Drop teacher labels whose shape disagrees with a reference student (median-scaled AbsRel >= thr).

    The reference is a student already fine-tuned on measured depth, so it needs no ground truth on the
    frames being checked. The labels' true error against the measured depth is computed only as a
    diagnostic, to show whether the check removes the bad labels."""
    from transformers import AutoModelForDepthEstimation

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ref = AutoModelForDepthEstimation.from_pretrained(reference).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    rows = []
    for s in samples:
        if s.teacher is None:
            continue
        pred = _predict(ref, s.rgb, device, mean, std)
        t, gt = s.teacher.astype(np.float32), s.gt.astype(np.float32)
        agree = median_scaled_abs_rel(t, pred, valid_depth(t, TEACHER_RANGE) & valid_depth(pred, TEACHER_RANGE))
        true = median_scaled_abs_rel(t, gt, valid_depth(gt, GT_RANGE) & np.isfinite(t))
        s.teacher_ok = math.isfinite(agree) and agree < thr
        rows.append((agree, true, s.teacher_ok))
    del ref
    torch.cuda.empty_cache()
    kept = [t for _, t, ok in rows if ok and math.isfinite(t)]
    dropped = [t for _, t, ok in rows if not ok and math.isfinite(t)]
    return {"checked": len(rows), "kept": len(kept), "dropped": len(dropped),
            "agree_median": float(np.median([a for a, _, _ in rows])) if rows else float("nan"),
            "true_error_kept_median": float(np.median(kept)) if kept else float("nan"),
            "true_error_dropped_median": float(np.median(dropped)) if dropped else float("nan"),
            "bad_labels_total": sum(t > 0.3 for _, t, _ in rows),  # true scaled error > 0.3 = clearly wrong shape
            "bad_labels_dropped": sum(t > 0.3 for _, t, ok in rows if not ok)}


class DistillDataset(torch.utils.data.Dataset):
    """Random 518x518 crop + horizontal flip + brightness/contrast jitter (training),
    or the full 518x924 frame (augment=False). use_gt=False hides the measured depth."""

    def __init__(self, samples: list[Sample], use_gt: bool, crop: int = 518, augment: bool = True):
        self.samples, self.use_gt, self.crop, self.augment = samples, use_gt, crop, augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        rgb = torch.from_numpy(s.rgb).permute(2, 0, 1).float() / 255.0  # 3HW
        gt = torch.from_numpy(np.nan_to_num(s.gt.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
        if not self.use_gt:
            gt = torch.zeros_like(gt)
        has_teacher = s.teacher is not None and s.teacher_ok
        teacher = (torch.from_numpy(np.nan_to_num(s.teacher.astype(np.float32), nan=0.0, posinf=0.0))
                   if has_teacher else torch.zeros_like(gt))
        if self.augment:  # torch RNG: each DataLoader worker is seeded differently
            w = rgb.shape[2]
            x0 = int(torch.randint(0, w - self.crop + 1, (1,)))
            rgb, gt, teacher = rgb[:, :, x0:x0 + self.crop], gt[:, x0:x0 + self.crop], teacher[:, x0:x0 + self.crop]
            if torch.rand(1) < 0.5:
                rgb, gt, teacher = rgb.flip(-1), gt.flip(-1), teacher.flip(-1)
            b, c = 0.8 + 0.4 * torch.rand(1), 0.8 + 0.4 * torch.rand(1)
            mean = rgb.mean()
            rgb = ((rgb * b - mean) * c + mean).clamp(0, 1)
        return {"rgb": rgb.contiguous(), "gt": gt.contiguous(), "teacher": teacher.contiguous(),
                "has_teacher": torch.tensor(has_teacher)}


# --------------------------------------------------------------------------- losses

def silog(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, lam: float, min_px: int = 100):
    """Per-image SiLog: sqrt(mean(d^2) - lam * mean(d)^2), d = log(pred) - log(target), averaged over the
    images with at least min_px valid pixels. lam=1 -> fully scale-invariant (shape only)."""
    v = valid.float()
    d = (torch.log(pred.clamp_min(1e-3)) - torch.log(torch.where(valid, target, torch.ones_like(target)))) * v
    n = v.sum((1, 2))
    ok = n >= min_px
    if not ok.any():
        return pred.sum() * 0.0
    n = n.clamp_min(1)
    m1 = d.sum((1, 2)) / n
    var = (((d - m1.view(-1, 1, 1)) * v) ** 2).sum((1, 2)) / n  # stable form of mean(d^2) - mean(d)^2
    return torch.sqrt((var + (1 - lam) * m1 * m1).clamp_min(1e-8))[ok].mean()


# --------------------------------------------------------------------------- training

def _loop(loader):
    while True:
        yield from loader


def _predict(model, rgb: np.ndarray, device, mean, std) -> np.ndarray:
    """Depth (meters) for one cached 518x924 uint8 frame."""
    x = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred = model(pixel_values=(x - mean) / std).predicted_depth
    return pred[0].float().cpu().numpy()


@torch.no_grad()
def evaluate_val(model, samples: list[Sample], device) -> dict:
    """Raw and median-scaled AbsRel / delta1 on full 518x924 frames against the measured depth."""
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    raw, med, d1 = [], [], []
    for s in samples:
        pred = _predict(model, s.rgb, device, mean, std)
        gt = s.gt.astype(np.float32)
        valid = valid_depth(gt, GT_RANGE)
        if valid.sum() < 100:
            continue
        raw.append(raw_abs_rel(pred, gt, valid))
        med.append(median_scaled_abs_rel(pred, gt, valid))
        p, g = np.maximum(pred[valid], 1e-6), gt[valid]
        d1.append(float(np.mean(np.maximum(p / g, g / p) < 1.25)))
    model.train()
    return {"abs_rel": float(np.mean(raw)), "abs_rel_median_frame": float(np.median(raw)),
            "d1": float(np.mean(d1)), "abs_rel_scaled": float(np.mean(med)), "n": len(raw)}


def train_student(out_dir: Path, gt_samples: list[Sample], teacher_pool: list[Sample], val_samples: list[Sample],
                  use_teacher: bool, steps: int = 1500, batch: int = 8, lr_encoder: float = 5e-6,
                  lr_decoder: float = 5e-5, w_teacher: float = 1.0, warmup: int = 100, val_every: int = 150,
                  log_every: int = 50, workers: int = 4, seed: int = 0) -> list[dict]:
    """Fine-tune the student. Every step uses `batch` frames with measured depth (metric loss; plus the
    teacher shape loss when use_teacher); use_teacher also adds `batch` teacher-only frames from
    teacher_pool. The checkpoint with the lowest raw val AbsRel is saved to out_dir/best."""
    from transformers import AutoModelForDepthEstimation

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    model = AutoModelForDepthEstimation.from_pretrained(STUDENT).to(device).train()
    enc = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": lr_encoder}, {"params": dec, "lr": lr_decoder}], weight_decay=0.01)
    base_lrs = [lr_encoder, lr_decoder]

    def make_loader(samples, use_gt):
        return torch.utils.data.DataLoader(DistillDataset(samples, use_gt), batch_size=batch, shuffle=True,
                                           drop_last=True, num_workers=workers, pin_memory=True,
                                           persistent_workers=workers > 0)

    gt_iter = _loop(make_loader(gt_samples, use_gt=True))
    u_iter = _loop(make_loader(teacher_pool, use_gt=False)) if use_teacher and teacher_pool else None
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    history = [{"step": 0, "val": evaluate_val(model, val_samples, device)}]
    best = history[0]["val"]["abs_rel"]
    print(f"step 0 (pretrained): val {history[0]['val']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "best")
    t0, running = time.time(), {}

    for step in range(1, steps + 1):
        scale = step / warmup if step <= warmup else (1 - (step - warmup) / max(1, steps - warmup)) ** 0.9
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale

        b = next(gt_iter)
        parts = [b]
        if u_iter is not None:
            parts.append(next(u_iter))
        rgb = torch.cat([p["rgb"] for p in parts]).to(device, non_blocking=True)
        gt = torch.cat([p["gt"] for p in parts]).to(device, non_blocking=True)
        teacher = torch.cat([p["teacher"] for p in parts]).to(device, non_blocking=True)
        has_t = torch.cat([p["has_teacher"] for p in parts]).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(pixel_values=(rgb - mean) / std).predicted_depth
        pred = pred.float()

        gt_valid = (gt > GT_RANGE[0]) & (gt < GT_RANGE[1])
        loss_metric = silog(pred, gt, gt_valid, lam=0.5)
        loss = loss_metric
        logs = {"metric": float(loss_metric)}
        if use_teacher:
            t_valid = (teacher > TEACHER_RANGE[0]) & has_t.view(-1, 1, 1)
            loss_shape = silog(pred, teacher.clamp(max=TEACHER_RANGE[1]), t_valid, lam=1.0)
            loss = loss + w_teacher * loss_shape
            logs["shape"] = float(loss_shape)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in logs.items():
            running[k] = running.get(k, 0.0) + v

        if step % log_every == 0:
            msg = "  ".join(f"{k}={v / log_every:.4f}" for k, v in running.items())
            print(f"step {step}/{steps}  {msg}  ({(time.time() - t0) / step:.2f} s/step)", flush=True)
            running = {}
        if step % val_every == 0 or step == steps:
            val = evaluate_val(model, val_samples, device)
            history.append({"step": step, "val": val})
            tag = ""
            if val["abs_rel"] < best:
                best, tag = val["abs_rel"], "  <- best, saved"
                model.save_pretrained(out_dir / "best")
            print(f"step {step}: val {val}{tag}", flush=True)

    model.save_pretrained(out_dir / "last")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history
