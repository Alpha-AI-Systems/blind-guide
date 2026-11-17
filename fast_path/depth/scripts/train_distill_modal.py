"""Teacher-student training on Modal: Depth Pro (teacher, frozen) -> DAv2-Small metric (student), SANPO-Real.

Steps
    1. prepare (CPU): pick walks from SANPO's official TRAIN split (chest camera), evenly spaced frames per walk,
       with roles given by --layout (in walk order, e.g. "v8,g60,u60,g240,u60,v12"):
         g = gt walks         measured depth used for the metric loss
         u = unlabeled walks  measured depth NOT used: only the teacher labels them. They stand in for walking
                              video with no depth (e.g. phone video, WAD), so the run shows what the teacher adds.
         v = val walks        measured depth used only to pick the best checkpoint
       Walk order is fixed by --seed, so a longer layout keeps the walks (and cached frames) of a shorter one.
    2. cache: gt/val frames are resized and cached on CPU containers; unlabeled frames also get a Depth Pro
       label (A100). Every frame is cached once in /vol/distill/frames/ and reused by later runs.
    3. train (A100, the configs run at the same time, same steps and same GT batches):
         gt-only    metric loss on the gt walks (no teacher)
         distill-u  metric loss + W_TEACHER * teacher shape loss on the unlabeled walks only. Teacher labels
                    that disagree with REFERENCE (a student already fine-tuned on measured depth) are dropped.
    4. evaluate each run's best checkpoint with eval_sanpo.py on the official TEST frames used before
       (300 random frames from 30 walks + 43 near-obstacle frames) and print it next to the pretrained
       student, the teacher and earlier runs.

Nothing from the official test split is used for training, filtering or checkpoint selection.

Run from the repo root:
    python -m modal run fast_path/depth/scripts/train_distill_modal.py
    python -m modal run fast_path/depth/scripts/train_distill_modal.py --steps 8000 --w-teacher 0.5
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

# =============================================================================
# SETTINGS (edit here). Command-line options override most of them.
# =============================================================================
VOLUME_NAME = "blind-guide-depth"
GPU = ["A100", "A100-80GB"]  # A100 only; Modal takes the 80 GB one if no 40 GB one is free
LABEL_WORKERS = 4  # Depth Pro containers labeling in parallel
PREP_FRAMES_PER_CONTAINER = 100  # gt/val frames resized per CPU container
CONFIGS = "gt-only,distill-u"
LAYOUT = "v8,g60,u60,g240,u60,v12"  # first 128 walks = the first run's walks: their frames are already cached
FRAMES_PER_WALK = 10  # evenly spaced over the whole walk
SEED = 0
STEPS = 6000
BATCH = 8  # frames with measured depth per step (distill-u adds as many teacher-only frames)
LR_ENCODER = 5e-6  # Depth Anything V2's metric fine-tuning rates
LR_DECODER = 5e-5
W_TEACHER = 0.25
FILTER = 0.35  # drop a teacher label if its scaled AbsRel to the reference student is at least this
REFERENCE = "/vol/distill/runs/train_g60_u60_v8_f10_seed0_steps1500_b8_wt1_f0.4/gt-only/best"  # first run
TEST_SETS = ["real_chest_test_s30_f10_stride15_seed0", "real_chest_near3m_test_s10_f5_seed0"]
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root")
VOLUME_MOUNT = "/vol"
FRAMES_DIR = f"{VOLUME_MOUNT}/distill/frames"
OLD_FRAMES_DIRS = [f"{VOLUME_MOUNT}/distill/train_g60_u60_v8_f10_seed0/frames"]  # first run's cache (moved once)

app = modal.App("blind-guide-depth-distill")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4,<2.9", "torchvision", "transformers>=4.50,<4.60", "opencv-python-headless>=4.8,<5",
                 "numpy>=1.26,<3", "pillow", "tqdm")
    .env({"HF_HOME": f"{VOLUME_MOUNT}/hf_cache"})
    .add_local_file(REPO_ROOT / "fast_path" / "__init__.py", "/root/fast_path/__init__.py")
    .add_local_dir(REPO_ROOT / "fast_path" / "depth", "/root/fast_path/depth", ignore=["**/__pycache__/**"])
)

ROLE_OF = {"g": "gt", "u": "unlabeled", "v": "val"}


def parse_layout(layout: str) -> list[tuple[str, int]]:
    parts = []
    for token in layout.split(","):
        m = re.fullmatch(r"\s*([guv])(\d+)\s*", token)
        if not m:
            raise SystemExit(f"--layout: bad part '{token}' (use e.g. v8,g60,u60)")
        parts.append((ROLE_OF[m.group(1)], int(m.group(2))))
    return parts


def data_name(layout: str, frames_per_walk: int, seed: int) -> str:
    return f"walks_{layout.replace(',', '-')}_f{frames_per_walk}_seed{seed}"


def _setup():
    import os
    import sys
    os.chdir("/root")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


def _rel(it: dict, folder: str, ext: str) -> str:
    return f"sanpo-real/{it['session']}/camera_chest/left/{folder}/{it['frame_id']}{ext}"


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=3600)
def prepare(name: str, layout: list[tuple[str, int]], frames_per_walk: int, seed: int) -> list[dict]:
    """Pick the walks and frames (listing only); the manifest is saved so later runs reuse it."""
    import json
    import random
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    _setup()
    from fast_path.depth import sanpo

    # one-time move of the first run's cache into the shared frame folder
    Path(FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    moved = 0
    for old in OLD_FRAMES_DIRS:
        for p in Path(old).glob("*.npz") if Path(old).exists() else []:
            if not (Path(FRAMES_DIR) / p.name).exists():
                shutil.move(str(p), Path(FRAMES_DIR) / p.name)
                moved += 1
    if moved:
        volume.commit()
        print(f"moved {moved} cached frames to {FRAMES_DIR}")

    manifest = Path(VOLUME_MOUNT) / "distill" / name / "manifest.json"
    if manifest.exists():
        print(f"reusing {manifest}")
        return json.loads(manifest.read_text())

    official = sanpo.official_split_sessions("sanpo-real", "train")
    sessions = sorted(s for s in sanpo.list_sessions("sanpo-real") if s in official)
    random.Random(seed).shuffle(sessions)
    need = sum(n for _, n in layout)
    print(f"{len(sessions)} official train walks; need {need}")

    def frames_of(session: str) -> tuple[str, list[str]]:
        if "chest" not in sanpo.session_cameras("sanpo-real", session):
            return session, []
        ids = sorted(set(sanpo.list_frame_ids("sanpo-real", session, "chest", "video_frames"))
                     & set(sanpo.list_frame_ids("sanpo-real", session, "chest", "depth_maps")))
        if len(ids) < frames_per_walk:
            return session, []
        step = len(ids) / frames_per_walk
        return session, [ids[int(step * (i + 0.5))] for i in range(frames_per_walk)]

    walks, start = [], 0
    with ThreadPoolExecutor(16) as pool:
        while len(walks) < need and start < len(sessions):
            batch = sessions[start:start + 2 * (need - len(walks))]
            start += len(batch)
            walks += [(s, f) for s, f in pool.map(frames_of, batch) if f]  # pool.map keeps the order
    if len(walks) < need:
        raise RuntimeError(f"only {len(walks)} usable walks, need {need}")

    roles = [role for role, n in layout for _ in range(n)]
    items = [{"session": s, "frame_id": f, "role": role, "key": f"{s}_{f}"}
             for (s, frame_ids), role in zip(walks, roles) for f in frame_ids]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(items, indent=1))
    volume.commit()
    print(f"{len(items)} frames from {need} walks -> {manifest}")
    return items


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=1800)
def cached(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split items into (to prepare on CPU, to label with the teacher): frames not cached yet, or unlabeled
    frames cached without a teacher label (or cut short by a stopped run)."""
    import numpy as np

    _setup()
    prep, teach = [], []
    for it in items:
        try:
            with np.load(Path(FRAMES_DIR) / f"{it['key']}.npz") as z:
                ok = {"rgb", "gt"} <= set(z.files) and (it["role"] != "unlabeled" or "teacher" in z.files)
        except Exception:
            ok = False
        if not ok:
            (teach if it["role"] == "unlabeled" else prep).append(it)
    return prep, teach


def _cache_frame(it: dict, raw: Path, teacher=None) -> None:
    import numpy as np

    from fast_path.depth import distill, sanpo
    rgb = sanpo.read_rgb(raw / _rel(it, "video_frames", ".png"))
    gt = distill.resize_depth_nearest(sanpo.read_depth(raw / _rel(it, "depth_maps", ".float16.gz")))
    arrays = {"key": np.array(it["key"]), "rgb": distill.resize_rgb(rgb), "gt": gt.astype(np.float16)}
    if teacher is not None:
        t = teacher.predict(rgb, out_size=distill.TRAIN_HW)
        ok = distill.valid_depth(gt, distill.GT_RANGE) & np.isfinite(t)
        arrays |= {"teacher": t.astype(np.float16),
                   # diagnostics only (never used for training or filtering):
                   "teacher_gt_scaled": np.array(distill.median_scaled_abs_rel(t, gt, ok)),
                   "teacher_gt_raw": np.array(distill.raw_abs_rel(t, gt, ok)),
                   "focal_px": np.array(teacher.last_focal_px or np.nan)}
    tmp = Path(FRAMES_DIR) / f"{it['key']}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(Path(FRAMES_DIR) / f"{it['key']}.npz")  # never leaves a half-written file


def _download(items: list[dict], raw: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from fast_path.depth import sanpo
    jobs = [_rel(it, "video_frames", ".png") for it in items] + [_rel(it, "depth_maps", ".float16.gz") for it in items]
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda j: sanpo.download(j, raw), jobs))


@app.function(image=image, volumes={VOLUME_MOUNT: volume}, timeout=3600, cpu=4, memory=8192)
def prep(items: list[dict]) -> int:
    """Download + resize + cache gt/val frames (no model needed)."""
    _setup()
    raw = Path("/tmp/raw")
    _download(items, raw)
    for it in items:
        _cache_frame(it, raw)
    volume.commit()
    return len(items)


@app.function(image=image, gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=3 * 3600)
def label(items: list[dict]) -> int:
    """Download, run Depth Pro once per frame, cache rgb + measured depth + teacher label."""
    _setup()
    from fast_path.depth.models import DepthProEstimator

    raw = Path("/tmp/raw")
    _download(items, raw)
    teacher = DepthProEstimator("depth-pro")
    for n, it in enumerate(items):
        _cache_frame(it, raw, teacher)
        if (n + 1) % 50 == 0:
            print(f"  labeled {n + 1}/{len(items)}", flush=True)
            volume.commit()
    volume.commit()
    return len(items)


@app.function(image=image, gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=6 * 3600, memory=49152)
def train(config: str, name: str, run: str, steps: int, batch: int, lr_encoder: float, lr_decoder: float,
          w_teacher: float, filter_thr: float, reference: str, seed: int, test_sets: list[str]) -> dict:
    """Train one config, then evaluate its best checkpoint on the test sets (zipped eval folders returned)."""
    import io
    import json
    import subprocess
    import sys
    import zipfile

    import torch

    _setup()
    from fast_path.depth import distill

    print(f"[{config}] started on {torch.cuda.get_device_name(0)}", flush=True)
    volume.reload()
    items = json.loads((Path(VOLUME_MOUNT) / "distill" / name / "manifest.json").read_text())
    use_teacher = config == "distill-u"
    roles = {"gt", "val", "unlabeled"} if use_teacher else {"gt", "val"}
    samples = distill.load_samples([(Path(FRAMES_DIR) / f"{it['key']}.npz", it["role"])
                                    for it in items if it["role"] in roles])
    gt = [s for s in samples if s.role == "gt"]
    val = [s for s in samples if s.role == "val"]
    pool = [s for s in samples if s.role == "unlabeled"]
    check = {}
    if use_teacher:
        check = distill.check_teacher(pool, reference, filter_thr)
        print(f"[{config}] teacher check vs reference student: {check}", flush=True)
        pool = [s for s in pool if s.teacher_ok]
    print(f"[{config}] {len(gt)} gt frames, {len(pool)} teacher-only frames, {len(val)} val frames", flush=True)

    ckpt = Path(VOLUME_MOUNT) / "distill" / "runs" / run / config
    history = distill.train_student(ckpt, gt, pool, val, use_teacher, steps=steps, batch=batch,
                                    lr_encoder=lr_encoder, lr_decoder=lr_decoder, w_teacher=w_teacher,
                                    val_every=500, log_every=250, seed=seed)
    volume.commit()

    evals = {}
    for test in test_sets:
        out = Path("/tmp/eval") / test
        done = subprocess.run([sys.executable, "fast_path/depth/scripts/eval_sanpo.py", "--root",
                               f"{VOLUME_MOUNT}/sanpo/{test}", "--split", "sanpo-real", "--camera", "chest",
                               "--model", str(ckpt / "best"), "--out", str(out)])
        if done.returncode != 0:
            print(f"[{config}] eval on {test} FAILED", flush=True)
            continue
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in out.rglob("*"):
                if p.is_file():
                    archive.write(p, p.relative_to(out))
        evals[test] = buffer.getvalue()
    return {"history": history, "evals": evals, "checkpoint": str(ckpt / "best"), "teacher_check": check}


# --------------------------------------------------------------------------- local side

def _per_frame_stats(csv_path: Path) -> dict:
    """Mean and per-frame median of the main metrics, raw ('none') and median-scaled ('median')."""
    import csv
    import math
    import statistics

    rows = list(csv.DictReader(csv_path.open()))
    out = {}
    for mode in ("none", "median"):
        sub = [r for r in rows if r["align"] == mode]
        for col in ("abs_rel", "d1", "abs_rel@obstacles<3m", "d1@obstacles<3m"):
            vals = [float(r[col]) for r in sub if r.get(col) not in (None, "") and math.isfinite(float(r[col]))]
            if vals:
                out[f"{mode}:{col}"] = (statistics.fmean(vals), statistics.median(vals), len(vals))
    return out


def _run_label(run: str, config: str) -> str:
    """e.g. 'gt-only (300 gt walks, 6000 st)' from the run folder name."""
    layout = re.match(r"walks_(.+?)_f\d+_seed", run)
    if layout:
        gt = sum(int(n) for n in re.findall(r"g(\d+)", layout.group(1)))
    else:  # first run: train_g60_u60_v8_...
        m = re.search(r"train_g(\d+)", run)
        gt = int(m.group(1)) if m else "?"
    steps = re.search(r"steps(\d+)", run)
    return f"{config} ({gt} gt walks, {steps.group(1) if steps else '?'} st)"


def _print_comparison() -> None:
    base = REPO_ROOT / "outputs"
    models = [("pretrained DAv2-S", base / "depth_modal" / "{t}" / "da2-small-metric-outdoor"),
              ("Depth Pro (teacher)", base / "depth_modal" / "{t}" / "depth-pro")]
    for run_dir in sorted((base / "depth_distill").glob("*"), key=lambda p: p.stat().st_mtime):
        for cfg in sorted(p.name for p in run_dir.iterdir() if p.is_dir()):
            models.append((_run_label(run_dir.name, cfg), run_dir / cfg / "{t}"))
    cols = [("none:abs_rel", "raw AbsRel"), ("none:d1", "raw d1"), ("none:abs_rel@obstacles<3m", "raw AbsRel <3m"),
            ("none:d1@obstacles<3m", "raw d1 <3m"), ("median:abs_rel", "scaled AbsRel"),
            ("median:abs_rel@obstacles<3m", "scaled AbsRel <3m")]
    for test in TEST_SETS:
        print(f"\n=== {test}   (per-frame MEDIAN, mean in brackets) ===")
        print(f"{'':<38}" + "".join(f"{label:>21}" for _, label in cols))
        for label, pattern in models:
            path = Path(str(pattern).format(t=test)) / "per_frame.csv"
            if not path.exists():
                continue
            stats = _per_frame_stats(path)
            cells = "".join(f"{stats[k][1]:>12.3f} ({stats[k][0]:.3f})" if k in stats else f"{'-':>21}"
                            for k, _ in cols)
            print(f"{label:<38}{cells}")


@app.local_entrypoint()
def main(configs: str = CONFIGS, layout: str = LAYOUT, frames_per_walk: int = FRAMES_PER_WALK, seed: int = SEED,
         steps: int = STEPS, batch: int = BATCH, lr_encoder: float = LR_ENCODER, lr_decoder: float = LR_DECODER,
         w_teacher: float = W_TEACHER, filter_thr: float = FILTER, reference: str = REFERENCE):
    import io
    import json
    import zipfile

    config_list = [c.strip() for c in configs.split(",") if c.strip()]
    if not set(config_list) <= {"gt-only", "distill-u"}:
        raise SystemExit("--configs: comma-separated from gt-only, distill-u")
    parts = parse_layout(layout)
    name = data_name(layout, frames_per_walk, seed)
    run = f"{name}_steps{steps}_b{batch}_wt{w_teacher:g}_f{filter_thr:g}"
    run_dir = REPO_ROOT / "outputs" / "depth_distill" / run

    print(f"1/4 picking frames ({name}) ...")
    items = prepare.remote(name, parts, frames_per_walk, seed)
    counts = {r: sum(it["role"] == r for it in items) for r in ("gt", "unlabeled", "val")}
    print(f"   {len(items)} frames: {counts}")

    to_prep, to_label = cached.remote(items)
    print(f"2/4 caching: {len(items) - len(to_prep) - len(to_label)} cached, {len(to_prep)} to prepare on CPU, "
          f"{len(to_label)} to label with Depth Pro on {GPU} ...")
    calls = []  # all started at once: Depth Pro on GPUs and the plain frames on CPUs in parallel
    if to_label:
        walks = sorted({it["session"] for it in to_label})
        chunks = [[it for it in to_label if walks.index(it["session"]) % LABEL_WORKERS == k]
                  for k in range(LABEL_WORKERS)]
        calls += [label.spawn(c) for c in chunks if c]
    n = PREP_FRAMES_PER_CONTAINER
    calls += [prep.spawn(to_prep[i:i + n]) for i in range(0, len(to_prep), n)]
    print(f"   cached {sum(call.get() for call in calls)} new frames")

    print(f"3/4 training {config_list} in parallel on {GPU} ({steps} steps) ...")
    calls = {c: train.spawn(c, name, run, steps, batch, lr_encoder, lr_decoder, w_teacher, filter_thr, reference,
                            seed, TEST_SETS) for c in config_list}
    results = {}
    for c, call in calls.items():
        try:
            results[c] = call.get()
        except Exception as e:  # keep the other config's results
            print(f"   {c} FAILED: {e}")

    print("4/4 results")
    for c, res in results.items():
        (run_dir / c).mkdir(parents=True, exist_ok=True)
        (run_dir / c / "history.json").write_text(json.dumps(res["history"], indent=2))
        if res.get("teacher_check"):
            (run_dir / c / "teacher_check.json").write_text(json.dumps(res["teacher_check"], indent=2))
        for test, data in res["evals"].items():
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                archive.extractall(run_dir / c / test)
        best = min(res["history"], key=lambda h: h["val"]["abs_rel"])
        print(f"   {c}: best val raw AbsRel {best['val']['abs_rel']:.3f} at step {best['step']} "
              f"(pretrained {res['history'][0]['val']['abs_rel']:.3f}); checkpoint on the Volume: {res['checkpoint']}")
    _print_comparison()
    print(f"\nSaved to {run_dir}")
