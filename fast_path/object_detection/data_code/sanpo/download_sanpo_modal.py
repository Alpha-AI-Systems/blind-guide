"""Download a general-purpose subset of SANPO-Real to a Modal Volume.

Stores the data GENERAL (not tied to boxes or any task):
    frames  : RGB frames as JPG, native resolution (2208x1242 or 1920x1080)
    masks   : the original SANPO segmentation masks, unchanged (PNG, lossless)
    index   : one CSV row per frame (session, frame id, split, size, paths)
Task-specific labels (YOLO boxes, segmentation views, ...) are made later from
these files by separate, fast scripts.

Which frames: SANPO-Real, CAMERA camera (left lens), only frames that have a
segmentation mask, every FRAME_STRIDE-th one (stride 3 = 5 fps from 15 fps).

Splits: SANPO's two official splits, unchanged: official train sessions -> train,
official test sessions -> test. No validation split is made here.

SANPO official links:
    Project page: https://google-research-datasets.github.io/sanpo_dataset/
    GitHub:       https://github.com/google-research-datasets/sanpo_dataset
    Paper:        https://arxiv.org/abs/2309.12172 (WACV 2025)
    Data bucket:  gs://gresearch/sanpo_dataset/v0 (public, read over HTTPS)
    License:      CC BY 4.0
Mask encoding (checked on a real file): R channel = class id (labelmap.json),
B channel = object id (> 0 only for "panoptic" classes, see labeltype.json).

Output on the Modal Volume VOLUME_NAME, under OUTPUT_SUBDIR:
    frames/{train,test}/<session>_<frame>.jpg
    masks/{train,test}/<session>_<frame>.png
    index/<session>.csv                one row per frame
    sanpo_meta/labelmap.json, labeltype.json, splits used (train/test session lists)

Run from the repo root (PowerShell). First a small test (2 sessions per split):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    modal run fast_path/object_detection/data_code/sanpo/download_sanpo_modal.py --max-sessions 2
Then the full run:
    modal run fast_path/object_detection/data_code/sanpo/download_sanpo_modal.py
Re-running skips frames that already exist.
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"  # Modal Volume that stores the dataset
OUTPUT_SUBDIR = "sanpo_real_chest_5fps"  # folder inside the volume
CAMERA = "chest"  # "chest" or "head"
FRAME_STRIDE = 3  # keep every 3rd annotated frame (15 fps -> 5 fps)
JPEG_QUALITY = 95
MAX_CONTAINERS = 10  # sessions processed in parallel
COMMIT_EVERY = 50  # save progress to the Volume every N frames (for resuming)
PRINT_EVERY = 25  # progress line every N frames inside a session
# =============================================================================

SANPO_BUCKET_URL = "https://storage.googleapis.com/gresearch"
SANPO_LIST_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
SANPO_ROOT = "sanpo_dataset/v0"
DATA_MOUNT = "/data"

app = modal.App("blind-guide-sanpo-download")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("pillow")


# ----------------------------------------------------------------------------- helpers

def _get(url: str, retries: int = 5) -> bytes:
    """HTTP GET with retries (exponential back-off)."""
    import time
    import urllib.request

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _list_names(prefix: str, max_results: int | None = None) -> list[str]:
    """List object names in the public SANPO bucket under a prefix."""
    import json
    import urllib.parse

    names, token = [], None
    while True:
        q = {"prefix": prefix, "maxResults": str(max_results or 1000), "fields": "nextPageToken,items(name)"}
        if token:
            q["pageToken"] = token
        d = json.loads(_get(f"{SANPO_LIST_API}?{urllib.parse.urlencode(q)}"))
        names += [i["name"] for i in d.get("items", [])]
        token = d.get("nextPageToken")
        if not token or max_results:
            return names


def _read_bytes(rel_path: str) -> bytes:
    import urllib.parse

    return _get(f"{SANPO_BUCKET_URL}/{urllib.parse.quote(f'{SANPO_ROOT}/{rel_path}')}")


# ----------------------------------------------------------------------------- Modal functions

def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=3 * 3600, max_containers=MAX_CONTAINERS, retries=2)
def download_session(session: str, split: str) -> dict:
    """Download one SANPO session (selected frames + masks) and write its index CSV.

    Resumable: files are written to a temp name and renamed when complete, so a
    killed run never leaves half-written files; existing files are skipped; the
    Volume is committed every COMMIT_EVERY frames; the index CSV is written last
    and marks the session as complete (a finished session is skipped quickly).
    """
    import csv
    import io
    import os
    import time
    from pathlib import Path

    from PIL import Image

    out = Path(DATA_MOUNT) / OUTPUT_SUBDIR
    frame_dir, mask_dir, index_dir = out / "frames" / split, out / "masks" / split, out / "index"
    for d in (frame_dir, mask_dir, index_dir):
        d.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{session}.csv"

    def summary_from(rows: list[dict], skipped: bool) -> dict:
        fmb = sum((out / r["frame"]).stat().st_size for r in rows) / 1e6
        mmb = sum((out / r["mask"]).stat().st_size for r in rows) / 1e6
        return {"session": session, "split": split, "frames": len(rows), "skipped": skipped,
                "frame_mb": round(fmb, 1), "mask_mb": round(mmb, 1)}

    if index_path.exists():  # session already complete
        with open(index_path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"[{split}] {session}: already complete ({len(rows)} frames), skipping", flush=True)
        return summary_from(rows, skipped=True)

    rel_cam = f"sanpo-real/{session}/camera_{CAMERA}/left"
    mask_names = sorted(_list_names(f"{SANPO_ROOT}/{rel_cam}/segmentation_masks/"))
    frame_ids = [n.rsplit("/", 1)[-1].split(".")[0] for n in mask_names][::FRAME_STRIDE]
    print(f"[{split}] {session}: {len(mask_names)} annotated frames, keeping {len(frame_ids)} "
          f"(every {FRAME_STRIDE})", flush=True)

    def atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    rows, start, done_before = [], time.time(), 0
    for i, frame_id in enumerate(frame_ids, 1):
        stem = f"{session}_{frame_id}"
        jpg_path, png_path = frame_dir / f"{stem}.jpg", mask_dir / f"{stem}.png"
        if jpg_path.exists() and png_path.exists():
            done_before += 1
        if not png_path.exists():  # original mask PNG, unchanged
            atomic_write(png_path, _read_bytes(f"{rel_cam}/segmentation_masks/{frame_id}.png"))
        if not jpg_path.exists():  # frame PNG -> JPG, native resolution
            frame = Image.open(io.BytesIO(_read_bytes(f"{rel_cam}/video_frames/{frame_id}.png"))).convert("RGB")
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=JPEG_QUALITY)
            atomic_write(jpg_path, buffer.getvalue())
        with Image.open(jpg_path) as im:
            width, height = im.size
        rows.append({
            "session": session, "frame_id": frame_id, "split": split, "camera": CAMERA,
            "width": width, "height": height,
            "frame": f"frames/{split}/{stem}.jpg", "mask": f"masks/{split}/{stem}.png",
        })
        if i % COMMIT_EVERY == 0:
            volume.commit()
        if i % PRINT_EVERY == 0 or i == len(frame_ids):
            elapsed = time.time() - start
            new = i - done_before
            eta = elapsed / new * (len(frame_ids) - i) if new else 0
            print(f"[{split}] {session}: {i}/{len(frame_ids)} ({100 * i / len(frame_ids):.0f}%) | "
                  f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)} | resumed {done_before}", flush=True)

    tmp_index = index_path.with_name(index_path.name + ".tmp")  # written last = completion marker
    with open(tmp_index, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["session"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_index, index_path)
    volume.commit()
    return summary_from(rows, skipped=False)


@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=1800)
def save_metadata(train: list[str], test: list[str]) -> str:
    """Store SANPO's label files and the session lists of the two official splits."""
    from pathlib import Path

    meta = Path(DATA_MOUNT) / OUTPUT_SUBDIR / "sanpo_meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("labelmap.json", "labeltype.json"):
        (meta / name).write_bytes(_read_bytes(name))
    for split, sessions in (("train", train), ("test", test)):
        (meta / f"{split}_sessions.txt").write_text("\n".join(sessions) + "\n")
    volume.commit()
    return str(meta)


@app.local_entrypoint()
def main(max_sessions: int = 0):
    """max_sessions > 0 = quick test with that many sessions per split."""
    # SANPO's official session lists (one session = one walk)
    train = sorted(_read_bytes("sanpo-real/splits/train_session_ids.txt").decode().split())
    test = sorted(_read_bytes("sanpo-real/splits/test_session_ids.txt").decode().split())

    def has_masks(session: str) -> bool:
        prefix = f"{SANPO_ROOT}/sanpo-real/{session}/camera_{CAMERA}/left/segmentation_masks/"
        return bool(_list_names(prefix, max_results=1))

    from concurrent.futures import ThreadPoolExecutor

    print(f"Checking {len(train) + len(test)} SANPO-Real sessions for {CAMERA} masks (about a minute) ...")
    with ThreadPoolExecutor(32) as pool:
        train_ok = list(pool.map(has_masks, train))
        test_ok = list(pool.map(has_masks, test))
    train = [s for s, ok in zip(train, train_ok) if ok]
    test = [s for s, ok in zip(test, test_ok) if ok]
    print(f"Sessions with {CAMERA} masks: train={len(train)} test={len(test)}")

    if max_sessions > 0:
        train, test = train[:max_sessions], test[:max_sessions]
        print(f"TEST RUN: {max_sessions} session(s) per split")
    print("Saving SANPO label files and split lists ...")
    print("Metadata saved to:", save_metadata.remote(train, test))

    jobs = [(s, "train") for s in train] + [(s, "test") for s in test]
    print(f"Starting {len(jobs)} session downloads ({MAX_CONTAINERS} in parallel) "
          f"to Volume '{VOLUME_NAME}' / {OUTPUT_SUBDIR} ...")
    import time

    totals = {"frames": 0, "frame_mb": 0.0, "mask_mb": 0.0}
    start, newly_done, failed = time.time(), 0, []
    for i, r in enumerate(download_session.starmap(jobs, order_outputs=False, return_exceptions=True), 1):
        if isinstance(r, Exception):
            failed.append(repr(r))
            print(f"[{i}/{len(jobs)}] FAILED: {r!r}  (re-run the command to retry; finished work is kept)", flush=True)
            continue
        for k in totals:
            totals[k] += r[k]
        newly_done += 0 if r["skipped"] else 1
        elapsed = time.time() - start
        remaining = len(jobs) - i
        # ETA from sessions actually downloaded in this run (skipped ones are instant)
        eta = elapsed / newly_done * remaining if newly_done else 0
        print(f"[{i}/{len(jobs)}] {r['split']:<5} {r['session']}: {r['frames']} frames, "
              f"{r['frame_mb']} MB frames, {r['mask_mb']} MB masks{' (already done)' if r['skipped'] else ''} | "
              f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)} "
              f"| total so far {totals['frame_mb'] / 1e3:.2f} GB", flush=True)
    print(f"TOTAL: {totals['frames']} frames | frames {totals['frame_mb'] / 1e3:.2f} GB | "
          f"masks {totals['mask_mb'] / 1e3:.2f} GB | time {_fmt_time(time.time() - start)}")
    if failed:
        print(f"{len(failed)} session(s) failed. Re-run the same command: completed sessions are skipped.")
