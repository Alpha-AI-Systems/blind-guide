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

ALL FRAMES (15 fps) ADD-ON, for the "which real frames" experiment (step 4C), run once:
    modal run fast_path/object_detection/data_code/sanpo/download_sanpo_modal.py --all-frames --max-sessions 2   # test
    modal run fast_path/object_detection/data_code/sanpo/download_sanpo_modal.py --all-frames
  - Only the TRAINING walks (official train minus the 12 validation walks): validation and test stay at 5 fps.
  - Every annotated frame (stride 1). The 5 fps frames already on the volume are kept, not downloaded again; the
    new frames go into the same frames/ and masks/ folders.
  - Their index goes to ALL_FRAMES_INDEX_DIR/<session>.csv (column in_5fps = the frame is also in index/), so
    index/ and every script that reads it stay exactly as before.
  - Checked per walk: the in_5fps frames must be exactly the frames in index/<session>.csv.
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"  # Modal Volume that stores the dataset
OUTPUT_SUBDIR = "sanpo_real_chest"  # folder inside the volume
CAMERA = "chest"  # "chest" or "head"
FRAME_STRIDE = 3  # keep every 3rd annotated frame (15 fps -> 5 fps)
ALL_FRAMES_INDEX_DIR = "index_15fps"  # --all-frames: index folder of the training walks at 15 fps (stride 1)
VALIDATION_WALKS_FILE = "validation_split/results/validation_walks.txt"  # next to this script; left out by --all-frames
JPEG_QUALITY = 95
MAX_CONTAINERS = 50  # sessions processed in parallel (104 training walks: about 2 rounds)
THREADS_PER_SESSION = 32  # frames downloaded at once inside one session (network-bound)
CPU_PER_CONTAINER = 4  # cores per container (PNG decode + JPEG encode); Modal's default is a fraction of a core
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


@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=3 * 3600, max_containers=MAX_CONTAINERS, retries=2,
              cpu=CPU_PER_CONTAINER)
def download_session(session: str, split: str, frame_stride: int = FRAME_STRIDE, index_name: str = "index") -> dict:
    """Download one SANPO session (selected frames + masks) and write its index CSV.

    frame_stride: keep every frame_stride-th annotated frame (3 = 5 fps, 1 = all 15 fps).
    index_name: the index folder (index = the 5 fps set; ALL_FRAMES_INDEX_DIR = the --all-frames add-on).
    Each row has in_5fps: the frame is one of the every-3rd frames, i.e. in the 5 fps set.

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

    import csv as csv_module

    out = Path(DATA_MOUNT) / OUTPUT_SUBDIR
    five_fps_index_path = out / "index" / f"{session}.csv"
    if index_name != "index" and five_fps_index_path.exists():
        # the add-on writes into the walk's EXISTING folder: an older download put some training walks in
        # frames/val/ (their index rows say split "val"); a second copy in frames/train/ would change train.txt
        with open(five_fps_index_path, newline="") as f:
            split = next(csv_module.DictReader(f))["split"]
    frame_dir, mask_dir, index_dir = out / "frames" / split, out / "masks" / split, out / index_name
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
    all_ids = [n.rsplit("/", 1)[-1].split(".")[0] for n in mask_names]
    frame_ids = all_ids[::frame_stride]
    five_fps_ids = set(all_ids[::3])                # the original download's rule (FRAME_STRIDE 3)
    five_fps_index = out / "index" / f"{session}.csv"
    if index_name != "index" and five_fps_index.exists():
        with open(five_fps_index, newline="") as f:
            existing = {row["frame_id"] for row in csv.DictReader(f)}
        if existing != five_fps_ids:                # the bucket listing changed since the 5 fps download
            raise RuntimeError(f"{session}: the every-3rd frames ({len(five_fps_ids)}) differ from "
                               f"index/{session}.csv ({len(existing)} frames); nothing written")
    print(f"[{split}] {session}: {len(mask_names)} annotated frames, keeping {len(frame_ids)} "
          f"(every {frame_stride}), {len(five_fps_ids & set(frame_ids))} of them in the 5 fps set", flush=True)

    def atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def fetch_frame(frame_id: str) -> tuple[dict, bool]:
        """Download one frame and its mask unless both exist. Returns (index row, True if both existed)."""
        stem = f"{session}_{frame_id}"
        jpg_path, png_path = frame_dir / f"{stem}.jpg", mask_dir / f"{stem}.png"
        existed = jpg_path.exists() and png_path.exists()
        if not png_path.exists():  # original mask PNG, unchanged
            atomic_write(png_path, _read_bytes(f"{rel_cam}/segmentation_masks/{frame_id}.png"))
        if not jpg_path.exists():  # frame PNG -> JPG, native resolution
            frame = Image.open(io.BytesIO(_read_bytes(f"{rel_cam}/video_frames/{frame_id}.png"))).convert("RGB")
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=JPEG_QUALITY)
            atomic_write(jpg_path, buffer.getvalue())
        with Image.open(jpg_path) as im:   # reads only the header
            width, height = im.size
        return {
            "session": session, "frame_id": frame_id, "split": split, "camera": CAMERA,
            "width": width, "height": height,
            "frame": f"frames/{split}/{stem}.jpg", "mask": f"masks/{split}/{stem}.png",
            "in_5fps": frame_id in five_fps_ids,
        }, existed

    from concurrent.futures import ThreadPoolExecutor

    rows, start, done_before = [], time.time(), 0
    # THREADS_PER_SESSION frames at once: each mostly waits on the network. map keeps the frame order.
    with ThreadPoolExecutor(THREADS_PER_SESSION) as pool:
        results = pool.map(fetch_frame, frame_ids)
        for i, (row, existed) in enumerate(results, 1):
            rows.append(row)
            done_before += int(existed)
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
def main(max_sessions: int = 0, all_frames: bool = False, sessions: str = ""):
    """max_sessions > 0 = quick test with that many sessions per split.
    all_frames = the 15 fps add-on: training walks only, every frame, index in ALL_FRAMES_INDEX_DIR.
    sessions = comma-separated walk ids: only these (a test on chosen walks)."""
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

    if all_frames:
        from pathlib import Path

        validation = set((Path(__file__).resolve().parent / VALIDATION_WALKS_FILE).read_text(encoding="utf-8").split())
        train, test = [s for s in train if s not in validation], []
        print(f"ALL FRAMES (15 fps): {len(train)} training walks (the {len(validation)} validation walks and the "
              f"test walks stay at 5 fps), index -> {ALL_FRAMES_INDEX_DIR}/")
    if max_sessions > 0:
        train, test = train[:max_sessions], test[:max_sessions]
        print(f"TEST RUN: {max_sessions} session(s) per split")
    if sessions:
        chosen = set(sessions.split(","))
        train, test = [s for s in train if s in chosen], [s for s in test if s in chosen]
        print(f"TEST RUN: only the chosen walks: train {train}, test {test}")
    if not all_frames and not sessions:           # the add-on and chosen-walk tests leave sanpo_meta as it is
        print("Saving SANPO label files and split lists ...")
        print("Metadata saved to:", save_metadata.remote(train, test))

    stride, index_name = (1, ALL_FRAMES_INDEX_DIR) if all_frames else (FRAME_STRIDE, "index")
    jobs = [(s, "train", stride, index_name) for s in train] + [(s, "test", stride, index_name) for s in test]
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
