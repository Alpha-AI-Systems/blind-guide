"""Download SANPO's dense depth maps for the frames we already have (runs on Modal).

Which frames: exactly the frames listed in <DATASET>/index/*.csv (made by download_sanpo_modal.py),
so every frame gets its depth map. Files are stored UNCHANGED (SANPO format):
    <DATASET>/depth/<split>/<session>_<frame>.float16.gz

SANPO depth format ("depth_maps", dense, computed with CREStereo): gzip of float16 values;
the first two values are (height, width) = (720, 1280), the rest is height x width depth in
METERS for the left chest camera. To use it with a 2208x1242 frame, resize it to the frame size.

Used for distance-aware scores ("objects within 5 m / 10 m").

SANPO official links:
    Project page: https://google-research-datasets.github.io/sanpo_dataset/
    GitHub:       https://github.com/google-research-datasets/sanpo_dataset
    Data bucket:  gs://gresearch/sanpo_dataset/v0 (public, read over HTTPS), license CC BY 4.0

Resumable: existing files are skipped; files are written to a temp name then renamed;
the Volume is committed regularly; a finished session writes a small done-marker.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/download_sanpo_depth_modal.py --max-sessions 2   # quick test
    python -m modal run fast_path/object_detection/data_code/sanpo/download_sanpo_depth_modal.py                    # all sessions
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"  # folder made by download_sanpo_modal.py
DEPTH_FOLDER = "depth_maps"  # SANPO dense depth (CREStereo); "zed_depth_maps" = sparse sensor depth
MAX_CONTAINERS = 60  # sessions processed in parallel (146 sessions in total)
DOWNLOAD_THREADS = 32  # files downloaded at the same time inside each container
COMMIT_EVERY = 100  # save progress to the Volume every N files
# =============================================================================

SANPO_BUCKET_URL = "https://storage.googleapis.com/gresearch"
SANPO_ROOT = "sanpo_dataset/v0"
VOLUME_MOUNT_PATH = "/data"

app = modal.App("blind-guide-sanpo-depth-download")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12")


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s" (used in progress lines)."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def download_bytes(url: str, retries: int = 5) -> bytes:
    """HTTP GET with retries (exponential back-off)."""
    import time
    import urllib.request

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=3 * 3600,
              max_containers=MAX_CONTAINERS, retries=2)
def download_session(index_csv_name: str) -> dict:
    """Download the depth map of every frame of one session (one index CSV).

    Steps:
        1. Read the session's frames from its index CSV; skip the session if its done-marker exists.
        2. Download the missing depth files in parallel threads (temp file, then rename).
        3. Commit regularly; write the done-marker at the end.
    """
    import csv
    import os
    import time
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    session = index_csv_name.removesuffix(".csv")

    # Step 1: frames of the session
    with open(dataset_root / "index" / index_csv_name, newline="") as index_file:
        frames = list(csv.DictReader(index_file))
    split = frames[0]["split"]
    depth_directory = dataset_root / "depth" / split
    depth_directory.mkdir(parents=True, exist_ok=True)
    done_marker = dataset_root / "depth" / "_done" / f"{session}.done"
    if done_marker.exists():
        print(f"[{split}] {session}: already done, skipping", flush=True)
        return {"session": session, "split": split, "files": len(frames), "megabytes": 0.0, "skipped": True}

    # Step 2: download missing files in parallel
    def download_one(frame: dict) -> int:
        stem = Path(frame["frame"]).stem  # <session>_<frame id>
        target = depth_directory / f"{stem}.float16.gz"
        if target.exists() and target.stat().st_size > 0:
            return 0
        remote = f"{SANPO_ROOT}/sanpo-real/{session}/camera_chest/left/{DEPTH_FOLDER}/{frame['frame_id']}.float16.gz"
        data = download_bytes(f"{SANPO_BUCKET_URL}/{urllib.parse.quote(remote)}")
        temporary_path = target.with_name(target.name + ".tmp")
        temporary_path.write_bytes(data)  # original file, unchanged
        os.replace(temporary_path, target)
        return len(data)

    print(f"[{split}] {session}: {len(frames)} depth maps", flush=True)
    start_time, downloaded_bytes, finished = time.time(), 0, 0
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        futures = [pool.submit(download_one, frame) for frame in frames]
        for future in as_completed(futures):
            downloaded_bytes += future.result()
            finished += 1
            # Step 3: commit regularly + progress
            if finished % COMMIT_EVERY == 0 or finished == len(frames):
                volume.commit()
                elapsed = time.time() - start_time
                print(f"[{split}] {session}: {finished}/{len(frames)} | {downloaded_bytes / 1e6:.0f} MB | "
                      f"elapsed {format_duration(elapsed)} | "
                      f"ETA {format_duration(elapsed / finished * (len(frames) - finished))}", flush=True)

    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(f"{len(frames)} files\n")
    volume.commit()
    return {"session": session, "split": split, "files": len(frames),
            "megabytes": round(downloaded_bytes / 1e6, 1), "skipped": False}


@app.local_entrypoint()
def main(max_sessions: int = 0):
    """max_sessions > 0 = quick test on that many sessions."""
    import time

    print(f"Listing sessions in Volume '{VOLUME_NAME}' / {DATASET}/index ...", flush=True)
    index_files = sorted(entry.path.rsplit("/", 1)[-1] for entry in volume.listdir(f"{DATASET}/index")
                         if entry.path.endswith(".csv"))
    if max_sessions > 0:
        index_files = index_files[:max_sessions]
        print(f"TEST RUN: {len(index_files)} session(s)")
    print(f"Downloading SANPO '{DEPTH_FOLDER}' for {len(index_files)} sessions "
          f"({MAX_CONTAINERS} in parallel, {DOWNLOAD_THREADS} files per container) ...", flush=True)

    start_time, total_files, total_megabytes, newly_done, failed = time.time(), 0, 0.0, 0, 0
    results = download_session.map(index_files, order_outputs=False, return_exceptions=True)
    for session_number, result in enumerate(results, 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{session_number}/{len(index_files)}] FAILED: {result!r}", flush=True)
            continue
        total_files += result["files"]
        total_megabytes += result["megabytes"]
        newly_done += 0 if result["skipped"] else 1
        elapsed = time.time() - start_time
        remaining_seconds = elapsed / newly_done * (len(index_files) - session_number) if newly_done else 0
        status = "already done" if result["skipped"] else f"{result['megabytes']:.0f} MB"
        print(f"[{session_number}/{len(index_files)}] {result['split']:<5} {result['session']}: "
              f"{result['files']} files, {status} | total {total_megabytes / 1e3:.2f} GB | "
              f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining_seconds)}", flush=True)
    print(f"DONE: {total_files:,} files, {total_megabytes / 1e3:.2f} GB new | time {format_duration(time.time() - start_time)}"
          f"{f' | {failed} failed: re-run to retry' if failed else ''}")
