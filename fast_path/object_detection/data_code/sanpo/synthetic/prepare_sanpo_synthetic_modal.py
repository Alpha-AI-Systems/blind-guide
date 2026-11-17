"""Prepare SANPO-Synthetic for training, in ONE script (runs on Modal): download, convert, resize, pack, list.

WHAT IT MAKES (on the Modal Volume VOLUME_NAME, everything under OUTPUT_DIR):
    frames/<split>/<walk>_<frame>.jpg          full-size frame (2208x1242), JPG quality JPEG_QUALITY (like the real data)
    masks/<split>/<walk>_<frame>.png           SANPO's original mask, unchanged (red = class id, blue = object id)
    index/<walk>.csv                           one row per frame: camera, split, frame id, size, paths, camera position,
                                               distance walked since the previous frame (reveals the walk's FPS)
    views/<VIEW_PREFIX>_short<size>/           one training view per image size in IMAGE_SIZES:
        tars/<walk>.tar                        the walk's resized frames (PNG) and ONE-channel masks (PNG) holding the
                                               raw SANPO class ids (the red channel), exactly like the real views;
                                               members frames/<split>/<stem>.png and masks/<split>/<stem>.png
                                               (what training/shared_training_code.py copies to local disk)
        <list>.txt                             image lists (see LISTS below), paths "./frames/<split>/<stem>.png"
        data.yaml, view.yaml                   Ultralytics dataset file (same 29 classes and label mapping as the real
                                               chest views) and the settings and counts of this view
    sanpo_meta/                                labelmap.json, labeltype.json, the official split lists, run settings

HOW (per walk, one container, nothing read back from the volume):
    1. list the walk's files in the public bucket (gs://gresearch/sanpo_dataset/v0/sanpo-synthetic/<walk>/camera_<cam>)
    2. per frame (THREADS_PER_WALK at once): download frame PNG + mask PNG -> save the frame as JPG and the mask as is
       -> for every image size: resize like Ultralytics 8.4.208 does at training (short side = size, cv2.INTER_LINEAR;
       mask: red channel = raw class id, cv2.INTER_NEAREST) -> keep the PNG bytes in memory
    3. write the walk's index CSV, then one tar per image size (written last = the walk is complete)
    Then one final step writes the lists, data.yaml and view.yaml from all index CSVs.

FACTS this relies on (checked on 8 walks, 2026-11-17, scratchpad synthetic_format_check.py):
    frames and masks are 2208x1242 RGB PNG; masks use the real data's encoding (red = class id from the shared
    labelmap.json, ids 0..30; blue = object id); every frame has a mask (annotation type "SYNTHETIC");
    each walk has exactly one camera (chest or head); official splits: sanpo-synthetic/splits/{train,test}_session_ids.txt.

LISTS (written in each view; the synthetic validation walks come from VALIDATION_WALKS_FILE when it exists):
    train.txt / val.txt                        all cameras in CAMERAS (training walks / validation walks)
    train_chest.txt, train_head.txt, ...       per camera
    without a validation file: only train*.txt and no data.yaml (Ultralytics needs a val list); run again after
    the validation split notebook made the file (only the lists step runs: finished walks are skipped).

RESUMABLE: a walk whose index CSV and all its tars exist is skipped; files are written to temporary names and renamed.

Run from the repo root (PowerShell). Quick test first (2 walks per camera), then everything:
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/synthetic/prepare_sanpo_synthetic_modal.py --max-walks 2
    python -m modal run fast_path/object_detection/data_code/sanpo/synthetic/prepare_sanpo_synthetic_modal.py

SANPO: https://google-research-datasets.github.io/sanpo_dataset/ | https://github.com/google-research-datasets/sanpo_dataset
       paper https://arxiv.org/abs/2309.12172 | license CC BY 4.0
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"                  # Modal Volume (one per Modal session/workspace)
OUTPUT_DIR = "sanpo_synthetic"         # folder inside the volume: everything goes here
CAMERAS = ["chest", "head"]            # which camera's walks ("chest", "head" or both)
SPLITS = ["train"]                     # official splits to prepare ("train"; "test" only if ever needed)
JPEG_QUALITY = 95                      # full-size frames: PNG -> JPG at this quality (as the real data)
IMAGE_SIZES = [352]                    # one training view per size (short side, multiple of 32); e.g. [352, 544]
VIEW_PREFIX = "semantic_v1"            # view folder name: <VIEW_PREFIX>_short<size>
IGNORED_CLASSES = ["unlabeled", "bus stop"]   # -> 255 (ignored), same 29 classes as the real chest views
VALIDATION_WALKS_FILE = "validation_split/results/validation_walks.txt"  # next to this script; made by the notebook
MAX_CONTAINERS = 20                    # walks processed in parallel (50 made the v1 Volume's commits fail)
COMMIT_RETRIES = 8                     # a failed volume commit is retried this many times (growing random wait)
THREADS_PER_WALK = 16                  # frames processed at once inside one walk (downloads wait on the network)
CPU_PER_CONTAINER = 1                  # cores per container: mostly waiting on downloads; Modal bills reserved cores
                                       # (4 cores x 50 containers cost about $12.7/h in the first run)
MEMORY_MB_PER_CONTAINER = 4096         # a walk's resized PNGs are held in memory until its tar is written
PRINT_EVERY = 25                       # progress line every N frames inside a walk
# =============================================================================

SANPO_ROOT = "sanpo_dataset/v0"
BUCKET_URL = "https://storage.googleapis.com/gresearch"
LIST_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
VOLUME_MOUNT_PATH = "/data"

app = modal.App("blind-guide-sanpo-synthetic-prepare")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install(
    "numpy", "opencv-python-headless", "pyyaml")


# ----------------------------------------------------------------------------- helpers (bucket, sizes, time)

def http_get(url: str, retries: int = 5) -> bytes:
    """HTTP GET with retries (exponential back-off). Returns the body."""
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


def bucket_file(relative_path: str) -> bytes:
    """A file of the public SANPO bucket, path relative to sanpo_dataset/v0."""
    import urllib.parse

    return http_get(f"{BUCKET_URL}/{urllib.parse.quote(f'{SANPO_ROOT}/{relative_path}')}")


def bucket_listing(prefix: str, delimiter: bool = False) -> tuple[list[str], list[str]]:
    """(sub-folders, object names) under a bucket prefix, all pages."""
    import json
    import urllib.parse

    folders, names, token = [], [], None
    while True:
        query = {"prefix": prefix, "maxResults": "1000", "fields": "prefixes,items(name),nextPageToken"}
        if delimiter:
            query["delimiter"] = "/"
        if token:
            query["pageToken"] = token
        data = json.loads(http_get(f"{LIST_API}?{urllib.parse.urlencode(query)}"))
        folders += data.get("prefixes", [])
        names += [item["name"] for item in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            return folders, names


def training_size(height: int, width: int, image_size: int) -> tuple[int, int]:
    """(width, height) Ultralytics 8.4.208 gives a frame at TRAINING time (short side = image_size).

    Same formula as ultralytics/data/base.py load_image with resize_short (and as make_resized_view_modal.py),
    so the resized frame is exactly what Ultralytics would produce from the full-size frame.
    """
    import math

    ratio = image_size / min(height, width)
    if ratio == 1:
        return width, height
    return (math.ceil(width * ratio), image_size) if height < width else (image_size, math.ceil(height * ratio))


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s"."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def view_name(image_size: int) -> str:
    return f"{VIEW_PREFIX}_short{image_size}"


def commit_with_retry() -> None:
    """volume.commit(), retried with a growing random wait. A v1 Volume allows only a few commits at the same time
    (Modal: "no more than five writers"); with many walks finishing together a commit can fail with
    DataLossError ("failed to publish commit"). The walk's files are still in this container, so retrying is safe."""
    import random
    import time

    for attempt in range(COMMIT_RETRIES):
        try:
            volume.commit()
            return
        except Exception as error:
            if attempt == COMMIT_RETRIES - 1:
                raise
            wait = min(60, 2 ** attempt) + random.uniform(0, 5)
            print(f"  commit failed ({type(error).__name__}), retry {attempt + 1}/{COMMIT_RETRIES - 1} in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)


# ----------------------------------------------------------------------------- Modal functions

@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=3 * 3600, max_containers=MAX_CONTAINERS,
              retries=2, cpu=CPU_PER_CONTAINER, memory=MEMORY_MB_PER_CONTAINER)
def prepare_walk(walk: str, split: str) -> dict:
    """Download, convert, resize and pack ONE walk (all frames). Returns a summary dict.

    walk: session id; split: official split ("train"/"test").
    """
    import csv
    import io
    import math
    import os
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import cv2
    import numpy as np

    root = Path(VOLUME_MOUNT_PATH) / OUTPUT_DIR
    index_path = root / "index" / f"{walk}.csv"
    tar_paths = {size: root / "views" / view_name(size) / "tars" / f"{walk}.tar" for size in IMAGE_SIZES}
    if index_path.exists() and all(path.exists() for path in tar_paths.values()):
        with open(index_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {"walk": walk, "split": split, "camera": rows[0]["camera"] if rows else "", "frames": len(rows),
                "skipped": True, "seconds": 0.0}

    start = time.time()
    cameras, _ = bucket_listing(f"{SANPO_ROOT}/sanpo-synthetic/{walk}/", delimiter=True)
    if len(cameras) != 1:
        raise RuntimeError(f"{walk}: expected one camera folder, found {cameras}")
    camera = cameras[0].rstrip("/").rsplit("/", 1)[-1].removeprefix("camera_")
    relative = f"sanpo-synthetic/{walk}/camera_{camera}"
    _, names = bucket_listing(f"{SANPO_ROOT}/{relative}/left/")
    frame_ids = sorted(Path(name).stem for name in names if "/video_frames/" in name and name.endswith(".png"))
    mask_ids = {Path(name).stem for name in names if "/segmentation_masks/" in name and name.endswith(".png")}
    if set(frame_ids) != mask_ids:
        raise RuntimeError(f"{walk}: {len(frame_ids)} frames but {len(mask_ids)} masks")
    poses = list(csv.DictReader(io.StringIO(bucket_file(f"{relative}/camera_poses.csv").decode())))
    if len(poses) != len(frame_ids):
        raise RuntimeError(f"{walk}: {len(poses)} camera poses for {len(frame_ids)} frames")
    print(f"[{camera}] {walk[:16]}: {len(frame_ids)} frames", flush=True)

    frame_dir, mask_dir = root / "frames" / split, root / "masks" / split
    for folder in (frame_dir, mask_dir, index_path.parent):
        folder.mkdir(parents=True, exist_ok=True)
    def atomic_write(path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def process(frame_id: str) -> dict:
        """One frame: save full-size JPG + original mask; return its resized PNG bytes per size and its size."""
        stem = f"{walk}_{frame_id}"
        frame_png = bucket_file(f"{relative}/left/video_frames/{frame_id}.png")
        mask_png = bucket_file(f"{relative}/left/segmentation_masks/{frame_id}.png")
        frame = cv2.imdecode(np.frombuffer(frame_png, np.uint8), cv2.IMREAD_COLOR)          # BGR
        mask = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_COLOR)            # BGR: red = index 2
        if frame is None or mask is None or frame.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"{stem}: unreadable frame or mask, or different sizes")
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"{stem}: JPG encoding failed")
        atomic_write(frame_dir / f"{stem}.jpg", jpg.tobytes())
        atomic_write(mask_dir / f"{stem}.png", mask_png)
        # the training frame is decoded from the saved JPG, like the real data (its views start from the JPGs)
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        class_ids = np.ascontiguousarray(mask[:, :, 2])      # one channel: raw SANPO class ids (as the real views)
        resized = {}
        for size in IMAGE_SIZES:
            width, height = training_size(frame.shape[0], frame.shape[1], size)
            small_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            small_mask = cv2.resize(class_ids, (width, height), interpolation=cv2.INTER_NEAREST)
            resized[size] = (cv2.imencode(".png", small_frame)[1].tobytes(),
                             cv2.imencode(".png", small_mask)[1].tobytes())
        return {"frame_id": frame_id, "stem": stem, "height": frame.shape[0], "width": frame.shape[1],
                "resized": resized}

    results = []
    with ThreadPoolExecutor(THREADS_PER_WALK) as pool:
        for number, result in enumerate(pool.map(process, frame_ids), 1):
            results.append(result)
            if number % PRINT_EVERY == 0 or number == len(frame_ids):
                elapsed = time.time() - start
                print(f"[{camera}] {walk[:16]}: {number}/{len(frame_ids)} frames | elapsed {format_duration(elapsed)} | "
                      f"ETA {format_duration(elapsed / number * (len(frame_ids) - number))}", flush=True)

    rows, previous = [], None
    for result, pose in zip(results, poses):
        position = (float(pose["pos_x"]), float(pose["pos_y"]), float(pose["pos_z"]))
        step = math.dist(position, previous) if previous is not None else ""
        previous = position
        rows.append({"session": walk, "frame_id": result["frame_id"], "split": split, "camera": camera,
                     "width": result["width"], "height": result["height"],
                     "frame": f"frames/{split}/{result['stem']}.jpg", "mask": f"masks/{split}/{result['stem']}.png",
                     "pos_x": position[0], "pos_y": position[1], "pos_z": position[2],
                     "step_m": round(step, 5) if step != "" else "", "tracking_state": pose.get("tracking_state", "")})
    temporary_index = index_path.with_name(index_path.name + ".tmp")
    with open(temporary_index, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_index, index_path)

    for size, tar_path in tar_paths.items():                 # the tars are written last: the walk is then complete
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tar_path.with_name(tar_path.stem + ".tmp.tar")
        with tarfile.open(temporary, "w") as archive:
            for result in results:
                for kind, content in zip(("frames", "masks"), result["resized"][size]):
                    info = tarfile.TarInfo(f"{kind}/{split}/{result['stem']}.png")
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
        os.replace(temporary, tar_path)
    # commit_with_retry takes: nothing; returns: nothing (raises after COMMIT_RETRIES failed commits)
    commit_with_retry()
    steps =[row["step_m"] for row in rows if row["step_m"] != ""]
    median_step = sorted(steps)[len(steps) // 2] if steps else None
    return {"walk": walk, "split": split, "camera": camera, "frames": len(rows), "skipped": False,
            "seconds": round(time.time() - start, 1), "median_step_m": median_step}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def save_metadata(settings: dict, split_lists: dict) -> dict:
    """Store the label files, official split lists and this run's settings in OUTPUT_DIR/sanpo_meta.
    Returns {raw id: training id or 255} and the training class names (the real chest views' mapping)."""
    import json
    from pathlib import Path

    meta = Path(VOLUME_MOUNT_PATH) / OUTPUT_DIR / "sanpo_meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("labelmap.json", "labeltype.json"):
        (meta / name).write_bytes(bucket_file(name))
    for split, walks in split_lists.items():
        (meta / f"{split}_session_ids.txt").write_text("\n".join(walks) + "\n")
    (meta / "prepare_settings.json").write_text(json.dumps(settings, indent=2))
    class_ids = json.loads((meta / "labelmap.json").read_text())
    kept = [name for name, class_id in sorted(class_ids.items(), key=lambda item: item[1]) if name not in IGNORED_CLASSES]
    mapping = {int(class_id): (kept.index(name) if name in kept else 255) for name, class_id in class_ids.items()}
    volume.commit()
    return {"label_mapping": mapping, "names": kept}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def write_lists(validation_walks: list[str], names: list[str], label_mapping: dict) -> dict:
    """Write each view's image lists, data.yaml (only with validation walks) and view.yaml from all index CSVs.
    Returns {view: {list: frames}}."""
    import csv
    from datetime import datetime, timezone
    from pathlib import Path

    import yaml

    root = Path(VOLUME_MOUNT_PATH) / OUTPUT_DIR
    validation = set(validation_walks)
    lists: dict[str, list[str]] = {"train": [], "val": []}
    walks_per_list: dict[str, set] = {}
    for index_path in sorted((root / "index").glob("*.csv")):
        with open(index_path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or rows[0]["split"] != "train":
            continue
        part = "val" if rows[0]["session"] in validation else "train"
        camera = rows[0]["camera"]
        for row in sorted(rows, key=lambda item: item["frame_id"]):
            line = f"./frames/{row['split']}/{Path(row['frame']).stem}.png"
            for name in (part, f"{part}_{camera}"):
                lists.setdefault(name, []).append(line)
                walks_per_list.setdefault(name, set()).add(rows[0]["session"])
    missing = validation - {walk for walks in walks_per_list.values() for walk in walks}
    counts = {}
    for size in IMAGE_SIZES:
        view = root / "views" / view_name(size)
        view.mkdir(parents=True, exist_ok=True)
        for name, lines in lists.items():
            (view / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lists["val"]:
            data = {"train": "train.txt", "val": "val.txt", "image_dir_name": "frames", "masks_dir": "masks",
                    "nc": len(names), "names": dict(enumerate(names)),
                    "label_mapping": dict(sorted({int(k): v for k, v in label_mapping.items()}.items())),
                    "image_size": size}
            header = (f"# SANPO-Synthetic view '{view_name(size)}' (prepare_sanpo_synthetic_modal.py): masks keep the raw\n"
                      f"# SANPO class ids; label_mapping turns them into the real chest views' 29 training ids\n")
            (view / "data.yaml").write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                                            encoding="utf-8")
        elif (view / "data.yaml").exists():
            (view / "data.yaml").unlink()
        counts[view_name(size)] = {name: len(lines) for name, lines in lists.items()}
        info = {"made": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "dataset": OUTPUT_DIR,
                "cameras": CAMERAS, "splits": SPLITS, "image_size": size, "jpeg_quality": JPEG_QUALITY,
                "resize": "short side = image_size, frames cv2.INTER_LINEAR (from the saved JPG), masks INTER_NEAREST",
                "ignored_classes": IGNORED_CLASSES, "validation_walks": sorted(validation),
                "validation_walks_not_found": sorted(missing),
                "walks_per_list": {name: len(walks) for name, walks in walks_per_list.items()},
                "frames_per_list": counts[view_name(size)]}
        (view / "view.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    volume.commit()
    return {"counts": counts, "validation_walks_not_found": sorted(missing)}


@app.local_entrypoint()
def main(max_walks: int = 0):
    """max_walks > 0 = quick test: that many walks per camera (and split)."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    print(f"SANPO-Synthetic -> Volume '{VOLUME_NAME}' / {OUTPUT_DIR} | cameras {CAMERAS} | splits {SPLITS} | "
          f"sizes {IMAGE_SIZES} | JPG quality {JPEG_QUALITY}")
    split_lists = {split: sorted(bucket_file(f"sanpo-synthetic/splits/{split}_session_ids.txt").decode().split())
                   for split in ("train", "test")}
    print(f"official splits: train {len(split_lists['train'])} walks, test {len(split_lists['test'])} walks")

    def camera_of(walk: str) -> str:
        folders, _ = bucket_listing(f"{SANPO_ROOT}/sanpo-synthetic/{walk}/", delimiter=True)
        return folders[0].rstrip("/").rsplit("/", 1)[-1].removeprefix("camera_") if folders else ""

    jobs = []
    for split in SPLITS:
        walks = split_lists[split]
        print(f"finding the camera of {len(walks)} {split} walks (about a minute) ...", flush=True)
        with ThreadPoolExecutor(32) as pool:
            cameras = list(pool.map(camera_of, walks))
        for camera in CAMERAS:
            chosen = [walk for walk, walk_camera in zip(walks, cameras) if walk_camera == camera]
            print(f"  {split} {camera}: {len(chosen)} walks")
            if max_walks > 0:
                chosen = chosen[:max_walks]
            jobs += [(walk, split) for walk in chosen]
    if max_walks > 0:
        print(f"TEST RUN: {max_walks} walk(s) per camera and split")

    settings = {"cameras": CAMERAS, "splits": SPLITS, "jpeg_quality": JPEG_QUALITY, "image_sizes": IMAGE_SIZES,
                "view_prefix": VIEW_PREFIX, "ignored_classes": IGNORED_CLASSES, "max_walks": max_walks}
    # save_metadata takes: settings, official split lists; returns: the label mapping and the 29 class names
    metadata = save_metadata.remote(settings, split_lists)

    print(f"{len(jobs)} walks, {MAX_CONTAINERS} in parallel ...", flush=True)
    start, done, failed, frames = time.time(), 0, 0, 0
    for number, result in enumerate(prepare_walk.starmap(jobs,
                                                         order_outputs=False, return_exceptions=True), 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{number}/{len(jobs)}] FAILED: {result!r}", flush=True)
            continue
        done += 0 if result["skipped"] else 1
        frames += result["frames"]
        elapsed = time.time() - start
        eta = elapsed / done * (len(jobs) - number) if done else 0
        step = f", median step {result['median_step_m']:.3f} m" if result.get("median_step_m") else ""
        took = " (already done)" if result["skipped"] else " in {:.0f}s".format(result["seconds"])
        print(f"[{number}/{len(jobs)}] {result['camera']:<5} {result['walk'][:16]}: {result['frames']} frames"
              f"{took}{step} | "
              f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}", flush=True)
    if failed:
        print(f"{failed} walk(s) failed: lists NOT written. Run the same command again (finished walks are skipped).")
        return

    validation_file = Path(__file__).resolve().parent / VALIDATION_WALKS_FILE
    validation_walks = validation_file.read_text(encoding="utf-8").split() if validation_file.exists() else []
    print(f"validation walks: {len(validation_walks)}" + ("" if validation_walks else
          f" ({validation_file} not found: only train lists, no data.yaml; run again after the validation split)"))
    # write_lists takes: the validation walks, the class names, the label mapping; returns: frames per list per view
    result = write_lists.remote(validation_walks, metadata["names"], metadata["label_mapping"])
    if result["validation_walks_not_found"]:
        print(f"WARNING: validation walks not prepared here: {result['validation_walks_not_found'][:5]} ...")
    print(f"DONE in {format_duration(time.time() - start)}: {frames:,} frames | lists: {result['counts']}")
