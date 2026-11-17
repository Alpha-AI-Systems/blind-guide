"""Prepare SANPO-Real for ONE camera, in ONE script (runs on Modal): download, convert, resize, pack, list.
Native frame rate (every labeled frame, 15 fps) and native resolution. Made for the head camera; works for chest too.

WHAT IT MAKES (on the Modal Volume VOLUME_NAME, everything under OUTPUT_DIR):
    frames/<split>/<walk>_<frame>.jpg          full-size frame (2208x1242 or 1920x1080), JPG quality JPEG_QUALITY
    masks/<split>/<walk>_<frame>.png           SANPO's original mask, unchanged (red = class id, blue = object id)
    index/<walk>.csv                           one row per frame: camera, split, frame id, size, paths, label source
                                               (HUMAN / MACHINE), in_5fps (position 0 mod 3 = the 5 fps set, the
                                               chest download's rule), camera position, distance since the last frame
    sanpo_meta/                                labelmap.json, labeltype.json, official splits, per-walk
                                               descriptions/ and annotation_types/, run settings
    views/<VIEW_PREFIX>_short<size>/           one training view per image size in IMAGE_SIZES:
        tars/<walk>.tar                        the walk's resized frames (PNG) and one-channel masks (PNG) with the raw
                                               SANPO class ids, members frames/<split>/<stem>.png, masks/<split>/<stem>.png
        train.txt / val.txt                    5 fps frames of the training / validation walks (val = VALIDATION_FILE)
        train_15fps.txt                        every frame of the training walks, in order per walk (data.train_fps)
        train_human.txt                        human-labeled training frames (5 fps set)
        test.txt / test_human.txt              5 fps frames of the official test walks (only if "test" in SPLITS)
        data.yaml, view.yaml                   same classes and label mapping as the real chest views

HOW (per walk, one container, nothing read back from the volume): list the walk's files in the public bucket ->
per frame (THREADS_PER_WALK at once): download frame + mask -> frame as JPG, mask as is -> resize like Ultralytics
8.4.208 at training (short side = size; frames cv2.INTER_LINEAR from the saved JPG, masks INTER_NEAREST) -> index CSV ->
one tar per size (written last = the walk is complete). Then the lists, data.yaml and view.yaml from all index CSVs.

FACTS (checked on a head walk, 2026-11-17): sanpo-real/<walk>/camera_<cam>/left/{video_frames,segmentation_masks}/
<6-digit id>.png with the same ids (all frames labeled, 15 fps), left/frame_segmentation_annotation_type.json
(frame number -> HUMAN_ANNOTATED / MACHINE_ANNOTATED), camera_<cam>/camera_poses.csv, <walk>/description.json.
Each walk has labels for one camera only (the walks of the other camera are skipped).

RESUMABLE: a walk whose index CSV and all its tars exist is skipped; files are written to temporary names first.
COST (lesson from the synthetic run): Modal bills reserved cores and memory; 20 containers x 1 core x 4 GB.

Run from the repo root (PowerShell). Quick test first (2 walks per split), then everything:
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session5.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/head/prepare_sanpo_real_head_modal.py --max-walks 2
    python -m modal run fast_path/object_detection/data_code/sanpo/head/prepare_sanpo_real_head_modal.py

SANPO: https://google-research-datasets.github.io/sanpo_dataset/ | https://github.com/google-research-datasets/sanpo_dataset
       paper https://arxiv.org/abs/2309.12172 | license CC BY 4.0
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
CAMERA = "head"                        # "head" or "chest": the camera whose labeled walks are prepared
VOLUME_NAME = "sanpo"                  # Modal Volume (one per Modal session/workspace)
OUTPUT_DIR = f"sanpo_real_{CAMERA}"    # folder inside the volume: everything goes here
SPLITS = ["train"]                     # official splits to prepare ("train"; add "test" to evaluate on this camera)
VALIDATION_FILE = "validation_split/results/validation_walks.txt"   # next to this script (this camera's split)
JPEG_QUALITY = 95                      # full-size frames: PNG -> JPG at this quality (as the real chest data)
IMAGE_SIZES = [352]                    # one training view per size (short side, multiple of 32); e.g. [352, 544]
VIEW_PREFIX = "semantic_v1"            # view folder name: <VIEW_PREFIX>_short<size>
IGNORED_CLASSES = ["unlabeled", "bus stop"]   # -> 255 (ignored): the same 29 classes as every other dataset
MAX_CONTAINERS = 20                    # walks processed in parallel (a v1 Volume allows only a few commits at once)
COMMIT_RETRIES = 8                     # a failed volume commit is retried this many times (growing random wait)
THREADS_PER_WALK = 16                  # frames processed at once inside one walk (downloads wait on the network)
CPU_PER_CONTAINER = 1                  # cores per container (mostly waiting on downloads; Modal bills reserved cores)
MEMORY_MB_PER_CONTAINER = 4096         # a walk's resized PNGs are held in memory until its tar is written
PRINT_EVERY = 100                      # progress line every N frames inside a walk
# =============================================================================

SANPO_ROOT = "sanpo_dataset/v0"
BUCKET_URL = "https://storage.googleapis.com/gresearch"
LIST_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
VOLUME_MOUNT_PATH = "/data"

app = modal.App(f"blind-guide-sanpo-real-{CAMERA}-prepare")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install(
    "numpy", "opencv-python-headless", "pyyaml")


# ----------------------------------------------------------------------------- helpers

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


def bucket_names(prefix: str, max_results: int = 0) -> list[str]:
    """Object names under a bucket prefix (all pages, or the first max_results)."""
    import json
    import urllib.parse

    names, token = [], None
    while True:
        query = {"prefix": prefix, "maxResults": str(max_results or 1000), "fields": "items(name),nextPageToken"}
        if token:
            query["pageToken"] = token
        data = json.loads(http_get(f"{LIST_API}?{urllib.parse.urlencode(query)}"))
        names += [item["name"] for item in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token or max_results:
            return names


def training_size(height: int, width: int, image_size: int) -> tuple[int, int]:
    """(width, height) Ultralytics 8.4.208 gives a frame at TRAINING time (short side = image_size); the same formula
    as ultralytics/data/base.py load_image with resize_short and as make_resized_view_modal.py."""
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
    """volume.commit(), retried with a growing random wait: a v1 Volume allows only a few commits at the same time
    (Modal: "no more than five writers") and a commit can fail with DataLossError; the files are still here."""
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
    """Download, convert, resize and pack ONE walk of CAMERA (every labeled frame). Returns a summary dict.
    walk: session id; split: official split folder ("train" / "test")."""
    import csv
    import io
    import json
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
            frames_done = sum(1 for _ in csv.DictReader(handle))
        return {"walk": walk, "split": split, "frames": frames_done, "skipped": True, "seconds": 0.0}

    start = time.time()
    relative = f"sanpo-real/{walk}/camera_{CAMERA}"
    names = bucket_names(f"{SANPO_ROOT}/{relative}/left/")
    frame_ids = sorted(Path(n).stem for n in names if "/video_frames/" in n and n.endswith(".png"))
    mask_ids = {Path(n).stem for n in names if "/segmentation_masks/" in n and n.endswith(".png")}
    if not mask_ids or not mask_ids <= set(frame_ids):
        raise RuntimeError(f"{walk}: {len(mask_ids)} masks, {len(frame_ids)} frames (masks must be a subset)")
    labeled = sorted(mask_ids)                               # only frames with a mask are used (all, at 15 fps)
    sources = {int(k): v for k, v in json.loads(
        bucket_file(f"{relative}/left/frame_segmentation_annotation_type.json")).items()}
    poses = {}
    try:                                                     # camera_poses.csv: one row per video frame, in order
        rows = list(csv.DictReader(io.StringIO(bucket_file(f"{relative}/camera_poses.csv").decode())))
        if len(rows) == len(frame_ids):
            poses = dict(zip(frame_ids, rows))
    except Exception:
        pass
    meta = root / "sanpo_meta"
    for folder, name in (("descriptions", f"sanpo-real/{walk}/description.json"),
                         ("annotation_types", f"{relative}/left/frame_segmentation_annotation_type.json")):
        (meta / folder).mkdir(parents=True, exist_ok=True)
        try:
            (meta / folder / f"{walk}.json").write_bytes(bucket_file(name))
        except Exception:
            pass
    print(f"[{split}] {walk[:16]}: {len(labeled)} labeled frames, poses {'yes' if poses else 'no'}", flush=True)

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
        frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)          # the views start from the saved JPG, like the chest views
        class_ids = np.ascontiguousarray(mask[:, :, 2])      # one channel: raw SANPO class ids
        resized = {}
        for size in IMAGE_SIZES:
            width, height = training_size(frame.shape[0], frame.shape[1], size)
            small_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            small_mask = cv2.resize(class_ids, (width, height), interpolation=cv2.INTER_NEAREST)
            resized[size] = (cv2.imencode(".png", small_frame)[1].tobytes(), cv2.imencode(".png", small_mask)[1].tobytes())
        return {"frame_id": frame_id, "stem": stem, "height": frame.shape[0], "width": frame.shape[1], "resized": resized}

    results = []
    with ThreadPoolExecutor(THREADS_PER_WALK) as pool:
        for number, result in enumerate(pool.map(process, labeled), 1):
            results.append(result)
            if number % PRINT_EVERY == 0 or number == len(labeled):
                elapsed = time.time() - start
                print(f"[{split}] {walk[:16]}: {number}/{len(labeled)} frames | elapsed {format_duration(elapsed)} | "
                      f"ETA {format_duration(elapsed / number * (len(labeled) - number))}", flush=True)

    rows, previous = [], None
    for position, result in enumerate(results):
        pose = poses.get(result["frame_id"])
        position_xyz = (float(pose["pos_x"]), float(pose["pos_y"]), float(pose["pos_z"])) if pose else None
        step = math.dist(position_xyz, previous) if position_xyz and previous else ""
        previous = position_xyz
        source = sources.get(int(result["frame_id"]), "")
        rows.append({"session": walk, "frame_id": result["frame_id"], "split": split, "camera": CAMERA,
                     "width": result["width"], "height": result["height"],
                     "frame": f"frames/{split}/{result['stem']}.jpg", "mask": f"masks/{split}/{result['stem']}.png",
                     "label_source": {"HUMAN_ANNOTATED": "human", "MACHINE_ANNOTATED": "machine"}.get(source, source),
                     "in_5fps": position % 3 == 0,
                     "pos_x": position_xyz[0] if position_xyz else "", "pos_y": position_xyz[1] if position_xyz else "",
                     "pos_z": position_xyz[2] if position_xyz else "", "step_m": round(step, 5) if step != "" else ""})
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
    human = sum(row["label_source"] == "human" for row in rows)
    return {"walk": walk, "split": split, "frames": len(rows), "human": human, "skipped": False,
            "seconds": round(time.time() - start, 1)}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def save_metadata(settings: dict, split_lists: dict) -> dict:
    """Store the label files, official split lists and this run's settings in OUTPUT_DIR/sanpo_meta.
    Returns {raw id: training id or 255} and the 29 training class names (the real chest views' mapping)."""
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
    commit_with_retry()
    return {"label_mapping": mapping, "names": kept}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def write_lists(validation_walks: list[str], names: list[str], label_mapping: dict) -> dict:
    """Write each view's image lists, data.yaml (only with validation walks) and view.yaml from all index CSVs.
    Returns {view: {list: frames}} and the validation walks that were not found."""
    import csv
    from datetime import datetime, timezone
    from pathlib import Path

    import yaml

    root = Path(VOLUME_MOUNT_PATH) / OUTPUT_DIR
    validation = set(validation_walks)
    lists = {name: [] for name in ("train", "train_15fps", "train_human", "val", "test", "test_human")}
    walks_seen = set()
    for index_path in sorted((root / "index").glob("*.csv")):
        with open(index_path, newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: row["frame_id"])
        if not rows:
            continue
        walk, split = rows[0]["session"], rows[0]["split"]
        walks_seen.add(walk)
        part = "test" if split == "test" else ("val" if walk in validation else "train")
        for row in rows:
            line = f"./frames/{row['split']}/{Path(row['frame']).stem}.png"
            five_fps = row["in_5fps"] == "True"
            if part == "train":
                lists["train_15fps"].append(line)
            if not five_fps:
                continue
            lists[part].append(line)
            if row["label_source"] == "human" and part in ("train", "test"):
                lists[f"{part}_human"].append(line)
    missing = sorted(validation - walks_seen)
    counts = {}
    for size in IMAGE_SIZES:
        view = root / "views" / view_name(size)
        view.mkdir(parents=True, exist_ok=True)
        for name, lines in lists.items():
            if lines or name in ("train", "val"):
                (view / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lists["val"]:
            data = {"train": "train.txt", "val": "val.txt"}
            if lists["test_human"]:
                data["test"] = "test_human.txt"
            data.update({"image_dir_name": "frames", "masks_dir": "masks", "nc": len(names), "names": dict(enumerate(names)),
                         "label_mapping": dict(sorted({int(k): v for k, v in label_mapping.items()}.items())),
                         "image_size": size})
            header = (f"# SANPO-Real {CAMERA} view '{view_name(size)}' (prepare_sanpo_real_modal.py): masks keep the raw SANPO\n"
                      f"# class ids; label_mapping turns them into the real chest views' 29 training ids (255 = ignored)\n")
            (view / "data.yaml").write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                                            encoding="utf-8")
        elif (view / "data.yaml").exists():
            (view / "data.yaml").unlink()
        counts[view_name(size)] = {name: len(lines) for name, lines in lists.items()}
        info = {"made": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "dataset": OUTPUT_DIR,
                "camera": CAMERA, "splits": SPLITS, "image_size": size, "jpeg_quality": JPEG_QUALITY,
                "resize": "short side = image_size, frames cv2.INTER_LINEAR (from the saved JPG), masks INTER_NEAREST",
                "lists": "5 fps = position 0 mod 3 of each walk's labeled frames; train_15fps = all, in order per walk",
                "ignored_classes": IGNORED_CLASSES, "validation_walks": sorted(validation),
                "validation_walks_not_found": missing, "frames_per_list": counts[view_name(size)]}
        (view / "view.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    commit_with_retry()
    return {"counts": counts, "validation_walks_not_found": missing}


@app.local_entrypoint()
def main(max_walks: int = 0):
    """max_walks > 0 = quick test: that many walks per split."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    print(f"SANPO-Real {CAMERA} -> Volume '{VOLUME_NAME}' / {OUTPUT_DIR} | splits {SPLITS} | sizes {IMAGE_SIZES} | "
          f"JPG quality {JPEG_QUALITY}")
    split_lists = {split: sorted(bucket_file(f"sanpo-real/splits/{split}_session_ids.txt").decode().split())
                   for split in ("train", "test")}

    def has_labels(walk: str) -> bool:
        return bool(bucket_names(f"{SANPO_ROOT}/sanpo-real/{walk}/camera_{CAMERA}/left/segmentation_masks/", 1))

    jobs = []
    for split in SPLITS:
        print(f"finding the {split} walks with {CAMERA} labels (about a minute) ...", flush=True)
        with ThreadPoolExecutor(32) as pool:
            labeled = [walk for walk, ok in zip(split_lists[split], pool.map(has_labels, split_lists[split])) if ok]
        print(f"  {split}: {len(labeled)} of {len(split_lists[split])} walks have {CAMERA} labels")
        jobs += [(walk, split) for walk in (labeled[:max_walks] if max_walks > 0 else labeled)]
    if max_walks > 0:
        print(f"TEST RUN: {max_walks} walk(s) per split")

    validation_file = Path(__file__).resolve().parent / VALIDATION_FILE
    validation_walks = validation_file.read_text(encoding="utf-8").split() if validation_file.exists() else []
    print(f"validation walks ({CAMERA}): {len(validation_walks)} from {validation_file}"
          + ("" if validation_walks else " (NOT FOUND: no val list and no data.yaml)"))
    settings = {"camera": CAMERA, "splits": SPLITS, "jpeg_quality": JPEG_QUALITY, "image_sizes": IMAGE_SIZES,
                "view_prefix": VIEW_PREFIX, "ignored_classes": IGNORED_CLASSES, "max_walks": max_walks,
                "validation_file": VALIDATION_FILE}
    # save_metadata takes: settings, official split lists; returns: the label mapping and the 29 class names
    metadata = save_metadata.remote(settings, split_lists)

    print(f"{len(jobs)} walks, {MAX_CONTAINERS} in parallel ...", flush=True)
    start, done, failed, frames, human = time.time(), 0, 0, 0, 0
    for number, result in enumerate(prepare_walk.starmap(jobs, order_outputs=False, return_exceptions=True), 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{number}/{len(jobs)}] FAILED: {result!r}", flush=True)
            continue
        done += 0 if result["skipped"] else 1
        frames += result["frames"]
        human += result.get("human", 0)
        elapsed = time.time() - start
        eta = elapsed / done * (len(jobs) - number) if done else 0
        took = " (already done)" if result["skipped"] else " in {:.0f}s".format(result["seconds"])
        print(f"[{number}/{len(jobs)}] {result['split']:<5} {result['walk'][:16]}: {result['frames']} frames{took} | "
              f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}", flush=True)
    if failed:
        print(f"{failed} walk(s) failed: lists NOT written. Run the same command again (finished walks are skipped).")
        return
    # write_lists takes: the validation walks, the class names, the label mapping; returns: frames per list per view
    result = write_lists.remote(validation_walks, metadata["names"], metadata["label_mapping"])
    if result["validation_walks_not_found"]:
        print(f"WARNING: validation walks not prepared here: {result['validation_walks_not_found'][:5]} ...")
    print(f"DONE in {format_duration(time.time() - start)}: {frames:,} frames | lists: {result['counts']}")
