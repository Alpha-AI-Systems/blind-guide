"""Prepare SANPO-Real, HEAD camera, in ONE script (runs on Modal): download, convert, resize, pack, list.
Every labeled frame (native frame rate, 15 fps) at native resolution, plus resized training views.
The chest camera has the same script in ../chest/prepare_sanpo_real_chest_modal.py (only CAMERA differs).

WHAT IT MAKES (on the Modal Volume VOLUME_NAME, under sanpo_real_<CAMERA>/):
    native/<split>/<walk>.tar                  the walk's NATIVE data in ONE file: frames/<split>/<stem>.jpg (full size,
                                               2208x1242 or 1920x1080, JPG quality JPEG_QUALITY) and masks/<split>/<stem>.png
                                               (SANPO's original mask, unchanged: red = class id, blue = object id)
    index_v2/<walk>.csv                        one row per frame: camera, split, frame id, size, label source
                                               (HUMAN / MACHINE), in_5fps (position 0 mod 3 = the 5 fps set, as before),
                                               camera position, distance since the last frame
    sanpo_meta/                                labelmap.json, labeltype.json, official splits, per-walk descriptions/
                                               and annotation_types/ (read by training for the human-labeled lists)
    views/<VIEW_PREFIX>_short<size>/           one training view per size in IMAGE_SIZES (352 and 544):
        tars/<walk>.tar                        the walk's resized frames (PNG) and one-channel masks (PNG, raw SANPO
                                               class ids) for EVERY labeled frame (15 fps), members
                                               frames/<split>/<stem>.png and masks/<split>/<stem>.png
        train.txt / val.txt                    5 fps frames of the training / validation walks (val = VALIDATION_FILE)
        train_15fps.txt                        every frame of the training walks, in order per walk (data.train_fps 10)
        train_human.txt                        human-labeled training frames (5 fps set)
        test.txt / test_human.txt              5 fps frames of the official test walks
        data.yaml, view.yaml                   the same 29 classes and label mapping as every other view
Only a few files per walk: a session's volume is limited to 500,000 files (sessions 5 and 6 are at about 87%).

DETECTS WHAT IS THERE (run the same command again any time; nothing is deleted):
    - a walk whose index, native archive and every view archive exist is skipped
    - a walk whose native archive exists but a view is missing: the view is made from the native archive (no download)
    - otherwise each frame is taken from the volume if an earlier script saved it there (frames/<split>/<stem>.jpg +
      masks/<split>/<stem>.png: the 5 fps chest frames, all head frames), else downloaded from SANPO's bucket
    The earlier views (semantic_v1_*) are not touched, so earlier runs stay reproducible.

HOW (per walk, one container): list the walk's files in the public bucket -> per frame (THREADS_PER_WALK at once):
frame as JPG (the views start from the JPG, as every earlier view) + original mask -> resize like Ultralytics 8.4.208
at training (short side = size; frames cv2.INTER_LINEAR, masks INTER_NEAREST) -> written straight into the walk's
archives (temporary names, renamed at the end: an archive that exists is complete). Then the lists and data.yaml.

FACTS (checked on a head walk, 2026-11-17): sanpo-real/<walk>/camera_<cam>/left/{video_frames,segmentation_masks}/
<6-digit id>.png with the same ids (all frames labeled, 15 fps), left/frame_segmentation_annotation_type.json
(frame number -> HUMAN_ANNOTATED / MACHINE_ANNOTATED), camera_<cam>/camera_poses.csv, <walk>/description.json.
Each walk has labels for one camera only (the walks of the other camera are skipped).
COST: Modal bills reserved cores and memory; MAX_CONTAINERS x CPU_PER_CONTAINER cores x MEMORY_MB_PER_CONTAINER.

Run from the repo root (PowerShell). Quick test first (2 walks per split), then everything:
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session5.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/head/prepare_sanpo_real_head_modal.py --max-walks 2
    python -m modal run fast_path/object_detection/data_code/sanpo/head/prepare_sanpo_real_head_modal.py
Option: --image-sizes 352,544,736 (other sizes than IMAGE_SIZES).

SANPO: https://google-research-datasets.github.io/sanpo_dataset/ | https://github.com/google-research-datasets/sanpo_dataset
       paper https://arxiv.org/abs/2309.12172 | license CC BY 4.0
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
CAMERA = "head"                        # the camera whose labeled walks are prepared, in sanpo_real_<CAMERA>
VOLUME_NAME = "sanpo"                  # Modal Volume (one per Modal session/workspace)
SPLITS = ["train", "test"]             # official splits ("test" = the official test walks, scored by every test.py)
VALIDATION_FILE = "validation_split/results/validation_walks.txt"   # next to this script: this camera's validation walks
JPEG_QUALITY = 95                      # native frames: PNG -> JPG at this quality (as every earlier SANPO-Real copy)
IMAGE_SIZES = [352, 544]               # one training view per size (short side, multiple of 32): 640x352 and 960x544
VIEW_PREFIX = "semantic_v2"            # views <VIEW_PREFIX>_short<size>; v2 = every frame (15 fps) in tars/
                                       # (semantic_v1 views: 5 fps in tars/, 10 / 15 fps chest frames in tars_15fps_extra/)
NATIVE_DIR = "native"                  # per-walk native archives
INDEX_DIR = "index_v2"                 # per-walk index CSVs of this script (index/ holds the earlier scripts' indexes)
IGNORED_CLASSES = ["unlabeled", "bus stop"]   # -> 255 (ignored): the same 29 classes as every other dataset
MAX_CONTAINERS = 20                    # walks processed in parallel (a v1 Volume allows only a few commits at once)
COMMIT_RETRIES = 8                     # a failed volume commit is retried this many times (growing random wait)
THREADS_PER_WALK = 16                  # frames processed at once inside one walk (downloads wait on the network)
CPU_PER_CONTAINER = 2                  # cores per container (JPG and PNG encoding of two sizes)
MEMORY_MB_PER_CONTAINER = 6144         # frames are written to the archives as they are done (only a few in memory)
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

def dataset_dir(camera: str) -> str:
    """The camera's dataset folder inside the volume (each camera is its own dataset)."""
    return f"sanpo_real_{camera}"


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
def prepare_walk(camera: str, walk: str, split: str, sizes: list) -> dict:
    """Make ONE walk's native archive, index and view archives, whichever are missing. Returns a summary dict.
    camera: "chest" / "head"; walk: session id; split: official split folder; sizes: the views' image sizes."""
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

    root = Path(VOLUME_MOUNT_PATH) / dataset_dir(camera)
    index_path = root / INDEX_DIR / f"{walk}.csv"
    native_path = root / NATIVE_DIR / split / f"{walk}.tar"
    tar_paths = {size: root / "views" / view_name(size) / "tars" / f"{walk}.tar" for size in sizes}
    missing_sizes = [size for size, path in tar_paths.items() if not path.exists()]
    have_native = index_path.exists() and native_path.exists()
    tag = f"[{camera} {split}] {walk[:16]}"
    if have_native and not missing_sizes:
        with open(index_path, newline="") as handle:
            frames_done = sum(1 for _ in csv.DictReader(handle))
        return {"camera": camera, "walk": walk, "split": split, "frames": frames_done, "skipped": True, "seconds": 0.0}

    start = time.time()
    relative = f"sanpo-real/{walk}/camera_{camera}"
    native_members = {}
    if have_native:
        # a view is missing: everything comes from the walk's native archive (no download)
        with tarfile.open(native_path) as archive:
            for member in archive.getmembers():
                native_members[member.name] = archive.extractfile(member).read()
        with open(index_path, newline="") as handle:
            index_rows = list(csv.DictReader(handle))
        labeled = [row["frame_id"] for row in index_rows]
        print(f"{tag}: {len(labeled)} frames from the native archive, making views {missing_sizes}", flush=True)
    else:
        names = bucket_names(f"{SANPO_ROOT}/{relative}/left/")
        frame_ids = sorted(Path(name).stem for name in names if "/video_frames/" in name and name.endswith(".png"))
        mask_ids = {Path(name).stem for name in names if "/segmentation_masks/" in name and name.endswith(".png")}
        if not mask_ids or not mask_ids <= set(frame_ids):
            raise RuntimeError(f"{walk}: {len(mask_ids)} masks, {len(frame_ids)} frames (masks must be a subset)")
        labeled = sorted(mask_ids)                           # only frames with a mask are used (all, at 15 fps)
        sources = {int(key): value for key, value in json.loads(
            bucket_file(f"{relative}/left/frame_segmentation_annotation_type.json")).items()}
        poses = {}
        try:                                                 # camera_poses.csv: one row per video frame, in order
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
        print(f"{tag}: {len(labeled)} labeled frames, poses {'yes' if poses else 'no'}, views {missing_sizes}",
              flush=True)

    def process(frame_id: str) -> dict:
        """One frame: its JPG and original mask (native archive, an earlier script's file on the volume, or the
        bucket) and its resized PNGs for every missing view."""
        stem = f"{walk}_{frame_id}"
        jpg_name, mask_name = f"frames/{split}/{stem}.jpg", f"masks/{split}/{stem}.png"
        source = "native"
        if have_native:
            jpg, mask_png = native_members[jpg_name], native_members[mask_name]
        elif (root / jpg_name).exists() and (root / mask_name).exists():
            jpg, mask_png, source = (root / jpg_name).read_bytes(), (root / mask_name).read_bytes(), "volume"
        else:
            frame_png = bucket_file(f"{relative}/left/video_frames/{frame_id}.png")
            mask_png = bucket_file(f"{relative}/left/segmentation_masks/{frame_id}.png")
            frame = cv2.imdecode(np.frombuffer(frame_png, np.uint8), cv2.IMREAD_COLOR)      # BGR
            if frame is None:
                raise RuntimeError(f"{stem}: unreadable frame")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"{stem}: JPG encoding failed")
            jpg, source = encoded.tobytes(), "bucket"
        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)   # the views start from the JPG
        mask = cv2.imdecode(np.frombuffer(mask_png, np.uint8), cv2.IMREAD_COLOR)   # BGR: red = index 2
        if frame is None or mask is None or frame.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"{stem}: unreadable frame or mask, or different sizes")
        class_ids = np.ascontiguousarray(mask[:, :, 2])      # one channel: raw SANPO class ids
        resized = {}
        for size in missing_sizes:
            width, height = training_size(frame.shape[0], frame.shape[1], size)
            small_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            small_mask = cv2.resize(class_ids, (width, height), interpolation=cv2.INTER_NEAREST)
            resized[size] = (cv2.imencode(".png", small_frame)[1].tobytes(), cv2.imencode(".png", small_mask)[1].tobytes())
        return {"frame_id": frame_id, "stem": stem, "height": frame.shape[0], "width": frame.shape[1], "source": source,
                "native": None if have_native else (jpg, mask_png), "resized": resized}

    def add(archive, name: str, content: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    # every archive is written under a temporary name while the frames come in, then renamed: an archive that
    # exists is complete
    temporaries = {size: tar_paths[size].with_name(tar_paths[size].stem + ".tmp.tar") for size in missing_sizes}
    if not have_native:
        temporaries["native"] = native_path.with_name(native_path.stem + ".tmp.tar")
    for path in temporaries.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    archives = {key: tarfile.open(path, "w") for key, path in temporaries.items()}
    results, counted_sources = [], {}
    try:
        with ThreadPoolExecutor(THREADS_PER_WALK) as pool:
            for number, result in enumerate(pool.map(process, labeled), 1):
                for size in missing_sizes:
                    for kind, content in zip(("frames", "masks"), result["resized"][size]):
                        add(archives[size], f"{kind}/{split}/{result['stem']}.png", content)
                if not have_native:
                    jpg, mask_png = result["native"]
                    add(archives["native"], f"frames/{split}/{result['stem']}.jpg", jpg)
                    add(archives["native"], f"masks/{split}/{result['stem']}.png", mask_png)
                counted_sources[result["source"]] = counted_sources.get(result["source"], 0) + 1
                results.append({key: value for key, value in result.items() if key not in ("native", "resized")})
                if number % PRINT_EVERY == 0 or number == len(labeled):
                    elapsed = time.time() - start
                    print(f"{tag}: {number}/{len(labeled)} frames | elapsed {format_duration(elapsed)} | "
                          f"ETA {format_duration(elapsed / number * (len(labeled) - number))}", flush=True)
    finally:
        for archive in archives.values():
            archive.close()

    if not have_native:
        rows, previous = [], None
        for position, result in enumerate(results):
            pose = poses.get(result["frame_id"])
            position_xyz = (float(pose["pos_x"]), float(pose["pos_y"]), float(pose["pos_z"])) if pose else None
            step = math.dist(position_xyz, previous) if position_xyz and previous else ""
            previous = position_xyz
            source = sources.get(int(result["frame_id"]), "")
            rows.append({"session": walk, "frame_id": result["frame_id"], "split": split, "camera": camera,
                         "width": result["width"], "height": result["height"],
                         "frame": f"frames/{split}/{result['stem']}.jpg", "mask": f"masks/{split}/{result['stem']}.png",
                         "label_source": {"HUMAN_ANNOTATED": "human", "MACHINE_ANNOTATED": "machine"}.get(source, source),
                         "in_5fps": position % 3 == 0,
                         "pos_x": position_xyz[0] if position_xyz else "", "pos_y": position_xyz[1] if position_xyz else "",
                         "pos_z": position_xyz[2] if position_xyz else "", "step_m": round(step, 5) if step != "" else ""})
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_index = index_path.with_name(index_path.name + ".tmp")
        with open(temporary_index, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporaries["native"], native_path)
        os.replace(temporary_index, index_path)
    for size in missing_sizes:                                # the view archives last: the walk is then complete
        os.replace(temporaries[size], tar_paths[size])
    # commit_with_retry takes: nothing; returns: nothing (raises after COMMIT_RETRIES failed commits)
    commit_with_retry()
    return {"camera": camera, "walk": walk, "split": split, "frames": len(results), "skipped": False,
            "sources": counted_sources, "views": missing_sizes, "seconds": round(time.time() - start, 1)}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def save_metadata(camera: str, settings: dict, split_lists: dict) -> dict:
    """Store the label files, official split lists and this run's settings in sanpo_real_<camera>/sanpo_meta.
    Returns {raw id: training id or 255} and the 29 training class names (the real chest views' mapping)."""
    import json
    from pathlib import Path

    meta = Path(VOLUME_MOUNT_PATH) / dataset_dir(camera) / "sanpo_meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("labelmap.json", "labeltype.json"):
        (meta / name).write_bytes(bucket_file(name))
    for split, walks in split_lists.items():
        (meta / f"{split}_session_ids.txt").write_text("\n".join(walks) + "\n")
    (meta / "prepare_v2_settings.json").write_text(json.dumps(settings, indent=2))
    class_ids = json.loads((meta / "labelmap.json").read_text())
    kept = [name for name, class_id in sorted(class_ids.items(), key=lambda item: item[1]) if name not in IGNORED_CLASSES]
    mapping = {int(class_id): (kept.index(name) if name in kept else 255) for name, class_id in class_ids.items()}
    commit_with_retry()
    return {"label_mapping": mapping, "names": kept}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def write_lists(camera: str, validation_walks: list[str], names: list[str], label_mapping: dict, sizes: list) -> dict:
    """Write each view's image lists, data.yaml and view.yaml from the camera's index CSVs, then compare the 5 fps
    lists with the earlier view semantic_v1_short352 (when it is on this volume): they must hold the same frames.
    Returns {view: {list: frames}}, the validation walks not found and the comparison."""
    import csv
    from datetime import datetime, timezone
    from pathlib import Path

    import yaml

    root = Path(VOLUME_MOUNT_PATH) / dataset_dir(camera)
    validation = set(validation_walks)
    lists = {name: [] for name in ("train", "train_15fps", "train_human", "val", "test", "test_human")}
    walks_seen = set()
    for index_path in sorted((root / INDEX_DIR).glob("*.csv")):
        with open(index_path, newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: row["frame_id"])
        if not rows:
            continue
        walk, split = rows[0]["session"], rows[0]["split"]
        walks_seen.add(walk)
        part = "test" if split == "test" else ("val" if walk in validation else "train")
        for row in rows:
            line = f"./frames/{row['split']}/{Path(row['frame']).stem}.png"
            if part == "train":
                lists["train_15fps"].append(line)
            if row["in_5fps"] != "True":
                continue
            lists[part].append(line)
            if row["label_source"] == "human" and part in ("train", "test"):
                lists[f"{part}_human"].append(line)
    missing = sorted(validation - walks_seen)
    counts = {}
    for size in sizes:
        view = root / "views" / view_name(size)
        view.mkdir(parents=True, exist_ok=True)
        for name, lines in lists.items():
            (view / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        data = {"train": "train.txt", "val": "val.txt", "test": "test_human.txt", "image_dir_name": "frames",
                "masks_dir": "masks", "nc": len(names), "names": dict(enumerate(names)),
                "label_mapping": dict(sorted({int(key): value for key, value in label_mapping.items()}.items())),
                "image_size": size}
        header = (f"# SANPO-Real {camera} view '{view_name(size)}' (prepare_sanpo_real_{camera}_modal.py): every labeled frame (15 fps)\n"
                  f"# in tars/; masks keep the raw SANPO class ids; label_mapping turns them into the 29 training ids\n")
        (view / "data.yaml").write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                                        encoding="utf-8")
        counts[view_name(size)] = {name: len(lines) for name, lines in lists.items()}
        info = {"made": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "dataset": dataset_dir(camera),
                "camera": camera, "splits": SPLITS, "image_size": size, "jpeg_quality": JPEG_QUALITY,
                "resize": "short side = image_size, frames cv2.INTER_LINEAR (from the JPG), masks INTER_NEAREST",
                "lists": "5 fps = position 0 mod 3 of each walk's labeled frames; train_15fps = all, in order per walk",
                "ignored_classes": IGNORED_CLASSES, "validation_walks": sorted(validation),
                "validation_walks_not_found": missing, "frames_per_list": counts[view_name(size)]}
        (view / "view.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    comparison = {}
    earlier = root / "views" / "semantic_v1_short352"
    for name in ("train", "val", "test", "test_human", "train_human", "train_15fps"):
        earlier_list = earlier / f"{name}.txt"
        if earlier_list.exists():
            old = set(earlier_list.read_text(encoding="utf-8").split())
            new = set(lists[name])
            comparison[name] = ("identical" if old == new else
                                f"DIFFERENT: {len(new - old)} only new, {len(old - new)} only in semantic_v1")
    commit_with_retry()
    return {"counts": counts, "validation_walks_not_found": missing, "compared_with_semantic_v1": comparison}


@app.local_entrypoint()
def main(max_walks: int = 0, image_sizes: str = ""):
    """max_walks > 0 = quick test: that many walks per split. image_sizes "352,544" replaces IMAGE_SIZES for this run."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    sizes = [int(size) for size in image_sizes.split(",") if size.strip()] or list(IMAGE_SIZES)
    chosen_cameras = [CAMERA]
    print(f"SANPO-Real {chosen_cameras} -> Volume '{VOLUME_NAME}' | splits {SPLITS} | views {[view_name(s) for s in sizes]} "
          f"| native JPG quality {JPEG_QUALITY}")
    split_lists = {split: sorted(bucket_file(f"sanpo-real/splits/{split}_session_ids.txt").decode().split())
                   for split in ("train", "test")}

    jobs, validation, metadata = [], {}, {}
    for camera in chosen_cameras:
        def has_labels(walk: str, camera: str = camera) -> bool:
            return bool(bucket_names(f"{SANPO_ROOT}/sanpo-real/{walk}/camera_{camera}/left/segmentation_masks/", 1))

        for split in SPLITS:
            print(f"finding the {split} walks with {camera} labels (about a minute) ...", flush=True)
            with ThreadPoolExecutor(32) as pool:
                labeled = [walk for walk, ok in zip(split_lists[split], pool.map(has_labels, split_lists[split])) if ok]
            print(f"  {camera} {split}: {len(labeled)} of {len(split_lists[split])} walks have {camera} labels")
            jobs += [(camera, walk, split, sizes) for walk in (labeled[:max_walks] if max_walks > 0 else labeled)]
        validation_file = Path(__file__).resolve().parent / VALIDATION_FILE
        validation[camera] = validation_file.read_text(encoding="utf-8").split() if validation_file.exists() else []
        if not validation[camera]:
            raise SystemExit(f"validation walks of the {camera} camera not found: {validation_file}")
        print(f"  {camera} validation walks: {len(validation[camera])} ({VALIDATION_FILE})")
        settings = {"camera": camera, "splits": SPLITS, "jpeg_quality": JPEG_QUALITY, "image_sizes": sizes,
                    "view_prefix": VIEW_PREFIX, "ignored_classes": IGNORED_CLASSES, "max_walks": max_walks,
                    "validation_file": VALIDATION_FILE}
        # save_metadata takes: camera, settings, official split lists; returns: the label mapping and 29 class names
        metadata[camera] = save_metadata.remote(camera, settings, split_lists)
    if max_walks > 0:
        print(f"TEST RUN: {max_walks} walk(s) per camera and split")

    print(f"{len(jobs)} walks, {MAX_CONTAINERS} in parallel ...", flush=True)
    start, done, failed, frames = time.time(), 0, 0, 0
    for number, result in enumerate(prepare_walk.starmap(jobs, order_outputs=False, return_exceptions=True), 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{number}/{len(jobs)}] FAILED: {result!r}", flush=True)
            continue
        done += 0 if result["skipped"] else 1
        frames += result["frames"]
        elapsed = time.time() - start
        eta = elapsed / done * (len(jobs) - number) if done else 0
        took = (" (already done)" if result["skipped"] else
                f" in {result['seconds']:.0f}s | views {result['views']} | frames from {result['sources']}")
        print(f"[{number}/{len(jobs)}] {result['camera']:<5} {result['split']:<5} {result['walk'][:16]}: "
              f"{result['frames']} frames{took} | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
              flush=True)
    if failed:
        print(f"{failed} walk(s) failed: lists NOT written. Run the same command again (finished walks are skipped).")
        return
    for camera in chosen_cameras:
        # write_lists takes: camera, validation walks, class names, label mapping, sizes; returns: frames per list,
        # the validation walks not found, and the comparison with the earlier semantic_v1_short352 lists
        result = write_lists.remote(camera, validation[camera], metadata[camera]["names"],
                                    metadata[camera]["label_mapping"], sizes)
        if result["validation_walks_not_found"]:
            print(f"WARNING {camera}: validation walks not prepared: {result['validation_walks_not_found'][:5]} ...")
        print(f"{camera}: lists {result['counts']}")
        print(f"{camera}: compared with semantic_v1_short352: {result['compared_with_semantic_v1'] or 'no earlier view'}")
    print(f"DONE in {format_duration(time.time() - start)}: {frames:,} frames")
