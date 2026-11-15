"""Make a RESIZED copy of a semantic view (runs on Modal, once per image size): small frames and masks, so
training does not decode and resize full-size files (2208x1242 or 1920x1080) at every step.

INPUT (on the Modal Volume):
    <DATASET>/views/<SOURCE_VIEW>/{train,train_human,val,test,test_human}.txt   made by make_semantic_view_modal.py
    <DATASET>/views/<SOURCE_VIEW>/masks/<folder>/<stem>.png                       one-channel raw SANPO class ids
    <DATASET>/frames/<folder>/<stem>.jpg                                          native-size frames

OUTPUT (default name "<SOURCE_VIEW>_short<IMAGE_SIZE>", e.g. semantic_v1_short352):
    <DATASET>/views/<view>/frames/<folder>/<stem>.png    resized frame (PNG = lossless, no second JPEG loss)
    <DATASET>/views/<view>/masks/<folder>/<stem>.png     resized mask (nearest neighbour, class ids unchanged)
    <DATASET>/views/<view>/<same lists>.txt              paths "./frames/<folder>/<stem>.png"
    <DATASET>/views/<view>/data.yaml, view.yaml          same classes and label_mapping as the source view
    <DATASET>/views/<view>/tars/<walk>.tar               the walk's frames/ and masks/ in ONE file: a Modal Volume
                                                         reads many small files slowly but big files fast, so
                                                         training copies these to the container's local disk

Each frame is resized EXACTLY as Ultralytics 8.4.208 would resize it at imgsz = IMAGE_SIZE
(ultralytics/data/base.py load_image, cv2.INTER_LINEAR; masks: dataset.py get_image_and_label, INTER_NEAREST):
    every list (train, val, test): SHORT side = IMAGE_SIZE, long side = ceil(long * ratio)  (Ultralytics' training
    rule). Ultralytics itself validates at the LONG side (352x198), a smaller scale than training; the training code
    validates at the training scale instead (data.validate_at_training_scale), so val/test frames are sized like the
    training frames (626x352). Set a list to "eval" in LISTS for Ultralytics' long-side rule.
    (The first view, semantic_v1_352, used the long side for val/test; it stays on the volumes but is not used.)
The training and validation walks are different walks, so no frame needs two sizes. When training runs with the
same imgsz, Ultralytics finds the size already right and does not resize again: the pixels the model sees are
identical to training on the native-size frames. With another imgsz, train on another resized view.

Resumable: an output file that already exists is not written again; lists and YAML are rewritten at the end.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session2.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/make_resized_view_modal.py
    python -m modal run fast_path/object_detection/data_code/sanpo/make_resized_view_modal.py --source-view semantic_smoke   # quick test
"""

from __future__ import annotations

import io
import math
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import modal
import yaml

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"
SOURCE_VIEW = "semantic_v1"        # override with --source-view
IMAGE_SIZE = 352                   # = ultralytics.imgsz used in training; override with --image-size
LISTS = {"train": "train", "train_human": "train", "val": "train", "test": "train", "test_human": "train"}
MAX_CONTAINERS = 20                # walks processed in parallel
PRINT_EVERY = 50                   # progress line every N frames inside a walk
PACK_THREADS = 32                  # parallel file reads when packing a walk's archive
# =============================================================================

VOLUME_MOUNT_PATH = "/data"

app = modal.App("blind-guide-sanpo-resized-view")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install(
    "numpy", "opencv-python-headless", "pyyaml")


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s" (used in progress lines)."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def target_size(height: int, width: int, image_size: int, mode: str) -> tuple[int, int]:
    """(width, height) Ultralytics 8.4.208 load_image gives a (height, width) frame at imgsz = image_size.

    mode "train": short side = image_size (resize_short=True). mode "eval": long side = image_size.
    Same formulas as ultralytics/data/base.py, so the result matches to the pixel.
    """
    if mode == "train":
        ratio = image_size / min(height, width)
        if ratio == 1:
            return width, height
        return (math.ceil(width * ratio), image_size) if height < width else (image_size, math.ceil(height * ratio))
    ratio = image_size / max(height, width)
    if ratio == 1:
        return width, height
    return min(math.ceil(width * ratio), image_size), min(math.ceil(height * ratio), image_size)


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=3 * 3600,
              max_containers=MAX_CONTAINERS, retries=2)
def resize_walk(walk: str, items: list[list[str]], source_view: str, target_view: str, image_size: int) -> dict:
    """Resize the frames and masks of one walk.

    items: [frame path relative to the dataset ("frames/<folder>/<stem>.jpg"), mode ("train" or "eval")].
    Returns {"walk", "frames", "written", "skipped", "sizes": {"WxH": count}}.
    """
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    source_root, target_root = dataset_root / "views" / source_view, dataset_root / "views" / target_view
    tar_path = target_root / "tars" / f"{walk}.tar"
    if tar_path.exists():                           # walk finished earlier (files and archive)
        print(f"{walk}: archive already exists, skipped", flush=True)
        return {"walk": walk, "frames": len(items), "written": 0, "skipped": len(items), "sizes": {}}
    written, skipped, sizes, start_time = 0, 0, {}, time.time()
    for item_number, (frame_path, mode) in enumerate(items, 1):
        folder, stem = Path(frame_path).parent.name, Path(frame_path).stem
        frame_target = target_root / "frames" / folder / f"{stem}.png"
        mask_target = target_root / "masks" / folder / f"{stem}.png"
        if frame_target.exists() and mask_target.exists():
            skipped += 1
        else:
            frame = cv2.imread(str(dataset_root / frame_path))             # BGR, as Ultralytics reads it
            mask = cv2.imread(str(source_root / "masks" / folder / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if frame is None or mask is None:
                raise FileNotFoundError(f"{frame_path} or its mask in view {source_view}")
            # target_size takes: frame height, width, image size, mode; returns: (width, height) Ultralytics uses
            width, height = target_size(frame.shape[0], frame.shape[1], image_size, mode)
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if (mask.shape[1], mask.shape[0]) != (width, height):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            for target, array in ((frame_target, frame), (mask_target, mask)):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.stem + ".tmp.png")
                cv2.imwrite(str(temporary), array)
                os.replace(temporary, target)
            written += 1
            sizes[f"{width}x{height}"] = sizes.get(f"{width}x{height}", 0) + 1
        if item_number % PRINT_EVERY == 0 or item_number == len(items):
            elapsed = time.time() - start_time
            print(f"{walk}: {item_number}/{len(items)} frames | {written} written, {skipped} existed | "
                  f"elapsed {format_duration(elapsed)} | "
                  f"ETA {format_duration(elapsed / item_number * (len(items) - item_number))}", flush=True)
    # pack_walk takes: view folder, frame paths, archive path; returns: the archive size in bytes
    archive_bytes = pack_walk(target_root, [frame_path for frame_path, _ in items], tar_path)
    print(f"{walk}: archive {tar_path.name} written ({archive_bytes / 1e6:.0f} MB)", flush=True)
    volume.commit()
    return {"walk": walk, "frames": len(items), "written": written, "skipped": skipped, "sizes": sizes}


def pack_walk(view_root: Path, frame_paths: list[str], tar_path: Path) -> int:
    """Pack one walk's resized frames and masks into ONE uncompressed tar file on the volume.

    Why: a Modal Volume reads many small files slowly (Ultralytics measured 1.3 MB/s; Modal advises fewer than
    50,000 files per volume), but big files fast. Training copies these archives to the container's local disk.
    Members are "frames/<folder>/<stem>.png" and "masks/<folder>/<stem>.png" (relative to the view folder).
    Files are read with many threads, because each small read mostly waits on the network.
    Returns the archive size in bytes.
    """
    members = []
    for frame_path in frame_paths:
        folder, stem = Path(frame_path).parent.name, Path(frame_path).stem
        members += [f"frames/{folder}/{stem}.png", f"masks/{folder}/{stem}.png"]
    with ThreadPoolExecutor(max_workers=PACK_THREADS) as pool:
        contents = list(pool.map(lambda member: (view_root / member).read_bytes(), members))
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tar_path.with_name(tar_path.stem + ".tmp.tar")
    with tarfile.open(temporary, "w") as archive:
        for member, content in zip(members, contents):
            info = tarfile.TarInfo(member)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    os.replace(temporary, tar_path)
    return tar_path.stat().st_size


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def write_lists_and_yaml(source_view: str, target_view: str, image_size: int, size_counts: dict) -> dict:
    """Write the lists, data.yaml and view.yaml of the resized view. Returns the frames per list."""
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    source_root, target_root = dataset_root / "views" / source_view, dataset_root / "views" / target_view
    counts = {}
    for name in LISTS:
        source_list = source_root / f"{name}.txt"
        if not source_list.exists():
            continue
        lines = []
        for line in source_list.read_text(encoding="utf-8").split():
            frame_path = Path(line.removeprefix("./../../"))
            lines.append(f"./frames/{frame_path.parent.name}/{frame_path.stem}.png")
        (target_root / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[name] = len(lines)

    data = yaml.safe_load((source_root / "data.yaml").read_text(encoding="utf-8"))
    data["image_dir_name"], data["masks_dir"] = "frames", "masks"
    data["image_size"] = image_size                 # read by the training code: warns if imgsz differs
    header = (f"# Resized copy of the semantic view '{source_view}' at imgsz {image_size} "
              f"(made by make_resized_view_modal.py)\n"
              f"# every list: short side {image_size} (val/test at the training scale)\n")
    (target_root / "data.yaml").write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                                           encoding="utf-8")
    view_info = {"made": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "dataset": DATASET,
                 "source_view": source_view, "image_size": image_size,
                 "resize": {"lists": {name: ("short side" if mode == "train" else "long side") + " = image_size"
                                      for name, mode in LISTS.items()},
                            "interpolation": "frames cv2.INTER_LINEAR",
                            "masks": "same size as their frame, cv2.INTER_NEAREST",
                            "formulas": "ultralytics 8.4.208 data/base.py load_image"},
                 "frame_sizes_written": size_counts, "frames_per_list": counts,
                 "archives": "tars/<walk>.tar: the walk's frames/ and masks/ files, for copying to local disk"}
    if not size_counts and (target_root / "view.yaml").exists():   # re-run that only added the archives
        view_info["frame_sizes_written"] = (yaml.safe_load((target_root / "view.yaml").read_text(encoding="utf-8"))
                                            or {}).get("frame_sizes_written", {})
    (target_root / "view.yaml").write_text(yaml.safe_dump(view_info, sort_keys=False), encoding="utf-8")
    volume.commit()
    return counts


@app.local_entrypoint()
def main(source_view: str = SOURCE_VIEW, image_size: int = IMAGE_SIZE, view_name: str = ""):
    target_view = view_name or f"{source_view}_short{image_size}"
    print(f"Resizing view '{source_view}' -> '{target_view}' at imgsz {image_size} on Volume '{VOLUME_NAME}'")
    walks: dict[str, dict[str, str]] = {}           # walk -> {frame path: mode}
    for name, mode in LISTS.items():
        try:
            text = b"".join(volume.read_file(f"{DATASET}/views/{source_view}/{name}.txt")).decode("utf-8")
        except Exception:
            print(f"  list {name}.txt not found in the source view, skipped")
            continue
        lines = text.split()
        for line in lines:
            frame_path = line.removeprefix("./../../")
            walk = Path(frame_path).stem.rsplit("_", 1)[0]
            walks.setdefault(walk, {})[frame_path] = mode
        print(f"  {name}.txt: {len(lines):,} frames ({mode} size)")
    jobs = sorted(walks.items())
    total = sum(len(items) for _, items in jobs)
    print(f"{len(jobs)} walks, {total:,} unique frames, {MAX_CONTAINERS} walks in parallel ...", flush=True)

    start_time, failed, written, size_counts = time.time(), 0, 0, {}
    arguments = [(walk, [[path, mode] for path, mode in sorted(items.items())], source_view, target_view, image_size)
                 for walk, items in jobs]
    for walk_number, result in enumerate(resize_walk.starmap(arguments, order_outputs=False,
                                                             return_exceptions=True), 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{walk_number}/{len(jobs)}] FAILED: {result!r}", flush=True)
            continue
        written += result["written"]
        for size, count in result["sizes"].items():
            size_counts[size] = size_counts.get(size, 0) + count
        elapsed = time.time() - start_time
        print(f"[{walk_number}/{len(jobs)}] {result['walk']}: {result['frames']} frames "
              f"({result['written']} written, {result['skipped']} existed) | elapsed {format_duration(elapsed)} | "
              f"ETA {format_duration(elapsed / walk_number * (len(jobs) - walk_number))}", flush=True)
    if failed:
        print(f"{failed} walk(s) failed: lists NOT written. Re-run the same command (finished files are kept).")
        return
    counts = write_lists_and_yaml.remote(source_view, target_view, image_size, size_counts)
    print(f"DONE in {format_duration(time.time() - start_time)}: {written:,} frames written | sizes {size_counts} | "
          f"frames per list: {counts}")
    print(f"Dataset file: /{DATASET}/views/{target_view}/data.yaml on Volume '{VOLUME_NAME}'")
