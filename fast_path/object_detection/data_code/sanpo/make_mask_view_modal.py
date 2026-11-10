"""Make a "mask view" from the stored SANPO masks (runs on Modal): one mask per object, for
segmentation evaluation and training.

INPUT (already on the Modal Volume, made by download_sanpo_modal.py):
    <DATASET>/index/<session>.csv        one row per frame (frame + mask paths, split, size)
    <DATASET>/masks/<split>/<stem>.png   original SANPO masks
    <DATASET>/sanpo_meta/labelmap.json   class name -> SANPO class id

OUTPUT (a "view" = one way of turning masks into labels; frames are NOT copied):
    <DATASET>/views/<VIEW_NAME>/objects/<split>/<session>.json   per frame, every object:
        class, class_id, source ("panoptic" / "blob"), bbox [x, y, width, height] (pixels),
        area (pixels), segmentation = the object's mask as COCO RLE at FULL resolution
    <DATASET>/views/<VIEW_NAME>/view.yaml                        class list + the settings used

Same classes and rules as the box view boxes_v1 (make_box_view_modal.py), so box and mask
results can be compared object for object:
  - PANOPTIC_CLASSES: one object per object id (pixels with that class AND that object id).
  - SEMANTIC_BLOB_CLASSES (no object ids): one object per blob of touching pixels
    (8-connectivity); blobs smaller than MIN_BLOB_AREA_PX are ignored as noise.
  - Objects whose box is thinner than MIN_BOX_SIDE_PX are dropped (as in boxes_v1).

RLE (run-length encoding) = the standard compact COCO format for a binary mask: it stores the
lengths of the runs of 0s and 1s instead of every pixel. Stored at full resolution so the
evaluation can compare masks at any resolution later.

Resumable: a session whose JSON exists is skipped; files are written atomically.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/make_mask_view_modal.py --max-sessions 2   # quick test
    python -m modal run fast_path/object_detection/data_code/sanpo/make_mask_view_modal.py                    # all sessions
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"  # folder made by download_sanpo_modal.py
VIEW_NAME = "masks_v1"  # new name = new view; old views are kept

# Same classes as boxes_v1 (keep in sync with make_box_view_modal.py).
# SANPO "panoptic" classes (have object ids) -> one mask per object.
# Not used: "bus stop", "opening-gate" (rare, not hazards).
PANOPTIC_CLASSES = [
    "pedestrian", "rider", "animal", "vehicle", "obstacle", "pole", "tree",
    "traffic sign", "traffic light", "bike rack", "stairs", "crosswalk", "opening-door",
]
# SANPO "semantic" classes (no object ids) -> one mask per blob of touching pixels.
SEMANTIC_BLOB_CLASSES = ["guard rail/road barrier", "wall/fence", "water body"]

MIN_BLOB_AREA_PIXELS = 400  # ignore blobs smaller than this many pixels (noise), as in boxes_v1
MIN_BOX_SIDE_PIXELS = 4  # ignore objects whose box is narrower or shorter than this, as in boxes_v1
MAX_CONTAINERS = 20  # sessions processed in parallel
PRINT_EVERY = 50  # progress line every N frames inside a session
# =============================================================================

VOLUME_MOUNT_PATH = "/data"
VIEW_CLASSES = PANOPTIC_CLASSES + SEMANTIC_BLOB_CLASSES  # class_id = position in this list (same as boxes_v1)

app = modal.App("blind-guide-sanpo-mask-view")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy", "pillow", "opencv-python-headless", "pycocotools"
)


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s" (used in progress lines)."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def mask_to_objects(sanpo_mask, sanpo_class_ids: dict[str, int]) -> list[dict]:
    """Turn one SANPO mask (H x W x 3, uint8) into objects, each with its own binary mask as COCO RLE.

    Steps:
        1. Red channel = class id of every pixel, blue channel = object id (0 = none).
        2. Panoptic classes: one object per object id inside the class.
        3. Semantic blob classes: one object per group of touching pixels (blob); small blobs dropped.
        4. Each object: box [x, y, width, height], area (pixels), mask encoded as RLE.
    """
    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    # Step 1: the two label channels
    class_map = sanpo_mask[..., 0]  # red channel: SANPO class id of every pixel
    object_map = sanpo_mask[..., 2]  # blue channel: object id (0 = no object id)

    def make_object(class_id: int, class_name: str, source: str, object_mask) -> dict | None:
        """Box, area and RLE of one binary object mask (None if the object is too thin)."""
        rows, columns = np.nonzero(object_mask)
        x1, y1, x2, y2 = int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1
        if x2 - x1 < MIN_BOX_SIDE_PIXELS or y2 - y1 < MIN_BOX_SIDE_PIXELS:
            return None  # sliver, dropped as in boxes_v1
        # Step 4: RLE needs a Fortran-ordered uint8 array; "counts" is bytes -> stored as text
        run_length_encoding = mask_utils.encode(np.asfortranarray(object_mask.astype(np.uint8)))
        run_length_encoding["counts"] = run_length_encoding["counts"].decode("ascii")
        return {"class_id": class_id, "class": class_name, "source": source,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": int(rows.size),
                "segmentation": run_length_encoding}

    objects = []
    for class_id, class_name in enumerate(VIEW_CLASSES):
        is_class = class_map == sanpo_class_ids[class_name]  # True where the pixel belongs to this class
        if not is_class.any():
            continue

        if class_name in PANOPTIC_CLASSES:
            # Step 2: one object per object id found inside this class
            for object_id in np.unique(object_map[is_class]):
                if object_id == 0:  # class pixels without an object id: cannot form an object
                    continue
                new_object = make_object(class_id, class_name, "panoptic", is_class & (object_map == object_id))
                if new_object:
                    objects.append(new_object)
        else:
            # Step 3: no object ids: groups of touching pixels ("blobs"), each a separate object.
            # blob_labels = blob number of every pixel (0 = background); stats[k] = (x, y, width, height, area)
            blob_count, blob_labels, stats, _ = cv2.connectedComponentsWithStats(
                is_class.astype(np.uint8), connectivity=8)
            for blob_number in range(1, blob_count):
                if stats[blob_number, cv2.CC_STAT_AREA] < MIN_BLOB_AREA_PIXELS:
                    continue
                new_object = make_object(class_id, class_name, "blob", blob_labels == blob_number)
                if new_object:
                    objects.append(new_object)
    return objects


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=3 * 3600,
              max_containers=MAX_CONTAINERS, retries=2)
def process_session(index_csv_name: str) -> dict:
    """Make the object masks for every frame of one session (one index CSV)."""
    import csv
    import json
    import os
    import time
    from pathlib import Path

    import numpy as np
    from PIL import Image

    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    view_root = dataset_root / "views" / VIEW_NAME
    session = index_csv_name.removesuffix(".csv")
    with open(dataset_root / "index" / index_csv_name, newline="") as index_file:
        frames = list(csv.DictReader(index_file))
    split = frames[0]["split"]
    output_path = view_root / "objects" / split / f"{session}.json"
    if output_path.exists():  # finished in an earlier run
        print(f"[{split}] {session}: already done, skipping", flush=True)
        return {"session": session, "split": split, "frames": len(frames), "objects": None, "skipped": True}

    sanpo_class_ids = json.loads((dataset_root / "sanpo_meta" / "labelmap.json").read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_records, object_count, start_time = [], 0, time.time()
    for frame_number, frame in enumerate(frames, 1):
        sanpo_mask = np.asarray(Image.open(dataset_root / frame["mask"]).convert("RGB"))
        objects = mask_to_objects(sanpo_mask, sanpo_class_ids)
        frame_records.append({"image": Path(frame["frame"]).name, "width": int(frame["width"]),
                              "height": int(frame["height"]), "objects": objects})
        object_count += len(objects)
        if frame_number % PRINT_EVERY == 0 or frame_number == len(frames):
            elapsed = time.time() - start_time
            print(f"[{split}] {session}: {frame_number}/{len(frames)} frames | {object_count:,} objects | "
                  f"elapsed {format_duration(elapsed)} | "
                  f"ETA {format_duration(elapsed / frame_number * (len(frames) - frame_number))}", flush=True)

    # Written last (temp file, then rename): marks this session as complete
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    temporary_path.write_text(json.dumps({"session": session, "split": split, "frames": frame_records}))
    os.replace(temporary_path, output_path)
    volume.commit()
    return {"session": session, "split": split, "frames": len(frames), "objects": object_count, "skipped": False}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=600)
def write_view_yaml() -> str:
    """Record the class list and settings used for this view."""
    from pathlib import Path

    view_root = Path(VOLUME_MOUNT_PATH) / DATASET / "views" / VIEW_NAME
    view_root.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in enumerate(VIEW_CLASSES))
    (view_root / "view.yaml").write_text(
        f"# Made by make_mask_view_modal.py from {DATASET}/masks (masks stored as COCO RLE, full resolution)\n"
        f"panoptic_classes: {PANOPTIC_CLASSES}\nsemantic_blob_classes: {SEMANTIC_BLOB_CLASSES}\n"
        f"min_blob_area_px: {MIN_BLOB_AREA_PIXELS}\nmin_box_side_px: {MIN_BOX_SIDE_PIXELS}\nnames:\n{names}\n"
    )
    volume.commit()
    return str(view_root / "view.yaml")


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
    print(f"View '{VIEW_NAME}': {len(VIEW_CLASSES)} classes -> {write_view_yaml.remote()}")
    print(f"Processing {len(index_files)} sessions ({MAX_CONTAINERS} in parallel) ...", flush=True)

    start_time, total_frames, total_objects, newly_done, failed = time.time(), 0, 0, 0, 0
    results = process_session.map(index_files, order_outputs=False, return_exceptions=True)
    for session_number, result in enumerate(results, 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{session_number}/{len(index_files)}] FAILED: {result!r}", flush=True)
            continue
        total_frames += result["frames"]
        total_objects += result["objects"] or 0
        newly_done += 0 if result["skipped"] else 1
        elapsed = time.time() - start_time
        remaining_seconds = elapsed / newly_done * (len(index_files) - session_number) if newly_done else 0
        status = "already done" if result["skipped"] else f"{result['objects']:,} objects"
        print(f"[{session_number}/{len(index_files)}] {result['split']:<5} {result['session']}: "
              f"{result['frames']} frames, {status} | elapsed {format_duration(elapsed)} | "
              f"ETA {format_duration(remaining_seconds)}", flush=True)
    print(f"DONE: {total_frames:,} frames, {total_objects:,} new objects | time {format_duration(time.time() - start_time)}"
          f"{f' | {failed} failed: re-run to retry' if failed else ''}")
