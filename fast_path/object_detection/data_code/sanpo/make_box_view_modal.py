"""Make a YOLO "box view" from the stored SANPO masks (runs on Modal).

INPUT (already on the Modal Volume, made by download_sanpo_modal.py):
    <DATASET>/index/<session>.csv        one row per frame (frame + mask paths, split, size)
    <DATASET>/masks/<split>/<stem>.png   original SANPO masks
    <DATASET>/sanpo_meta/labelmap.json   class name -> SANPO class id

OUTPUT (a "view" = one way of turning masks into labels; frames are NOT copied):
    <DATASET>/views/<VIEW_NAME>/labels/<split>/<stem>.txt   YOLO boxes, one line per object:
                                                            class_id x_center y_center width height (0 to 1)
    <DATASET>/views/<VIEW_NAME>/boxes/<split>/<session>.csv every box in pixels (for the notebook)
    <DATASET>/views/<VIEW_NAME>/view.yaml                   class list + the settings used

Change the classes or blob rules below, set a new VIEW_NAME, and re-run: the
frames stay untouched, only this small labels folder is created.

How a mask becomes boxes (SANPO mask PNG: red channel = class id, blue channel
= object id, object id > 0 only for SANPO's official "panoptic" classes):
  - PANOPTIC_CLASSES: one box per object id. The box is the smallest rectangle
    around all pixels with that clas
    s AND that object id.
  - SEMANTIC_BLOB_CLASSES (no object ids): pixels of the class are grouped into
    "blobs" (pixels that touch each other, 8-connectivity). One box per blob.
    Blobs smaller than MIN_BLOB_AREA_PX are ignored as noise.

Resumable: a session whose boxes CSV exists is skipped; files are written
atomically (temp file, then rename).

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    modal run fast_path/object_detection/data_code/sanpo/make_box_view_modal.py --max-sessions 2   # quick test
    modal run --detach fast_path/object_detection/data_code/sanpo/make_box_view_modal.py           # all sessions
"""

from __future__ import annotations

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"  # folder made by download_sanpo_modal.py
VIEW_NAME = "boxes_v1"  # new name = new view; old views are kept

# SANPO "panoptic" classes (have object ids) -> one box per object.
# Not used: "bus stop", "opening-gate" (rare, not hazards).
PANOPTIC_CLASSES = [
    "pedestrian", "rider", "animal", "vehicle", "obstacle", "pole", "tree",
    "traffic sign", "traffic light", "bike rack", "stairs", "crosswalk", "opening-door",
]
# SANPO "semantic" classes (no object ids) -> one box per blob of touching pixels.
SEMANTIC_BLOB_CLASSES = ["guard rail/road barrier", "wall/fence", "water body"]

MIN_BLOB_AREA_PX = 400  # ignore blobs smaller than this many pixels (noise)
MIN_BOX_SIDE_PX = 4  # ignore boxes narrower or shorter than this (pixels)
MAX_CONTAINERS = 20  # sessions processed in parallel
PRINT_EVERY = 50  # progress line every N frames inside a session
# =============================================================================

DATA_MOUNT = "/data"
YOLO_CLASSES = PANOPTIC_CLASSES + SEMANTIC_BLOB_CLASSES  # YOLO class id = position in this list

app = modal.App("blind-guide-sanpo-box-view")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy", "pillow", "opencv-python-headless"
)


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def mask_to_boxes(mask, sanpo_ids: dict[str, int]) -> list[dict]:
    """Turn one SANPO mask (H x W x 3, uint8) into boxes in pixel coordinates.

    Returns a list of dicts: yolo_id, class, source ("panoptic" or "blob"),
    x1, y1, x2, y2 (pixels, x2/y2 exclusive) and pixels (object size).
    """
    import cv2
    import numpy as np

    class_map = mask[..., 0]  # red channel: SANPO class id of every pixel
    object_map = mask[..., 2]  # blue channel: object id (0 = no object id)
    boxes = []

    for yolo_id, name in enumerate(YOLO_CLASSES):
        is_class = class_map == sanpo_ids[name]  # True where the pixel belongs to this class
        if not is_class.any():
            continue

        if name in PANOPTIC_CLASSES:
            # One box per object id found inside this class.
            for object_id in np.unique(object_map[is_class]):
                if object_id == 0:  # class pixels without an object id: cannot form an object box
                    continue
                ys, xs = np.nonzero(is_class & (object_map == object_id))
                boxes.append({"yolo_id": yolo_id, "class": name, "source": "panoptic",
                              "x1": int(xs.min()), "y1": int(ys.min()),
                              "x2": int(xs.max()) + 1, "y2": int(ys.max()) + 1, "pixels": int(xs.size)})
        else:
            # No object ids: find groups of touching pixels ("blobs").
            # stats[k] = (x, y, width, height, area) of blob k; blob 0 is the background.
            n_blobs, _, stats, _ = cv2.connectedComponentsWithStats(is_class.astype(np.uint8), connectivity=8)
            for k in range(1, n_blobs):
                x, y, w, h, area = (int(v) for v in stats[k])
                if area < MIN_BLOB_AREA_PX:
                    continue
                boxes.append({"yolo_id": yolo_id, "class": name, "source": "blob",
                              "x1": x, "y1": y, "x2": x + w, "y2": y + h, "pixels": area})

    # Drop slivers that are too thin to be useful boxes.
    return [b for b in boxes if b["x2"] - b["x1"] >= MIN_BOX_SIDE_PX and b["y2"] - b["y1"] >= MIN_BOX_SIDE_PX]


def to_yolo_line(box: dict, width: int, height: int) -> str:
    """Pixel box -> YOLO line: class_id x_center y_center width height, all as fractions of the image."""
    xc = (box["x1"] + box["x2"]) / 2 / width
    yc = (box["y1"] + box["y2"]) / 2 / height
    bw = (box["x2"] - box["x1"]) / width
    bh = (box["y2"] - box["y1"]) / height
    return f"{box['yolo_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=2 * 3600, max_containers=MAX_CONTAINERS, retries=2)
def process_session(index_csv_name: str) -> dict:
    """Make labels for every frame of one session (one index CSV)."""
    import csv
    import json
    import os
    import time
    from pathlib import Path

    import numpy as np
    from PIL import Image

    root = Path(DATA_MOUNT) / DATASET
    view = root / "views" / VIEW_NAME
    session = index_csv_name.removesuffix(".csv")
    with open(root / "index" / index_csv_name, newline="") as f:
        frames = list(csv.DictReader(f))
    split = frames[0]["split"]
    boxes_csv = view / "boxes" / split / f"{session}.csv"
    if boxes_csv.exists():  # finished in an earlier run
        print(f"[{split}] {session}: already done, skipping", flush=True)
        return {"session": session, "split": split, "frames": len(frames), "boxes": None, "skipped": True}

    sanpo_ids = json.loads((root / "sanpo_meta" / "labelmap.json").read_text())
    label_dir = view / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    boxes_csv.parent.mkdir(parents=True, exist_ok=True)

    rows, start = [], time.time()
    for i, fr in enumerate(frames, 1):
        width, height = int(fr["width"]), int(fr["height"])
        mask = np.asarray(Image.open(root / fr["mask"]).convert("RGB"))
        boxes = mask_to_boxes(mask, sanpo_ids)

        stem = Path(fr["frame"]).stem
        tmp = label_dir / f"{stem}.txt.tmp"
        tmp.write_text("".join(to_yolo_line(b, width, height) + "\n" for b in boxes))  # empty file = no objects
        os.replace(tmp, label_dir / f"{stem}.txt")
        rows += [{"split": split, "image": f"{stem}.jpg", "width": width, "height": height, **b} for b in boxes]

        if i % PRINT_EVERY == 0 or i == len(frames):
            elapsed = time.time() - start
            print(f"[{split}] {session}: {i}/{len(frames)} frames | elapsed {_fmt_time(elapsed)} | "
                  f"ETA {_fmt_time(elapsed / i * (len(frames) - i))}", flush=True)

    # The boxes CSV is written last: it marks this session as complete.
    fields = ["split", "image", "width", "height", "yolo_id", "class", "source", "x1", "y1", "x2", "y2", "pixels"]
    tmp = boxes_csv.with_name(boxes_csv.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, boxes_csv)
    volume.commit()
    return {"session": session, "split": split, "frames": len(frames), "boxes": len(rows), "skipped": False}


@app.function(image=image, volumes={DATA_MOUNT: volume}, timeout=600)
def write_view_yaml() -> str:
    """Record the class list and settings used for this view."""
    from pathlib import Path

    view = Path(DATA_MOUNT) / DATASET / "views" / VIEW_NAME
    view.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_CLASSES))
    (view / "view.yaml").write_text(
        f"# Made by make_box_view_modal.py from {DATASET}/masks\n"
        f"panoptic_classes: {PANOPTIC_CLASSES}\nsemantic_blob_classes: {SEMANTIC_BLOB_CLASSES}\n"
        f"min_blob_area_px: {MIN_BLOB_AREA_PX}\nmin_box_side_px: {MIN_BOX_SIDE_PX}\nnames:\n{names}\n"
    )
    volume.commit()
    return str(view / "view.yaml")


@app.local_entrypoint()
def main(max_sessions: int = 0):
    """max_sessions > 0 = quick test on that many sessions."""
    import time

    print(f"Listing sessions in Volume '{VOLUME_NAME}' / {DATASET}/index ...", flush=True)
    index_files = sorted(e.path.rsplit("/", 1)[-1] for e in volume.listdir(f"{DATASET}/index") if e.path.endswith(".csv"))
    if max_sessions > 0:
        index_files = index_files[:max_sessions]
        print(f"TEST RUN: {len(index_files)} session(s)")
    print(f"View '{VIEW_NAME}': {len(YOLO_CLASSES)} classes -> {write_view_yaml.remote()}")
    print(f"Processing {len(index_files)} sessions ({MAX_CONTAINERS} in parallel) ...", flush=True)

    start, total_frames, total_boxes, new, failed = time.time(), 0, 0, 0, 0
    for i, r in enumerate(process_session.map(index_files, order_outputs=False, return_exceptions=True), 1):
        if isinstance(r, Exception):
            failed += 1
            print(f"[{i}/{len(index_files)}] FAILED: {r!r}", flush=True)
            continue
        total_frames += r["frames"]
        total_boxes += r["boxes"] or 0
        new += 0 if r["skipped"] else 1
        elapsed = time.time() - start
        eta = elapsed / new * (len(index_files) - i) if new else 0
        print(f"[{i}/{len(index_files)}] {r['split']:<5} {r['session']}: {r['frames']} frames, "
              f"{'already done' if r['skipped'] else str(r['boxes']) + ' boxes'} | "
              f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(eta)}", flush=True)
    print(f"DONE: {total_frames} frames, {total_boxes} new boxes | time {_fmt_time(time.time() - start)}"
          f"{f' | {failed} failed: re-run to retry' if failed else ''}")
