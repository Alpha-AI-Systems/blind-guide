"""Make the "semantic view" of SANPO (runs on Modal): what Ultralytics needs to TRAIN a semantic model.

INPUT (already on the Modal Volume, made by download_sanpo_modal.py):
    <DATASET>/index/<session>.csv        one row per frame (frame + mask paths, split, size)
    <DATASET>/masks/<folder>/<stem>.png  original SANPO masks (red channel = class id, blue = object id)
    <DATASET>/sanpo_meta/labelmap.json   class name -> SANPO class id

OUTPUT (frames are NOT copied; a view is one way of turning SANPO's labels into training labels):
    <DATASET>/views/<VIEW_NAME>/masks/<folder>/<stem>.png   ONE channel per pixel = the raw SANPO class id
                                                            (the red channel), same folders as frames/
    <DATASET>/views/<VIEW_NAME>/train.txt, val.txt          image lists (training and validation walks)
    <DATASET>/views/<VIEW_NAME>/train_human.txt             training frames with HUMAN-made labels only
    <DATASET>/views/<VIEW_NAME>/test.txt, test_human.txt    official test split: all frames / human-labeled only
    <DATASET>/views/<VIEW_NAME>/train_15fps.txt             training walks at 15 fps, frames in order per walk; the
                                                            training code derives 5 / 10 fps from it (data.train_fps)
                                                            (only when every training walk
                                                            has ALL_FRAMES_INDEX_DIR/<walk>.csv, from
                                                            download_sanpo_modal.py --all-frames)
  Every other list holds only the 5 fps frames (index/), so it stays exactly as before the 15 fps add-on.
    <DATASET>/views/<VIEW_NAME>/data.yaml                   the Ultralytics dataset file
    <DATASET>/views/<VIEW_NAME>/view.yaml                   the settings used and the counts

Decisions (md_files/decisions_for_training_yolo.md):
  - Splits: SANPO's official train split minus the 12 validation walks (validation_split/results/
    validation_walks.txt) = train; those 12 walks = val; SANPO's official test split = test.
  - Classes: 29. "unlabeled" (id 0) and "bus stop" (id 25, no pixels in the chest data) map to 255,
    Ultralytics' ignore label. The mapping from raw SANPO ids to training ids is written into data.yaml
    (label_mapping), so the masks themselves keep the raw ids.
  - Human-labeled frames come from SANPO's per-walk frame_segmentation_annotation_type.json, downloaded
    once into <DATASET>/sanpo_meta/annotation_types/.

Portable paths: the image lists hold paths relative to the list file ("./../../frames/<folder>/<stem>.jpg";
Ultralytics resolves a leading "./" against the list's folder) and data.yaml has no "path" key (Ultralytics then
uses the YAML's own folder), so the same view works on Modal and in a local copy of the dataset.
Ultralytics finds each mask from the image path with the keys image_dir_name and masks_dir, through the
trainer subclass in training/shared_training_code.py.

Resumable: a mask that already exists is not written again; lists and YAML are rewritten at the end.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python -m modal run fast_path/object_detection/data_code/sanpo/make_semantic_view_modal.py --max-sessions 2 --view-name semantic_smoke   # quick test
    python -m modal run fast_path/object_detection/data_code/sanpo/make_semantic_view_modal.py                                                 # full view
"""

from __future__ import annotations

from pathlib import Path

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest"  # folder made by download_sanpo_modal.py
VIEW_NAME = "semantic_v1"  # new name = new view; old views are kept (override with --view-name)
IGNORED_CLASSES = ["unlabeled", "bus stop"]  # mapped to 255 = ignored in training and validation
VALIDATION_WALKS_FILE = Path(__file__).resolve().parent / "validation_split" / "results" / "validation_walks.txt"
SANPO_URL = "https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real"
CAMERA = "chest"
ALL_FRAMES_INDEX_DIR = "index_15fps"  # the 15 fps add-on's index (download_sanpo_modal.py --all-frames)
MAX_CONTAINERS = 50  # walks processed in parallel
THREADS_PER_WALK = 16  # masks converted at once inside one walk
CPU_PER_CONTAINER = 4  # cores per container (PNG decode + encode); Modal's default is a fraction of a core
PRINT_EVERY = 50  # progress line every N frames inside a session
# =============================================================================

VOLUME_MOUNT_PATH = "/data"

app = modal.App("blind-guide-sanpo-semantic-view")
volume = modal.Volume.from_name(VOLUME_NAME)
image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install(
    "numpy", "opencv-python-headless", "pyyaml")


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s" (used in progress lines)."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def human_labeled_frames(dataset_root: Path, session: str) -> set[int]:
    """Frame numbers of one walk whose labels were made by people (SANPO's annotation-type file).

    Read from <DATASET>/sanpo_meta/annotation_types/<session>.json, downloaded first if missing.
    Returns a set of frame numbers.
    """
    import json
    import time
    import urllib.parse
    import urllib.request

    cache_path = dataset_root / "sanpo_meta" / "annotation_types" / f"{session}.json"
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = (f"{SANPO_URL}/{urllib.parse.quote(session)}/camera_{CAMERA}/left/"
               f"frame_segmentation_annotation_type.json")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        temporary_path = cache_path.with_suffix(".tmp")
        temporary_path.write_bytes(data)
        temporary_path.replace(cache_path)
    return {int(number) for number, source in json.loads(cache_path.read_text()).items()
            if source == "HUMAN_ANNOTATED"}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=3 * 3600,
              max_containers=MAX_CONTAINERS, retries=2, cpu=CPU_PER_CONTAINER)
def process_session(index_csv_name: str, view_name: str) -> dict:
    """Write the one-channel class-id masks of one walk, and return its frames with their split and label source.

    Steps:
        1. Read the walk's index CSV (frame and mask paths, split folder): ALL_FRAMES_INDEX_DIR/<walk>.csv when it
           exists (15 fps, a superset), else index/<walk>.csv (5 fps).
        2. Per frame: read the SANPO mask, keep the red channel (class id), write it as a one-channel PNG.
        3. Mark which frames are human-labeled, and which are in the 5 fps set (index/).
    Returns {"session", "folder", "has_15fps", "frames": [{"frame": path relative to the dataset, "frame_id": int,
             "human": bool, "in_5fps": bool}], "written": masks written now, "skipped": masks that already existed}.
    """
    import csv
    import os
    import time

    import cv2

    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    view_root = dataset_root / "views" / view_name
    session = index_csv_name.removesuffix(".csv")
    with open(dataset_root / "index" / index_csv_name, newline="") as index_file:
        frames = list(csv.DictReader(index_file))
    five_fps_ids = {frame["frame_id"] for frame in frames}
    all_frames_csv = dataset_root / ALL_FRAMES_INDEX_DIR / index_csv_name
    has_15fps = all_frames_csv.exists()
    if has_15fps:
        with open(all_frames_csv, newline="") as index_file:
            frames = list(csv.DictReader(index_file))
        marked = {frame["frame_id"] for frame in frames if frame["in_5fps"] == "True"}
        if marked != five_fps_ids:
            raise RuntimeError(f"{session}: {ALL_FRAMES_INDEX_DIR}/ marks {len(marked)} frames as 5 fps, "
                               f"index/ has {len(five_fps_ids)}")
    folder = frames[0]["split"]                     # the frames/ subfolder of this walk (train / val / test)
    # human_labeled_frames takes: dataset folder, walk id; returns: set of human-labeled frame numbers
    human = human_labeled_frames(dataset_root, session)

    (view_root / "masks" / folder).mkdir(parents=True, exist_ok=True)

    def convert(frame: dict) -> bool:
        """Write one frame's class-id mask unless it exists. Returns True if it was written now."""
        target = view_root / "masks" / folder / f"{Path(frame['frame']).stem}.png"
        if target.exists():
            return False
        sanpo_mask = cv2.imread(str(dataset_root / frame["mask"]))       # BGR: red (the class id) is index 2
        if sanpo_mask is None:
            raise FileNotFoundError(dataset_root / frame["mask"])
        temporary = target.with_name(target.stem + ".tmp.png")
        cv2.imwrite(str(temporary), sanpo_mask[:, :, 2])                 # one channel, raw SANPO class ids
        os.replace(temporary, target)
        return True

    from concurrent.futures import ThreadPoolExecutor

    records, written, skipped, start_time = [], 0, 0, time.time()
    # THREADS_PER_WALK frames at once (file reads wait on the volume; OpenCV releases the GIL); map keeps the order
    with ThreadPoolExecutor(THREADS_PER_WALK) as pool:
        for frame_number, (frame, was_written) in enumerate(zip(frames, pool.map(convert, frames)), 1):
            written += int(was_written)
            skipped += int(not was_written)
            records.append({"frame": frame["frame"], "frame_id": int(frame["frame_id"]),
                            "human": int(frame["frame_id"]) in human, "in_5fps": frame["frame_id"] in five_fps_ids})
            if frame_number % PRINT_EVERY == 0 or frame_number == len(frames):
                elapsed = time.time() - start_time
                print(f"[{folder}] {session}: {frame_number}/{len(frames)} frames | {written} written, {skipped} "
                      f"existed | elapsed {format_duration(elapsed)} | "
                      f"ETA {format_duration(elapsed / frame_number * (len(frames) - frame_number))}", flush=True)
    volume.commit()
    return {"session": session, "folder": folder, "has_15fps": has_15fps, "frames": records, "written": written,
            "skipped": skipped}


@app.function(image=image, volumes={VOLUME_MOUNT_PATH: volume}, timeout=1800)
def write_lists_and_yaml(view_name: str, walks: list[dict], validation_walks: list[str]) -> dict:
    """Write the image lists, data.yaml and view.yaml from every walk's frames.

    walks: process_session results. Returns the number of frames per list.
    """
    import json
    from datetime import datetime, timezone

    import yaml

    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    view_root = dataset_root / "views" / view_name
    view_root.mkdir(parents=True, exist_ok=True)
    validation = set(validation_walks)
    lists = {"train": [], "train_human": [], "val": [], "test": [], "test_human": [], "train_15fps": []}
    training_walks, training_walks_with_15fps = 0, 0
    for walk in sorted(walks, key=lambda item: item["session"]):
        official = "test" if walk["folder"] == "test" else "train"   # our old "val" folder is official train
        split = "val" if (official == "train" and walk["session"] in validation) else official
        if split == "train":
            training_walks += 1
            training_walks_with_15fps += int(walk.get("has_15fps", False))
        for position, record in enumerate(sorted(walk["frames"], key=lambda item: item["frame_id"])):
            line = "./../../" + record["frame"]                      # relative to the list file's folder
            if split == "train" and walk.get("has_15fps"):
                # position in the walk's 15 fps sequence: 0 mod 3 = the 5 fps frames (download rule all_ids[::3])
                if (position % 3 == 0) != record["in_5fps"]:
                    raise RuntimeError(f"{walk['session']}: frame {record['frame_id']} at position {position} "
                                       f"breaks the every-3rd rule of the 5 fps set")
                lists["train_15fps"].append(line)                   # every frame of the walk, in order: the
                                                                     # training code picks 5/10 fps from it
            if not record.get("in_5fps", True):                      # 15 fps add-on frame: only in train_15fps
                continue
            lists[split].append(line)
            if record["human"] and split in ("train", "test"):
                lists[f"{split}_human"].append(line)
    if training_walks_with_15fps < training_walks:                   # a mixed 5/15 fps list would mislead
        print(f"train_15fps.txt not written: {training_walks_with_15fps} of {training_walks} training walks have "
              f"{ALL_FRAMES_INDEX_DIR}/ (run download_sanpo_modal.py --all-frames first)", flush=True)
        del lists["train_15fps"]
    for name, lines in lists.items():
        (view_root / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    class_ids = json.loads((dataset_root / "sanpo_meta" / "labelmap.json").read_text())
    kept = [name for name, class_id in sorted(class_ids.items(), key=lambda item: item[1])
            if name not in IGNORED_CLASSES]
    label_mapping = {int(class_id): (kept.index(name) if name in kept else 255) for name, class_id in class_ids.items()}
    data = {
        "train": "train.txt", "val": "val.txt", "test": "test_human.txt",
        "image_dir_name": "frames", "masks_dir": f"views/{view_name}/masks",
        "nc": len(kept), "names": {index: name for index, name in enumerate(kept)},
        "label_mapping": dict(sorted(label_mapping.items())),
    }
    header = (f"# SANPO semantic view '{view_name}' for Ultralytics (made by make_semantic_view_modal.py)\n"
              f"# masks keep the raw SANPO class ids; label_mapping turns them into training ids (255 = ignored)\n")
    (view_root / "data.yaml").write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                                         encoding="utf-8")
    counts = {name: len(lines) for name, lines in lists.items()}
    view_info = {"made": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), "dataset": DATASET,
                 "ignored_classes": IGNORED_CLASSES, "validation_walks": sorted(validation),
                 "walks": len(walks), "frames_per_list": counts}
    (view_root / "view.yaml").write_text(yaml.safe_dump(view_info, sort_keys=False), encoding="utf-8")
    volume.commit()
    return counts


@app.local_entrypoint()
def main(max_sessions: int = 0, view_name: str = VIEW_NAME):
    """max_sessions > 0 = quick test: that many walks from each of train, validation and test."""
    import time

    validation_walks = VALIDATION_WALKS_FILE.read_text(encoding="utf-8").split()
    print(f"Validation walks ({len(validation_walks)}) from {VALIDATION_WALKS_FILE}")
    print(f"Listing walks in Volume '{VOLUME_NAME}' / {DATASET}/index ...", flush=True)
    index_files = sorted(entry.path.rsplit("/", 1)[-1] for entry in volume.listdir(f"{DATASET}/index")
                         if entry.path.endswith(".csv"))
    if max_sessions > 0:
        validation_files = [name for name in index_files if name.removesuffix(".csv") in validation_walks]
        others = [name for name in index_files if name not in validation_files]
        index_files = others[:max_sessions] + validation_files[:max_sessions] + others[-max_sessions:]
        print(f"TEST RUN: {len(index_files)} walk(s) into view '{view_name}'")
    print(f"View '{view_name}': processing {len(index_files)} walks ({MAX_CONTAINERS} in parallel) ...", flush=True)

    start_time, walks, failed, total_written = time.time(), [], 0, 0
    results = process_session.map(index_files, kwargs={"view_name": view_name}, order_outputs=False,
                                  return_exceptions=True)
    for walk_number, result in enumerate(results, 1):
        if isinstance(result, Exception):
            failed += 1
            print(f"[{walk_number}/{len(index_files)}] FAILED: {result!r}", flush=True)
            continue
        walks.append(result)
        total_written += result["written"]
        elapsed = time.time() - start_time
        print(f"[{walk_number}/{len(index_files)}] {result['folder']:<5} {result['session']}: "
              f"{len(result['frames'])} frames ({result['written']} masks written, {result['skipped']} existed) | "
              f"elapsed {format_duration(elapsed)} | "
              f"ETA {format_duration(elapsed / walk_number * (len(index_files) - walk_number))}", flush=True)
    if failed:
        print(f"{failed} walk(s) failed: lists NOT written. Re-run the same command (finished masks are kept).")
        return
    counts = write_lists_and_yaml.remote(view_name, walks, validation_walks)
    print(f"DONE in {format_duration(time.time() - start_time)}: {total_written:,} masks written | frames per list: {counts}")
    print(f"Dataset file: /{DATASET}/views/{view_name}/data.yaml on Volume '{VOLUME_NAME}'")
