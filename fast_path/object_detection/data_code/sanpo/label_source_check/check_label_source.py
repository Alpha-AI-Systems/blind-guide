"""Check which of our SANPO frames have HUMAN-made labels and which have MACHINE-made labels.

Why
    In SANPO-Real, people labeled only every 6th frame of a walk (frames 0, 6, 12, ...). The labels of
    the frames in between were made by a model (AOT) that carries the human labels forward
    (SANPO paper, appendix A.3.3). SANPO publishes, for every walk, a file that says which is which:
        sanpo-real/<session>/camera_chest/left/frame_segmentation_annotation_type.json
        {"0": "HUMAN_ANNOTATED", "1": "MACHINE_ANNOTATED", ...}   (key = frame number)
    Our copy keeps every 3rd labeled frame (5 per second), so we need to know how many of OUR frames
    are human-labeled.

Steps
    1. List our frames (index CSVs of the local dataset copy).
    2. Download SANPO's annotation-type file of every walk (small JSON files, kept in
       results/annotation_types/ so a second run does not download again).
    3. Give every one of our frames its label source (human / machine).
    4. Count per official split (train = our train + val folders, test) and save.

Output (in ./results/ next to this script)
    annotation_types/<session>.json   SANPO's files, unchanged
    label_source_per_frame.csv        split, official_split, session, frame_id, image, label_source
    summary.md                        the counts

Run from the repo root:
    python fast_path/object_detection/data_code/sanpo/label_source_check/check_label_source.py
"""

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "sanpo" / "sanpo_real_chest_5fps"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
SANPO_BUCKET_URL = "https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real"
CAMERA = "chest"
DOWNLOAD_THREADS = 16
# =============================================================================


def download_annotation_types(session):
    """Get SANPO's annotation-type file of one walk (from results/annotation_types/ if already downloaded).

    Returns {frame number (int): "HUMAN_ANNOTATED" or "MACHINE_ANNOTATED"}.
    """
    cache_path = RESULTS_DIRECTORY / "annotation_types" / f"{session}.json"
    if not cache_path.exists():
        url = f"{SANPO_BUCKET_URL}/{urllib.parse.quote(session)}/camera_{CAMERA}/left/frame_segmentation_annotation_type.json"
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
    return {int(frame_number): source for frame_number, source in json.loads(cache_path.read_text()).items()}


def main():
    import pandas as pd

    start_time = time.time()
    (RESULTS_DIRECTORY / "annotation_types").mkdir(parents=True, exist_ok=True)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Results: {RESULTS_DIRECTORY}")

    # Step 1: our frames
    print("\nStep 1/4: listing our frames ...", flush=True)
    index_files = sorted((DATASET_ROOT / "index").glob("*.csv"))
    frames = pd.concat([pd.read_csv(index_file, dtype={"frame_id": str}) for index_file in index_files],
                       ignore_index=True)
    frames["official_split"] = frames["split"].replace({"val": "train"})  # our val folder is official train
    frames["frame_number"] = frames["frame_id"].astype(int)
    frames["image"] = frames["frame"].str.rsplit("/", n=1).str[-1]
    sessions = sorted(frames["session"].unique())
    print(f"  {len(frames):,} frames in {len(sessions)} walks", flush=True)

    # Step 2: SANPO's annotation-type file of every walk
    print("\nStep 2/4: getting SANPO's annotation-type files ...", flush=True)
    annotation_types = {}
    with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
        # download_annotation_types takes: session id; returns: {frame number: "HUMAN_ANNOTATED" / "MACHINE_ANNOTATED"}
        for number, (session, types) in enumerate(zip(sessions, pool.map(download_annotation_types, sessions)), start=1):
            annotation_types[session] = types
            if number % 25 == 0 or number == len(sessions):
                elapsed = time.time() - start_time
                print(f"  [{number}/{len(sessions)}] elapsed {elapsed:.0f}s | "
                      f"ETA {elapsed / number * (len(sessions) - number):.0f}s", flush=True)

    # Step 3: label source of every one of our frames
    print("\nStep 3/4: matching our frames ...", flush=True)
    frames["label_source"] = [annotation_types[session].get(frame_number, "NOT_LISTED")
                              for session, frame_number in zip(frames["session"], frames["frame_number"])]
    frames["label_source"] = frames["label_source"].replace(
        {"HUMAN_ANNOTATED": "human", "MACHINE_ANNOTATED": "machine", "NOT_LISTED": "not listed"})
    frames[["split", "official_split", "session", "frame_id", "image", "label_source"]].to_csv(
        RESULTS_DIRECTORY / "label_source_per_frame.csv", index=False)

    # Step 4: counts
    print("\nStep 4/4: counting ...", flush=True)
    lines = ["# Human-made and machine-made labels", "",
             "Source: SANPO's `frame_segmentation_annotation_type.json` of every walk (chest camera).", "",
             "## Our copy (every 3rd labeled frame, 5 per second)", "",
             "| Split | Frames | Human-labeled | Machine-labeled | Not listed |", "| --- | --- | --- | --- | --- |"]
    for split in ["train", "test", "total"]:
        part = frames if split == "total" else frames[frames["official_split"] == split]
        counts = part["label_source"].value_counts()
        cells = [f"{int(counts.get(source, 0)):,} ({100 * counts.get(source, 0) / len(part):.1f}%)"
                 for source in ["human", "machine", "not listed"]]
        lines.append(f"| {split} | {len(part):,} | " + " | ".join(cells) + " |")

    # The native data (all 15 frames per second), from the same files
    split_by_session = frames.groupby("session")["official_split"].first()
    lines += ["", "## Native data (all labeled frames, 15 per second)", "",
              "| Split | Frames | Human-labeled | Machine-labeled |", "| --- | --- | --- | --- |"]
    native = {"train": {"human": 0, "machine": 0}, "test": {"human": 0, "machine": 0}}
    for session, types in annotation_types.items():
        for source in types.values():
            native[split_by_session[session]]["human" if source == "HUMAN_ANNOTATED" else "machine"] += 1
    native["total"] = {key: native["train"][key] + native["test"][key] for key in ["human", "machine"]}
    for split in ["train", "test", "total"]:
        total = native[split]["human"] + native[split]["machine"]
        lines.append(f"| {split} | {total:,} | {native[split]['human']:,} ({100 * native[split]['human'] / total:.1f}%) | "
                     f"{native[split]['machine']:,} ({100 * native[split]['machine'] / total:.1f}%) |")

    # Which frame numbers are human-labeled (remainder when the frame number is divided by 6)
    human_remainders, our_remainders = {}, frames["frame_number"].mod(6).value_counts().sort_index().to_dict()
    for types in annotation_types.values():
        for frame_number, source in types.items():
            if source == "HUMAN_ANNOTATED":
                human_remainders[frame_number % 6] = human_remainders.get(frame_number % 6, 0) + 1
    lines += ["", "## Alignment check", "",
              f"- Human-labeled frame numbers, by remainder of (frame number / 6): {dict(sorted(human_remainders.items()))}",
              f"- Our frame numbers, by remainder of (frame number / 6): {our_remainders}",
              f"- Walks where our copy has no human-labeled frame: "
              f"{int((frames.groupby('session')['label_source'].apply(lambda column: (column == 'human').sum()) == 0).sum())}"]
    summary_text = "\n".join(lines) + "\n"
    (RESULTS_DIRECTORY / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Done in {time.time() - start_time:.0f}s. Results in {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
