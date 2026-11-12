"""Measure how sharp or blurry every SANPO frame is (local copy of the dataset, no model, CPU only).

Why
    Many SANPO frames are blurry (walking motion, rain, ...). We keep them in the dataset; this script
    only counts them and shows examples, so we know how many there are.

Sharpness score (one number per frame)
    "Variance of the Laplacian": the frame is turned to grayscale and resized to SCORE_WIDTH pixels wide,
    the Laplacian filter marks the edges, and the score is how much the edge strength varies.
    Sharp frame = strong edges = high score. Blurry frame = weak edges = low score.
    Limits: the score has no built-in line between "blurry" and "sharp" (look at the sheets to choose one),
    and a smooth scene (plain wall, fog) also scores low without being blurry.

Steps
    1. List all frames of all splits (index CSVs).
    2. Score every frame, in parallel threads. Scores are appended to results/sharpness_scores.csv
       while running, so a stopped run continues where it stopped.
    3. Summary: how the scores are spread (percentiles), overall and per split -> results/summary.md
    4. Example sheets: frames grouped from most blurry to sharpest (score bands by percentile),
       SHEET_ROWS x SHEET_COLUMNS random frames per band -> results/example_sheets/

Output (in ./results/ next to this script)
    sharpness_scores.csv   split, session, image, score (one row per frame)
    summary.md             percentile tables
    example_sheets/        band_01_... (most blurry) to band_08_... (sharpest)

Run from the repo root:
    python fast_path/object_detection/data_code/sanpo/blur_analysis/measure_blur.py
"""

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
DATASET_ROOT = Path(__file__).resolve().parents[3] / "data" / "sanpo" / "sanpo_real_chest_5fps"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
SCORE_WIDTH = 640  # frames are resized to this width before scoring (same scale for every frame, and fast)
WORKER_THREADS = max(1, (os.cpu_count() or 4) - 2)  # frames scored in parallel (OpenCV releases the Python lock)
FRAMES_PER_CHUNK = 500  # scores are saved and progress is printed after every chunk
# Score bands for the example sheets, as percentiles of all scores (0 = most blurry frame, 100 = sharpest)
BANDS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 75), (75, 100)]
PERCENTILES = [1, 5, 10, 20, 30, 50, 75, 90, 99]  # reported in the summary
SHEET_ROWS, SHEET_COLUMNS = 4, 4  # frames per example sheet
TILE_WIDTH = 480  # width of one frame on a sheet
RANDOM_SEED = 0  # same example frames on every run
# =============================================================================


def format_duration(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def sharpness_score(frame_path):
    """Score one frame: variance of the Laplacian of the grayscale frame resized to SCORE_WIDTH wide.

    Returns the score (float); higher = sharper.
    """
    import cv2

    gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    height, width = gray.shape
    gray = cv2.resize(gray, (SCORE_WIDTH, round(height * SCORE_WIDTH / width)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_sheet(rows, title):
    """Paste frames into one sheet image, each with its score written on it.

    rows: list of dictionaries (split, image, score). Returns the sheet (BGR image).
    """
    import cv2
    import numpy as np

    tile_height = round(TILE_WIDTH * 1242 / 2208)
    sheet = np.zeros((40 + SHEET_ROWS * tile_height, SHEET_COLUMNS * TILE_WIDTH, 3), dtype=np.uint8)
    cv2.putText(sheet, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for position, row in enumerate(rows):
        frame = cv2.imread(str(DATASET_ROOT / "frames" / row["split"] / row["image"]))
        tile = cv2.resize(frame, (TILE_WIDTH, tile_height), interpolation=cv2.INTER_AREA)
        label = f"score {row['score']:.0f} | {row['split']}"
        cv2.rectangle(tile, (0, 0), (250, 26), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        top = 40 + (position // SHEET_COLUMNS) * tile_height
        left = (position % SHEET_COLUMNS) * TILE_WIDTH
        sheet[top:top + tile_height, left:left + TILE_WIDTH] = tile
    return sheet


def main():
    import cv2
    import numpy as np
    import pandas as pd

    start_time = time.time()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIRECTORY / "sharpness_scores.csv"
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Results: {RESULTS_DIRECTORY}")
    print(f"Settings: score width {SCORE_WIDTH} px | {WORKER_THREADS} threads | bands {BANDS}")

    # Step 1: all frames of all splits
    print("\nStep 1/4: listing frames ...", flush=True)
    index_files = sorted((DATASET_ROOT / "index").glob("*.csv"))
    frames = pd.concat([pd.read_csv(index_file) for index_file in index_files], ignore_index=True)
    frames["image"] = frames["frame"].str.rsplit("/", n=1).str[-1]
    print(f"  {len(frames):,} frames in {frames['session'].nunique()} sessions "
          f"({', '.join(f'{split}: {count:,}' for split, count in frames['split'].value_counts().items())})", flush=True)

    # Step 2: score every frame (resumable)
    print("\nStep 2/4: scoring frames ...", flush=True)
    done_images = set()
    if scores_path.exists():
        done_images = set(pd.read_csv(scores_path)["image"])
        print(f"  {len(done_images):,} frames already scored, continuing", flush=True)
    else:
        with open(scores_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["split", "session", "image", "score"])
    todo = frames[~frames["image"].isin(done_images)].reset_index(drop=True)
    scoring_start = time.time()
    with ThreadPoolExecutor(WORKER_THREADS) as pool:
        for chunk_start in range(0, len(todo), FRAMES_PER_CHUNK):
            chunk = todo.iloc[chunk_start:chunk_start + FRAMES_PER_CHUNK]
            frame_paths = [str(DATASET_ROOT / frame_path) for frame_path in chunk["frame"]]
            # sharpness_score takes: frame path; returns: the sharpness score (float, higher = sharper)
            chunk_scores = list(pool.map(sharpness_score, frame_paths))
            with open(scores_path, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                for (split, session, image), score in zip(chunk[["split", "session", "image"]].itertuples(index=False),
                                                          chunk_scores):
                    writer.writerow([split, session, image, round(score, 2)])
            frames_done = min(chunk_start + FRAMES_PER_CHUNK, len(todo))
            elapsed = time.time() - scoring_start
            frames_per_second = frames_done / elapsed if elapsed else 0
            remaining = (len(todo) - frames_done) / frames_per_second if frames_per_second else 0
            print(f"  [{frames_done:,}/{len(todo):,}] {frames_per_second:.0f} frames/s | "
                  f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining)}", flush=True)
    scores = pd.read_csv(scores_path)
    print(f"  {len(scores):,} frames scored", flush=True)

    # Step 3: summary (percentiles overall and per split)
    print("\nStep 3/4: writing summary ...", flush=True)
    groups = [("all", scores)] + [(split, scores[scores["split"] == split]) for split in ["train", "val", "test"]]
    lines = ["# Sharpness of the SANPO frames", "",
             f"Score = variance of the Laplacian (grayscale frame resized to {SCORE_WIDTH} px wide). "
             "Higher = sharper, lower = blurrier. There is no built-in line between blurry and sharp: "
             "look at `example_sheets/` to choose one.", "",
             "## Score at each percentile (percentile 10 = 10% of the frames score below this value)", "",
             "| Frames | Count | " + " | ".join(f"p{percentile}" for percentile in PERCENTILES) + " |",
             "| --- | --- | " + " | ".join("---" for _ in PERCENTILES) + " |"]
    for name, group in groups:
        values = np.percentile(group["score"], PERCENTILES)
        lines.append(f"| {name} | {len(group):,} | " + " | ".join(f"{value:.0f}" for value in values) + " |")
    band_edges = np.percentile(scores["score"], sorted({edge for band in BANDS for edge in band}))
    edge_by_percentile = dict(zip(sorted({edge for band in BANDS for edge in band}), band_edges))
    lines += ["", "## Frames below each score, per split", "",
              "| Score below | all | train | val | test |", "| --- | --- | --- | --- | --- |"]
    for percentile in [1, 5, 10, 20, 30, 50]:
        limit = edge_by_percentile[percentile]
        cells = [f"{int((group['score'] < limit).sum()):,} ({100 * (group['score'] < limit).mean():.1f}%)"
                 for _, group in groups]
        lines.append(f"| {limit:.0f} | " + " | ".join(cells) + " |")
    summary_text = "\n".join(lines) + "\n"
    (RESULTS_DIRECTORY / "summary.md").write_text(summary_text, encoding="utf-8")
    print(summary_text, flush=True)

    # Step 4: example sheets per score band
    print("Step 4/4: drawing example sheets ...", flush=True)
    sheets_directory = RESULTS_DIRECTORY / "example_sheets"
    sheets_directory.mkdir(parents=True, exist_ok=True)
    for band_number, (low, high) in enumerate(BANDS, start=1):
        low_score, high_score = edge_by_percentile[low], edge_by_percentile[high]
        in_band = scores[(scores["score"] >= low_score) & (scores["score"] <= high_score)]
        picked = in_band.sample(n=min(SHEET_ROWS * SHEET_COLUMNS, len(in_band)), random_state=RANDOM_SEED)
        picked = picked.sort_values("score").to_dict("records")
        title = (f"Band {band_number}: percentile {low} to {high} | score {low_score:.0f} to {high_score:.0f} | "
                 f"{len(in_band):,} frames (0 = most blurry, 100 = sharpest)")
        # make_sheet takes: list of frames (split, image, score), title; returns: the sheet image (BGR)
        sheet = make_sheet(picked, title)
        file_path = sheets_directory / f"band_{band_number:02d}_percentile_{low:02d}_to_{high:02d}.jpg"
        cv2.imwrite(str(file_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  [{band_number}/{len(BANDS)}] saved {file_path.name}", flush=True)
    print(f"\nDone in {format_duration(time.time() - start_time)}. Results in {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
