"""Download the ROD obstacle-detection dataset (YOLO format) from Kaggle.

Dataset: abtinzandi/obstacle-detection-dataset (arXiv 2611.00681), about
1.5 GB, 24,326 images, 25 classes, train/valid/test splits, MIT license.
The same data is on Hugging Face, but it is ~48k small files and hits the
Hugging Face rate limit (1000 requests / 5 min), so we use Kaggle's single zip.

Needs Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_API_TOKEN).
Saved to fast_path/object_detection/data/rod/ (git-ignored).

Run from the repo root:
    python fast_path/object_detection/data_code/rod/download_rod.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

# =============================================================================
# SETTINGS (edit here). Command-line options override these.
# =============================================================================
KAGGLE_DATASET = "abtinzandi/obstacle-detection-dataset"
OBJECT_DETECTION_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = OBJECT_DETECTION_DIR / "data" / "rod"
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ROD dataset from Kaggle")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Target folder")
    args = parser.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} to {output} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(output), unzip=True, quiet=False)

    yamls = sorted(output.rglob("data.yaml"))
    print("data.yaml found at:", *yamls, sep="\n  ")
    root = yamls[0].parent if yamls else output
    for split in ("train", "valid", "test"):
        images = len(list((root / split / "images").glob("*")))
        labels = len(list((root / split / "labels").glob("*.txt")))
        print(f"{split:<6} images={images:<6} labels={labels}")


if __name__ == "__main__":
    main()
