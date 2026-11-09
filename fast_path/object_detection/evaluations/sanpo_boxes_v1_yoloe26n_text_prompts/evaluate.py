"""Evaluation B: YOLOE-26n with text prompts (open vocabulary, no training) on SANPO boxes (view boxes_v1), on Modal.

What is measured
    All 16 box classes of the view. Each SANPO class gets one text prompt (PROMPTS below);
    the model's class i = PROMPTS entry i = SANPO class i, so predictions map 1:1 to ground truth.

Model
    YOLOE-26n loaded box-only, the official way: YOLOE("yoloe-26n.yaml").load("yoloe-26n-seg.pt")
    (all released YOLOE checkpoints are segmentation models; this drops the unused mask branch).
    model.set_classes(prompts) runs the text encoder once (MobileCLIP, downloaded on first use).

How
    1. Reads the frame list (index CSVs) and ground-truth boxes (view boxes CSVs) from the Volume.
    2. Runs YOLOE-26n on every frame of each split (GPU, batched).
    3. Scores with pycocotools (standard COCO metrics): AP50-95, AP50, AP75, AR, AP by object size,
       overall and per class. Predictions use confidence 0.001 (standard for mAP).
    4. Operating points: precision / recall / F1 / false alarms per frame at fixed confidences.
    Steps shared with the other evaluations live in ../evaluate_shared_code.py.

Output (written into ./results/<RUN_NAME>/ next to this script)
    metrics.json   all numbers (settings, per split overall + per class, operating points, speed)
    summary.txt    the same as readable tables

Resumable: predictions are saved on the Volume while running (evaluations/<name>/<run>/),
so a stopped run continues where it stopped; a finished split is not predicted again.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python -m modal run fast_path/object_detection/evaluations/sanpo_boxes_v1_yoloe26n_text_prompts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# The shared code is one folder up (evaluations/evaluate_shared_code.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_shared_code as shared  # noqa: E402

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"  # folder in the Volume (download_sanpo_modal.py)
VIEW_NAME = "boxes_v1"  # box view (make_box_view_modal.py)
SPLITS = ["val", "test"]  # splits to evaluate

ARCHITECTURE = "yoloe-26n.yaml"  # box-only YOLOE architecture
WEIGHTS = "yoloe-26n-seg.pt"  # official released weights (loaded into the box-only architecture)
IMAGE_SIZE = 640  # model input size (long side)
CONFIDENCE_THRESHOLD = 0.001  # low threshold: standard for mAP (keeps the full precision/recall curve)
NMS_IOU_THRESHOLD = 0.7  # overlap threshold used to remove duplicate predictions (Ultralytics default)
MAX_DETECTIONS = 300  # per frame
HALF_PRECISION = True  # FP16
BATCH_SIZE = 32
GPU = "L4"
RUN_NAME = f"yoloe26n_prompts_v1_img{IMAGE_SIZE}"  # results/<RUN_NAME>/

# SANPO class (ground truth) -> text prompt given to YOLOE. Order = class ids of the view (boxes_v1).
PROMPTS = {
    "pedestrian": "person",
    "rider": "person riding a bicycle",
    "animal": "animal",
    "vehicle": "vehicle",
    "obstacle": "obstacle",
    "pole": "pole",
    "tree": "tree",
    "traffic sign": "traffic sign",
    "traffic light": "traffic light",
    "bike rack": "bike rack",
    "stairs": "stairs",
    "crosswalk": "crosswalk",
    "opening-door": "door",
    "guard rail/road barrier": "guard rail",
    "wall/fence": "fence",
    "water body": "puddle",
}
# Same structure as evaluation A: evaluation class -> (SANPO ground-truth classes, [prompt])
EVALUATION_CLASSES = {sanpo_name: ([sanpo_name], [prompt]) for sanpo_name, prompt in PROMPTS.items()}
SAVE_EVERY_BATCHES = 20  # save partial predictions every N batches (for resuming)
# Operating points: what the user would get with one fixed confidence threshold on the phone.
OPERATING_CONFIDENCES = [0.25, 0.35, 0.5]  # confidence thresholds to report precision / recall / F1 at
MATCH_IOU = 0.5  # a prediction counts as correct if it overlaps a ground-truth box by IoU >= this
# =============================================================================

EVALUATION_NAME = Path(__file__).resolve().parent.name
VOLUME_MOUNT_PATH = "/data"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

app = modal.App(f"blind-guide-eval-{EVALUATION_NAME}".replace("_", "-")[:60])
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    # CLIP tokenizer is needed for text prompts (installed from GitHub, needs git)
    .pip_install("ultralytics==8.4.11", "pycocotools", "pandas", "git+https://github.com/ultralytics/CLIP.git")
    .add_local_python_source("evaluate_shared_code")  # the shared code, available inside the container
)


# ----------------------------------------------------------------------------- Modal function

@app.function(image=image, gpu=GPU, volumes={VOLUME_MOUNT_PATH: volume}, timeout=4 * 3600, retries=1)
def run_evaluation() -> dict:
    """The whole evaluation, on a Modal GPU.

    Steps:
        1. Paths: dataset, a work folder for predictions (resume), a weights cache on the Volume.
        2. Load YOLOE-26n box-only and set the 16 text prompts (text encoder runs once).
        3. Class mapping: model class i = prompt i = evaluation category i + 1; SANPO class -> evaluation category.
        4. For each split (shared.evaluate_splits):
           a. load frames + ground-truth boxes,
           b. convert ground truth to COCO format,
           c. predict all frames (batched; saved regularly so a stopped run resumes),
           d. score with COCO metrics, operating points, and average speed.
        5. Return all results (settings, per-split overall + per-class numbers, speed).
    """
    import os

    import torch
    from ultralytics import YOLOE

    # Step 1: paths
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    work_directory = Path(VOLUME_MOUNT_PATH) / "evaluations" / EVALUATION_NAME / RUN_NAME  # predictions (resume)
    work_directory.mkdir(parents=True, exist_ok=True)
    weights_cache = Path(VOLUME_MOUNT_PATH) / "_cache" / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_cache)  # Ultralytics downloads weights into the current folder: keep them on the Volume

    # Step 2: model + text prompts
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | {ARCHITECTURE} + {WEIGHTS} | image size {IMAGE_SIZE} | "
          f"half precision {HALF_PRECISION}", flush=True)
    model = YOLOE(ARCHITECTURE).load(WEIGHTS)  # box-only YOLOE (official method)
    print(f"Setting {len(PROMPTS)} text prompts: {list(PROMPTS.values())}", flush=True)
    model.set_classes(list(PROMPTS.values()))  # text encoder runs once; downloads into the weights cache
    volume.commit()

    # Step 3: class mapping. Model class i = PROMPTS entry i -> evaluation category id i + 1 (1-based for COCO)
    evaluation_class_names = list(EVALUATION_CLASSES)
    # Each evaluation class gets a number: pedestrian -> 1, rider -> 2, ... water body -> 16
    category_ids_by_name = {}
    for index, name in enumerate(evaluation_class_names):
        category_ids_by_name[name] = index + 1

    # Model class id -> evaluation category id. Model class i is prompt i, so it is simply i -> i + 1
    model_class_to_category_id = {}
    for index in range(len(evaluation_class_names)):
        model_class_to_category_id[index] = index + 1

    # SANPO ground-truth class -> evaluation category id (e.g. pole -> 6)
    sanpo_class_to_category_id = {}
    for evaluation_name, (sanpo_names, prompts) in EVALUATION_CLASSES.items():
        for sanpo_name in sanpo_names:
            sanpo_class_to_category_id[sanpo_name] = category_ids_by_name[evaluation_name]

    # Results dict: settings first, then one entry per split
    results = {
        "evaluation": EVALUATION_NAME, "run": RUN_NAME, "gpu": gpu_name,
        "settings": {
            "architecture": ARCHITECTURE, "weights": WEIGHTS, "image_size": IMAGE_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS, "half_precision": HALF_PRECISION, "batch_size": BATCH_SIZE,
            "dataset": DATASET, "view": VIEW_NAME,
            "operating_confidences": OPERATING_CONFIDENCES, "match_iou": MATCH_IOU,
            "evaluation_classes": {evaluation_name: {"sanpo_ground_truth": sanpo_names, "prompt": prompts[0]}
                                   for evaluation_name, (sanpo_names, prompts) in EVALUATION_CLASSES.items()},
        },
        "splits": {},
    }

    # Print the full setup before anything is evaluated
    shared.print_setup(results, category_ids_by_name, VOLUME_NAME, DATASET, VIEW_NAME, SPLITS,
                       OPERATING_CONFIDENCES, MATCH_IOU)

    # Step 4: evaluate each split (ground truth -> predictions -> scores), shared with the other evaluations
    shared.evaluate_splits(
        model=model, volume=volume, results=results, dataset_root=dataset_root, work_directory=work_directory,
        view_name=VIEW_NAME, splits=SPLITS, evaluation_classes=EVALUATION_CLASSES,
        category_ids_by_name=category_ids_by_name, model_class_to_category_id=model_class_to_category_id,
        sanpo_class_to_category_id=sanpo_class_to_category_id, image_size=IMAGE_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD,
        max_detections=MAX_DETECTIONS, half_precision=HALF_PRECISION, batch_size=BATCH_SIZE,
        save_every_batches=SAVE_EVERY_BATCHES, operating_confidences=OPERATING_CONFIDENCES, match_iou=MATCH_IOU)

    # Step 5: return everything to the local computer
    return results


# ----------------------------------------------------------------------------- local: entry point

@app.local_entrypoint()
def main():
    """Entry point of `modal run`: run the evaluation on Modal, then save the results locally.

    Steps:
        1. run_evaluation.remote(): runs on a Modal GPU, returns the results dict.
        2. shared.write_results(): writes metrics.json + summary.txt into ./results/<RUN_NAME>/.
        3. Print the summary.
    """
    import time

    start_time = time.time()
    print(f"Running {EVALUATION_NAME} / {RUN_NAME} on Modal ({GPU}) ...", flush=True)
    results = run_evaluation.remote()
    output_directory = shared.write_results(results, RESULTS_DIRECTORY / RUN_NAME)
    print((output_directory / "summary.txt").read_text())
    print(f"Results saved to {output_directory} | total time {shared.format_duration(time.time() - start_time)}")
