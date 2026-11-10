"""Evaluation D: YOLOE-26n-seg with text prompts (open vocabulary, no training) on SANPO MASKS (view masks_v1), on Modal.

What is measured
    All 16 box classes of the view. Each SANPO class gets one text prompt (PROMPTS below);
    the model's class i = PROMPTS entry i = SANPO class i, so predictions map 1:1 to ground truth.

Model
    YOLOE-26n-seg used as released, WITH its mask branch: YOLOE("yoloe-26n-seg.pt")
    (the box evaluation instead loaded it box-only from yoloe-26n.yaml).
    model.set_classes(prompts) runs the text encoder once (MobileCLIP, downloaded on first use).

How
    1. Reads the frame list (index CSVs) and ground-truth object masks (view masks_v1) from the Volume.
       Masks are compared at the model input size (frame resized so its long side = image size).
    2. Runs YOLOE-26n-seg on every frame of each split (GPU, batched).
    3. Scores MASKS with pycocotools (COCO "segm" metrics): AP50-95, AP50, AP75, AR, AP by object size,
       overall and per class. Predictions use confidence 0.001 (standard for mAP).
    4. Operating points: precision / recall / F1 / false alarms per frame at fixed confidences (mask IoU).
    5. Near objects: the same scores counting only objects within 5 m / 10 m (SANPO depth).
    Steps shared with the other evaluations live in ../evaluate_shared_code.py.

Output (written into ./results/<RUN_NAME>/ next to this script)
    metrics.json   all numbers (settings, per split overall + per class, operating points, speed)
    metrics.txt    all numbers as readable tables
    summary.md     short summary (main scores, near objects, per class)
    example_images/ ground truth vs prediction, 5 files x 5 rows

Resumable: predictions are saved on the Volume while running (evaluations/<name>/<run>/),
so a stopped run continues where it stopped; a finished split is not predicted again.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    (add --image-size 1280 to test another model input size; results go to a separate folder)
    python -m modal run fast_path/object_detection/evaluations/sanpo_masks_v1_yoloe26n_seg_text_prompts/evaluate.py
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
VIEW_NAME = "masks_v1"  # mask view (make_mask_view_modal.py): one mask per object, same objects as boxes_v1
TASK = "segment"  # compare masks (COCO "segm" scores), not boxes
SPLITS = ["val", "test"]  # splits to evaluate

WEIGHTS = "yoloe-26n-seg.pt"  # official released segmentation weights, used as they are (with masks)
IMAGE_SIZE = 640  # model input size (long side); override with --image-size, e.g. 1280
CONFIDENCE_THRESHOLD = 0.001  # low threshold: standard for mAP (keeps the full precision/recall curve)
NMS_IOU_THRESHOLD = 0.7  # overlap threshold used to remove duplicate predictions (Ultralytics default)
MAX_DETECTIONS = 100  # per frame (COCO scores use the top 100 anyway; limits the stored masks)
HALF_PRECISION = True  # FP16 (tested: works on ultralytics 8.4.208; it crashed on 8.4.11)
BATCH_SIZE = 64  # frames per GPU call (smaller than boxes: every prediction also has a mask)
GPU = "H100"
CPU_CORES = 16  # reading + resizing frames is the bottleneck, so give the container many CPUs
LOADER_THREADS = 32  # frames read and resized in parallel
RUN_NAME = f"yoloe26n_seg_prompts_v1_img{IMAGE_SIZE}"  # results/<RUN_NAME>/ (default image size)


def run_name_for(image_size: int) -> str:
    """Run name (results folder) for an image size, e.g. yoloe26n_seg_prompts_v1_img1280."""
    return f"yoloe26n_seg_prompts_v1_img{image_size}"

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

EVALUATION_NAME = "sanpo_masks_v1_yoloe26n_seg_text_prompts"  # written out: inside the Modal container the script folder is /root
VOLUME_MOUNT_PATH = "/data"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

app = modal.App(f"blind-guide-eval-{EVALUATION_NAME}".replace("_", "-")[:60])
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    # CLIP tokenizer is needed for text prompts (installed from GitHub, needs git)
    .pip_install("ultralytics==8.4.208", "pycocotools", "pandas", "nvidia-ml-py", "git+https://github.com/ultralytics/CLIP.git")
    .add_local_python_source("evaluate_shared_code")  # the shared code, available inside the container
)


# ----------------------------------------------------------------------------- Modal function

@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=32768, volumes={VOLUME_MOUNT_PATH: volume},
              timeout=4 * 3600, retries=1)
def run_evaluation(image_size: int = IMAGE_SIZE) -> dict:
    """The whole evaluation, on a Modal GPU (image_size = model input size, long side).

    Steps:
        1. Paths: dataset, a work folder for predictions (resume), a weights cache on the Volume.
        2. Load YOLOE-26n-seg (with masks) and set the 16 text prompts (text encoder runs once).
        3. Class mapping: model class i = prompt i = evaluation category i + 1; SANPO class -> evaluation category.
        4. For each split (shared.evaluate_splits):
           a. load frames + ground-truth masks,
           b. convert ground truth to COCO format,
           c. predict all frames (batched; saved regularly so a stopped run resumes),
           d. score masks with COCO metrics, operating points, near objects, and average speed.
        5. Return all results (settings, per-split overall + per-class numbers, speed).
    """
    import os

    import torch
    from ultralytics import YOLOE

    # Step 1: paths (the run name includes the image size)
    run_name = run_name_for(image_size)
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    work_directory = Path(VOLUME_MOUNT_PATH) / "evaluations" / EVALUATION_NAME / run_name  # predictions (resume)
    work_directory.mkdir(parents=True, exist_ok=True)
    weights_cache = Path(VOLUME_MOUNT_PATH) / "_cache" / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_cache)  # Ultralytics downloads weights into the current folder: keep them on the Volume

    # Step 2: model + text prompts
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | {WEIGHTS} (with masks) | image size {image_size} | "
          f"half precision {HALF_PRECISION}", flush=True)
    model = YOLOE(WEIGHTS)  # full segmentation model (boxes + masks)
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
        "evaluation": EVALUATION_NAME, "run": run_name, "gpu": gpu_name,
        "settings": {
            "weights": WEIGHTS, "image_size": image_size,
            "confidence_threshold": CONFIDENCE_THRESHOLD, "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS, "half_precision": HALF_PRECISION, "batch_size": BATCH_SIZE, "cpu_cores": CPU_CORES,
            "loader_threads": LOADER_THREADS,
            "dataset": DATASET, "view": VIEW_NAME, "task": TASK,
            "operating_confidences": OPERATING_CONFIDENCES, "match_iou": MATCH_IOU,
            "evaluation_classes": {evaluation_name: {"sanpo_ground_truth": sanpo_names, "prompt": prompts[0]}
                                   for evaluation_name, (sanpo_names, prompts) in EVALUATION_CLASSES.items()},
        },
        "splits": {},
    }

    # Print the full setup before anything is evaluated
    # shared.print_setup takes: results (settings), class ids, Volume, dataset, view, splits, confidences, match IoU
    #   returns: nothing (prints the setup)
    shared.print_setup(results, category_ids_by_name, VOLUME_NAME, DATASET, VIEW_NAME, SPLITS,
                       OPERATING_CONFIDENCES, MATCH_IOU)

    # Step 4: evaluate each split (ground truth -> predictions -> scores), shared with the other evaluations
    # shared.evaluate_splits takes: model, Volume, results, paths, classes and class mappings, prediction settings,
    #   task ("segment" here)
    #   returns: results, filled with results["splits"][split] (scores), results["near_objects"],
    #   results["example_images_jpeg"]
    shared.evaluate_splits(
        model=model, volume=volume, results=results, dataset_root=dataset_root, work_directory=work_directory,
        view_name=VIEW_NAME, splits=SPLITS, evaluation_classes=EVALUATION_CLASSES,
        category_ids_by_name=category_ids_by_name, model_class_to_category_id=model_class_to_category_id,
        sanpo_class_to_category_id=sanpo_class_to_category_id, image_size=image_size,
        confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD,
        max_detections=MAX_DETECTIONS, half_precision=HALF_PRECISION, batch_size=BATCH_SIZE,
        save_every_batches=SAVE_EVERY_BATCHES, operating_confidences=OPERATING_CONFIDENCES, match_iou=MATCH_IOU,
        loader_threads=LOADER_THREADS, task=TASK)

    # Step 5: return everything to the local computer
    return results


# ----------------------------------------------------------------------------- local: entry point

@app.local_entrypoint()
def main(image_size: int = IMAGE_SIZE):
    """Entry point of `modal run`: run the evaluation on Modal, then save the results locally.

    Steps:
        1. run_evaluation.remote(image_size): runs on a Modal GPU, returns the results dict.
        2. shared.write_results(): writes metrics.json, metrics.txt, summary.md, example_images/ into ./results/<RUN_NAME>/.
        3. Print the overview tables (aligned plain text; summary.md is for the editor).
    """
    import time

    start_time = time.time()
    run_name = run_name_for(image_size)
    print(f"Running {EVALUATION_NAME} / {run_name} on Modal ({GPU}) ...", flush=True)
    results = run_evaluation.remote(image_size)
    # shared.write_results takes: results dict, output folder
    #   returns: the output folder (writes metrics.json, metrics.txt, summary.md, example_images/)
    output_directory = shared.write_results(results, RESULTS_DIRECTORY / run_name)
    # shared.terminal_summary takes: results dict; returns: the overview as aligned plain-text tables
    print(shared.terminal_summary(results))
    print(f"Results saved to {output_directory} | total time {shared.format_duration(time.time() - start_time)}")
