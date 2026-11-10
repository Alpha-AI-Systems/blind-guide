"""Evaluation A: plain YOLO26n (COCO-pretrained, no training) on SANPO boxes (view boxes_v1), on Modal.

What is measured
    YOLO26n only knows COCO's 80 classes, so it is scored only on the SANPO classes
    that COCO can match. COCO "person" covers both SANPO pedestrian and rider, so
    those two are merged into one "person" class here:

        evaluation class   SANPO ground-truth classes   COCO classes counted as a match
        person             pedestrian, rider            person
        vehicle            vehicle                      bicycle, car, motorcycle, bus, train, truck
        animal             animal                       bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
        traffic light      traffic light                traffic light

    (SANPO "vehicle" includes bicycle, motorcycle, bus, train: SANPO paper, Appendix A.4.1.)
    All other SANPO classes are ignored, and all other COCO predictions are dropped.

How
    1. Reads the frame list (index CSVs) and ground-truth boxes (view boxes CSVs) from the Volume.
    2. Runs YOLO26n on every frame of each split (GPU, batched), keeps mapped predictions.
    3. Scores with pycocotools (standard COCO metrics): AP50-95, AP50, AP75, AR, AP by object size,
       overall and per class. Predictions use confidence 0.001 (standard for mAP).
    4. Operating points: precision / recall / F1 / false alarms per frame at fixed confidences.
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
    python -m modal run fast_path/object_detection/evaluations/sanpo_boxes_v1_yolo26n_coco_zero_shot/evaluate.py
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

WEIGHTS = "yolo26n.pt"
IMAGE_SIZE = 640  # model input size (long side); override with --image-size, e.g. 1280
CONFIDENCE_THRESHOLD = 0.001  # low threshold: standard for mAP (keeps the full precision/recall curve)
NMS_IOU_THRESHOLD = 0.7  # overlap threshold used to remove duplicate predictions (Ultralytics default)
MAX_DETECTIONS = 300  # per frame
HALF_PRECISION = True  # FP16
BATCH_SIZE = 256  # frames per GPU call (the GPU was mostly idle at 32)
GPU = "H100"
CPU_CORES = 16  # reading + resizing frames is the bottleneck, so give the container many CPUs
LOADER_THREADS = 32  # frames read and resized in parallel
RUN_NAME = f"yolo26n_img{IMAGE_SIZE}"  # results/<RUN_NAME>/ (default image size)


def run_name_for(image_size: int) -> str:
    """Run name (results folder) for an image size, e.g. yolo26n_img1280."""
    return f"yolo26n_img{image_size}"

# evaluation class -> (SANPO ground-truth classes, COCO class names counted as a match)
EVALUATION_CLASSES = {
    "person": (["pedestrian", "rider"], ["person"]),
    "vehicle": (["vehicle"], ["bicycle", "car", "motorcycle", "bus", "train", "truck"]),
    "animal": (["animal"], ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]),
    "traffic light": (["traffic light"], ["traffic light"]),
}
SAVE_EVERY_BATCHES = 20  # save partial predictions every N batches (for resuming)
# Operating points: what the user would get with one fixed confidence threshold on the phone.
OPERATING_CONFIDENCES = [0.25, 0.35, 0.5]  # confidence thresholds to report precision / recall / F1 at
MATCH_IOU = 0.5  # a prediction counts as correct if it overlaps a ground-truth box by IoU >= this
# =============================================================================

EVALUATION_NAME = "sanpo_boxes_v1_yolo26n_coco_zero_shot"  # written out: inside the Modal container the script folder is /root
VOLUME_MOUNT_PATH = "/data"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

app = modal.App(f"blind-guide-eval-{EVALUATION_NAME}".replace("_", "-")[:60])
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.11", "pycocotools", "pandas", "nvidia-ml-py")
    .add_local_python_source("evaluate_shared_code")  # the shared code, available inside the container
)


# ----------------------------------------------------------------------------- Modal function

@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=32768, volumes={VOLUME_MOUNT_PATH: volume},
              timeout=4 * 3600, retries=1)
def run_evaluation(image_size: int = IMAGE_SIZE) -> dict:
    """The whole evaluation, on a Modal GPU (image_size = model input size, long side).

    Steps:
        1. Paths: dataset, a work folder for predictions (resume), a weights cache on the Volume.
        2. Load YOLO26n (COCO-pretrained).
        3. Class mapping: COCO class id -> evaluation category, SANPO class -> evaluation category.
        4. For each split (shared.evaluate_splits):
           a. load frames + ground-truth boxes,
           b. convert ground truth to COCO format,
           c. predict all frames (batched; saved regularly so a stopped run resumes),
           d. score with COCO metrics, operating points, and average speed.
        5. Return all results (settings, per-split overall + per-class numbers, speed).
    """
    import os

    import torch
    from ultralytics import YOLO

    # Step 1: paths (the run name includes the image size)
    run_name = run_name_for(image_size)
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    work_directory = Path(VOLUME_MOUNT_PATH) / "evaluations" / EVALUATION_NAME / run_name  # predictions (resume)
    work_directory.mkdir(parents=True, exist_ok=True)
    weights_cache = Path(VOLUME_MOUNT_PATH) / "_cache" / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_cache)  # Ultralytics downloads weights into the current folder: keep them on the Volume

    # Step 2: model
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | weights: {WEIGHTS} | image size {image_size} | half precision {HALF_PRECISION}", flush=True)
    model = YOLO(WEIGHTS)
    volume.commit()

    # Step 3: class mapping. COCO class id -> evaluation category id (1-based for COCO format)
    evaluation_class_names = list(EVALUATION_CLASSES)

    # Each evaluation class gets a number: person -> 1, vehicle -> 2, animal -> 3, traffic light -> 4
    category_ids_by_name = {}
    for index, name in enumerate(evaluation_class_names):
        category_ids_by_name[name] = index + 1

    # COCO class name -> COCO class id used by the model (e.g. "car" -> 2)
    coco_class_id_by_name = {}
    for class_id, name in model.names.items():
        coco_class_id_by_name[name] = class_id

    # Model (COCO) class id -> evaluation category id (e.g. car id 2 -> vehicle 2, bus id 5 -> vehicle 2)
    model_class_to_category_id = {}
    for evaluation_name, (sanpo_names, coco_names) in EVALUATION_CLASSES.items():
        for coco_name in coco_names:
            coco_class_id = coco_class_id_by_name[coco_name]
            model_class_to_category_id[coco_class_id] = category_ids_by_name[evaluation_name]

    # SANPO ground-truth class -> evaluation category id (e.g. pedestrian -> 1, rider -> 1)
    sanpo_class_to_category_id = {}
    for evaluation_name, (sanpo_names, coco_names) in EVALUATION_CLASSES.items():
        for sanpo_name in sanpo_names:
            sanpo_class_to_category_id[sanpo_name] = category_ids_by_name[evaluation_name]

    # Results dict: settings first, then one entry per split
    results = {
        "evaluation": EVALUATION_NAME, "run": run_name, "gpu": gpu_name,
        "settings": {
            "weights": WEIGHTS, "image_size": image_size, "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD, "max_detections": MAX_DETECTIONS,
            "half_precision": HALF_PRECISION, "batch_size": BATCH_SIZE, "cpu_cores": CPU_CORES,
            "loader_threads": LOADER_THREADS, "dataset": DATASET, "view": VIEW_NAME,
            "operating_confidences": OPERATING_CONFIDENCES, "match_iou": MATCH_IOU,
            "evaluation_classes": {evaluation_name: {"sanpo_ground_truth": sanpo_names, "coco": coco_names}
                                   for evaluation_name, (sanpo_names, coco_names) in EVALUATION_CLASSES.items()},
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
        sanpo_class_to_category_id=sanpo_class_to_category_id, image_size=image_size,
        confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD,
        max_detections=MAX_DETECTIONS, half_precision=HALF_PRECISION, batch_size=BATCH_SIZE,
        save_every_batches=SAVE_EVERY_BATCHES, operating_confidences=OPERATING_CONFIDENCES, match_iou=MATCH_IOU,
        loader_threads=LOADER_THREADS)

    # Step 5: return everything to the local computer
    return results


# ----------------------------------------------------------------------------- local: entry point

@app.local_entrypoint()
def main(image_size: int = IMAGE_SIZE):
    """Entry point of `modal run`: run the evaluation on Modal, then save the results locally.

    Steps:
        1. run_evaluation.remote(image_size): runs on a Modal GPU, returns the results dict.
        2. shared.write_results(): writes metrics.json, metrics.txt, summary.md, example_images/ into ./results/<RUN_NAME>/.
        3. Print the summary.
    """
    import time

    start_time = time.time()
    run_name = run_name_for(image_size)
    print(f"Running {EVALUATION_NAME} / {run_name} on Modal ({GPU}) ...", flush=True)
    results = run_evaluation.remote(image_size)
    output_directory = shared.write_results(results, RESULTS_DIRECTORY / run_name)
    print((output_directory / "summary.md").read_text(encoding="utf-8"))
    print(f"Results saved to {output_directory} | total time {shared.format_duration(time.time() - start_time)}")
