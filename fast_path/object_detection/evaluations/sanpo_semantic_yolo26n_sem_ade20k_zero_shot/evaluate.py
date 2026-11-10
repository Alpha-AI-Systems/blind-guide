"""Evaluation F: YOLO26n-sem-ade20k (semantic model trained on ADE20K, no training by us) on SANPO, on Modal.

What is measured
    A semantic model gives ONE CLASS PER PIXEL (no objects, no confidence per object), so there is
    no AP here. The scores are IoU per class, mIoU (main score), pixel accuracy, precision and recall.

    YOLO26n-sem-ade20k knows the 150 ADE20K classes. It is scored on the SANPO classes they can match:

        evaluation class   SANPO ground-truth classes   ADE20K classes counted as a match
        road               road                         road
        sidewalk           sidewalk                     sidewalk
        paved trail        paved trail                  path
        building           building                     building, house, skyscraper, hovel
        wall/fence         wall/fence                   wall, fence
        hand rail          hand rail                    railing, bannister
        opening-door       opening-door                 door, screen door
        person             pedestrian, rider            person   (ADE20K has no rider class)
        animal             animal                       animal
        stairs             stairs                       stairs, stairway, step
        water body         water body                   water, sea, river, lake, swimming pool, waterfall
        vehicle            vehicle                      car, bus, truck, van, bicycle, minibike
        traffic sign       traffic sign                 signboard   (also covers shop signs)
        traffic light      traffic light                traffic light
        pole               pole                         pole, streetlight
        obstacle           obstacle                     bench, ashcan, box, barrel, rock
        sky                sky                          sky
        tree               tree                         tree, palm
        vegetation         vegetation                   plant, flower
        terrain            terrain                      grass, earth, sand, land, field

    The obstacle list comes from looking at real SANPO obstacles (trash cans, benches, boxes, rocks);
    SANPO obstacles such as snow piles and bollards have no ADE20K class.
    Pixels of all other SANPO classes (curb, crosswalk, guard rail/road barrier, bike rack, ...) and
    "unlabeled" pixels are ignored: not counted as right or wrong.
    All other ADE20K classes (about 100, mostly indoor) have no match: they count as wrong on scored pixels.

How
    1. Reads the frame list (index CSVs), the raw SANPO masks (red channel = class id per pixel)
       and the depth maps from the Volume.
    2. Runs YOLO26n-sem-ade20k on every frame of each split (GPU, batched). Frames, labels and depth are
       resized to the model input size (long side = image size).
    3. Counts pixels per (SANPO class, model class), then computes the scores:
       all pixels, and only pixels within 5 m / 10 m (SANPO depth).
    Steps shared with the other evaluations live in ../evaluate_shared_code.py (section SEMANTIC EVALUATIONS).

Output (written into ./results/<RUN_NAME>/ next to this script)
    metrics.json   all numbers, with the raw count tables
    metrics.txt    the result tables
    summary.md     the result tables as markdown
    example_images/ frame | ground truth | prediction, 5 files x 5 rows

Resumable: the pixel counts are saved on the Volume while running (evaluations/<name>/<run>/),
so a stopped run continues where it stopped; a finished split is not predicted again.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    (add --image-size 1280 to test another model input size; results go to a separate folder)
    python -m modal run fast_path/object_detection/evaluations/sanpo_semantic_yolo26n_sem_ade20k_zero_shot/evaluate.py
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
SPLITS = ["val", "test"]  # splits to evaluate

WEIGHTS = "yolo26n-sem-ade20k.pt"  # official semantic model trained on ADE20K (150 classes)
IMAGE_SIZE = 640  # model input size (long side); override with --image-size, e.g. 1280
HALF_PRECISION = True  # FP16
# Frames per GPU call, per image size (big batches, to use the H100's 80 GB).
# Measured GPU memory: 640: about 0.033 GB per frame, 1280: about 0.11 GB per frame.
BATCH_SIZES = {640: 128, 1280: 64}
DEFAULT_BATCH_SIZE = 64  # for any other image size
GPU = "H100"
CPU_CORES = 16  # reading + resizing frames, masks and depth is the bottleneck, so give the container many CPUs
LOADER_THREADS = 32  # frames read and resized in parallel
DISTANCE_LIMITS_METERS = (5, 10)  # extra scores that count only pixels within N meters
MAX_FRAMES = 0  # 0 = all frames; N = only the first N frames of each split (quick test)


def run_name_for(image_size: int) -> str:
    """Run name (results folder) for an image size, e.g. yolo26n_sem_ade20k_img1280."""
    return f"yolo26n_sem_ade20k_img{image_size}"


# evaluation class -> (SANPO ground-truth classes, model (ADE20K) classes counted as a match)
EVALUATION_CLASSES = {
    "road": (["road"], ["road"]),
    "sidewalk": (["sidewalk"], ["sidewalk"]),
    "paved trail": (["paved trail"], ["path"]),
    "building": (["building"], ["building", "house", "skyscraper", "hovel"]),
    "wall/fence": (["wall/fence"], ["wall", "fence"]),
    "hand rail": (["hand rail"], ["railing", "bannister"]),
    "opening-door": (["opening-door"], ["door", "screen door"]),
    "person": (["pedestrian", "rider"], ["person"]),
    "animal": (["animal"], ["animal"]),
    "stairs": (["stairs"], ["stairs", "stairway", "step"]),
    "water body": (["water body"], ["water", "sea", "river", "lake", "swimming pool", "waterfall"]),
    "vehicle": (["vehicle"], ["car", "bus", "truck", "van", "bicycle", "minibike"]),
    "traffic sign": (["traffic sign"], ["signboard"]),
    "traffic light": (["traffic light"], ["traffic light"]),
    "pole": (["pole"], ["pole", "streetlight"]),
    "obstacle": (["obstacle"], ["bench", "ashcan", "box", "barrel", "rock"]),
    "sky": (["sky"], ["sky"]),
    "tree": (["tree"], ["tree", "palm"]),
    "vegetation": (["vegetation"], ["plant", "flower"]),
    "terrain": (["terrain"], ["grass", "earth", "sand", "land", "field"]),
}
SAVE_EVERY_BATCHES = 2  # save the pixel counts every N batches (for resuming)
# =============================================================================

EVALUATION_NAME = "sanpo_semantic_yolo26n_sem_ade20k_zero_shot"  # written out: inside the Modal container the script folder is /root
VOLUME_MOUNT_PATH = "/data"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

app = modal.App(f"blind-guide-eval-{EVALUATION_NAME}".replace("_", "-")[:60])
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.208", "pycocotools", "pandas", "nvidia-ml-py")
    .add_local_python_source("evaluate_shared_code")  # the shared code, available inside the container
)


# ----------------------------------------------------------------------------- Modal function

@app.function(image=image, gpu=GPU, cpu=CPU_CORES, memory=32768, volumes={VOLUME_MOUNT_PATH: volume},
              timeout=4 * 3600, retries=1)
def run_evaluation(image_size: int = IMAGE_SIZE) -> dict:
    """The whole evaluation, on a Modal GPU (image_size = model input size, long side).

    Steps:
        1. Paths: dataset, a work folder for the pixel counts (resume), a weights cache on the Volume.
        2. Load the semantic model.
        3. Print the setup (settings and class mapping).
        4. For each split (shared.evaluate_semantic_splits): predict all frames, count pixels, score.
        5. Return all results.
    """
    import os

    import torch
    from ultralytics import YOLO

    # Step 1: paths (the run name includes the image size)
    run_name = run_name_for(image_size)
    batch_size = BATCH_SIZES.get(image_size, DEFAULT_BATCH_SIZE)
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    work_directory = Path(VOLUME_MOUNT_PATH) / "evaluations" / EVALUATION_NAME / run_name  # pixel counts (resume)
    work_directory.mkdir(parents=True, exist_ok=True)
    weights_cache = Path(VOLUME_MOUNT_PATH) / "_cache" / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_cache)  # Ultralytics downloads weights into the current folder: keep them on the Volume

    # Step 2: model
    gpu_name = torch.cuda.get_device_name(0)
    model = YOLO(WEIGHTS)
    volume.commit()

    # Results dict: settings first, then one entry per split
    results = {
        "evaluation": EVALUATION_NAME, "run": run_name, "gpu": gpu_name,
        "settings": {
            "weights": WEIGHTS, "image_size": image_size, "half_precision": HALF_PRECISION,
            "batch_size": batch_size, "cpu_cores": CPU_CORES, "loader_threads": LOADER_THREADS,
            "dataset": DATASET, "task": "semantic", "distance_limits_meters": list(DISTANCE_LIMITS_METERS),
            "max_frames": MAX_FRAMES,
            "evaluation_classes": {evaluation_name: {"sanpo_ground_truth": sanpo_names, "model": model_names}
                                   for evaluation_name, (sanpo_names, model_names) in EVALUATION_CLASSES.items()},
        },
        "splits": {},
    }

    # Step 3: print the full setup before anything is evaluated
    print("\n" + "=" * 80, flush=True)
    print(f"EVALUATION SETUP: {EVALUATION_NAME} / run {run_name}", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {gpu_name}", flush=True)
    print(f"Dataset: Volume '{VOLUME_NAME}' / {DATASET} | splits: {', '.join(SPLITS)}", flush=True)
    print(f"Model: {WEIGHTS} ({len(model.names)} classes) | image size {image_size} | half precision {HALF_PRECISION} | "
          f"batch size {batch_size}", flush=True)
    print(f"Evaluation classes ({len(EVALUATION_CLASSES)}): SANPO ground truth <- model classes counted as a match", flush=True)
    for number, (evaluation_name, (sanpo_names, model_names)) in enumerate(EVALUATION_CLASSES.items(), start=1):
        print(f"  {number:>2}. {evaluation_name:<20} {', '.join(sanpo_names):<24} <- {', '.join(model_names)}", flush=True)
    print(f"Scores: IoU per class, mIoU, pixel accuracy, precision, recall; also for pixels within "
          f"{' / '.join(str(limit) for limit in DISTANCE_LIMITS_METERS)} m", flush=True)
    print("=" * 80, flush=True)

    # Step 4: evaluate each split
    # shared.evaluate_semantic_splits takes: model, Volume, results, paths, splits, class mapping, prediction settings
    #   returns: results, filled with results["splits"][split] (scores, speed, raw count tables)
    #   and results["example_images_jpeg"]
    shared.evaluate_semantic_splits(
        model=model, volume=volume, results=results, dataset_root=dataset_root, work_directory=work_directory,
        splits=SPLITS, evaluation_classes=EVALUATION_CLASSES, image_size=image_size,
        half_precision=HALF_PRECISION, batch_size=batch_size, save_every_batches=SAVE_EVERY_BATCHES,
        loader_threads=LOADER_THREADS, distance_limits_meters=DISTANCE_LIMITS_METERS, max_frames=MAX_FRAMES)

    # Step 5: return everything to the local computer
    return results


# ----------------------------------------------------------------------------- local: entry point

@app.local_entrypoint()
def main(image_size: int = IMAGE_SIZE):
    """Entry point of `modal run`: run the evaluation on Modal, then save the results locally.

    Steps:
        1. run_evaluation.remote(image_size): runs on a Modal GPU, returns the results dict.
        2. shared.write_semantic_results(): writes metrics.json, metrics.txt, summary.md, example_images/
           into ./results/<RUN_NAME>/.
        3. Print the result tables.
    """
    import time

    start_time = time.time()
    run_name = run_name_for(image_size)
    print(f"Running {EVALUATION_NAME} / {run_name} on Modal ({GPU}) ...", flush=True)
    results = run_evaluation.remote(image_size)
    # shared.write_semantic_results takes: results dict, output folder
    #   returns: the output folder (writes metrics.json, metrics.txt, summary.md, example_images/)
    output_directory = shared.write_semantic_results(results, RESULTS_DIRECTORY / run_name)
    # shared.semantic_terminal_summary takes: results dict; returns: the result tables as text with borders
    print(shared.semantic_terminal_summary(results))
    print(f"Results saved to {output_directory} | total time {shared.format_duration(time.time() - start_time)}")
