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

Output (written into ./results/<RUN_NAME>/ next to this script)
    metrics.json   all numbers (settings, per split overall + per class, speed)
    summary.txt    the same as readable tables

Resumable: predictions are saved on the Volume while running (evaluations/<name>/<run>/),
so a stopped run continues where it stopped; a finished split is not predicted again.

Run from the repo root (PowerShell):
    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session1.toml"
    python -m modal run fast_path/object_detection/evaluations/sanpo_boxes_v1_yolo26n_coco_zero_shot/evaluate.py
"""

from __future__ import annotations

from pathlib import Path

import modal

# =============================================================================
# SETTINGS (edit here)
# =============================================================================
VOLUME_NAME = "sanpo"
DATASET = "sanpo_real_chest_5fps"  # folder in the Volume (download_sanpo_modal.py)
VIEW_NAME = "boxes_v1"  # box view (make_box_view_modal.py)
SPLITS = ["val", "test"]  # splits to evaluate

WEIGHTS = "yolo26n.pt"
IMAGE_SIZE = 640  # model input size (long side)
CONFIDENCE_THRESHOLD = 0.001  # low threshold: standard for mAP (keeps the full precision/recall curve)
NMS_IOU_THRESHOLD = 0.7  # overlap threshold used to remove duplicate predictions (Ultralytics default)
MAX_DETECTIONS = 300  # per frame
HALF_PRECISION = True  # FP16
BATCH_SIZE = 32
GPU = "L4"
RUN_NAME = f"yolo26n_img{IMAGE_SIZE}"  # results/<RUN_NAME>/

# evaluation class -> (SANPO ground-truth classes, COCO class names counted as a match)
EVALUATION_CLASSES = {
    "person": (["pedestrian", "rider"], ["person"]),
    "vehicle": (["vehicle"], ["bicycle", "car", "motorcycle", "bus", "train", "truck"]),
    "animal": (["animal"], ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]),
    "traffic light": (["traffic light"], ["traffic light"]),
}
SAVE_EVERY_BATCHES = 20  # save partial predictions every N batches (for resuming)
# =============================================================================

EVALUATION_NAME = Path(__file__).resolve().parent.name
VOLUME_MOUNT_PATH = "/data"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

app = modal.App(f"blind-guide-eval-{EVALUATION_NAME}".replace("_", "-")[:60])
volume = modal.Volume.from_name(VOLUME_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.11", "pycocotools", "pandas")
)


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s" (used in progress lines)."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


# ----------------------------------------------------------------------------- helpers (run on Modal)

def load_split(dataset_root: Path, split: str):
    """Load the frames and the ground-truth boxes of one split.

    Steps:
        1. Read every index CSV (one per session) -> one table, one row per frame.
        2. Keep only the frames of this split.
        3. Read every boxes CSV of the view for this split -> one table, one row per box (pixels).
    Returns (frames, ground_truth).
    """
    import pandas as pd

    # Step 1: all index CSV files (one file per session = one walk), e.g. index/abc123.csv
    index_files = sorted((dataset_root / "index").glob("*.csv"))
    # Read each CSV into a table and stack them into one big table (one row per frame, all sessions)
    all_frames = pd.concat([pd.read_csv(index_file) for index_file in index_files], ignore_index=True)
    # Step 2: keep only the rows (frames) of this split; renumber rows 0, 1, 2, ...
    frames = all_frames[all_frames["split"] == split].reset_index(drop=True)
    # Step 3: all boxes CSV files of this split (one per session), stacked into one table (one row per box)
    box_files = sorted((dataset_root / "views" / VIEW_NAME / "boxes" / split).glob("*.csv"))
    ground_truth = pd.concat([pd.read_csv(box_file) for box_file in box_files], ignore_index=True)
    return frames, ground_truth


def coco_evaluation(ground_truth_coco: dict, detections: list[dict], category_names_by_id: dict[int, str]) -> dict:
    """Score predictions against ground truth with the standard COCO method (pycocotools).

    Steps:
        1. Build the ground-truth COCO object from ground_truth_coco (images, annotations, categories).
        2. Load the predictions (image_id, category_id, bbox [x, y, width, height], score).
        3. evaluate(): per image and class, match predictions to ground-truth boxes at
           10 IoU thresholds (0.50, 0.55, ..., 0.95).
           accumulate(): build precision/recall curves. summarize(): the 12 standard numbers.
        4. Overall numbers (AP50-95, AP50, AP75, AP by size, AR).
        5. Per-class numbers read from the precision/recall arrays.
    Returns {"overall": {...}, "per_class": {class: {...}}}.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # Step 1: ground truth
    coco_ground_truth = COCO()
    coco_ground_truth.dataset = ground_truth_coco
    coco_ground_truth.createIndex()
    if not detections:
        return {"overall": None, "per_class": {}, "note": "no predictions"}
    # Step 2: predictions
    coco_detections = coco_ground_truth.loadRes(detections)
    # Step 3: match, build curves, summarize
    coco_evaluator = COCOeval(coco_ground_truth, coco_detections, iouType="bbox")
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()  # prints the standard COCO table to the log

    # Step 4: overall numbers (same order as coco_evaluator.stats)
    metric_names = ["AP50-95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    overall = {name: round(float(value), 4) for name, value in zip(metric_names, coco_evaluator.stats)}

    # Step 5: per-class numbers (-1 in the arrays = no data, ignored by mean_of_valid_values)
    precision = coco_evaluator.eval["precision"]  # [IoU thresholds, recall points, classes, area ranges, max detections]
    recall = coco_evaluator.eval["recall"]  # [IoU thresholds, classes, area ranges, max detections]

    def mean_of_valid_values(values):
        values = values[values > -1]
        return round(float(values.mean()), 4) if values.size else None

    # Area range index: 0 = all, 1 = small, 2 = medium, 3 = large. Last index = 100 detections per image.
    per_class = {}
    for class_index, category_id in enumerate(coco_evaluator.params.catIds):
        per_class[category_names_by_id[category_id]] = {
            "AP50-95": mean_of_valid_values(precision[:, :, class_index, 0, -1]),
            "AP50": mean_of_valid_values(precision[0, :, class_index, 0, -1]),
            "AP75": mean_of_valid_values(precision[5, :, class_index, 0, -1]),
            "AR100": mean_of_valid_values(recall[:, class_index, 0, -1]),
            "AP_small": mean_of_valid_values(precision[:, :, class_index, 1, -1]),
            "AP_medium": mean_of_valid_values(precision[:, :, class_index, 2, -1]),
            "AP_large": mean_of_valid_values(precision[:, :, class_index, 3, -1]),
            "ground_truth_boxes": len(coco_ground_truth.getAnnIds(catIds=[category_id])),
            "predictions": sum(1 for detection in detections if detection["category_id"] == category_id),
        }
    return {"overall": overall, "per_class": per_class}


# ----------------------------------------------------------------------------- Modal function

@app.function(image=image, gpu=GPU, volumes={VOLUME_MOUNT_PATH: volume}, timeout=4 * 3600, retries=1)
def run_evaluation() -> dict:
    """The whole evaluation, on a Modal GPU.

    Steps:
        1. Paths: dataset, a work folder for predictions (resume), a weights cache on the Volume.
        2. Load YOLO26n (COCO-pretrained).
        3. Class mapping: COCO class id -> evaluation category, SANPO class -> evaluation category.
        4. For each split:
           a. load frames + ground-truth boxes,
           b. convert ground truth to COCO format,
           c. predict all frames (batched; saved regularly so a stopped run resumes),
           d. score with COCO metrics + average speed.
        5. Return all results (settings, per-split overall + per-class numbers, speed).
    """
    import json
    import os
    import time

    import torch
    from ultralytics import YOLO

    # Step 1: paths
    dataset_root = Path(VOLUME_MOUNT_PATH) / DATASET
    work_directory = Path(VOLUME_MOUNT_PATH) / "evaluations" / EVALUATION_NAME / RUN_NAME  # predictions (resume)
    work_directory.mkdir(parents=True, exist_ok=True)
    weights_cache = Path(VOLUME_MOUNT_PATH) / "_cache" / "weights"
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.chdir(weights_cache)  # Ultralytics downloads weights into the current folder: keep them on the Volume

    # Step 2: model
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} | weights: {WEIGHTS} | image size {IMAGE_SIZE} | half precision {HALF_PRECISION}", flush=True)
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
        "evaluation": EVALUATION_NAME, "run": RUN_NAME, "gpu": gpu_name,
        "settings": {
            "weights": WEIGHTS, "image_size": IMAGE_SIZE, "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD, "max_detections": MAX_DETECTIONS,
            "half_precision": HALF_PRECISION, "batch_size": BATCH_SIZE, "dataset": DATASET, "view": VIEW_NAME,
            "evaluation_classes": {evaluation_name: {"sanpo_ground_truth": sanpo_names, "coco": coco_names}
                                   for evaluation_name, (sanpo_names, coco_names) in EVALUATION_CLASSES.items()},
        },
        "splits": {},
    }

    # Step 4: evaluate each split
    for split in SPLITS:
        print(f"\n===== split: {split} =====", flush=True)
        # Step 4a: frames and ground truth of the split
        frames, ground_truth = load_split(dataset_root, split)
        frame_names = [Path(frame_path).stem for frame_path in frames["frame"]]
        image_ids = {frame_name: index + 1 for index, frame_name in enumerate(frame_names)}

        # Step 4b: ground truth in COCO format (only SANPO classes that map to an evaluation class)
        # Keep only boxes whose SANPO class is evaluated here (pedestrian, rider, vehicle, animal, traffic light);
        # boxes of other classes (pole, tree, ...) are removed because this model cannot detect them
        ground_truth = ground_truth[ground_truth["class"].isin(sanpo_class_to_category_id)]
        # COCO format = one dict with 3 lists: images, annotations (ground-truth boxes), categories (classes)
        ground_truth_coco = {
            # One entry per frame: a number (id), its name and size. Frames with no boxes are kept too
            # (a prediction on them counts as a false positive)
            "images": [{"id": image_ids[frame_name], "file_name": frame_name, "width": int(width), "height": int(height)}
                       for frame_name, width, height in zip(frame_names, frames["width"], frames["height"])],
            # One entry per ground-truth box:
            #   image_id    = which frame it is in
            #   category_id = its evaluation class number (e.g. pedestrian -> 1 person)
            #   bbox        = [x, y, width, height] in pixels (COCO uses top-left corner + size, not x1, y1, x2, y2)
            #   area        = box area in pixels (COCO uses it to split boxes into small / medium / large)
            #   iscrowd     = 0: a normal single object
            # to_dict("records"): the column "class" is a Python keyword, so attribute access cannot be used
            "annotations": [
                {"id": index + 1, "image_id": image_ids[Path(box["image"]).stem],
                 "category_id": sanpo_class_to_category_id[box["class"]],
                 "bbox": [float(box["x1"]), float(box["y1"]), float(box["x2"] - box["x1"]), float(box["y2"] - box["y1"])],
                 "area": float((box["x2"] - box["x1"]) * (box["y2"] - box["y1"])), "iscrowd": 0}
                for index, box in enumerate(ground_truth.to_dict("records"))
            ],
            # One entry per evaluation class: its number and name (e.g. {"id": 2, "name": "vehicle"})
            "categories": [{"id": category_ids_by_name[name], "name": name} for name in evaluation_class_names],
        }
        print(f"{len(frame_names):,} frames | {len(ground_truth_coco['annotations']):,} ground-truth boxes "
              f"in evaluation classes", flush=True)

        # Step 4c: predictions, batched on the GPU (reused if finished, resumed if partial)
        # 4c.1: files on the Volume: final = all frames predicted, partial = progress saved so far
        final_path = work_directory / f"predictions_{split}.json"
        partial_path = work_directory / f"predictions_{split}.partial.json"
        # 4c.2: if this split was fully predicted in an earlier run, reuse it (skip prediction)
        if final_path.exists():
            saved = json.loads(final_path.read_text())
            print(f"Predictions already done ({len(saved['detections']):,}), reusing", flush=True)
        else:
            # 4c.3: start from the saved progress if a partial file exists, otherwise from zero.
            #   saved = {"frames_done": how many frames are predicted, "detections": predictions so far,
            #            "speed": per-frame timings}
            saved = (json.loads(partial_path.read_text()) if partial_path.exists()
                     else {"frames_done": 0, "detections": [], "speed": []})
            if saved["frames_done"]:
                print(f"Resuming from frame {saved['frames_done']:,}", flush=True)
            # 4c.4: full path of every frame of the split, and timers for progress / ETA
            frame_paths = [str(dataset_root / frame_path) for frame_path in frames["frame"]]
            start_time, batches_done, frames_done_at_start = time.time(), 0, saved["frames_done"]
            
            # 4c.5: loop over the frames in batches of BATCH_SIZE, starting after the frames already done
            for batch_start in range(saved["frames_done"], len(frame_paths), BATCH_SIZE):
                batch_paths = frame_paths[batch_start:batch_start + BATCH_SIZE]
                # 4c.6: run the model on the whole batch (one GPU call); one result per frame
                batch_results = model.predict(batch_paths, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD,
                                              iou=NMS_IOU_THRESHOLD, max_det=MAX_DETECTIONS, half=HALF_PRECISION,
                                              verbose=False)
                # 4c.7: for each frame of the batch: store its timings, then its predicted boxes
                for frame_path, result in zip(batch_paths, batch_results):
                    image_id = image_ids[Path(frame_path).stem]
                    saved["speed"].append([result.speed["preprocess"], result.speed["inference"],
                                           result.speed["postprocess"]])
                    # Each predicted box: corners (pixels), confidence score, model class id
                    for (x1, y1, x2, y2), score, model_class in zip(result.boxes.xyxy.tolist(),
                                                                    result.boxes.conf.tolist(),
                                                                    result.boxes.cls.tolist()):
                        # Keep it only if its class maps to an evaluation class; store it in COCO format
                        # (bbox = [x, y, width, height], same as the ground truth)
                        if int(model_class) in model_class_to_category_id:
                            saved["detections"].append({
                                "image_id": image_id, "category_id": model_class_to_category_id[int(model_class)],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
                # 4c.8: update progress and print speed, elapsed time and ETA
                saved["frames_done"] = min(batch_start + BATCH_SIZE, len(frame_paths))
                batches_done += 1
                elapsed = time.time() - start_time
                frames_per_second = (saved["frames_done"] - frames_done_at_start) / elapsed if elapsed else 0
                remaining_seconds = (len(frame_paths) - saved["frames_done"]) / frames_per_second if frames_per_second else 0
                print(f"[{split}] {saved['frames_done']:,}/{len(frame_paths):,} frames | {frames_per_second:.1f} frames/s | "
                      f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining_seconds)}", flush=True)
                # 4c.9: every SAVE_EVERY_BATCHES batches, save progress to the partial file (for resuming).
                # Written to a .tmp file first, then renamed, so a crash never leaves a broken file.
                if batches_done % SAVE_EVERY_BATCHES == 0:
                    temporary_path = partial_path.with_suffix(".tmp")
                    temporary_path.write_text(json.dumps(saved))
                    os.replace(temporary_path, partial_path)
                    volume.commit()
            # 4c.10: all frames done: save the final file and delete the partial one
            temporary_path = final_path.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(saved))
            os.replace(temporary_path, final_path)
            if partial_path.exists():
                partial_path.unlink()
            volume.commit()

        # Step 4d: score with COCO metrics, and average speed per frame
        print(f"Scoring {split} with pycocotools ...", flush=True)
        category_names_by_id = {category_id: name for name, category_id in category_ids_by_name.items()}
        metrics = coco_evaluation(ground_truth_coco, saved["detections"], category_names_by_id)
        speeds = saved["speed"]  # per frame: [preprocess, inference, postprocess] in ms
        metrics["frames"] = len(frame_names)
        metrics["speed_ms_per_frame"] = {
            "preprocess": round(sum(speed[0] for speed in speeds) / len(speeds), 2),
            "inference": round(sum(speed[1] for speed in speeds) / len(speeds), 2),
            "postprocess": round(sum(speed[2] for speed in speeds) / len(speeds), 2),
        } if speeds else None
        results["splits"][split] = metrics
    # Step 5: return everything to the local computer
    return results


# ----------------------------------------------------------------------------- local: save results

def write_results(results: dict) -> Path:
    """Save the results next to this script (runs on the local computer).

    Steps:
        1. results/<RUN_NAME>/metrics.json: everything, machine-readable.
        2. results/<RUN_NAME>/summary.txt: settings, class mapping, and per split an overall
           line, a speed line and a per-class table, human-readable.
    """
    import json
    from datetime import datetime

    output_directory = RESULTS_DIRECTORY / RUN_NAME
    output_directory.mkdir(parents=True, exist_ok=True)
    results["finished_at"] = datetime.now().isoformat(timespec="seconds")
    # Step 1: machine-readable
    (output_directory / "metrics.json").write_text(json.dumps(results, indent=2))

    # Step 2: readable summary
    settings = results["settings"]
    lines = [f"Evaluation: {results['evaluation']}  |  run: {results['run']}  |  GPU: {results['gpu']}",
             f"Finished: {results['finished_at']}",
             "Settings: " + ", ".join(f"{name}={value}" for name, value in settings.items()
                                      if name != "evaluation_classes"),
             "Evaluation classes (SANPO ground truth <- COCO predictions):"]
    for evaluation_name, mapping in settings["evaluation_classes"].items():
        lines.append(f"  {evaluation_name:<14} ground truth: {', '.join(mapping['sanpo_ground_truth'])}  "
                     f"<-  COCO: {', '.join(mapping['coco'])}")

    def format_metric(value):
        return f"{value:.3f}" if isinstance(value, float) else "  n/a"

    for split, metrics in results["splits"].items():
        lines += ["", f"=== {split}: {metrics['frames']:,} frames ==="]
        if metrics["overall"] is None:
            lines.append("no predictions")
            continue
        overall = metrics["overall"]
        lines.append(f"Overall: AP50-95 {overall['AP50-95']:.3f} | AP50 {overall['AP50']:.3f} | "
                     f"AP75 {overall['AP75']:.3f} | AR100 {overall['AR100']:.3f} | AP small/medium/large "
                     f"{overall['AP_small']:.3f}/{overall['AP_medium']:.3f}/{overall['AP_large']:.3f}")
        if metrics.get("speed_ms_per_frame"):
            speed = metrics["speed_ms_per_frame"]
            lines.append(f"Speed (ms/frame, {results['gpu']}): preprocess {speed['preprocess']} | "
                         f"inference {speed['inference']} | postprocess {speed['postprocess']}")
        lines.append(f"{'class':<16}{'GT boxes':>10}{'preds':>9}{'AP50-95':>9}{'AP50':>8}{'AP75':>8}{'AR100':>8}"
                     f"{'AP_s':>7}{'AP_m':>7}{'AP_l':>7}")
        for class_name, class_metrics in metrics["per_class"].items():
            lines.append(
                f"{class_name:<16}{class_metrics['ground_truth_boxes']:>10,}{class_metrics['predictions']:>9,}"
                f"{format_metric(class_metrics['AP50-95']):>9}{format_metric(class_metrics['AP50']):>8}"
                f"{format_metric(class_metrics['AP75']):>8}{format_metric(class_metrics['AR100']):>8}"
                f"{format_metric(class_metrics['AP_small']):>7}{format_metric(class_metrics['AP_medium']):>7}"
                f"{format_metric(class_metrics['AP_large']):>7}")
    (output_directory / "summary.txt").write_text("\n".join(lines) + "\n")
    return output_directory


@app.local_entrypoint()
def main():
    """Entry point of `modal run`: run the evaluation on Modal, then save the results locally.

    Steps:
        1. run_evaluation.remote(): runs on a Modal GPU, returns the results dict.
        2. write_results(): writes metrics.json + summary.txt into ./results/<RUN_NAME>/.
        3. Print the summary.
    """
    import time

    start_time = time.time()
    print(f"Running {EVALUATION_NAME} / {RUN_NAME} on Modal ({GPU}) ...", flush=True)
    results = run_evaluation.remote()
    output_directory = write_results(results)
    print((output_directory / "summary.txt").read_text())
    print(f"Results saved to {output_directory} | total time {format_duration(time.time() - start_time)}")
