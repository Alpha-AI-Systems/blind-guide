"""Code shared by all evaluation scripts in this folder (each subfolder has its own evaluate.py).

Each evaluate.py keeps only what is specific to its model: settings, model loading and the
class mapping (steps 1 to 3). Everything else lives here:

    format_duration()          seconds -> "0h01m05s"
    load_split()               frames + ground-truth boxes of one split
    coco_evaluation()          standard COCO scores (pycocotools)
    operating_point_metrics()  precision / recall / F1 / false alarms at fixed confidences
    print_setup()              prints the full setup before anything is evaluated
    evaluate_splits()          step 4: ground truth, predictions (resumable), scoring, for every split
    write_results()            saves results/<run>/metrics.json, metrics.txt, summary.md, example_images/

How the scripts import it: they add this folder to sys.path (local run), and their Modal image
adds this file with Image.add_local_python_source("evaluate_shared_code") (container run).
"""

from __future__ import annotations

from pathlib import Path


def format_duration(seconds: float) -> str:
    """Seconds -> "0h01m05s" (used in progress lines)."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


# ----------------------------------------------------------------------------- helpers (run on Modal)

def load_split(dataset_root: Path, split: str, view_name: str):
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
    box_files = sorted((dataset_root / "views" / view_name / "boxes" / split).glob("*.csv"))
    ground_truth = pd.concat([pd.read_csv(box_file) for box_file in box_files], ignore_index=True)
    return frames, ground_truth


def load_mask_split(dataset_root: Path, split: str, view_name: str):
    """Load the frames and the ground-truth object MASKS of one split (mask view, e.g. masks_v1).

    Same frames as load_split(). Ground truth: one row per object with image, class, source,
    box x1, y1, x2, y2 and area (full-resolution pixels) and segmentation (COCO RLE, full resolution).
    Returns (frames, ground_truth).
    """
    import json

    import pandas as pd

    # Steps 1-2: frames of the split (same as load_split)
    index_files = sorted((dataset_root / "index").glob("*.csv"))
    all_frames = pd.concat([pd.read_csv(index_file) for index_file in index_files], ignore_index=True)
    frames = all_frames[all_frames["split"] == split].reset_index(drop=True)
    # Step 3: every object of every frame, from the per-session JSON files of the mask view
    rows = []
    for object_file in sorted((dataset_root / "views" / view_name / "objects" / split).glob("*.json")):
        for frame in json.loads(object_file.read_text())["frames"]:
            for item in frame["objects"]:
                x, y, width, height = item["bbox"]
                rows.append({"image": frame["image"], "class": item["class"], "source": item["source"],
                             "x1": x, "y1": y, "x2": x + width, "y2": y + height, "area": item["area"],
                             "segmentation": item["segmentation"]})
    return frames, pd.DataFrame(rows)


def comparison_size(width: int, height: int, image_size: int) -> tuple[int, int]:
    """Size (width, height) of a frame after resizing its long side to image_size (as load_and_resize_frame)."""
    ratio = image_size / max(width, height)
    return round(width * ratio), round(height * ratio)


def resize_rle(arguments: tuple) -> dict:
    """Resize one COCO RLE mask to (width, height). Top-level so it can run in a process pool.

    RLE (run-length encoding) = COCO's compact mask format: instead of every pixel, it stores the
    lengths of the runs of 0s and 1s ("1200 zeros, 30 ones, 500 zeros, ...").
    """
    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    # One tuple argument (so a process pool can map over a list): the mask and the target size
    segmentation, width, height = arguments
    # 1. RLE -> full binary image (H x W, values 0 / 1)
    mask = mask_utils.decode(segmentation)
    # 2. Resize. Nearest neighbor keeps the values exactly 0 or 1 (no blurry in-between values)
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    # 3. Binary image -> RLE again. pycocotools needs a uint8 array in Fortran (column-major) order
    encoded = mask_utils.encode(np.asfortranarray(resized.astype(np.uint8)))
    # 4. "counts" comes back as bytes; store it as text so it can be saved in JSON
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def box_iou_matrix(first_boxes, second_boxes):
    """IoU between every box in first_boxes and every box in second_boxes (boxes as [x, y, width, height]).

    IoU (intersection over union) = overlap area / combined area: 1 = identical boxes, 0 = no overlap.
    Returns a matrix: row = a box of first_boxes, column = a box of second_boxes.
    """
    import numpy as np

    # Boxes as arrays: column 0 = x (left), 1 = y (top), 2 = width, 3 = height
    first = np.asarray(first_boxes, dtype=float)
    second = np.asarray(second_boxes, dtype=float)
    # Right and bottom edges (x + width, y + height)
    first_right, first_bottom = first[:, 0] + first[:, 2], first[:, 1] + first[:, 3]
    second_right, second_bottom = second[:, 0] + second[:, 2], second[:, 1] + second[:, 3]
    # Overlap rectangle of every pair: [:, None] and [None] compare each first box with each second box.
    # Width = smaller right edge - larger left edge; negative means no overlap, so clip at 0.
    overlap_width = np.clip(np.minimum(first_right[:, None], second_right[None])
                            - np.maximum(first[:, 0][:, None], second[:, 0][None]), 0, None)
    overlap_height = np.clip(np.minimum(first_bottom[:, None], second_bottom[None])
                             - np.maximum(first[:, 1][:, None], second[:, 1][None]), 0, None)
    intersection = overlap_width * overlap_height
    # Union = area of first + area of second - the overlap (otherwise counted twice)
    union = (first[:, 2] * first[:, 3])[:, None] + (second[:, 2] * second[:, 3])[None] - intersection
    # Divide; the tiny minimum avoids dividing by zero for empty boxes
    return intersection / np.maximum(union, 1e-9)


def iou_matrix(first_items, second_items, iou_type: str):
    """IoU between two lists of shapes: boxes ([x, y, width, height], iou_type "bbox") or COCO RLE masks ("segm").

    Returns a matrix: row = an item of first_items, column = an item of second_items.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    if iou_type == "segm":
        # Masks: pycocotools computes the overlap directly on the RLE masks (no need to decode them).
        # Third argument = "is crowd" flag per second item; 0 = normal objects (standard IoU).
        return np.asarray(mask_utils.iou(list(first_items), list(second_items), [0] * len(second_items)))
    # Boxes: plain rectangle overlap
    return box_iou_matrix(first_items, second_items)


def coco_evaluation(ground_truth_coco: dict, detections: list[dict], category_names_by_id: dict[int, str],
                    iou_type: str = "bbox", detection_area_scale: float = 1.0) -> dict:
    """Score predictions against ground truth with the standard COCO method (pycocotools).

    Steps:
        1. Build the ground-truth COCO object from ground_truth_coco (images, annotations, categories).
        2. Load the predictions (image_id, category_id, bbox [x, y, width, height], score).
        3. evaluate(): per image and class, match predictions to ground-truth boxes at
           10 IoU thresholds (0.50, 0.55, ..., 0.95).
           accumulate(): build precision/recall curves. summarize(): the 12 standard numbers.
        4. Overall numbers (AP50-95, AP50, AP75, AP by size, AR).
        5. Per-class numbers read from the precision/recall arrays.
    iou_type: "bbox" (compare boxes) or "segm" (compare masks, COCO RLE in "segmentation").
    detection_area_scale: masks can be compared at a smaller resolution than the frame, while the
        ground-truth "area" is in full-resolution pixels; prediction areas are multiplied by this
        factor so the small / medium / large size groups stay the same as for boxes.
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
    # Step 2: predictions (pycocotools computes each mask's area; scale it to full-resolution pixels)
    coco_detections = coco_ground_truth.loadRes(detections)
    if detection_area_scale != 1.0:
        for annotation in coco_detections.anns.values():
            annotation["area"] *= detection_area_scale
    # Step 3: match, build curves, summarize
    coco_evaluator = COCOeval(coco_ground_truth, coco_detections, iouType=iou_type)
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


def operating_point_metrics(ground_truth_coco: dict, detections: list[dict], category_names_by_id: dict[int, str],
                            confidences: list[float], match_iou: float, iou_type: str = "bbox") -> dict:
    """Precision, recall, F1 and false alarms per frame at fixed confidence thresholds.

    COCO AP averages over all thresholds; on the phone we use ONE threshold, so these numbers show
    what the user would actually get.

    Steps:
        1. Group ground-truth boxes and predictions by (frame, class).
        2. In each group, sort predictions by score (high first) and match each one to the
           best still-unmatched ground-truth box with IoU >= match_iou (greedy, one-to-one).
           A matched prediction = true positive (TP), an unmatched one = false positive (FP).
        3. For each confidence c: keep predictions with score >= c and count
           TP, FP, and FN (missed ground-truth boxes = ground truth - TP).
        4. precision = TP / (TP + FP), recall = TP / ground truth, F1 = 2 P R / (P + R),
           false alarms per frame = FP / number of frames. Per class and overall.
        5. "all_predictions": the same with every prediction kept (confidence >= the run's
           low threshold) = the highest recall the model can reach (safety number: how many
           objects it finds at all).
    iou_type: "bbox" compares boxes, "segm" compares masks (COCO RLE in "segmentation").
    """
    import numpy as np
    from collections import defaultdict

    frame_count = len(ground_truth_coco["images"])
    shape_key = "segmentation" if iou_type == "segm" else "bbox"  # what is compared: masks or boxes

    # Step 1: group by (frame, class)
    ground_truth_boxes = defaultdict(list)
    for annotation in ground_truth_coco["annotations"]:
        ground_truth_boxes[(annotation["image_id"], annotation["category_id"])].append(annotation[shape_key])
    predictions = defaultdict(list)
    for detection in detections:
        predictions[(detection["image_id"], detection["category_id"])].append(
            (detection["score"], detection[shape_key]))

    # Step 2: greedy matching once per group (in score order), remember (score, is_true_positive)
    scored_predictions = defaultdict(list)  # category id -> list of (score, is_true_positive)
    for (image_id, category_id), group in predictions.items():
        group.sort(key=lambda item: item[0], reverse=True)
        truth = ground_truth_boxes.get((image_id, category_id), [])
        if not truth:
            scored_predictions[category_id] += [(score, False) for score, _ in group]
            continue
        overlaps = iou_matrix([shape for _, shape in group], truth, iou_type)
        matched = np.zeros(len(truth), dtype=bool)
        for prediction_index, (score, _) in enumerate(group):
            candidate_overlaps = np.where(matched, -1.0, overlaps[prediction_index])
            best = int(candidate_overlaps.argmax())
            is_true_positive = candidate_overlaps[best] >= match_iou
            if is_true_positive:
                matched[best] = True
            scored_predictions[category_id].append((score, bool(is_true_positive)))

    ground_truth_count = defaultdict(int)
    for (_, category_id), boxes in ground_truth_boxes.items():
        ground_truth_count[category_id] += len(boxes)

    def summarize(true_positives, false_positives, total_ground_truth):
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else None
        recall = true_positives / total_ground_truth if total_ground_truth else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        return {"true_positives": true_positives, "false_positives": false_positives,
                "missed": total_ground_truth - true_positives, "ground_truth": total_ground_truth,
                "precision": None if precision is None else round(precision, 4),
                "recall": None if recall is None else round(recall, 4),
                "f1": None if f1 is None else round(f1, 4),
                "false_alarms_per_frame": round(false_positives / frame_count, 4) if frame_count else None}

    # Steps 3-5: count at each confidence (+ all predictions)
    report = {"match_iou": match_iou, "frames": frame_count, "by_confidence": {}}
    labels_and_confidences = [(f"{confidence:g}", confidence) for confidence in confidences]
    labels_and_confidences.append(("all_predictions", float("-inf")))
    for label, minimum_confidence in labels_and_confidences:
        per_class = {}
        total_true_positives, total_false_positives, total_ground_truth = 0, 0, 0
        for category_id, class_name in category_names_by_id.items():
            kept = [is_true_positive for score, is_true_positive in scored_predictions.get(category_id, [])
                    if score >= minimum_confidence]
            true_positives = sum(kept)
            false_positives = len(kept) - true_positives
            per_class[class_name] = summarize(true_positives, false_positives, ground_truth_count[category_id])
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_ground_truth += ground_truth_count[category_id]
        report["by_confidence"][label] = {
            "overall": summarize(total_true_positives, total_false_positives, total_ground_truth),
            "per_class": per_class,
        }
    return report


def print_setup(results: dict, category_ids_by_name: dict[str, int], volume_name: str, dataset: str,
                view_name: str, splits: list[str], operating_confidences: list[float], match_iou: float) -> None:
    """Print the full setup before anything is evaluated."""
    print("\n" + "=" * 80, flush=True)
    print(f"EVALUATION SETUP: {results['evaluation']} / run {results['run']}", flush=True)
    print("=" * 80, flush=True)
    print(f"GPU: {results['gpu']}", flush=True)
    print(f"Dataset: Volume '{volume_name}' / {dataset} | box view: {view_name} | splits: {', '.join(splits)}",
          flush=True)
    print("Settings:", flush=True)
    for setting_name, setting_value in results["settings"].items():
        if setting_name != "evaluation_classes":
            print(f"  {setting_name:<22} {setting_value}", flush=True)
    print(f"Evaluation classes ({len(results['settings']['evaluation_classes'])}), "
          f"ground truth <- what the model predicts:", flush=True)
    for evaluation_name, mapping in results["settings"]["evaluation_classes"].items():
        parts = [f"{key}: {', '.join(value) if isinstance(value, list) else value}" for key, value in mapping.items()]
        print(f"  {category_ids_by_name[evaluation_name]:>2}. {evaluation_name:<24} " + " | ".join(parts), flush=True)
    print(f"Operating points: precision / recall / F1 / false alarms per frame at confidence {operating_confidences} "
          f"and with all predictions (IoU >= {match_iou})", flush=True)
    print("Metrics: COCO (pycocotools): AP50-95, AP50, AP75, AR100, AP small/medium/large, overall and per class; "
          "speed in ms/frame", flush=True)
    print(f"Results will be saved to: results/{results['run']}/ (metrics.json, metrics.txt, summary.md, example_images/)", flush=True)
    print("=" * 80, flush=True)


def load_and_resize_frame(frame_path: str, image_size: int):
    """Read one frame and resize it so its long side = image_size (what Ultralytics would do anyway).

    Runs in a loader thread (OpenCV releases the Python lock, so threads really run in parallel).
    Returns (resized BGR image, x scale, y scale); multiplying a box in the resized image by the
    scales gives the box in the original frame.
    """
    import cv2

    image = cv2.imread(frame_path)
    height, width = image.shape[:2]
    ratio = image_size / max(height, width)
    new_width, new_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # same as Ultralytics
    return resized, width / new_width, height / new_height


def gpu_usage() -> str:
    """Current GPU memory (used by this process / total) and GPU utilization, for progress lines."""
    try:
        import torch

        used_gb = torch.cuda.memory_reserved() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        try:
            utilization = f"{torch.cuda.utilization()}%"  # needs nvidia-ml-py
        except Exception:
            utilization = "n/a"
        return f"GPU mem {used_gb:.1f}/{total_gb:.0f} GB, util {utilization}"
    except Exception:
        return "GPU n/a"


def gpu_memory_peak_gb() -> float | None:
    """Highest GPU memory reserved by this process so far, in GB."""
    try:
        import torch

        return round(torch.cuda.max_memory_reserved() / 1e9, 2)
    except Exception:
        return None


# ----------------------------------------------------------------------------- step 4 (run on Modal)

def evaluate_splits(*, model, volume, results: dict, dataset_root: Path, work_directory: Path, view_name: str,
                    splits: list[str], evaluation_classes: dict, category_ids_by_name: dict[str, int],
                    model_class_to_category_id: dict[int, int], sanpo_class_to_category_id: dict[str, int],
                    image_size: int, confidence_threshold: float, nms_iou_threshold: float, max_detections: int,
                    half_precision: bool, batch_size: int, save_every_batches: int,
                    operating_confidences: list[float], match_iou: float, loader_threads: int = 32,
                    task: str = "box", distance_limits_meters: tuple = (5, 10),
                    distances_view_name: str = "boxes_v1") -> dict:
    """Step 4 of every evaluation: for each split, ground truth -> predictions -> scores.

    task: "box" (compare boxes, view like boxes_v1) or "segment" (compare masks, view like masks_v1).
        Segment: masks are compared at the model's input size (the frame resized so its long side
        = image_size), using Ultralytics' retina_masks (masks returned at the input frame's size);
        the ground-truth masks are resized to the same size.
    distance_limits_meters: extra "near objects" scores, counting only ground truth within N meters
        (distance = median SANPO depth over the object's pixels, from views/<distances_view_name>/distances;
        skipped if those files do not exist). Predictions that match a farther object are ignored.

    Steps (per split):
        a. load frames + ground truth (boxes or masks), attach object distances,
        b. convert ground truth to COCO format,
        c. predict all frames (batched; saved regularly so a stopped run resumes),
        d. score with COCO metrics, operating points, near-object scores, and average speed,
        e. example images (last split).
    Fills results["splits"][split] (and results["near_objects"]) and returns results.
    """
    import json
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    import numpy as np
    from pycocotools import mask as mask_utils

    evaluation_class_names = list(evaluation_classes)
    iou_type = "segm" if task == "segment" else "bbox"

    # Step 4: evaluate each split
    for split in splits:
        print(f"\n===== split: {split} =====", flush=True)
        # Step 4a: frames and ground truth of the split (boxes or masks)
        if task == "segment":
            frames, ground_truth = load_mask_split(dataset_root, split, view_name)
        else:
            frames, ground_truth = load_split(dataset_root, split, view_name)
        frame_names = [Path(frame_path).stem for frame_path in frames["frame"]]
        image_ids = {frame_name: index + 1 for index, frame_name in enumerate(frame_names)}

        # Step 4b: ground truth in COCO format (only SANPO classes that map to an evaluation class)
        # Keep only objects whose SANPO class is evaluated here; objects of other classes are removed
        # because this model is not scored on them
        ground_truth = ground_truth[ground_truth["class"].isin(sanpo_class_to_category_id)].reset_index(drop=True)
        # Distance of each object (meters), if computed for this split (same objects as boxes_v1)
        distance_files = sorted((dataset_root / "views" / distances_view_name / "distances" / split).glob("*.csv"))
        if distance_files:
            import pandas as pd

            distances = pd.concat([pd.read_csv(path) for path in distance_files], ignore_index=True)
            ground_truth = ground_truth.merge(distances[["image", "class", "x1", "y1", "x2", "y2", "distance_m"]],
                                              on=["image", "class", "x1", "y1", "x2", "y2"], how="left")
            print(f"Object distances attached ({int(ground_truth['distance_m'].notna().sum()):,} of "
                  f"{len(ground_truth):,} objects)", flush=True)
        else:
            ground_truth["distance_m"] = float("nan")
            print("No object distances for this split: near-object scores skipped", flush=True)

        # Segment: ground-truth masks resized to the comparison size (model input size), in parallel processes
        frame_sizes = {f"{Path(path).stem}.jpg": (int(width), int(height))
                       for path, width, height in zip(frames["frame"], frames["width"], frames["height"])}
        comparison_sizes = {image: comparison_size(width, height, image_size)
                            for image, (width, height) in frame_sizes.items()}
        detection_area_scale = 1.0
        if task == "segment":
            print(f"Resizing {len(ground_truth):,} ground-truth masks to the model input size ...", flush=True)
            resize_jobs = [(segmentation, *comparison_sizes[image])
                           for segmentation, image in zip(ground_truth["segmentation"], ground_truth["image"])]
            with ProcessPoolExecutor(os.cpu_count()) as pool:
                ground_truth["segmentation_compared"] = list(pool.map(resize_rle, resize_jobs, chunksize=256))
            # Ground-truth areas stay in full-resolution pixels; predicted mask areas are scaled to match
            full_width, full_height = next(iter(frame_sizes.values()))
            compared_width, compared_height = next(iter(comparison_sizes.values()))
            detection_area_scale = (full_width * full_height) / (compared_width * compared_height)

        def image_size_entry(frame_name, width, height):
            # Segment: masks are compared at the comparison size, so the image size is that size
            if task == "segment":
                width, height = comparison_sizes[f"{frame_name}.jpg"]
            return {"id": image_ids[frame_name], "file_name": frame_name, "width": int(width), "height": int(height)}

        def annotation_entry(index, box):
            entry = {"id": index + 1, "image_id": image_ids[Path(box["image"]).stem],
                     "category_id": sanpo_class_to_category_id[box["class"]],
                     "bbox": [float(box["x1"]), float(box["y1"]), float(box["x2"] - box["x1"]),
                              float(box["y2"] - box["y1"])],
                     "area": float((box["x2"] - box["x1"]) * (box["y2"] - box["y1"])), "iscrowd": 0,
                     "distance_m": box["distance_m"]}  # extra field (ignored by pycocotools)
            if task == "segment":
                entry["segmentation"] = box["segmentation_compared"]  # mask at the comparison size
                entry["area"] = float(box["area"])  # mask pixel count, full resolution
            return entry

        # COCO format = one dict with 3 lists: images, annotations (ground-truth objects), categories (classes)
        ground_truth_coco = {
            # One entry per frame: a number (id), its name and size. Frames with no objects are kept too
            # (a prediction on them counts as a false positive)
            "images": [image_size_entry(frame_name, width, height)
                       for frame_name, width, height in zip(frame_names, frames["width"], frames["height"])],
            # One entry per ground-truth object:
            #   image_id     = which frame it is in
            #   category_id  = its evaluation class number (e.g. pedestrian -> 1 person)
            #   bbox         = [x, y, width, height] in pixels (COCO uses top-left corner + size, not x1, y1, x2, y2)
            #   area         = box area (boxes) or mask pixel count (masks), full-resolution pixels
            #                  (COCO uses it to split objects into small / medium / large)
            #   segmentation = (masks only) the object's mask as COCO RLE
            #   iscrowd      = 0: a normal single object
            # to_dict("records"): the column "class" is a Python keyword, so attribute access cannot be used
            "annotations": [annotation_entry(index, box) for index, box in enumerate(ground_truth.to_dict("records"))],
            # One entry per evaluation class: its number and name (e.g. {"id": 2, "name": "vehicle"})
            "categories": [{"id": category_ids_by_name[name], "name": name} for name in evaluation_class_names],
        }
        print(f"{len(frame_names):,} frames | {len(ground_truth_coco['annotations']):,} ground-truth boxes "
              f"in evaluation classes", flush=True)
        # Per-class ground truth of this split: boxes, and frames that contain the class
        frames_without_boxes = len(frame_names) - ground_truth["image"].nunique()
        print(f"Frames with no ground-truth box in the evaluation classes: {frames_without_boxes:,}", flush=True)
        print(f"  {'class':<24}{'GT boxes':>10}{'frames':>9}", flush=True)
        for evaluation_name, (sanpo_names, _) in evaluation_classes.items():
            class_boxes = ground_truth[ground_truth["class"].isin(sanpo_names)]
            print(f"  {evaluation_name:<24}{len(class_boxes):>10,}{class_boxes['image'].nunique():>9,}", flush=True)

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
            batch_starts = list(range(saved["frames_done"], len(frame_paths), batch_size))

            # Speed: reading + resizing the 2208x1242 JPGs on the CPU is the slow part, not the GPU.
            # So frames are read and resized in loader_threads parallel threads, and the NEXT batch is
            # prepared while the GPU works on the current one (prefetch).
            loader = ThreadPoolExecutor(loader_threads)

            def submit_batch(batch_index):
                batch_paths = frame_paths[batch_starts[batch_index]:batch_starts[batch_index] + batch_size]
                return batch_paths, [loader.submit(load_and_resize_frame, path, image_size) for path in batch_paths]

            next_batch = submit_batch(0) if batch_starts else None

            # 4c.5: loop over the frames in batches of batch_size, starting after the frames already done
            for batch_index, batch_start in enumerate(batch_starts):
                batch_paths, loading = next_batch
                loaded = [future.result() for future in loading]  # (resized image, x scale, y scale) per frame
                if batch_index + 1 < len(batch_starts):
                    next_batch = submit_batch(batch_index + 1)  # start reading the next batch now
                # 4c.6: run the model on the whole batch (one GPU call); one result per frame.
                # Frames are already resized to the model size, so Ultralytics only pads them.
                # Segment: retina_masks=True returns each mask at the input frame's size (= comparison size).
                # Precision: quantize=16 = FP16, None = FP32 ("quantize" replaces the deprecated "half"
                # since recent ultralytics 8.4 releases).
                # model.predict takes: list of BGR images + settings; returns: one Results object per image
                #   (result.boxes.xyxy / .conf / .cls, result.masks.data for segmentation, result.speed)
                batch_results = model.predict([image for image, _, _ in loaded], imgsz=image_size,
                                              conf=confidence_threshold, iou=nms_iou_threshold,
                                              max_det=max_detections, quantize=16 if half_precision else None,
                                              verbose=False, retina_masks=(task == "segment"))
                # 4c.7: for each frame of the batch: store its timings, then its predicted boxes (and masks)
                for frame_path, (_, x_scale, y_scale), result in zip(batch_paths, loaded, batch_results):
                    image_id = image_ids[Path(frame_path).stem]
                    saved["speed"].append([result.speed["preprocess"], result.speed["inference"],
                                           result.speed["postprocess"]])
                    boxes = result.boxes.xyxy.tolist()
                    scores = result.boxes.conf.tolist()
                    model_classes = [int(model_class) for model_class in result.boxes.cls.tolist()]
                    # Keep only predictions whose class maps to an evaluation class
                    kept_indices = [index for index, model_class in enumerate(model_classes)
                                    if model_class in model_class_to_category_id]
                    # Segment: encode the kept masks (N x H x W, values 0/1) as COCO RLE, all at once
                    encoded_masks = []
                    if task == "segment" and kept_indices and result.masks is not None:
                        mask_array = result.masks.data[kept_indices].cpu().numpy().astype(np.uint8)
                        encoded_masks = mask_utils.encode(np.asfortranarray(mask_array.transpose(1, 2, 0)))
                        for encoded in encoded_masks:
                            encoded["counts"] = encoded["counts"].decode("ascii")  # bytes -> text (JSON)
                    for position, index in enumerate(kept_indices):
                        # Scale the box back to the original frame size and store it in COCO format
                        # (bbox = [x, y, width, height])
                        x1, y1, x2, y2 = boxes[index]
                        x1, x2 = x1 * x_scale, x2 * x_scale
                        y1, y2 = y1 * y_scale, y2 * y_scale
                        detection = {"image_id": image_id,
                                     "category_id": model_class_to_category_id[model_classes[index]],
                                     "bbox": [x1, y1, x2 - x1, y2 - y1], "score": scores[index]}
                        if encoded_masks:
                            detection["segmentation"] = encoded_masks[position]  # mask at the comparison size
                        saved["detections"].append(detection)
                # 4c.8: update progress and print speed, GPU use, elapsed time and ETA
                saved["frames_done"] = min(batch_start + batch_size, len(frame_paths))
                batches_done += 1
                elapsed = time.time() - start_time
                frames_per_second = (saved["frames_done"] - frames_done_at_start) / elapsed if elapsed else 0
                remaining_seconds = (len(frame_paths) - saved["frames_done"]) / frames_per_second if frames_per_second else 0
                print(f"[{split}] {saved['frames_done']:,}/{len(frame_paths):,} frames | {frames_per_second:.1f} frames/s | "
                      f"{gpu_usage()} | elapsed {format_duration(elapsed)} | ETA {format_duration(remaining_seconds)}",
                      flush=True)
                # 4c.9: every save_every_batches batches, save progress to the partial file (for resuming).
                # Written to a .tmp file first, then renamed, so a crash never leaves a broken file.
                if batches_done % save_every_batches == 0:
                    temporary_path = partial_path.with_suffix(".tmp")
                    temporary_path.write_text(json.dumps(saved))
                    os.replace(temporary_path, partial_path)
                    volume.commit()
            loader.shutdown()
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
        metrics = coco_evaluation(ground_truth_coco, saved["detections"], category_names_by_id,
                                  iou_type=iou_type, detection_area_scale=detection_area_scale)
        speeds = saved["speed"]  # per frame: [preprocess, inference, postprocess] in ms
        metrics["frames"] = len(frame_names)
        # Operating points: precision / recall / F1 / false alarms per frame at fixed confidences
        print(f"Operating points at confidence {operating_confidences} (IoU >= {match_iou}) ...", flush=True)
        metrics["operating_points"] = operating_point_metrics(
            ground_truth_coco, saved["detections"], category_names_by_id, operating_confidences, match_iou,
            iou_type=iou_type)
        # Near objects: only ground truth within N meters (if distances exist)
        if ground_truth["distance_m"].notna().any():
            near_scores = near_object_scores(
                ground_truth_coco, saved["detections"], category_names_by_id, iou_type=iou_type,
                detection_area_scale=detection_area_scale, distance_limits_meters=distance_limits_meters,
                operating_confidences=operating_confidences, match_iou=match_iou)
            for distance_label, near_metrics in near_scores.items():
                results.setdefault("near_objects", {}).setdefault(distance_label, {})[split] = near_metrics
                near_overall = near_metrics.get("overall") or {}
                print(f"  within {distance_label} m: {near_metrics['ground_truth_boxes']:,} GT | "
                      f"AP50 {near_overall.get('AP50')} | AP50-95 {near_overall.get('AP50-95')}", flush=True)
        for label, operating_point in metrics["operating_points"]["by_confidence"].items():
            overall_point = operating_point["overall"]
            print(f"  confidence {label:<16} precision {overall_point['precision']} | recall {overall_point['recall']} | "
                  f"F1 {overall_point['f1']} | false alarms/frame {overall_point['false_alarms_per_frame']}", flush=True)
        metrics["speed_ms_per_frame"] = {
            "preprocess": round(sum(speed[0] for speed in speeds) / len(speeds), 2),
            "inference": round(sum(speed[1] for speed in speeds) / len(speeds), 2),
            "postprocess": round(sum(speed[2] for speed in speeds) / len(speeds), 2),
        } if speeds else None
        metrics["gpu_memory_peak_gb"] = gpu_memory_peak_gb()
        print(f"GPU memory peak so far: {metrics['gpu_memory_peak_gb']} GB", flush=True)
        results["splits"][split] = metrics

        # Step 4e: example images (ground truth left, prediction right) for the last split (usually test),
        # drawn from the saved predictions (no extra model run). Saved locally by write_results().
        if split == splits[-1]:
            print(f"Drawing example images from {split} ...", flush=True)
            sanpo_to_evaluation_name = {sanpo_name: evaluation_name
                                        for evaluation_name, (sanpo_names, _) in evaluation_classes.items()
                                        for sanpo_name in sanpo_names}
            results["example_images_jpeg"] = make_example_images(
                frames=frames, dataset_root=dataset_root, ground_truth=ground_truth,
                sanpo_to_evaluation_name=sanpo_to_evaluation_name, detections=saved["detections"],
                image_ids=image_ids, category_names_by_id=category_names_by_id,
                evaluation_class_names=evaluation_class_names, title=f"{results['run']} | {split}")
            print(f"Example images: {len(results['example_images_jpeg'])} files", flush=True)
    return results


def near_object_scores(ground_truth_coco: dict, detections: list[dict], category_names_by_id: dict[int, str], *,
                       iou_type: str, detection_area_scale: float, distance_limits_meters,
                       operating_confidences: list[float], match_iou: float) -> dict:
    """Scores that count only NEAR ground truth: for each limit N, objects within N meters.

    Steps (per distance limit):
        1. Split ground truth into near (distance <= N) and far (farther, or no depth).
        2. Ignore predictions that match a FAR object (same class, IoU >= match_iou): they found a
           real object, just one we do not score here, so they are neither correct nor false alarms.
        3. Score the rest against the near ground truth only (COCO metrics + operating points).
    Returns {"5": metrics, "10": metrics, ...}.
    """
    from collections import defaultdict

    shape_key = "segmentation" if iou_type == "segm" else "bbox"
    scores = {}
    for limit in distance_limits_meters:
        # Step 1: near / far ground truth (NaN distance = no depth = treated as far)
        near_annotations, far_shapes = [], defaultdict(list)
        for annotation in ground_truth_coco["annotations"]:
            distance = annotation.get("distance_m")
            if distance is not None and distance == distance and distance <= limit:  # distance == distance: not NaN
                near_annotations.append(annotation)
            else:
                far_shapes[(annotation["image_id"], annotation["category_id"])].append(annotation[shape_key])

        # Step 2: drop predictions that match a far object (grouped per frame and class)
        grouped_detections = defaultdict(list)
        for detection in detections:
            grouped_detections[(detection["image_id"], detection["category_id"])].append(detection)
        kept = []
        for key, group in grouped_detections.items():
            if key not in far_shapes:
                kept += group
                continue
            overlaps = iou_matrix([detection[shape_key] for detection in group], far_shapes[key], iou_type)
            kept += [detection for detection, best in zip(group, overlaps.max(axis=1)) if best < match_iou]

        # Step 3: score against near ground truth only
        near_coco = {**ground_truth_coco, "annotations": near_annotations}
        # coco_evaluation takes: ground truth (COCO dict), predictions, class names by id, "bbox"/"segm", area scale
        #   returns: {"overall": {"AP50", "AP50-95", "AP75", "AR100", AP by size...}, "per_class": {class: {...}}}
        metrics = coco_evaluation(near_coco, kept, category_names_by_id, iou_type=iou_type,
                                  detection_area_scale=detection_area_scale)
        # operating_point_metrics takes: ground truth, predictions, class names, confidences, match IoU, "bbox"/"segm"
        #   returns: {"match_iou", "frames", "by_confidence": {"0.25": {"overall": {precision, recall, f1,
        #            false_alarms_per_frame, ...}, "per_class": {...}}, ..., "all_predictions": {...}}}
        metrics["operating_points"] = operating_point_metrics(near_coco, kept, category_names_by_id,
                                                              operating_confidences, match_iou, iou_type=iou_type)
        metrics["ground_truth_boxes"] = len(near_annotations)
        metrics["predictions_ignored_far"] = len(detections) - len(kept)
        scores[str(limit)] = metrics
    return scores


def make_example_images(*, frames, dataset_root: Path, ground_truth, sanpo_to_evaluation_name: dict[str, str],
                        detections: list[dict], image_ids: dict[str, int], category_names_by_id: dict[int, str],
                        evaluation_class_names: list[str], title: str, files: int = 5, rows_per_file: int = 5,
                        confidence: float = 0.25, panel_width: int = 1100, seed: int = 7) -> dict[str, bytes]:
    """Draw example images: each file has rows_per_file rows, ground truth on the left, prediction on the right.

    Steps:
        1. Pick files x rows_per_file different frames, cycling through the evaluation classes
           from rarest to most common, so rare classes are shown too.
        2. Per frame: ground-truth boxes (left) and predictions with confidence >= confidence (right),
           colored by class, with the class name (and score for predictions).
        3. Stack the rows under a legend; encode each file as JPEG.
    Returns {file name: JPEG bytes}, e.g. {"examples_01.jpg": b"..."}.
    """
    import cv2
    import numpy as np

    # One distinct color per evaluation class (BGR)
    palette = [(60, 220, 60), (255, 160, 30), (230, 60, 230), (40, 210, 255), (0, 0, 255), (0, 200, 255),
               (34, 139, 34), (255, 255, 0), (0, 140, 0), (180, 105, 255), (0, 69, 255), (255, 255, 255),
               (203, 192, 255), (128, 0, 128), (0, 128, 255), (255, 0, 0)]
    colors = {name: palette[index % len(palette)] for index, name in enumerate(evaluation_class_names)}

    # Step 1: frames that contain each class; cycle classes from rarest to most common
    ground_truth = ground_truth.assign(evaluation_class=ground_truth["class"].map(sanpo_to_evaluation_name))
    frames_by_class = {name: list(ground_truth.loc[ground_truth["evaluation_class"] == name, "image"].unique())
                       for name in evaluation_class_names}
    classes_by_rarity = sorted((name for name in evaluation_class_names if frames_by_class[name]),
                               key=lambda name: len(frames_by_class[name]))
    random_generator = np.random.default_rng(seed)
    chosen = []  # (image name, class it was chosen for)
    total = files * rows_per_file
    while len(chosen) < total and classes_by_rarity:
        added = False
        for class_name in classes_by_rarity:
            candidates = [image for image in frames_by_class[class_name] if image not in {c[0] for c in chosen}]
            if candidates and len(chosen) < total:
                chosen.append((candidates[random_generator.integers(len(candidates))], class_name))
                added = True
        if not added:
            break

    # Predictions per frame (confidence >= confidence), boxes back to x1, y1, x2, y2
    image_names = {image_id: frame_name for frame_name, image_id in image_ids.items()}
    predictions_by_image = {}
    for detection in detections:
        if detection["score"] >= confidence:
            x, y, width, height = detection["bbox"]
            predictions_by_image.setdefault(image_names[detection["image_id"]], []).append(
                (x, y, x + width, y + height, category_names_by_id[detection["category_id"]], detection["score"],
                 detection.get("segmentation")))  # mask (COCO RLE) for segmentation runs, else None
    frame_path_by_name = {Path(path).stem: path for path in frames["frame"]}

    def draw(image, items, scale):
        """Draw each item: its mask (if any) as a semi-transparent color fill, then its box and label."""
        from pycocotools import mask as mask_utils

        canvas = image.copy()
        for x1, y1, x2, y2, name, score, segmentation in items:
            color = colors.get(name, (200, 200, 200))
            if segmentation is not None:
                # RLE -> 0/1 mask (comparison size) -> resized to the drawing size -> blend 45 % of the color
                mask = mask_utils.decode(segmentation)
                mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array(color)).astype(np.uint8)
            point1, point2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(canvas, point1, point2, color, 2)
            label = name if score is None else f"{name} {score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (point1[0], max(0, point1[1] - text_height - 6)),
                          (point1[0] + text_width + 4, point1[1]), color, -1)
            cv2.putText(canvas, label, (point1[0] + 2, point1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    def title_bar(text, width):
        bar = np.full((34, width, 3), 30, np.uint8)
        cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return bar

    # Step 2: one row per chosen frame
    rows = []
    for image_name, class_name in chosen:
        frame_name = Path(image_name).stem
        frame = cv2.imread(str(dataset_root / frame_path_by_name[frame_name]))
        scale = panel_width / frame.shape[1]
        small = cv2.resize(frame, (panel_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        truth = ground_truth[ground_truth["image"] == image_name]
        # Ground truth: box, class, no score, and its mask at the comparison size (segmentation runs only)
        truth_items = [(box["x1"], box["y1"], box["x2"], box["y2"], box["evaluation_class"], None,
                        box.get("segmentation_compared")) for box in truth.to_dict("records")]
        predicted_items = predictions_by_image.get(frame_name, [])
        left = np.vstack([title_bar(f"GROUND TRUTH ({len(truth_items)} boxes) | chosen for: {class_name} | "
                                    f"{frame_name[:32]}", panel_width), draw(small, truth_items, scale)])
        right = np.vstack([title_bar(f"PREDICTION ({len(predicted_items)} boxes, confidence >= {confidence})",
                                     panel_width), draw(small, predicted_items, scale)])
        rows.append(np.hstack([left, np.full((left.shape[0], 8, 3), 255, np.uint8), right]))
    if not rows:
        return {}

    # Step 3: legend + rows, one JPEG per group of rows_per_file rows
    width = rows[0].shape[1]
    legend_lines = (len(evaluation_class_names) + 7) // 8
    legend = np.full((40 + 30 * legend_lines, width, 3), 255, np.uint8)
    cv2.putText(legend, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for index, name in enumerate(evaluation_class_names):
        x, y = 10 + (index % 8) * (width // 8), 36 + (index // 8) * 30
        cv2.rectangle(legend, (x, y), (x + 20, y + 18), colors[name], -1)
        cv2.rectangle(legend, (x, y), (x + 20, y + 18), (0, 0, 0), 1)
        cv2.putText(legend, name[:26], (x + 26, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    separator = np.full((8, width, 3), 255, np.uint8)
    images = {}
    for file_index in range(0, len(rows), rows_per_file):
        parts = [legend]
        for row in rows[file_index:file_index + rows_per_file]:
            parts += [row, separator]
        ok, encoded = cv2.imencode(".jpg", np.vstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            images[f"examples_{file_index // rows_per_file + 1:02d}.jpg"] = encoded.tobytes()
    return images


# ----------------------------------------------------------------------------- local: save results

def write_results(results: dict, output_directory: Path) -> Path:
    """Save the results next to the evaluation script (runs on the local computer).

    Steps:
        0. <output_directory>/example_images/examples_XX.jpg: example images (if the run made them).
        1. <output_directory>/metrics.json: everything, machine-readable.
        2. <output_directory>/metrics.txt: all numbers, human-readable (overview sections, then
           settings, per-class tables and operating points).
        3. <output_directory>/summary.md: the short summary (a few tables, the main numbers only).
    """
    import json
    from datetime import datetime

    output_directory.mkdir(parents=True, exist_ok=True)
    # Step 0: example images (bytes, not stored in metrics.json)
    example_images = results.pop("example_images_jpeg", None) or {}
    if example_images:
        images_directory = output_directory / "example_images"
        images_directory.mkdir(parents=True, exist_ok=True)
        for file_name, jpeg_bytes in example_images.items():
            (images_directory / file_name).write_bytes(jpeg_bytes)
        print(f"Saved {len(example_images)} example images to {images_directory}", flush=True)
    results["finished_at"] = datetime.now().isoformat(timespec="seconds")
    # Step 1: machine-readable
    (output_directory / "metrics.json").write_text(json.dumps(results, indent=2))
    # Step 2: all numbers, readable
    (output_directory / "metrics.txt").write_text(build_summary_text(results), encoding="utf-8")
    # Step 3: short summary
    (output_directory / "summary.md").write_text(build_summary_markdown(results), encoding="utf-8")
    return output_directory


def build_summary_markdown(results: dict) -> str:
    """summary.md: the short summary, as a few markdown tables (all numbers are in metrics.txt).

    Sections:
        1. Run (model, GPU, classes, image size)
        2. Main scores per split: AP50 (main score), AP50-95, recall, and at confidence 0.25
        3. Near objects (if computed): AP50 and recall at 0.25 within 5 m / 10 m
        4. Per class AP50 on the last split (usually test)
    """
    def show(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    settings = results["settings"]
    splits = [split for split, metrics in results["splits"].items() if metrics.get("overall")]
    lines = [f"# Summary: {results['run']}", "",
             f"- **Evaluation:** {results['evaluation']}",
             f"- **GPU:** {results['gpu']}",
             f"- **Classes evaluated:** {len(settings['evaluation_classes'])}",
             f"- **Model input size (long side):** {settings.get('image_size')}",
             "- **Main score:** AP50 (higher is better, 0 to 1). All numbers: `metrics.txt`. "
             "Examples: `example_images/`.", ""]

    # 2. Main scores
    lines += ["## Main scores", "",
              "| Split | Frames | AP50 | AP50-95 | Recall (AR100) | Precision @0.25 | Recall @0.25 | False alarms/frame @0.25 |",
              "|---|---|---|---|---|---|---|---|"]
    for split in splits:
        metrics = results["splits"][split]
        overall = metrics["overall"]
        point = metrics.get("operating_points", {}).get("by_confidence", {}).get("0.25", {}).get("overall", {})
        lines.append(f"| {split} | {metrics['frames']:,} | **{show(overall['AP50'])}** | {show(overall['AP50-95'])} | "
                     f"{show(overall['AR100'])} | {show(point.get('precision'))} | {show(point.get('recall'))} | "
                     f"{show(point.get('false_alarms_per_frame'), 2)} |")
    lines.append("")

    # 3. Near objects
    near_objects = results.get("near_objects") or {}
    if near_objects:
        lines += ["## Near objects (farther objects ignored)", "",
                  "| Split | Within | GT boxes | AP50 | Recall @0.25 |", "|---|---|---|---|---|"]
        for distance_label, by_split in near_objects.items():
            for split, metrics in by_split.items():
                overall = metrics.get("overall") or {}
                point = (metrics.get("operating_points", {}).get("by_confidence", {}).get("0.25", {})
                         .get("overall", {}))
                lines.append(f"| {split} | {distance_label} m | {metrics.get('ground_truth_boxes', 0):,} | "
                             f"**{show(overall.get('AP50'))}** | {show(point.get('recall'))} |")
        lines.append("")

    # 4. Per class AP50 on the last split
    if splits:
        last_split = splits[-1]
        per_class = results["splits"][last_split]["per_class"]
        ranked = sorted(per_class.items(), key=lambda item: item[1].get("AP50") or 0, reverse=True)
        lines += [f"## Per class ({last_split}, sorted by AP50)", "", "| Class | GT boxes | AP50 |", "|---|---|---|"]
        for class_name, class_metrics in ranked:
            lines.append(f"| {class_name} | {class_metrics['ground_truth_boxes']:,} | {show(class_metrics.get('AP50'))} |")
        lines.append("")
    return "\n".join(lines)


def bordered_table(title: str, headers: list[str], rows: list[list], numeric_columns: set[int] | None = None,
                   group_column: int | None = None) -> list[str]:
    """Draw a table with box borders, e.g.

        1. MAIN SCORES
        ┌───────┬───────┐
        │ split │  AP50 │
        ├───────┼───────┤
        │ val   │ 0.425 │
        └───────┴───────┘

    numeric_columns: column indexes aligned to the right (numbers); others are aligned left.
    group_column: draw a separator line whenever the value in this column changes (e.g. per split).
    Returns the lines (title first).
    """
    numeric_columns = numeric_columns or set()
    cells = [[str(value) for value in row] for row in rows]
    widths = [max([len(header)] + [len(row[index]) for row in cells]) for index, header in enumerate(headers)]

    def line(left, middle, right):
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def row_line(values):
        parts = [f" {value.rjust(width) if index in numeric_columns else value.ljust(width)} "
                 for index, (value, width) in enumerate(zip(values, widths))]
        return "│" + "│".join(parts) + "│"

    lines = [title, line("┌", "┬", "┐"), row_line(headers), line("├", "┼", "┤")]
    for row_index, row in enumerate(cells):
        if group_column is not None and row_index > 0 and row[group_column] != cells[row_index - 1][group_column]:
            lines.append(line("├", "┼", "┤"))  # separator between groups (e.g. val | test)
        lines.append(row_line(row))
    lines.append(line("└", "┴", "┘"))
    return lines


def build_summary_lines(results: dict) -> list[str]:
    """The overview at the top of metrics.txt (also printed at the end of a run), as bordered tables.

    Sections:
        0. Run information (model run, evaluation, GPU, classes, image size, splits)
        1. Main scores per split (COCO, over all confidence thresholds): AP50, AP50-95, AP75, recall
        2. Fixed confidence (what the user would get): precision, recall, F1, false alarms per frame
        3. AP50 and recall per class, for every split (sorted by AP50 of the last split)
        4. Speed and GPU memory per split
        5. Near objects (if computed)
        6. How to read the numbers
    """
    def show(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    settings = results["settings"]
    splits = [split for split, metrics in results["splits"].items() if metrics.get("overall")]
    lines = ["=" * 100, "SUMMARY", "=" * 100, ""]

    # 0. Run information
    lines += bordered_table("0. RUN", ["item", "value"], [
        ["model run", results["run"]],
        ["evaluation", results["evaluation"]],
        ["GPU", results["gpu"]],
        ["task", settings.get("task", "box")],
        ["classes evaluated", len(settings["evaluation_classes"])],
        ["model input size (long side)", settings.get("image_size")],
        ["splits", ", ".join(f"{split} ({results['splits'][split]['frames']:,} frames)" for split in results["splits"])],
    ]) + [""]

    # 1. Main scores (COCO)
    lines += bordered_table(
        "1. MAIN SCORES (COCO, over all confidence thresholds; AP50 = main score)",
        ["split", "AP50", "AP50-95", "AP75", "recall (AR100)"],
        [[split, show(results["splits"][split]["overall"]["AP50"]), show(results["splits"][split]["overall"]["AP50-95"]),
          show(results["splits"][split]["overall"]["AP75"]), show(results["splits"][split]["overall"]["AR100"])]
         for split in splits],
        numeric_columns={1, 2, 3, 4}) + [""]

    # 2. Fixed confidence (operating points)
    rows = []
    for split in splits:
        by_confidence = results["splits"][split].get("operating_points", {}).get("by_confidence", {})
        for label, operating_point in by_confidence.items():
            point = operating_point["overall"]
            rows.append([split, "all (max recall)" if label == "all_predictions" else label, show(point["precision"]),
                         show(point["recall"]), show(point["f1"]), show(point["false_alarms_per_frame"], 2)])
    lines += bordered_table(
        f"2. AT A FIXED CONFIDENCE (what the user would get; correct if IoU >= {settings.get('match_iou', 0.5)})",
        ["split", "confidence", "precision", "recall", "F1", "false alarms/frame"], rows,
        numeric_columns={2, 3, 4, 5}, group_column=0) + [""]

    # 3. Per class: AP50 for every split, and recall at the first fixed confidence
    first_label = None
    if splits:
        labels = list(results["splits"][splits[-1]].get("operating_points", {}).get("by_confidence", {}))
        first_label = next((label for label in labels if label != "all_predictions"), None)
    headers = ["class"] + [f"{split} AP50" for split in splits]
    if first_label:
        headers += [f"{split} recall@{first_label}" for split in splits]
    class_names = list(results["splits"][splits[-1]]["per_class"]) if splits else []
    class_names.sort(key=lambda name: results["splits"][splits[-1]]["per_class"][name].get("AP50") or 0, reverse=True)
    rows = []
    for class_name in class_names:
        row = [class_name] + [show(results["splits"][split]["per_class"].get(class_name, {}).get("AP50")) for split in splits]
        if first_label:
            for split in splits:
                point = (results["splits"][split].get("operating_points", {}).get("by_confidence", {})
                         .get(first_label, {}).get("per_class", {}).get(class_name, {}))
                row.append(show(point.get("recall")))
        rows.append(row)
    lines += bordered_table(f"3. PER CLASS (sorted by {splits[-1] if splits else ''} AP50)", headers, rows,
                            numeric_columns=set(range(1, len(headers)))) + [""]

    # 4. Speed and GPU memory
    rows = []
    for split in splits:
        metrics = results["splits"][split]
        speed = metrics.get("speed_ms_per_frame") or {}
        rows.append([split, show(speed.get("inference"), 2), show(speed.get("preprocess"), 2),
                     show(speed.get("postprocess"), 2), show(metrics.get("gpu_memory_peak_gb"), 2)])
    lines += bordered_table("4. SPEED AND GPU (milliseconds per frame, model in a batch)",
                            ["split", "model", "preprocess", "postprocess", "GPU memory peak (GB)"], rows,
                            numeric_columns={1, 2, 3, 4}) + [""]

    # 5. Near objects (only if distance-aware scores exist)
    near_objects = results.get("near_objects") or {}
    if near_objects:
        rows = []
        for split in splits:
            for distance_label, by_split in near_objects.items():
                metrics = by_split.get(split)
                if not metrics:
                    continue
                overall = metrics.get("overall") or {}
                point = (metrics.get("operating_points", {}).get("by_confidence", {}).get("0.25", {})
                         .get("overall", {}))
                rows.append([split, f"{distance_label} m", f"{metrics.get('ground_truth_boxes', 0):,}",
                             show(overall.get("AP50")), show(overall.get("AP50-95")), show(overall.get("AR100")),
                             show(point.get("precision")), show(point.get("recall"))])
        lines += bordered_table(
            "5. NEAR OBJECTS (only ground truth within N meters counts; predictions of farther objects are ignored)",
            ["split", "within", "GT objects", "AP50", "AP50-95", "recall (AR100)", "precision@0.25", "recall@0.25"],
            rows, numeric_columns={1, 2, 3, 4, 5, 6, 7}, group_column=0) + [""]

    # 6. How to read
    lines += bordered_table("6. HOW TO READ", ["score", "meaning"], [
        ["AP50", "main score: how well objects are found (overlap >= 0.5), over all confidence thresholds"],
        ["AP50-95", "standard COCO score: also rewards tight boxes / masks (overlap 0.50 to 0.95)"],
        ["recall", "share of real objects that were found"],
        ["precision", "share of predictions that were correct"],
        ["false alarms/frame", "wrong predictions per frame at that confidence"],
        ["all (max recall)", "every prediction kept: the highest recall the model can reach"],
    ])
    lines += ["", "=" * 100, "DETAILS", "=" * 100, ""]
    return lines


def terminal_summary(results: dict) -> str:
    """Overview for the terminal at the end of a run: the aligned plain-text tables (sections 0 to 6 of
    metrics.txt), without the long DETAILS part. summary.md is meant for the editor, not the terminal."""
    lines = build_summary_lines(results)
    if "DETAILS" in lines:
        lines = lines[:lines.index("DETAILS") - 1]  # cut before the "=====" line above DETAILS
    return "\n".join(lines)


def build_summary_text(results: dict) -> str:
    """Full metrics.txt: the overview sections first, then all details (settings, per-class tables, operating points)."""
    settings = results["settings"]
    lines = build_summary_lines(results)
    lines += [f"Evaluation: {results['evaluation']}  |  run: {results['run']}  |  GPU: {results['gpu']}",
             f"Finished: {results['finished_at']}",
             "Settings: " + ", ".join(f"{name}={value}" for name, value in settings.items()
                                      if name != "evaluation_classes"),
             "Evaluation classes (ground truth <- what the model predicts):"]
    for evaluation_name, mapping in settings["evaluation_classes"].items():
        parts = [f"{key}: {', '.join(value) if isinstance(value, list) else value}" for key, value in mapping.items()]
        lines.append(f"  {evaluation_name:<24} " + " | ".join(parts))

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
                         f"inference {speed['inference']} | postprocess {speed['postprocess']} | "
                         f"GPU memory peak {metrics.get('gpu_memory_peak_gb')} GB")
        lines.append(f"{'class':<24}{'GT boxes':>10}{'preds':>9}{'AP50-95':>9}{'AP50':>8}{'AP75':>8}{'AR100':>8}"
                     f"{'AP_s':>7}{'AP_m':>7}{'AP_l':>7}")
        for class_name, class_metrics in metrics["per_class"].items():
            lines.append(
                f"{class_name:<24}{class_metrics['ground_truth_boxes']:>10,}{class_metrics['predictions']:>9,}"
                f"{format_metric(class_metrics['AP50-95']):>9}{format_metric(class_metrics['AP50']):>8}"
                f"{format_metric(class_metrics['AP75']):>8}{format_metric(class_metrics['AR100']):>8}"
                f"{format_metric(class_metrics['AP_small']):>7}{format_metric(class_metrics['AP_medium']):>7}"
                f"{format_metric(class_metrics['AP_large']):>7}")
    # Operating points (one fixed confidence threshold, as the phone would use)
    for split, metrics in results["splits"].items():
        operating_points = metrics.get("operating_points")
        if not operating_points:
            continue
        lines += ["", f"=== {split}: operating points (prediction correct if IoU >= {operating_points['match_iou']}) ===",
                  "all_predictions = every prediction kept: the highest recall the model can reach"]
        for label, operating_point in operating_points["by_confidence"].items():
            lines += ["", f"confidence {label}",
                      f"{'class':<24}{'GT':>8}{'TP':>8}{'FP':>8}{'missed':>8}{'prec':>7}{'recall':>8}{'F1':>7}{'FA/frame':>10}"]
            rows = list(operating_point["per_class"].items()) + [("OVERALL", operating_point["overall"])]
            for class_name, point in rows:
                lines.append(f"{class_name:<24}{point['ground_truth']:>8,}{point['true_positives']:>8,}"
                             f"{point['false_positives']:>8,}{point['missed']:>8,}{format_metric(point['precision']):>7}"
                             f"{format_metric(point['recall']):>8}{format_metric(point['f1']):>7}"
                             f"{format_metric(point['false_alarms_per_frame']):>10}")
    return "\n".join(lines) + "\n"

# =====================================================================================================================
# SEMANTIC EVALUATIONS (one class per pixel, e.g. yolo26n-sem)
# =====================================================================================================================
# Code shared by the SEMANTIC evaluation scripts (one class per pixel, e.g. yolo26n-sem).
#
# Semantic models have no objects and no confidence per object, so there is no AP here. The scores are:
#
#     IoU per class    pixels where prediction and label both say the class / pixels where either says it
#     mIoU             IoU averaged over the scored classes (main score)
#     pixel accuracy   share of all scored pixels that got the right class
#     precision        of the pixels predicted as the class, the share that really are the class
#     recall           of the real pixels of the class, the share predicted as the class
#
# How a pixel is scored:
#     ground truth  = SANPO class of the pixel (red channel of the raw mask)
#     prediction    = model class of the pixel, translated to an evaluation class by the mapping of the script
#     ignored       = pixels whose SANPO class is not in the mapping (and "unlabeled"): not counted at all
#     "no match"    = model classes that are not in the mapping: counted as wrong on scored pixels
#
# What is counted while predicting: one table per scope ("all" pixels, pixels within 5 m, within 10 m),
#     table[SANPO class id][model class id] = number of pixels.
# All scores are computed from these tables, so a changed mapping can be re-scored without predicting again.
#
# Functions:
#     load_frame_labels_depth()      one frame: resized image, SANPO class per pixel, depth per pixel
#     semantic_scores()              count table -> IoU / precision / recall per class, mIoU, pixel accuracy
#     evaluate_semantic_splits()     for every split: predict all frames (resumable), count, score
#     make_semantic_example_images() frame | ground truth | prediction
#     write_semantic_results()       saves results/<run>/metrics.json, metrics.txt, summary.md, example_images/
#     semantic_terminal_summary()    the tables printed at the end of a run

# ----------------------------------------------------------------------------- loading (runs in loader threads)

def load_frame_labels_depth(frame_path: str, mask_path: str, depth_path: str | None, image_size: int):
    """Read one frame with its labels and depth, all resized to the model input size.

    Steps:
        1. Frame: resized so its long side = image_size (same as the other evaluations).
        2. Labels: raw SANPO mask, red channel = class id per pixel; resized with nearest neighbor
           (never blends two class ids into a third one).
        3. Depth (if the file exists): gzip of float16 values, the first two are (height, width),
           the rest is the depth in meters; resized with nearest neighbor.
    Returns (resized BGR image, class id per pixel (uint8), depth in meters per pixel (float32) or None).
    """
    import gzip

    import cv2
    import numpy as np

    # Step 1: load_and_resize_frame takes: frame path, image size
    #   returns: (resized BGR image, x scale, y scale)
    image, _, _ = load_and_resize_frame(frame_path, image_size)
    height, width = image.shape[:2]
    # Step 2: OpenCV reads the PNG as BGR, so the red channel is index 2
    class_ids = cv2.imread(mask_path)[:, :, 2]
    class_ids = cv2.resize(class_ids, (width, height), interpolation=cv2.INTER_NEAREST)
    # Step 3
    depth = None
    if depth_path is not None and Path(depth_path).exists():
        values = np.frombuffer(gzip.decompress(Path(depth_path).read_bytes()), dtype=np.float16)
        depth_height, depth_width = int(values[0]), int(values[1])
        depth = values[2:].reshape(depth_height, depth_width).astype(np.float32)
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return image, class_ids, depth


# ----------------------------------------------------------------------------- scoring

def semantic_scores(counts, evaluation_classes: dict, sanpo_class_ids: dict[str, int],
                    model_class_ids: dict[str, int], top_predictions: int = 3) -> dict:
    """Scores from one count table.

    counts: table[SANPO class id][model class id] = number of pixels (list of lists or array).
    evaluation_classes: {evaluation class: (SANPO class names, model class names)}.
    sanpo_class_ids / model_class_ids: class name -> id (row / column of the table).

    Steps:
        1. Fold the table to evaluation classes: rows = ground-truth evaluation class,
           columns = predicted evaluation class, plus one last column "no match".
           Rows of ignored SANPO classes are dropped (their pixels are not scored).
        2. Per class: correct = diagonal; missed = rest of the row; wrong = rest of the column.
           IoU = correct / (correct + missed + wrong), precision = correct / (correct + wrong),
           recall = correct / (correct + missed).
        3. mIoU = mean IoU over the classes that have ground-truth pixels; pixel accuracy = all correct / all pixels.
        4. Per class: which model classes its pixels were given most often (to understand mistakes).
    Returns {"overall": {"mIoU", "pixel_accuracy", "classes_scored", "pixels_scored"},
             "per_class": {class: {"IoU", "precision", "recall", "ground_truth_pixel_share", "top_predictions"}}}.
    """
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64)
    names = list(evaluation_classes)
    model_class_names = {class_id: name for name, class_id in model_class_ids.items()}

    # Step 1: fold rows (SANPO ids -> evaluation class), then columns (model ids -> evaluation class / no match)
    rows = np.zeros((len(names), counts.shape[1]))
    for index, name in enumerate(names):
        for sanpo_name in evaluation_classes[name][0]:
            rows[index] += counts[sanpo_class_ids[sanpo_name]]
    folded = np.zeros((len(names), len(names) + 1))
    matched_columns = set()
    for index, name in enumerate(names):
        for model_name in evaluation_classes[name][1]:
            folded[:, index] += rows[:, model_class_ids[model_name]]
            matched_columns.add(model_class_ids[model_name])
    unmatched_columns = [column for column in range(counts.shape[1]) if column not in matched_columns]
    folded[:, len(names)] = rows[:, unmatched_columns].sum(axis=1)

    # Steps 2-4
    total_pixels = folded.sum()
    per_class, ious = {}, []
    for index, name in enumerate(names):
        correct = folded[index, index]
        ground_truth_pixels = folded[index].sum()
        predicted_pixels = folded[:, index].sum()
        union = ground_truth_pixels + predicted_pixels - correct
        entry = {
            "IoU": round(correct / union, 4) if ground_truth_pixels > 0 and union > 0 else None,
            "precision": round(correct / predicted_pixels, 4) if predicted_pixels > 0 else None,
            "recall": round(correct / ground_truth_pixels, 4) if ground_truth_pixels > 0 else None,
            "ground_truth_pixel_share": round(ground_truth_pixels / total_pixels, 5) if total_pixels > 0 else None,
            "top_predictions": [],
        }
        if ground_truth_pixels > 0:
            ious.append(correct / union)
            for column in np.argsort(rows[index])[::-1][:top_predictions]:
                if rows[index, column] > 0:
                    entry["top_predictions"].append({"model_class": model_class_names[int(column)],
                                                     "share": round(rows[index, column] / ground_truth_pixels, 4)})
        per_class[name] = entry
    return {
        "overall": {
            "mIoU": round(float(np.mean(ious)), 4) if ious else None,
            "pixel_accuracy": round(float(np.trace(folded[:, :len(names)]) / total_pixels), 4) if total_pixels > 0 else None,
            "classes_scored": len(ious),
            "pixels_scored": int(total_pixels),
        },
        "per_class": per_class,
    }


# ----------------------------------------------------------------------------- semantic: predict + count + score

def evaluate_semantic_splits(*, model, volume, results: dict, dataset_root: Path, work_directory: Path,
                             splits: list[str], evaluation_classes: dict, image_size: int, half_precision: bool,
                             batch_size: int, save_every_batches: int, loader_threads: int = 32,
                             distance_limits_meters: tuple = (5, 10), max_frames: int = 0) -> dict:
    """For each split: predict every frame, count pixels per (SANPO class, model class), then score.

    evaluation_classes: {evaluation class: (SANPO class names, model class names)}.
    max_frames: 0 = all frames; N = only the first N frames of each split (quick test).

    Steps (per split):
        a. frames of the split (index CSVs), with the path of their mask and depth file,
        b. predict in batches; per frame add its pixels to the count tables (GPU);
           progress is saved regularly so a stopped run resumes,
        c. scores from the count tables: all pixels, and pixels within N meters,
        d. example images (last split).
    Fills results["splits"][split] and returns results.
    """
    import json
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import torch

    # Class name -> id: SANPO (rows of the count table) and model (columns)
    sanpo_class_ids = json.loads((dataset_root / "sanpo_meta" / "labelmap.json").read_text())
    model_class_ids = {name: class_id for class_id, name in model.names.items()}
    for evaluation_name, (sanpo_names, model_names) in evaluation_classes.items():
        for sanpo_name in sanpo_names:
            assert sanpo_name in sanpo_class_ids, f"'{sanpo_name}' is not a SANPO class"
        for model_name in model_names:
            assert model_name in model_class_ids, f"'{model_name}' is not a class of this model"
    results["settings"]["sanpo_class_ids"] = sanpo_class_ids
    results["settings"]["model_class_ids"] = model_class_ids
    row_count, column_count = max(sanpo_class_ids.values()) + 1, max(model_class_ids.values()) + 1
    scopes = ["all"] + [str(limit) for limit in distance_limits_meters]  # "all" pixels, then within N meters

    for split in splits:
        print(f"\n===== split: {split} =====", flush=True)
        # Step a: frames of the split
        index_files = sorted((dataset_root / "index").glob("*.csv"))
        all_frames = pd.concat([pd.read_csv(index_file) for index_file in index_files], ignore_index=True)
        frames = all_frames[all_frames["split"] == split].reset_index(drop=True)
        if max_frames:
            frames = frames.head(max_frames)
        frame_paths = [str(dataset_root / path) for path in frames["frame"]]
        mask_paths = [str(dataset_root / path) for path in frames["mask"]]
        depth_paths = [str(dataset_root / "depth" / split / f"{Path(path).stem}.float16.gz") for path in frames["frame"]]
        print(f"{len(frame_paths):,} frames", flush=True)

        # Step b: predict and count (reused if finished, resumed if partial)
        final_path = work_directory / f"counts_{split}.json"
        partial_path = work_directory / f"counts_{split}.partial.json"
        if final_path.exists():
            saved = json.loads(final_path.read_text())
            print(f"Already done ({saved['frames_done']:,} frames), reusing", flush=True)
        else:
            # saved = {"frames_done", "counts": {scope: flat list}, "speed_sums": [preprocess, inference,
            #          postprocess] in ms (summed over frames), "frames_with_depth"}
            saved = (json.loads(partial_path.read_text()) if partial_path.exists() else
                     {"frames_done": 0, "counts": {scope: [0] * (row_count * column_count) for scope in scopes},
                      "speed_sums": [0.0, 0.0, 0.0], "frames_with_depth": 0})
            if saved["frames_done"]:
                print(f"Resuming from frame {saved['frames_done']:,}", flush=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            counts = {scope: torch.tensor(saved["counts"][scope], dtype=torch.int64, device=device) for scope in scopes}
            start_time, batches_done, frames_done_at_start = time.time(), 0, saved["frames_done"]
            batch_starts = list(range(saved["frames_done"], len(frame_paths), batch_size))

            def save_progress(path):
                # Written to a .tmp file first, then renamed, so a crash never leaves a broken file
                saved["counts"] = {scope: counts[scope].tolist() for scope in scopes}
                temporary_path = path.with_suffix(".tmp")
                temporary_path.write_text(json.dumps(saved))
                os.replace(temporary_path, path)
                volume.commit()

            # Frames, masks and depth are read and resized in parallel threads, and the NEXT batch is
            # prepared while the GPU works on the current one (prefetch)
            loader = ThreadPoolExecutor(loader_threads)

            def submit_batch(batch_index):
                first = batch_starts[batch_index]
                # load_frame_labels_depth takes: frame path, mask path, depth path, image size
                #   returns: (resized BGR image, SANPO class id per pixel, depth in meters per pixel or None)
                return [loader.submit(load_frame_labels_depth, frame_paths[index], mask_paths[index],
                                      depth_paths[index], image_size)
                        for index in range(first, min(first + batch_size, len(frame_paths)))]

            next_batch = submit_batch(0) if batch_starts else None
            for batch_index, batch_start in enumerate(batch_starts):
                loaded = [future.result() for future in next_batch]
                if batch_index + 1 < len(batch_starts):
                    next_batch = submit_batch(batch_index + 1)  # start reading the next batch now
                # model.predict takes: list of BGR images + settings; returns: one Results object per image
                #   (result.semantic_mask.data = model class id per pixel, at the input image size; result.speed)
                batch_results = model.predict([image for image, _, _ in loaded], imgsz=image_size,
                                              quantize=16 if half_precision else None, verbose=False)
                for (image, class_ids, depth), result in zip(loaded, batch_results):
                    prediction = result.semantic_mask.data.to(device=device, dtype=torch.int64)
                    truth = torch.from_numpy(class_ids).to(device=device, dtype=torch.int64)
                    assert prediction.shape == truth.shape, f"prediction {prediction.shape} vs labels {truth.shape}"
                    # One number per pixel that says both classes: SANPO id * number of model classes + model id.
                    # Counting how often each number appears gives the count table (flattened).
                    pair = truth * column_count + prediction
                    counts["all"] += torch.bincount(pair.flatten(), minlength=row_count * column_count)
                    if depth is not None:
                        depth_meters = torch.from_numpy(depth).to(device)
                        for limit in distance_limits_meters:
                            near = (depth_meters > 0) & (depth_meters <= limit)  # no depth (0, NaN, inf) = not near
                            counts[str(limit)] += torch.bincount(pair[near], minlength=row_count * column_count)
                        saved["frames_with_depth"] += 1
                    for position, key in enumerate(["preprocess", "inference", "postprocess"]):
                        saved["speed_sums"][position] += result.speed[key]
                # Progress: speed, GPU use, elapsed time and ETA
                saved["frames_done"] = min(batch_start + batch_size, len(frame_paths))
                batches_done += 1
                elapsed = time.time() - start_time
                frames_per_second = (saved["frames_done"] - frames_done_at_start) / elapsed if elapsed else 0
                remaining_seconds = (len(frame_paths) - saved["frames_done"]) / frames_per_second if frames_per_second else 0
                print(f"[{split}] {saved['frames_done']:,}/{len(frame_paths):,} frames | {frames_per_second:.1f} frames/s | "
                      f"{gpu_usage()} | elapsed {format_duration(elapsed)} | ETA {format_duration(remaining_seconds)}",
                      flush=True)
                if batches_done % save_every_batches == 0:
                    save_progress(partial_path)
            loader.shutdown()
            save_progress(final_path)
            if partial_path.exists():
                partial_path.unlink()
                volume.commit()

        # Step c: scores from the count tables
        tables = {scope: [saved["counts"][scope][row * column_count:(row + 1) * column_count]
                          for row in range(row_count)] for scope in scopes}
        # semantic_scores takes: count table, evaluation classes, SANPO class ids, model class ids
        #   returns: {"overall": {"mIoU", "pixel_accuracy", "classes_scored", "pixels_scored"},
        #             "per_class": {class: {"IoU", "precision", "recall", "ground_truth_pixel_share", "top_predictions"}}}
        metrics = semantic_scores(tables["all"], evaluation_classes, sanpo_class_ids, model_class_ids)
        metrics["frames"] = saved["frames_done"]
        metrics["frames_with_depth"] = saved["frames_with_depth"]
        metrics["near"] = {}
        if saved["frames_with_depth"]:
            for limit in distance_limits_meters:
                metrics["near"][str(limit)] = semantic_scores(tables[str(limit)], evaluation_classes,
                                                              sanpo_class_ids, model_class_ids)
        else:
            print("No depth files for this split: near-pixel scores skipped", flush=True)
        frame_count = max(1, saved["frames_done"])
        metrics["speed_ms_per_frame"] = {key: round(saved["speed_sums"][position] / frame_count, 2)
                                         for position, key in enumerate(["preprocess", "inference", "postprocess"])}
        metrics["gpu_memory_peak_gb"] = gpu_memory_peak_gb()
        metrics["counts"] = tables  # raw count tables (to re-score with another mapping without predicting again)
        results["splits"][split] = metrics
        print(f"[{split}] mIoU {metrics['overall']['mIoU']} | pixel accuracy {metrics['overall']['pixel_accuracy']} | "
              f"classes scored {metrics['overall']['classes_scored']}", flush=True)
        for limit, near_metrics in metrics["near"].items():
            print(f"[{split}] within {limit} m: mIoU {near_metrics['overall']['mIoU']} | "
                  f"pixel accuracy {near_metrics['overall']['pixel_accuracy']}", flush=True)
        print(f"  {'class':<26}{'IoU':>8}{'precision':>11}{'recall':>8}", flush=True)
        for class_name, entry in metrics["per_class"].items():
            print(f"  {class_name:<26}{str(entry['IoU']):>8}{str(entry['precision']):>11}{str(entry['recall']):>8}",
                  flush=True)

        # Step d: example images for the last split (usually test). Saved locally by write_semantic_results().
        if split == splits[-1]:
            print(f"Drawing example images from {split} ...", flush=True)
            # make_semantic_example_images takes: model, frame / mask paths, classes and class ids, image size, title
            #   returns: {file name: JPEG bytes}
            results["example_images_jpeg"] = make_semantic_example_images(
                model=model, frame_paths=frame_paths, mask_paths=mask_paths, evaluation_classes=evaluation_classes,
                sanpo_class_ids=sanpo_class_ids, model_class_ids=model_class_ids, image_size=image_size,
                half_precision=half_precision, title=f"{results['run']} | {split}")
            print(f"Example images: {len(results['example_images_jpeg'])} files", flush=True)
    return results


# ----------------------------------------------------------------------------- semantic: example images

def make_semantic_example_images(*, model, frame_paths: list[str], mask_paths: list[str], evaluation_classes: dict,
                                 sanpo_class_ids: dict[str, int], model_class_ids: dict[str, int], image_size: int,
                                 half_precision: bool, title: str, files: int = 5, rows_per_file: int = 5,
                                 panel_width: int = 640, seed: int = 7) -> dict[str, bytes]:
    """Draw example images: each row is one frame: the frame | ground truth | prediction (colored by class).

    Steps:
        1. Pick files x rows_per_file random frames (fixed seed, so every run shows the same frames).
        2. Per frame: predict, translate ground truth and prediction to evaluation classes, color them.
           Black = prediction with no matching class; gray = ground truth that is not scored.
        3. Stack the rows under a legend; encode each file as JPEG.
    Returns {file name: JPEG bytes}, e.g. {"examples_01.jpg": b"..."}.
    """
    import cv2
    import numpy as np

    names = list(evaluation_classes)
    palette = [(128, 64, 128), (232, 35, 244), (40, 90, 160), (156, 102, 102), (153, 153, 190), (0, 165, 255),
               (0, 230, 230), (35, 142, 107), (152, 251, 152), (180, 130, 70), (255, 80, 80), (0, 0, 255),
               (142, 0, 0), (255, 255, 0), (255, 255, 255), (140, 230, 240), (0, 255, 127), (203, 192, 255),
               (0, 100, 0), (255, 144, 30), (19, 69, 139), (211, 0, 148)]  # BGR, one per evaluation class
    no_match_index, ignored_index = len(names), len(names) + 1
    colors = np.array([palette[index % len(palette)] for index in range(len(names))] + [(0, 0, 0), (110, 110, 110)],
                      dtype=np.uint8)
    # Lookup tables: SANPO class id -> evaluation class index, model class id -> evaluation class index
    sanpo_lookup = np.full(256, ignored_index, dtype=np.int64)
    model_lookup = np.full(max(model_class_ids.values()) + 1, no_match_index, dtype=np.int64)
    for index, name in enumerate(names):
        for sanpo_name in evaluation_classes[name][0]:
            sanpo_lookup[sanpo_class_ids[sanpo_name]] = index
        for model_name in evaluation_classes[name][1]:
            model_lookup[model_class_ids[model_name]] = index

    def title_bar(text, width):
        bar = np.full((30, width, 3), 30, np.uint8)
        cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return bar

    def colored(image, class_indices):
        # 65 % class color + 35 % frame, so the scene stays visible under the colors
        return (0.65 * colors[class_indices] + 0.35 * image).astype(np.uint8)

    # Step 1
    total = min(files * rows_per_file, len(frame_paths))
    chosen = sorted(np.random.default_rng(seed).choice(len(frame_paths), size=total, replace=False).tolist())

    # Step 2
    rows = []
    for position in range(0, total, rows_per_file):
        group = chosen[position:position + rows_per_file]
        # load_frame_labels_depth takes: frame path, mask path, depth path (None here), image size
        #   returns: (resized BGR image, SANPO class id per pixel, None)
        loaded = [load_frame_labels_depth(frame_paths[index], mask_paths[index], None, image_size) for index in group]
        batch_results = model.predict([image for image, _, _ in loaded], imgsz=image_size,
                                      quantize=16 if half_precision else None, verbose=False)
        for index, (image, class_ids, _), result in zip(group, loaded, batch_results):
            prediction = result.semantic_mask.data.cpu().numpy().astype(np.int64)
            panel_height = round(image.shape[0] * panel_width / image.shape[1])
            panels = []
            for text, picture in [(f"FRAME | {Path(frame_paths[index]).stem[:40]}", image),
                                  ("GROUND TRUTH (gray = not scored)", colored(image, sanpo_lookup[class_ids])),
                                  ("PREDICTION (black = no matching class)", colored(image, model_lookup[prediction]))]:
                picture = cv2.resize(picture, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                panels += [np.vstack([title_bar(text, panel_width), picture]),
                           np.full((panel_height + 30, 6, 3), 255, np.uint8)]
            rows.append(np.hstack(panels[:-1]))
    if not rows:
        return {}

    # Step 3: legend + rows
    width = rows[0].shape[1]
    legend_names = names + ["no matching class (prediction)", "not scored (ground truth)"]
    legend_lines = (len(legend_names) + 5) // 6
    legend = np.full((40 + 28 * legend_lines, width, 3), 255, np.uint8)
    cv2.putText(legend, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for index, name in enumerate(legend_names):
        x, y = 10 + (index % 6) * (width // 6), 36 + (index // 6) * 28
        cv2.rectangle(legend, (x, y), (x + 20, y + 18), tuple(int(value) for value in colors[index]), -1)
        cv2.rectangle(legend, (x, y), (x + 20, y + 18), (0, 0, 0), 1)
        cv2.putText(legend, name[:34], (x + 26, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    separator = np.full((8, width, 3), 255, np.uint8)
    images = {}
    for file_index in range(0, len(rows), rows_per_file):
        parts = [legend]
        for row in rows[file_index:file_index + rows_per_file]:
            parts += [row, separator]
        ok, encoded = cv2.imencode(".jpg", np.vstack(parts), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            images[f"examples_{file_index // rows_per_file + 1:02d}.jpg"] = encoded.tobytes()
    return images


# ----------------------------------------------------------------------------- semantic: results files (local)

def semantic_summary_tables(results: dict) -> list[dict]:
    """The result tables of a semantic run, as data (drawn later with borders or as markdown).

    Tables:
        0. Run information
        1. Main scores per split: mIoU, pixel accuracy, and mIoU for near pixels
        2. Per class: IoU for every split; precision, recall and pixel share on the last split
        3. Near pixels per class (last split): IoU for all pixels, within 10 m, within 5 m
        4. What the model said on each class's pixels (last split): the most common model classes
        5. Speed and GPU memory
        6. How to read the numbers
    Returns a list of {"title", "headers", "rows", "numeric_columns", "group_column"}.
    """
    def show(value, digits=3):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"

    def percent(value):
        return f"{100 * value:.1f}%" if isinstance(value, (int, float)) else "n/a"

    settings = results["settings"]
    splits = list(results["splits"])
    last = splits[-1]
    near_limits = sorted(results["splits"][last].get("near", {}), key=float, reverse=True)  # e.g. ["10", "5"]
    tables = []

    tables.append({"title": "0. RUN", "headers": ["item", "value"], "rows": [
        ["model run", results["run"]], ["evaluation", results["evaluation"]], ["GPU", results["gpu"]],
        ["weights", settings["weights"]], ["task", "semantic (one class per pixel)"],
        ["classes scored", len(settings["evaluation_classes"])],
        ["model input size (long side)", settings["image_size"]],
        ["splits", ", ".join(f"{split} ({results['splits'][split]['frames']:,} frames)" for split in splits)],
    ], "numeric_columns": set(), "group_column": None})

    rows = []
    for split in splits:
        metrics = results["splits"][split]
        rows.append([split, show(metrics["overall"]["mIoU"]), show(metrics["overall"]["pixel_accuracy"])]
                    + [show(metrics["near"][limit]["overall"]["mIoU"]) if limit in metrics.get("near", {}) else "n/a"
                       for limit in near_limits])
    tables.append({"title": "1. MAIN SCORES (mIoU = main score)",
                   "headers": ["split", "mIoU", "pixel accuracy"] + [f"mIoU within {limit} m" for limit in near_limits],
                   "rows": rows, "numeric_columns": set(range(1, 3 + len(near_limits))), "group_column": None})

    class_names = sorted(results["splits"][last]["per_class"],
                         key=lambda name: results["splits"][last]["per_class"][name]["IoU"] or 0, reverse=True)
    rows = []
    for class_name in class_names:
        entry = results["splits"][last]["per_class"][class_name]
        rows.append([class_name] + [show(results["splits"][split]["per_class"][class_name]["IoU"]) for split in splits]
                    + [show(entry["precision"]), show(entry["recall"]), percent(entry["ground_truth_pixel_share"])])
    tables.append({"title": f"2. PER CLASS (sorted by {last} IoU)",
                   "headers": ["class"] + [f"{split} IoU" for split in splits]
                   + [f"{last} precision", f"{last} recall", f"{last} share of scored pixels"],
                   "rows": rows, "numeric_columns": set(range(1, len(splits) + 4)), "group_column": None})

    if near_limits:
        rows = []
        for class_name in class_names:
            near = results["splits"][last]["near"]
            rows.append([class_name, show(results["splits"][last]["per_class"][class_name]["IoU"])]
                        + [show(near[limit]["per_class"][class_name]["IoU"]) for limit in near_limits])
        tables.append({"title": f"3. NEAR PIXELS, IoU PER CLASS ({last}; only pixels within N meters are counted)",
                       "headers": ["class", "all pixels"] + [f"within {limit} m" for limit in near_limits],
                       "rows": rows, "numeric_columns": set(range(1, 2 + len(near_limits))), "group_column": None})

    rows = []
    for class_name in class_names:
        entry = results["splits"][last]["per_class"][class_name]
        said = ", ".join(f"{item['model_class']} {100 * item['share']:.0f}%" for item in entry["top_predictions"])
        expected = ", ".join(settings["evaluation_classes"][class_name]["model"])
        rows.append([class_name, expected, said or "n/a"])
    tables.append({"title": f"4. WHAT THE MODEL SAID ON EACH CLASS'S PIXELS ({last}; its 3 most common answers)",
                   "headers": ["class (ground truth)", "model classes counted as correct", "what the model said"],
                   "rows": rows, "numeric_columns": set(), "group_column": None})

    rows = []
    for split in splits:
        metrics = results["splits"][split]
        speed = metrics.get("speed_ms_per_frame") or {}
        rows.append([split, show(speed.get("inference"), 2), show(speed.get("preprocess"), 2),
                     show(speed.get("postprocess"), 2), show(metrics.get("gpu_memory_peak_gb"), 2)])
    tables.append({"title": "5. SPEED AND GPU (milliseconds per frame, model in a batch)",
                   "headers": ["split", "model", "preprocess", "postprocess", "GPU memory peak (GB)"],
                   "rows": rows, "numeric_columns": {1, 2, 3, 4}, "group_column": None})

    tables.append({"title": "6. HOW TO READ", "headers": ["score", "meaning"], "rows": [
        ["IoU", "pixels where prediction and label both say the class / pixels where either says it"],
        ["mIoU", "main score: IoU averaged over the scored classes"],
        ["pixel accuracy", "share of all scored pixels that got the right class (dominated by big classes)"],
        ["precision", "of the pixels predicted as the class, the share that really are the class"],
        ["recall", "of the real pixels of the class, the share predicted as the class"],
        ["not scored", "pixels of SANPO classes the model has no class for: ignored"],
    ], "numeric_columns": set(), "group_column": None})
    return tables


def semantic_terminal_summary(results: dict) -> str:
    """The result tables of a semantic run with box borders (printed at the end of a run, and saved as metrics.txt)."""
    lines = ["=" * 100, "SUMMARY", "=" * 100, ""]
    for table in semantic_summary_tables(results):
        # bordered_table takes: title, headers, rows, numeric columns, group column; returns: the lines of the table
        lines += bordered_table(table["title"], table["headers"], table["rows"], table["numeric_columns"],
                                table["group_column"]) + [""]
    return "\n".join(lines)


def semantic_summary_markdown(results: dict) -> str:
    """summary.md of a semantic run: the same tables as markdown (without "how to read")."""
    lines = [f"# {results['run']}", ""]
    for table in semantic_summary_tables(results)[:-1]:
        lines += [f"## {table['title']}", "", "| " + " | ".join(table["headers"]) + " |",
                  "| " + " | ".join("---" for _ in table["headers"]) + " |"]
        lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in table["rows"]]
        lines.append("")
    return "\n".join(lines)


def write_semantic_results(results: dict, output_directory: Path) -> Path:
    """Save the results of a semantic run next to the evaluation script (runs on the local computer).

    Writes example_images/examples_XX.jpg, metrics.json (everything, with the raw count tables),
    metrics.txt (the tables with borders) and summary.md (the tables as markdown).
    Returns the output folder.
    """
    import json
    from datetime import datetime

    output_directory.mkdir(parents=True, exist_ok=True)
    example_images = results.pop("example_images_jpeg", None) or {}
    if example_images:
        images_directory = output_directory / "example_images"
        images_directory.mkdir(parents=True, exist_ok=True)
        for file_name, jpeg_bytes in example_images.items():
            (images_directory / file_name).write_bytes(jpeg_bytes)
        print(f"Saved {len(example_images)} example images to {images_directory}", flush=True)
    results["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (output_directory / "metrics.json").write_text(json.dumps(results), encoding="utf-8")
    (output_directory / "metrics.txt").write_text(semantic_terminal_summary(results), encoding="utf-8")
    (output_directory / "summary.md").write_text(semantic_summary_markdown(results), encoding="utf-8")
    return output_directory
