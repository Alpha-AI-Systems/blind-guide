"""Code shared by all evaluation scripts in this folder (each subfolder has its own evaluate.py).

Each evaluate.py keeps only what is specific to its model: settings, model loading and the
class mapping (steps 1 to 3). Everything else lives here:

    format_duration()          seconds -> "0h01m05s"
    load_split()               frames + ground-truth boxes of one split
    coco_evaluation()          standard COCO scores (pycocotools)
    operating_point_metrics()  precision / recall / F1 / false alarms at fixed confidences
    print_setup()              prints the full setup before anything is evaluated
    evaluate_splits()          step 4: ground truth, predictions (resumable), scoring, for every split
    write_results()            saves results/<run>/metrics.json + summary.txt

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


def operating_point_metrics(ground_truth_coco: dict, detections: list[dict], category_names_by_id: dict[int, str],
                            confidences: list[float], match_iou: float) -> dict:
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
    """
    import numpy as np
    from collections import defaultdict

    frame_count = len(ground_truth_coco["images"])

    # Step 1: group by (frame, class)
    ground_truth_boxes = defaultdict(list)
    for annotation in ground_truth_coco["annotations"]:
        ground_truth_boxes[(annotation["image_id"], annotation["category_id"])].append(annotation["bbox"])
    predictions = defaultdict(list)
    for detection in detections:
        predictions[(detection["image_id"], detection["category_id"])].append((detection["score"], detection["bbox"]))

    def iou_matrix(first_boxes, second_boxes):
        """IoU between every box in first_boxes and every box in second_boxes (boxes as [x, y, width, height])."""
        first = np.asarray(first_boxes, dtype=float)
        second = np.asarray(second_boxes, dtype=float)
        first_right, first_bottom = first[:, 0] + first[:, 2], first[:, 1] + first[:, 3]
        second_right, second_bottom = second[:, 0] + second[:, 2], second[:, 1] + second[:, 3]
        overlap_width = np.clip(np.minimum(first_right[:, None], second_right[None])
                                - np.maximum(first[:, 0][:, None], second[:, 0][None]), 0, None)
        overlap_height = np.clip(np.minimum(first_bottom[:, None], second_bottom[None])
                                 - np.maximum(first[:, 1][:, None], second[:, 1][None]), 0, None)
        intersection = overlap_width * overlap_height
        union = (first[:, 2] * first[:, 3])[:, None] + (second[:, 2] * second[:, 3])[None] - intersection
        return intersection / np.maximum(union, 1e-9)

    # Step 2: greedy matching once per group (in score order), remember (score, is_true_positive)
    scored_predictions = defaultdict(list)  # category id -> list of (score, is_true_positive)
    for (image_id, category_id), group in predictions.items():
        group.sort(key=lambda item: item[0], reverse=True)
        truth = ground_truth_boxes.get((image_id, category_id), [])
        if not truth:
            scored_predictions[category_id] += [(score, False) for score, _ in group]
            continue
        overlaps = iou_matrix([bbox for _, bbox in group], truth)
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
    print(f"Results will be saved to: results/{results['run']}/ (metrics.json + summary.txt)", flush=True)
    print("=" * 80, flush=True)


# ----------------------------------------------------------------------------- step 4 (run on Modal)

def evaluate_splits(*, model, volume, results: dict, dataset_root: Path, work_directory: Path, view_name: str,
                    splits: list[str], evaluation_classes: dict, category_ids_by_name: dict[str, int],
                    model_class_to_category_id: dict[int, int], sanpo_class_to_category_id: dict[str, int],
                    image_size: int, confidence_threshold: float, nms_iou_threshold: float, max_detections: int,
                    half_precision: bool, batch_size: int, save_every_batches: int,
                    operating_confidences: list[float], match_iou: float) -> dict:
    """Step 4 of every evaluation: for each split, ground truth -> predictions -> scores.

    Steps (per split):
        a. load frames + ground-truth boxes,
        b. convert ground truth to COCO format,
        c. predict all frames (batched; saved regularly so a stopped run resumes),
        d. score with COCO metrics, operating points, and average speed.
    Fills results["splits"][split] and returns results.
    """
    import json
    import os
    import time

    evaluation_class_names = list(evaluation_classes)

    # Step 4: evaluate each split
    for split in splits:
        print(f"\n===== split: {split} =====", flush=True)
        # Step 4a: frames and ground truth of the split
        frames, ground_truth = load_split(dataset_root, split, view_name)
        frame_names = [Path(frame_path).stem for frame_path in frames["frame"]]
        image_ids = {frame_name: index + 1 for index, frame_name in enumerate(frame_names)}

        # Step 4b: ground truth in COCO format (only SANPO classes that map to an evaluation class)
        # Keep only boxes whose SANPO class is evaluated here; boxes of other classes are removed
        # because this model is not scored on them
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

            # 4c.5: loop over the frames in batches of batch_size, starting after the frames already done
            for batch_start in range(saved["frames_done"], len(frame_paths), batch_size):
                batch_paths = frame_paths[batch_start:batch_start + batch_size]
                # 4c.6: run the model on the whole batch (one GPU call); one result per frame
                batch_results = model.predict(batch_paths, imgsz=image_size, conf=confidence_threshold,
                                              iou=nms_iou_threshold, max_det=max_detections, half=half_precision,
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
                saved["frames_done"] = min(batch_start + batch_size, len(frame_paths))
                batches_done += 1
                elapsed = time.time() - start_time
                frames_per_second = (saved["frames_done"] - frames_done_at_start) / elapsed if elapsed else 0
                remaining_seconds = (len(frame_paths) - saved["frames_done"]) / frames_per_second if frames_per_second else 0
                print(f"[{split}] {saved['frames_done']:,}/{len(frame_paths):,} frames | {frames_per_second:.1f} frames/s | "
                      f"elapsed {format_duration(elapsed)} | ETA {format_duration(remaining_seconds)}", flush=True)
                # 4c.9: every save_every_batches batches, save progress to the partial file (for resuming).
                # Written to a .tmp file first, then renamed, so a crash never leaves a broken file.
                if batches_done % save_every_batches == 0:
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
        # Operating points: precision / recall / F1 / false alarms per frame at fixed confidences
        print(f"Operating points at confidence {operating_confidences} (IoU >= {match_iou}) ...", flush=True)
        metrics["operating_points"] = operating_point_metrics(
            ground_truth_coco, saved["detections"], category_names_by_id, operating_confidences, match_iou)
        for label, operating_point in metrics["operating_points"]["by_confidence"].items():
            overall_point = operating_point["overall"]
            print(f"  confidence {label:<16} precision {overall_point['precision']} | recall {overall_point['recall']} | "
                  f"F1 {overall_point['f1']} | false alarms/frame {overall_point['false_alarms_per_frame']}", flush=True)
        metrics["speed_ms_per_frame"] = {
            "preprocess": round(sum(speed[0] for speed in speeds) / len(speeds), 2),
            "inference": round(sum(speed[1] for speed in speeds) / len(speeds), 2),
            "postprocess": round(sum(speed[2] for speed in speeds) / len(speeds), 2),
        } if speeds else None
        results["splits"][split] = metrics
    return results


# ----------------------------------------------------------------------------- local: save results

def write_results(results: dict, output_directory: Path) -> Path:
    """Save the results next to the evaluation script (runs on the local computer).

    Steps:
        1. <output_directory>/metrics.json: everything, machine-readable.
        2. <output_directory>/summary.txt: settings, class mapping, and per split an overall
           line, a speed line, a per-class table and the operating points, human-readable.
    """
    import json
    from datetime import datetime

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
                         f"inference {speed['inference']} | postprocess {speed['postprocess']}")
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
    (output_directory / "summary.txt").write_text("\n".join(lines) + "\n")
    return output_directory
