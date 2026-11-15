"""Run the SANPO depth evaluation (eval_sanpo.py) on a Modal GPU, for several models on the SAME frames.

What it does
    1. Inside the container, gets a SANPO-Real subset (chest camera, dense depth + segmentation masks):
         --selection random : download_sanpo.py, evenly spaced frames (default 5 walks x 10 frames = 50,
                              one per second; the recipe of the earlier DAv2 baseline)
         --selection near   : select_near_sanpo.py, frames with an obstacle within 3 m (the frames where a
                              warning matters); --sessions walks x --max-frames frames per walk
       --official-split test limits either one to SANPO's official test walks.
       The frames are kept on the Modal Volume VOLUME_NAME, so a re-run with the same settings does not
       download them again.
    2. Runs eval_sanpo.py once per model on those frames (AbsRel, delta1, RMSE, per distance band, on obstacle
       pixels and on obstacle pixels within 3 m; raw meters "none" and per-frame median-scaled "median").
    3. Copies each model's results (summary.json, per_frame.csv, vis/) back to outputs/depth_modal/<run>/<model>/
       and prints a comparison table.

Model weights are cached on the Volume too (Hugging Face cache), so only the first run downloads them.

Run from the repo root:
    python -m modal run fast_path/depth/scripts/eval_depth_modal.py
    python -m modal run fast_path/depth/scripts/eval_depth_modal.py --seed 1
    python -m modal run fast_path/depth/scripts/eval_depth_modal.py --selection near --sessions 10 --max-frames 5
    python -m modal run fast_path/depth/scripts/eval_depth_modal.py --official-split test --sessions 30
"""

from __future__ import annotations

from pathlib import Path

import modal

# =============================================================================
# SETTINGS (edit here). Command-line options override the frame/model settings.
# =============================================================================
VOLUME_NAME = "blind-guide-depth"
GPU = "L4"
MODELS = "depth-pro,da2-small-metric-outdoor"  # comma-separated short names (models.py) or HF ids
SELECTION = "random"  # "random" (evenly spaced frames) or "near" (frames with an obstacle within 3 m)
OFFICIAL_SPLIT = ""  # "" = any walk, "train" or "test" = SANPO's official split only
SESSIONS = 5  # SANPO-Real walks
MAX_FRAMES = 10  # frames per walk
STRIDE = 15  # random selection: every 15th frame = 1 frame per second (real data is 15 fps)
SEED = 0  # which walks are picked (same seed = same walks)
INPUT_SIZE = 518  # DAv2 input short side; Depth Pro always uses its fixed 1536 input
# =============================================================================

# On this PC: the repo root. Inside the Modal container this file sits at /root/, where the
# fast_path package is also placed, so the root is /root there.
REPO_ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root")
VOLUME_MOUNT = "/vol"

app = modal.App("blind-guide-depth-eval")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # torchvision: needed by Depth Pro's image processor (DepthProImageProcessorFast)
    .pip_install("torch>=2.4,<2.9", "torchvision", "transformers>=4.50,<4.60", "opencv-python-headless>=4.8,<5",
                 "numpy>=1.26,<3", "pillow", "tqdm")
    .env({"HF_HOME": f"{VOLUME_MOUNT}/hf_cache"})  # model weights cached on the Volume
    .add_local_file(REPO_ROOT / "fast_path" / "__init__.py", "/root/fast_path/__init__.py")
    .add_local_dir(REPO_ROOT / "fast_path" / "depth", "/root/fast_path/depth", ignore=["**/__pycache__/**"])
)


def run_name(selection: str, official_split: str, sessions: int, max_frames: int, stride: int, seed: int) -> str:
    split = f"_{official_split}" if official_split else ""
    if selection == "near":
        return f"real_chest_near3m{split}_s{sessions}_f{max_frames}_seed{seed}"
    return f"real_chest{split}_s{sessions}_f{max_frames}_stride{stride}_seed{seed}"


@app.function(image=image, gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=3 * 3600)
def evaluate(models: list[str], selection: str, official_split: str, sessions: int, max_frames: int,
             stride: int, seed: int, input_size: int) -> dict:
    """Get the frames (if needed), evaluate every model, return each model's output folder as a zip."""
    import io
    import os
    import subprocess
    import sys
    import zipfile

    os.chdir("/root")
    name = run_name(selection, official_split, sessions, max_frames, stride, seed)
    data_root = f"{VOLUME_MOUNT}/sanpo/{name}"  # one folder per frame recipe, so runs never mix frames
    split_args = ["--official-split", official_split] if official_split else []

    # Step 1: frames + dense depth + masks (both scripts skip files that already exist)
    if selection == "near":
        # Skip walks already used by earlier runs, so this set is new
        used = sorted({p.name for p in Path(f"{VOLUME_MOUNT}/sanpo").glob("*/sanpo-real/*")
                       if p.is_dir() and not str(p).startswith(data_root)})
        command = ["fast_path/depth/scripts/select_near_sanpo.py", "--root", data_root,
                   "--sessions", str(sessions), "--per-session", str(max_frames), "--seed", str(seed),
                   f"--exclude={','.join(used)}", *split_args]
    else:
        command = ["fast_path/depth/scripts/download_sanpo.py", "--root", data_root, "--split", "sanpo-real",
                   "--camera", "chest", "--sessions", str(sessions), "--max-frames", str(max_frames),
                   "--stride", str(stride), "--seed", str(seed), "--gt", "depth_maps", "segmentation_masks",
                   *split_args]
    subprocess.run([sys.executable, *command], check=True)
    volume.commit()

    # Step 2: evaluate each model on exactly those frames
    results = {}
    for model in models:
        out_dir = Path("/tmp/eval") / model.replace("/", "_")
        print(f"\n===== {model} =====", flush=True)
        finished = subprocess.run([sys.executable, "fast_path/depth/scripts/eval_sanpo.py", "--root", data_root,
                                   "--split", "sanpo-real", "--camera", "chest", "--model", model,
                                   "--input-size", str(input_size), "--out", str(out_dir)])
        volume.commit()  # keep the downloaded weights in the cache
        if finished.returncode != 0:  # keep going: the other models still get evaluated
            print(f"FAILED: {model} (exit code {finished.returncode}), see the error above", flush=True)
            continue

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(out_dir))
        results[model] = buffer.getvalue()
    return results


@app.local_entrypoint()
def main(models: str = MODELS, selection: str = SELECTION, official_split: str = OFFICIAL_SPLIT,
         sessions: int = SESSIONS, max_frames: int = MAX_FRAMES, stride: int = STRIDE, seed: int = SEED,
         input_size: int = INPUT_SIZE):
    """Run on Modal, then unpack the results into outputs/depth_modal/<run>/ and print a comparison."""
    import io
    import json
    import zipfile

    if selection not in ("random", "near"):
        raise SystemExit("--selection must be 'random' or 'near'")
    if official_split not in ("", "train", "test"):
        raise SystemExit("--official-split must be '', 'train' or 'test'")
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    name = run_name(selection, official_split, sessions, max_frames, stride, seed)
    print(f"Evaluating {model_list} on {name} ({GPU}) ...")
    zipped = evaluate.remote(model_list, selection, official_split, sessions, max_frames, stride, seed, input_size)

    missing = [m for m in model_list if m not in zipped]
    if missing:
        print(f"No results for {missing} (failed on Modal, see the log above)")
    out_root = REPO_ROOT / "outputs" / "depth_modal" / name
    summaries = {}
    for model, data in zipped.items():
        target = out_root / model.replace("/", "_")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(target)
        summaries[model] = json.loads((target / "summary.json").read_text())

    # Comparison table: raw meters (what the phone gets) and median-scaled (shape only)
    columns = ["abs_rel", "d1", "abs_rel@0-2m", "abs_rel@2-5m", "abs_rel@5-10m", "abs_rel@obstacles",
               "abs_rel@obstacles<3m", "d1@obstacles<3m"]
    print(f"\n{'model':<28}{'align':<8}{'frames':>7}{'ms':>7}" + "".join(f"{c:>21}" for c in columns))
    for model, summary in summaries.items():
        for mode, values in summary["results"].items():
            cells = "".join(f"{values.get(c, float('nan')):>21.3f}" for c in columns)
            print(f"{model:<28}{mode:<8}{summary['n_frames']:>7}{summary['latency_ms_mean'] or 0:>7.0f}{cells}")
        if "focal_px" in summary["results"].get("none", {}):
            print(f"{'':<28}estimated focal length (mean): {summary['results']['none']['focal_px']:.0f} px")
    print(f"\nSaved to {out_root}")
