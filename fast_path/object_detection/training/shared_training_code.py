"""
shared_training_code.py - the shared training engine of the blind-guide object-detection training runs.

Adapted from the general training engine `ml_utils.py` (D:/My Projects/shared-code), which came from a
Kaggle / research project. Ultralytics does the training itself (its loop, augmentations, MuSGD optimizer,
loss, EMA, checkpoints, early stopping); this file does everything around it, through Ultralytics callbacks
and a small trainer subclass.

Layout (fast_path/object_detection/training/):
    shared_training_code.py          this file
    shared_config.yaml               settings shared by every run
    runs/<run name>/train.py         what is specific to one run (model, data, extra callbacks)
    runs/<run name>/config.yaml      only what differs from shared_config.yaml
    runs/<run name>/results/         everything the run writes (see "Outputs" below)

What it covers
    config        DEFAULTS_YAML < shared_config.yaml < runs/<run>/config.yaml, deep-merged, printed in boxes
    printing      boxed banners / key-value boxes / tables; the FULL console of every run is also written to
                  results/train_config/<run id>_<date and time>.txt
    run identity  run001, run002 ... per run folder, tagging every CSV row and every file
    training      Ultralytics model.train(...) with every setting from the config, a resume policy and a
                  guard that refuses to resume after a guarded setting changed
    diagnostics   per-step line (loss, learning rate, gradient norm, elapsed, ETA), throughput three ways
                  (compute-only, real with data wait, end-to-end), data-wait %, MFU, SM busy %, memory
                  controller busy %, peak GPU memory, update-to-weight ratio, head output scale probe
    outputs       train_log.csv (one row per step), val_log.csv (one row per validation, with IoU per class),
                  ud_ratio.csv, training_summary.json, plots, Ultralytics' own folder (weights, results.csv)
    modal         image, volumes, resources, run-id allocation, launch, Ctrl+C stops the GPU first and then
                  syncs, results sync home (CSV logs merged row by row), checkpoint pull, path remapping,
                  checkpoint upload from a local run to a Modal volume
    command line  python shared_training_code.py plot | sync | pull | leaderboard | config  <run folder>

Removed from ml_utils.py (not needed for Ultralytics semantic segmentation training, agreed with the user):
    the own training loop and everything that only it used (optimizer / scheduler builders, LLRD, LP-FT,
    SAM, own EMA, own checkpoints and resume, dataloaders, Hooks, predict), tabular / classification metrics
    and losses, cross-validation and ensembling, Kaggle tools.
"""

from __future__ import annotations

import os

# Cap OpenMP / MKL threads before torch is imported (oversubscription slows the data loader workers).
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, str(min(8, os.cpu_count() or 4)))

import copy
import csv
import json
import logging as _logging
import math
import random
import re
import shutil
import sys
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

try:
    import torch
except ImportError:
    # The Python that LAUNCHES a Modal run (python -m modal run train.py) may have no torch: the launcher only
    # needs the config, run ids, Modal glue, sync and plots. Training itself runs where torch exists.
    torch = None


class _ExitWhenAppStopped(_logging.Filter):
    """Swallow Modal's heartbeat tracebacks once the app is stopped, and exit the local client.

    Stopping a run from the Modal dashboard leaves the local `modal run` process heartbeating a dead app
    every 10 s, and each reply ("App state is APP_STATE_STOPPED") is logged as a full traceback until
    Ctrl+C. That reply means the app is already gone, so the right end state is simply exiting.
    Anything else passes through.
    """

    _done = False

    def filter(self, record):
        exception = record.exc_info[1] if record.exc_info else None
        text = "%s %s" % (record.getMessage(), exception or "")
        if "APP_STATE_STOPPED" not in text:
            return True
        if not _ExitWhenAppStopped._done:
            _ExitWhenAppStopped._done = True
            print("\n  app stopped on Modal - closing the local client", flush=True)
            os._exit(0)                  # the remote app is already stopped; nothing to clean up
        return False


for _logger_name in ("modal-utils", "modal-client"):
    _logging.getLogger(_logger_name).addFilter(_ExitWhenAppStopped())

try:
    import cv2
    cv2.setNumThreads(0)          # data loader workers do their own parallelism
except ImportError:
    cv2 = None

# box-drawing characters need utf-8; Windows consoles default to cp1252
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# pretty printing
# =============================================================================
WIDTH = 78


def _rule(character: str = "\u2500"):
    print(character * WIDTH)


def _banner(title: str, emoji: str = ""):
    """Big section header."""
    print()
    print("\u2554" + "\u2550" * (WIDTH - 2) + "\u2557")
    print("\u2551" + _pad(f" {emoji} {title} ", WIDTH - 2) + "\u2551")
    print("\u255A" + "\u2550" * (WIDTH - 2) + "\u255D")


def _kv(title: str, pairs, emoji: str = ""):
    """Boxed key / value block. `pairs` may contain (key, value) or None for a separator line."""
    head = f" {emoji} {title} "
    print("\u250C\u2500" + head + "\u2500" * max(0, WIDTH - _display_width(head) - 3) + "\u2510")
    for row in pairs:
        if row is None:
            print("\u251C" + "\u2500" * (WIDTH - 2) + "\u2524")
            continue
        key, value = row
        text = f" {key:<26}\u2502 {value}"
        if _display_width(text) > WIDTH - 2:
            text = text[:WIDTH - 5] + "..."
        print("\u2502" + _pad(text, WIDTH - 2) + "\u2502")
    print("\u2514" + "\u2500" * (WIDTH - 2) + "\u2518")


def _table(title: str, headers, rows, emoji: str = ""):
    """Boxed table. The title sits above the box so it never overflows the border."""
    if not rows:
        return
    columns = [[str(header)] + [str(row[index]) for row in rows] for index, header in enumerate(headers)]
    widths = [max(_display_width(cell) for cell in column) for column in columns]
    print(f"  {emoji} {title}".rstrip())
    print("┌" + "┬".join("─" * (width + 2) for width in widths) + "┐")
    print("│ " + " │ ".join(_pad(header, width) for header, width in zip(headers, widths)) + " │")
    print("├" + "┼".join("─" * (width + 2) for width in widths) + "┤")
    for row in rows:
        print("│ " + " │ ".join(_pad(cell, width) for cell, width in zip(row, widths)) + " │")
    print("└" + "┴".join("─" * (width + 2) for width in widths) + "┘")


def _display_width(text: str) -> int:
    # emoji take two terminal cells, so plain len() skews every border to their right. Emoji below
    # U+1F000 (e.g. U+1F5C2 is above, but U+2699 "gear" is below) are drawn wide only when followed by the
    # emoji selector U+FE0F, so those count as two cells in that case.
    text = str(text)
    width = 0
    for index, character in enumerate(text):
        code = ord(character)
        if code == 0xFE0F:
            continue
        if code >= 0x1F000:
            width += 2
        elif code >= 0x2300 and index + 1 < len(text) and ord(text[index + 1]) == 0xFE0F:
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, width: int) -> str:
    return str(text) + " " * max(0, width - _display_width(text))


# public names for the printing helpers, so run scripts can reuse the same look
rule, banner, kv, table = _rule, _banner, _kv, _table


# =============================================================================
# timing
# =============================================================================
class Timer:
    """`with Timer("decode"):` prints how long the block took; `.seconds` holds the number.

    verbose=False keeps it silent. cuda_sync=True waits for queued GPU work at both ends, because CUDA
    calls return before the kernels finish and would otherwise time nothing.
    """

    def __init__(self, label: str = "", verbose: bool = True, cuda_sync: bool = False):
        self.label, self.verbose, self.cuda_sync, self.seconds = label, verbose, cuda_sync, 0.0

    def __enter__(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exception):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        if self.verbose:
            print(f"  ⏱️  {self.label or 'block'}: {format_precise(self.seconds)}", flush=True)
        return False


def timed(function=None, *, label: str | None = None):
    """Decorator printing a function's wall time on every call: `@timed` or `@timed(label=...)`."""
    import functools

    def wrap(inner_function):
        @functools.wraps(inner_function)
        def inner(*arguments, **keyword_arguments):
            with Timer(label or inner_function.__name__):
                return inner_function(*arguments, **keyword_arguments)
        return inner
    return wrap(function) if function is not None else wrap


def format_duration(seconds: float) -> str:
    """Seconds -> "1h04m09s" / "4m09s"."""
    hours, remainder = divmod(int(max(0, seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s" if hours else f"{minutes}m{seconds:02d}s"


def format_precise(seconds: float) -> str:
    """Seconds -> "812 ms" / "4.21 s" / "4m09s" / "1h04m09s"."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    return format_duration(seconds)


def clock(seconds: float | None) -> str | None:
    """Seconds -> "44:32", or "1:35:34" once it runs past an hour."""
    if seconds is None:
        return None
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


# =============================================================================
# config
# =============================================================================
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; `override` wins on scalars, dicts are merged key by key.

    copy.deepcopy so an untouched nested dict is not shared with `base`; otherwise two configs made from
    one load would alias each other.
    """
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_scientific(node):
    """Turn YAML's accidental strings back into floats.

    YAML only reads scientific notation as a number when the mantissa has a decimal point: `8.0e-6` is a
    float, `8e-6` is the STRING "8e-6". One missing dot would silently hand a string to the optimizer.
    Only strings carrying an exponent marker are touched, so real strings ("auto", "cosine") stay.
    """
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, value in list(items):
        if isinstance(value, (dict, list)):
            _coerce_scientific(value)
        elif isinstance(value, str) and ("e" in value or "E" in value):
            try:
                node[key] = float(value)
            except ValueError:
                pass


# Every key this file reads, with its default and meaning. Merge order: DEFAULTS < shared_config.yaml <
# the run's config.yaml, so a run config only states what differs.
DEFAULTS_YAML = r"""
experiment:
  name: null                 # default: the run folder's name (also the Modal app name)
  run_id: null               # allocated at launch (run001, run002, ...)
  notes: ""

model:
  weights: yolo26n-sem-ade20k.pt   # starting weights (Ultralytics downloads official ones by name)
  weights_dir: null          # folder official weights are downloaded into (null = the current folder;
                             # on Modal: <runs volume>/_weights, so they are downloaded only once)
  task: semantic

data:
  yaml: null                 # Ultralytics dataset YAML (train / val image lists, names, masks_dir, label_mapping)
  image_dir_name: frames     # the folder name in every image path that the mask path replaces ...
  masks_dir: masks           # ... with this (a path segment), e.g. frames/train/x.jpg -> <masks_dir>/train/x.png
  validate_at_training_scale: true   # validation frames sized by the SHORT side like training (626x352 at
                             # imgsz 352); false = Ultralytics' long side (352x198, objects 1.8x smaller)

ultralytics:                 # passed as-is to Ultralytics model.train(...); see docs.ultralytics.com/modes/train
  epochs: 25
  batch: 32
  imgsz: 352                 # TRAINING sets the frame's SHORT side (352 -> 626x352); prediction sets the long side; multiples of 32 only
  optimizer: MuSGD
  lr0: 0.001                 # peak learning rate
  lrf: 0.01                  # final learning rate = lr0 * lrf
  cos_lr: true               # cosine decay (Ultralytics updates the learning rate once per epoch, after warmup)
  warmup_epochs: 2
  momentum: 0.9
  weight_decay: 0.0005
  nbs: 32                    # nominal batch size: Ultralytics accumulates gradients over round(nbs / batch) batches
  mosaic: 0.05               # share of training images that are 4-frame mosaics
  close_mosaic: 15           # mosaic off for the last N epochs
  fliplr: 0.5
  flipud: 0.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  scale: 0.25                # random zoom between 1 - scale and 1 + scale
  translate: 0.1
  degrees: 3.0               # random rotation, degrees either way
  shear: 0.0
  perspective: 0.0
  mixup: 0.0
  cutmix: 0.0
  copy_paste: 0.0
  cls_pw: 0.0                # class weighting power (0 = off)
  patience: 100              # early stopping, in epochs without improvement
  seed: 42
  deterministic: true
  amp: true
  workers: 8                 # training data loader worker processes (see dataloader: below for the rest)
  cache: false               # false | ram | disk: keep decoded images in RAM (fastest) or as .npy files on disk
  save_period: -1            # also keep a checkpoint every N epochs (-1 = only last.pt and best.pt)
  val: true
  plots: true
  verbose: true

dataloader:                  # the data loaders are built here (trainer subclass), not by Ultralytics, so all of this
                             # is configurable; workers / batch / cache come from the ultralytics section
  val_workers: null          # validation workers (null = the same as ultralytics.workers; Ultralytics itself uses 2x)
  prefetch_factor: 4         # batches each training worker prepares ahead (Ultralytics' fixed value: 4)
  val_prefetch_factor: 2     # the same for validation (Ultralytics: 2)
  pin_memory: true           # page-locked host memory: faster copies to the GPU
  drop_last: false           # drop the last, smaller training batch of each epoch
  cap_workers_at_cpu_count: false   # Ultralytics caps workers at os.cpu_count(); in a Modal container that reports
                                    # the host's cores, not the reserved ones, so the cap is off by default

logging:
  # Ultralytics' integrations with outside tracking tools: each one that is installed on the machine would
  # otherwise switch itself on (MLflow, for example, writes mlflow.db and mlruns/ into the current folder).
  # Listed ones are removed from the trainer; Ultralytics' own results.csv and plots stay.
  disable_integrations: [mlflow, comet, clearml, dvc, neptune, raytune, wandb, tensorboard, hub]
  unit: img                  # what ONE training item is called in throughput lines
  console_log_every: 20      # optimizer steps between console lines
  diagnostics_every: 200     # steps between update-to-weight ratio / scale probe readings (0 = off)
  debug_images_every: 500    # steps between debug images of the EXACT model input + mask (step 1 too; 0 = off)
  debug_images_per_save: 4   # images of the batch per debug PNG (one row each)
  flops_probe: true          # count FLOPs once (needed for MFU)
  print:
    config: true             # boxed config sections
    raw_config: true         # the merged YAML verbatim
    environment: true        # host, versions, CPU / RAM, GPUs
    model: true              # parameter counts
    step_line: true          # loss / lr / gradient norm / elapsed / ETA line
    throughput: true         # compute / real / end-to-end rate and data-wait %
    compute: true            # MFU, SM busy %, memory controller busy %
    ud_ratio: true           # update-to-weight ratio summary
    scale_probe: true        # output scale of the model's last module
    per_class_eval: true     # every class's IoU at each validation
    final_summary: true      # end-of-run table

output:
  run_dir: results/train      # training outputs (results/test/ holds the test evaluation of the run)
  ultralytics_dir: ultralytics   # Ultralytics' own folder inside results/ (weights/last.pt, best.pt, results.csv)
  train_log_csv: train_log.csv
  val_log_csv: val_log.csv
  summary_json: training_summary.json
  console_capture_dir: train_config   # full terminal capture per run, named <run id>_<date and time>.txt
  commit_every_steps: 50     # Modal: publish the runs volume this often (in a background thread)
  make_plots: true
  plots_dir: plots
  debug_dir: debug           # results/debug/<run_id>_step<N>_epoch<E>.png + .json (see debug_images_every)
  upload_checkpoints: null   # {modal_config, volume, patterns}: copy a LOCAL run's checkpoints to a Modal volume

modal:
  gpu: "H100"
  timeout_hours: 8
  cpu: 16
  memory_gb: 64
  scaledown_window: 2
  python_version: "3.12"
  pip: ["ultralytics==8.4.208", numpy, pandas, pyyaml, matplotlib, psutil, nvidia-ml-py]
  extra_pip: []
  apt: [libgl1, libglib2.0-0]
  runs_volume: "blind-guide-runs"
  runs_mount: "/runs"
  data_volume: "sanpo"
  data_mount: "/data"
  path_remap: {}             # {"data.yaml": "/data/...", ...}: local paths swapped for the container's
  local_data_copy: true      # copy the dataset view's tars/*.tar to the container's local disk before training
                             # (a Modal Volume reads many small files slowly); ignored when the view has no tars/
  local_data_dir: "/tmp/local_data"
  secrets: []
  sync_checkpoints: false    # also bring checkpoints home when syncing results
  prices_per_hour:           # US$, for the cost line of summary.md (modal.com/pricing)
    gpu: {B200: 6.25, H200: 4.54, H100: 3.95, A100-80GB: 2.50, A100: 2.10, L40S: 1.95, A10: 1.10, L4: 0.80, T4: 0.59}
    cpu_core: 0.0473
    memory_gb: 0.008

test:                        # runs/<run>/test.py: scores the run's checkpoints on these splits (same code as validation)
  splits: {val: val.txt, test_human: test_human.txt, test_all: test.txt}   # split name -> image list in the view
  leaderboard_split: test_human   # runs/leaderboard.md is sorted by this split's mIoU (the SANPO paper's protocol)
  checkpoints: [best]        # only best.pt (chosen on val): never choose a checkpoint on the test set
  leaderboard_checkpoint: best   # only this checkpoint of each run appears in runs/leaderboard.md
  in_leaderboard: true       # false = this run is left out of runs/leaderboard.md (listed below it, with the reason)
  leaderboard_note: ""       # why the run is left out (e.g. "learning-rate sweep: not chosen on val")
  batch: 32
  gpu: "T4"
  cpu: 4
  memory_gb: 16

resume: auto                 # auto = continue Ultralytics' last.pt if the run is unfinished | false = start fresh
resume_force: false          # resume even though guarded settings changed
resume_guard: [model.weights, data.yaml, ultralytics.epochs, ultralytics.batch, ultralytics.imgsz,
               ultralytics.optimizer, ultralytics.lr0, ultralytics.lrf, ultralytics.cos_lr, ultralytics.nbs]
"""
DEFAULTS = yaml.safe_load(DEFAULTS_YAML)


def with_defaults(config: dict) -> dict:
    """config on top of DEFAULTS (config wins). Safe to call twice."""
    merged = _deep_merge(DEFAULTS, config or {})
    _coerce_scientific(merged)
    return merged


def config_get(config: dict, dotted: str, default=None):
    """config_get(config, "ultralytics.lr0") with a default for any missing level."""
    node = config
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def config_set(config: dict, dotted: str, value) -> None:
    """config_set(config, "data.yaml", "/data/x.yaml"), creating missing levels."""
    node = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(run_dir, config_name: str = "config.yaml", shared_name: str = "shared_config.yaml",
                verbose: bool = False, overrides: dict | None = None) -> dict:
    """DEFAULTS < training/shared_config.yaml < runs/<run>/config.yaml < overrides.

    run_dir is the run folder (training/runs/<run name>); the shared file sits two levels up, beside this
    file. The run name defaults to the folder name, which is also what the Modal app is called.
    """
    run_dir = Path(run_dir)
    shared_path = run_dir.parent.parent / shared_name
    shared = yaml.safe_load(shared_path.read_text(encoding="utf-8")) if shared_path.exists() else {}
    own_path = run_dir / config_name
    own = yaml.safe_load(own_path.read_text(encoding="utf-8")) if own_path.exists() else {}
    config = _deep_merge(_deep_merge(DEFAULTS, shared or {}), own or {})
    if overrides:
        config = _deep_merge(config, overrides)
    _coerce_scientific(config)
    if not config["experiment"].get("name"):
        config["experiment"]["name"] = run_dir.name
    if verbose:
        print_config(config)
    return config


def save_config(config: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _flatten(node: dict, prefix: str = "") -> list:
    rows = []
    for key, value in node.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            rows += _flatten(value, full_key + ".")
        else:
            rows.append((full_key, value))
    return rows


_SECTION_EMOJI = {"experiment": "🎯", "model": "🧠", "data": "🗂️", "ultralytics": "🏋️", "logging": "📝",
                  "output": "📋", "modal": "☁️"}


def print_config(config: dict):
    """Every config section as a boxed key / value block, then the raw merged YAML.

    Walks whatever sections the config has, so run-specific sections print exactly like the shared ones.
    """
    print_switches = config_get(config, "logging.print", {}) or {}
    _banner("CONFIGURATION", "⚙️")
    loose = []
    for section, value in config.items():
        if isinstance(value, dict):
            if section == "logging":                     # the print switches are noise here
                value = {key: inner for key, inner in value.items() if key != "print"}
            rows = _flatten(value)
            if rows:
                _kv(section, rows, _SECTION_EMOJI.get(section, "•"))
        else:
            loose.append((section, value))
    if loose:
        _kv("top level", loose, "🔧")
    if print_switches.get("raw_config", True):
        print()
        print("── raw merged config (verbatim) " + "─" * (WIDTH - 31))
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True).rstrip())
        _rule()


# =============================================================================
# run identity
# =============================================================================
# Every launch of a run folder gets a sequential id: run001, run002, ... It tags every CSV row and prefixes
# every file the launch writes, so artifacts from two launches (a resume, a second container) never blend.
RUN_ID_PATTERN = re.compile(r"run\d+")


def run_id_of(name: str) -> str | None:
    """Leading runNNN of a filename, e.g. run007_2026-11-14_10-00-00.txt -> run007."""
    match = re.match(r"(run\d+)", str(name))
    return match.group(1) if match else None


def scan_run_ids(results_dir) -> set[str]:
    """Every runNNN this results/ folder already shows evidence of."""
    results_dir = Path(results_dir)
    ids = set()
    for subfolder in (".run_ids", "train_config"):
        folder = results_dir / subfolder
        if folder.is_dir():
            ids |= {run_id for path in folder.glob("run*") if (run_id := run_id_of(path.name))}
    for name in ("train_log.csv", "val_log.csv"):
        path = results_dir / name
        if path.exists():
            try:                                   # a truncated log must not block a launch
                ids |= set(pd.read_csv(path, usecols=["run_id"], engine="python", on_bad_lines="skip")
                           ["run_id"].dropna().astype(str))
            except Exception:
                pass
    return {run_id for run_id in ids if RUN_ID_PATTERN.fullmatch(str(run_id))}


def next_run_number(ids) -> int:
    return max((int(str(run_id)[3:]) for run_id in ids if RUN_ID_PATTERN.fullmatch(str(run_id))), default=0) + 1


def allocate_run_id(results_dir, known=()) -> str:
    """Reserve the next free runNNN for this run folder.

    Called on THIS PC at launch, never inside the container: the number is then passed into the run.
    `known` carries ids already seen on the Modal volume, so a relaunch cannot reuse one. The reservation
    is an exclusive file create, so two launches at the same moment cannot collide either.
    """
    import platform

    results_dir = Path(results_dir)
    lock_dir = results_dir / ".run_ids"            # markers only, never synced
    lock_dir.mkdir(parents=True, exist_ok=True)
    number = next_run_number(scan_run_ids(results_dir) | {str(run_id) for run_id in known})
    while True:
        candidate = f"run{number:03d}"
        try:
            with open(lock_dir / candidate, "x", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  "
                             f"{platform.node()}  {workspace_name()}\n")
            return candidate
        except FileExistsError:                    # another launch took it a moment ago
            number += 1


def hardware_info(config: dict) -> dict:
    """What this run actually trained on, asked of the driver, not the config.

    The config records what was REQUESTED; a run that silently landed on a different GPU, or fell back to
    the CPU, would otherwise look identical in the summary.
    """
    modal_settings = config.get("modal", {}) or {}
    info = {"gpu_requested": modal_settings.get("gpu"), "side": runtime_side(),
            "cpu_requested": modal_settings.get("cpu"), "memory_gb_requested": modal_settings.get("memory_gb"),
            "batch": config_get(config, "ultralytics.batch"), "workers": config_get(config, "ultralytics.workers")}
    try:
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 1)
            info["gpu_capability"] = f"sm_{properties.major}{properties.minor}"
            info["cuda"] = torch.version.cuda
            info["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        else:
            info["gpu"] = "cpu"
    except Exception as error:                     # bookkeeping must never kill a run
        info["gpu"] = f"unknown ({type(error).__name__})"
    return info


def workspace_name() -> str:
    """A console label for the active Modal workspace. Only for the console and results/.run_ids, which
    stay on this PC; never put it in a committed file (use runtime_side() there)."""
    workspace = os.environ.get("MODAL_WORKSPACE") or os.environ.get("MODAL_PROFILE")
    if workspace:
        return workspace
    config_path = os.environ.get("MODAL_CONFIG_PATH")
    return Path(config_path).stem if config_path else runtime_side()


def runtime_side() -> str:
    """Which side of the wire this is: MODAL_TASK_ID only exists inside a Modal container."""
    return "modal-container" if os.environ.get("MODAL_TASK_ID") else "local"


def set_seed(seed: int, deterministic: bool = False):
    """Python, numpy and torch seeds (Ultralytics seeds its own training with ultralytics.seed too)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True


# =============================================================================
# logging: console capture and CSV files
# =============================================================================
class _Tee:
    """Write to the console and the capture file at once, flushing every time."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass                      # a broken capture file must never kill a run

    def flush(self):
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False


class ConsoleCapture:
    """Everything printed to the terminal (stdout and stderr) is also appended to a text file.

    `with ConsoleCapture(results_dir / "train_config", run_id):` writes <run id>_<date and time>.txt.
    The streams are restored even when the run crashes, so the traceback lands in the file too.
    """

    def __init__(self, folder, run_id: str):
        self.folder, self.run_id = Path(folder), run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = self.folder / f"{run_id}_{stamp}.txt"

    def __enter__(self):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass
        self.original_out, self.original_error = sys.stdout, sys.stderr
        self.file = open(self.path, "a", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(self.original_out, self.file), _Tee(self.original_error, self.file)
        # Ultralytics logs through `logging` handlers that were bound to the ORIGINAL streams at import;
        # point them at the tee as well, so its lines also reach the file
        self._handlers = []
        for handler in _logging.getLogger("ultralytics").handlers:
            if isinstance(handler, _logging.StreamHandler):
                self._handlers.append((handler, handler.stream))
                handler.setStream(sys.stdout)
        print(f"[capture] full console of this run -> {self.path}")
        print(f"[capture] run id: {self.run_id}  side: {runtime_side()}")
        print(f"[capture] command: {' '.join(sys.argv)}")
        print(f"[capture] started {datetime.now():%Y-%m-%d %H:%M:%S}")
        return self

    def __exit__(self, *exception):
        if exception and exception[0] is not None:
            import traceback
            traceback.print_exception(*exception)      # into the file as well
        for handler, stream in self._handlers:
            try:
                handler.setStream(stream)
            except Exception:
                pass
        sys.stdout, sys.stderr = self.original_out, self.original_error
        self.file.close()
        print(f"[capture] saved -> {self.path}")
        return False


class CSVLogger:
    """Append-only CSV, header on the first row, flushed per row so a crash keeps the log.

    The first column is always run_id. Columns that appear later (on resume, or mid-run) widen the file
    instead of being dropped: the file is rewritten once with the union header, earlier rows left blank.
    """

    def __init__(self, path: Path, run_id: str = ""):
        self.path = Path(path)
        self.run_id = run_id
        self._file = None
        self._writer = None

    def log(self, row: dict):
        row = {"run_id": self.run_id, **row}
        try:
            if self._file is None:
                new = (not self.path.exists()) or self.path.stat().st_size == 0
                fields = list(row.keys())
                if not new:                        # reuse the file's own header, never mix two
                    with open(self.path, newline="", encoding="utf-8") as handle:
                        old = next(csv.reader(handle), [])
                    if old and old != fields:
                        extra = [column for column in fields if column not in old]
                        if extra:
                            self._migrate(old, old + extra)
                            fields = old + extra
                            print(f"  📝 {self.path.name}: added {len(extra)} new column(s), "
                                  f"existing rows back-filled blank")
                        else:
                            fields = old
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "a", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=fields, restval="", extrasaction="ignore")
                if new:
                    self._writer.writeheader()
            else:
                late = [key for key in row if key not in self._writer.fieldnames]
                if late:                           # columns that only appear on a later row
                    fields = list(self._writer.fieldnames) + late
                    self._file.close()
                    self._migrate(list(self._writer.fieldnames), fields)
                    self._file = open(self.path, "a", newline="", encoding="utf-8")
                    self._writer = csv.DictWriter(self._file, fieldnames=fields, restval="",
                                                  extrasaction="ignore")
                    print(f"  📝 {self.path.name}: added {len(late)} new column(s) mid-run, "
                          f"earlier rows back-filled blank")
            self._writer.writerow(row)
            self._file.flush()
        except Exception as error:                 # logging must never crash training
            print(f"  [warn] CSV log failed: {type(error).__name__}: {error}")
            self._file, self._writer = None, None

    def _migrate(self, old: list, new: list):
        """Rewrite the CSV under a wider header, through a temporary file so a kill cannot eat it."""
        temporary = self.path.with_suffix(".migrating")
        with open(self.path, newline="", encoding="utf-8") as handle, \
                open(temporary, "w", newline="", encoding="utf-8") as output:
            reader = csv.DictReader(handle)
            writer = csv.DictWriter(output, fieldnames=new, restval="", extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
        temporary.replace(self.path)

    def close(self):
        try:
            if self._file:
                self._file.close()
        except Exception:
            pass
        self._file, self._writer = None, None


def _json_safe(value):
    """Make a value JSON-valid: NaN / inf -> null, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _safe(label: str, function, *arguments, **keyword_arguments):
    """Run a function but never let it kill a run: warn and carry on."""
    try:
        return function(*arguments, **keyword_arguments)
    except Exception as error:
        print(f"  ⚠️  {label} failed (continuing): {type(error).__name__}: {error}")
        return None


# =============================================================================
# environment
# =============================================================================
def print_environment(config: dict):
    """Hardware and software this run is actually executing on."""
    import platform

    _banner("ENVIRONMENT", "\U0001F5A5\uFE0F")
    rows = [("host", platform.node()),
            ("platform", f"{platform.system()} {platform.release()}"),
            ("python", platform.python_version()),
            ("torch", torch.__version__),
            ("cuda (torch)", torch.version.cuda or "cpu build")]
    try:
        import ultralytics
        rows.append(("ultralytics", ultralytics.__version__))
    except Exception:
        rows.append(("ultralytics", "NOT INSTALLED"))
    rows.append(("numpy / pandas", f"{np.__version__} / {pd.__version__}"))
    if cv2 is not None:
        rows.append(("opencv", cv2.__version__))
    _kv("software", rows, "\U0001F4E6")

    cpu_rows = [("logical cores", os.cpu_count())]
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_rows += [("physical cores", psutil.cpu_count(logical=False)),
                     ("RAM total", f"{memory.total / 1024**3:.1f} GiB"),
                     ("RAM available", f"{memory.available / 1024**3:.1f} GiB")]
    except ImportError:
        cpu_rows.append(("RAM", "install psutil for memory details"))
    try:                                   # cpu.max = "quota period"; quota / period = real cores
        with open("/sys/fs/cgroup/cpu.max") as handle:
            quota, period = handle.read().split()
            if quota != "max":
                cpu_rows.append(("container CPU limit", f"{int(quota) / int(period):.2f} cores"))
    except Exception:
        pass
    try:                                   # the cgroup limit is what Modal actually gives
        with open("/sys/fs/cgroup/memory.max") as handle:
            value = handle.read().strip()
            if value.isdigit():
                cpu_rows.append(("container RAM limit", f"{int(value) / 1024**3:.1f} GiB"))
    except Exception:
        pass
    _kv("cpu / memory", cpu_rows, "\U0001F9EE")

    if torch.cuda.is_available():
        rows = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            rows.append([index, properties.name, f"{properties.total_memory / 1024**3:.1f} GiB",
                         f"{properties.major}.{properties.minor}", properties.multi_processor_count,
                         f"{_peak_tflops(properties.name):.0f}" if _peak_tflops(properties.name) else "?"])
        _table("gpu", ["#", "device", "memory", "capability", "SMs", "peak TFLOP/s (fp16)"], rows, "\U0001F3AE")
    else:
        _kv("gpu", [("available", "NO - running on CPU")], "\U0001F3AE")


# =============================================================================
# compute diagnostics: FLOPs, peak throughput, GPU busy
# =============================================================================
# Dense (non-sparse) tensor-core throughput at the precision mixed precision actually uses (fp16 / bf16
# with fp32 accumulate). Vendor headline numbers often quote 2x these through sparsity.
# Order matters: the first key contained in the device name wins (L40S before L40 before L4).
_PEAK_TFLOPS = {
    "B200": 2250.0, "B100": 1800.0, "H200": 989.0, "H100": 989.0,
    "A100": 312.0, "L40S": 362.0, "L40": 181.0, "L4": 121.0, "A10G": 125.0, "A10": 125.0,
    "A6000": 155.0, "6000 Ada": 364.0, "V100": 125.0, "P100": 18.7, "T4": 65.0,
    "5090": 209.5, "4090": 165.2, "4080": 97.5, "4070": 58.0, "4060": 60.5,
    "3090": 71.0, "3080": 59.5,
}


def _peak_tflops(name: str) -> float:
    """Peak dense fp16 / bf16 TFLOP/s of this device, 0.0 when there is no figure for it."""
    for key, value in _PEAK_TFLOPS.items():
        if key.lower() in (name or "").lower():
            return value
    return 0.0


def _gpu_busy() -> tuple:
    """(SM busy %, memory controller busy %) from NVML, (None, None) when unavailable.

    Both are TIME-based: the share of the sample window in which any kernel ran, and in which the memory
    controller was reading or writing. A kernel using 3% of the tensor cores still reports ~100% SM busy,
    which is exactly why MFU is printed beside them.
    """
    try:
        return torch.cuda.utilization(), torch.cuda.memory_usage()
    except Exception:
        return None, None


def measure_flops_per_image(model, height: int, width: int, device) -> float:
    """FLOPs of ONE image's forward pass at the training input size, counted once on a zero image.

    Counted, not looked up: published GFLOP figures are at 640x640, while training runs at its own size.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return 0.0
    was_training = model.training
    try:
        model.eval()
        image = torch.zeros(1, 3, int(height), int(width), device=device)
        with torch.no_grad():
            counter = FlopCounterMode(display=False)
            with counter:
                model(image)
        return float(counter.get_total_flops())
    except Exception as error:                     # a diagnostic must never kill a training run
        print(f"  [warn] could not count FLOPs: {type(error).__name__}: {error}")
        return 0.0
    finally:
        if was_training:
            model.train()


# =============================================================================
# diagnostics: update-to-weight ratio, output scale
# =============================================================================
class UpdateToWeightRatio:
    """log10 of the update-to-weight ratio of each weight tensor (the Karpathy diagnostic).

    Snapshots the weights BEFORE the optimizer step and diffs them AFTER, which is the real update whatever
    the optimizer (MuSGD, momentum, weight decay) did to produce it. Roughly -3 is healthy for layers that
    learn from scratch; a pretrained trunk that is being fine-tuned should only drift (-5 to -6). Much
    lower and the layer is effectively frozen; much higher and the step is large against the weights.
    Only 2D+ weights are tracked: biases and norms have their own scales and read as noise.
    """

    def __init__(self, model, every: int = 200, run_id: str = ""):
        self.every = int(every)
        self.run_id = run_id
        self.parameters = [(name, parameter) for name, parameter in model.named_parameters()
                           if parameter.requires_grad and parameter.ndim >= 2]
        # the semantic head is the model's last module; its index is the highest "model.<n>." prefix
        indexes = [int(match.group(1)) for name, _ in self.parameters
                   if (match := re.match(r"model\.(\d+)\.", name))]
        self.head_prefix = f"model.{max(indexes)}." if indexes else None
        self.snapshot, self.rows = None, []

    def is_head(self, name: str) -> bool:
        return bool(self.head_prefix) and name.startswith(self.head_prefix)

    def due(self, step: int) -> bool:
        return self.every > 0 and step % self.every == 0

    def before(self):
        self.snapshot = [parameter.detach().clone() for _, parameter in self.parameters]

    def after(self, epoch: int, step: int):
        if self.snapshot is None:
            return None
        readings = []
        for (name, parameter), old in zip(self.parameters, self.snapshot):
            update = (parameter.detach() - old).float().std().item()
            weight = parameter.detach().float().std().item()
            if weight > 0 and update > 0:
                readings.append((name, math.log10(update / weight)))
        self.snapshot = None
        for name, value in readings:
            self.rows.append({"run_id": self.run_id, "epoch": epoch, "step": step, "param": name,
                              "log10_ud": round(value, 4)})
        return readings

    def summarize(self, readings) -> str:
        if not readings:
            return ""

        def line(label, values, wanted):
            values = sorted(values)
            return ("     %-7s n=%-3d min %6.2f  mean %6.2f  median %6.2f  max %6.2f   %s"
                    % (label, len(values), values[0], sum(values) / len(values), values[len(values) // 2],
                       values[-1], wanted))

        head = [value for name, value in readings if self.is_head(name)]
        trunk = [value for name, value in readings if not self.is_head(name)]
        lines = ["  📏 update / weight ratio, log10"]
        if head:
            lines.append(line("head", head, "(want about -3)"))
        if trunk:
            lines.append(line("trunk", trunk, "(fine-tuning a pretrained trunk: -5..-6 is drift, not frozen)"))
        frozen = [name for name, value in readings if not self.is_head(name) and value < -7.0]
        hot = [name for name, value in readings if not self.is_head(name) and value > -3.0]
        if frozen:
            lines.append("     ⚠️  trunk frozen (< -7): %d tensors, e.g. %s" % (len(frozen), frozen[0]))
        if hot:
            lines.append("     ⚠️  trunk hot (> -3, may damage the pretrained features): %d tensors, e.g. %s"
                         % (len(hot), hot[0]))
        return "\n".join(lines)

    def dump(self, path: Path):
        """MERGE with what is already there (a resumed run keeps the earlier rows), never truncate."""
        if not self.rows:
            return
        try:
            frame = pd.DataFrame(self.rows)
            if path.exists():
                try:
                    frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
                except Exception as error:
                    print(f"  [warn] {path.name} unreadable, starting fresh: {type(error).__name__}: {error}")
            keys = [column for column in ("run_id", "step", "param") if column in frame.columns]
            frame = frame.drop_duplicates(subset=keys, keep="last").sort_values(keys)
            frame.to_csv(path, index=False)
            self.rows = []
        except Exception as error:
            print(f"  [warn] could not write {path.name}: {type(error).__name__}: {error}")


class OutputScaleProbe:
    """Scale of the model's output (the semantic head's class scores) every N steps.

    A forward hook on the model's last module reads the standard deviation and largest absolute value of
    the first tensor it outputs. Exploding or collapsing scores show up here long before they show in the
    loss.
    """

    def __init__(self, model, every: int = 200):
        self.every = int(every)
        self.reading, self.armed, self.handle, self.module_name = None, False, None, None
        try:
            last_name, last_module = list(model.named_children())[-1]
            inner = getattr(last_module, "__getitem__", None) and list(last_module.children())
            module = inner[-1] if inner else last_module
            self.module_name = f"{last_name}[-1]" if inner else last_name
            self.handle = module.register_forward_hook(self._hook)
        except Exception:
            self.handle = None

    @staticmethod
    def _first_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                found = OutputScaleProbe._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(output, dict):
            for item in output.values():
                found = OutputScaleProbe._first_tensor(item)
                if found is not None:
                    return found
        return None

    def _hook(self, _module, _inputs, output):
        if not self.armed:
            return
        tensor = self._first_tensor(output)
        if tensor is not None and tensor.is_floating_point():
            tensor = tensor.detach().float()
            self.reading = (float(tensor.std()), float(tensor.abs().max()))
        self.armed = False

    def arm(self, step: int):
        self.armed = self.handle is not None and self.every > 0 and step % self.every == 0

    def read(self):
        reading, self.reading = self.reading, None
        return reading

    @staticmethod
    def row(reading) -> dict:
        return {} if not reading else {"output_std": round(reading[0], 4), "output_absmax": round(reading[1], 3)}

    def summarize(self, reading) -> str:
        if not reading:
            return ""
        return (f"  \U0001F4CF output scale of {self.module_name}: std {reading[0]:.3f}  |max| {reading[1]:.2f}")


# =============================================================================
# volume flushes and checkpoint uploads (background threads)
# =============================================================================
class AsyncPersist:
    """Publish the Modal runs volume (volume.commit) off the training thread.

    commit() is a blocking network flush; inline every N steps it stalls the GPU for its whole duration.
    The files are already written to the container's filesystem, commit only publishes them, so nothing
    is lost by doing it slightly later. One flush at a time: if one is still running when the next is due,
    skip it (the next one publishes the same files plus more).
    """

    def __init__(self, function):
        self.function, self.thread, self.skipped, self.done = function, None, 0, 0
        self.last_error = None

    def __call__(self, blocking: bool = False):
        if self.function is None:
            return
        if blocking:
            self.join()
            self._run()
            return
        if self.thread is not None and self.thread.is_alive():
            self.skipped += 1
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.function()
            self.done += 1
        except Exception as error:                 # a failed flush must not kill training
            self.last_error = f"{type(error).__name__}: {error}"

    def join(self, timeout: float = 300.0):
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout)


UPLOAD_POINTER = Path.home() / ".ml_utils_upload_modal_config"   # one line: a Modal config path


def resolve_upload_modal_config(value: str) -> str:
    """`modal_config: "local"` reads the real path from ~/.ml_utils_upload_modal_config, a file outside the
    repo, so no workspace-specific path is ever committed. Any other value is used as given."""
    if value != "local":
        return value
    if not UPLOAD_POINTER.exists():
        raise SystemExit(f"upload_checkpoints.modal_config is 'local' but {UPLOAD_POINTER} does not exist: "
                         f"write the Modal config path into it")
    return UPLOAD_POINTER.read_text(encoding="utf-8").strip()


class CheckpointUploader:
    """Copy freshly written checkpoints of a LOCAL run to a Modal volume (output.upload_checkpoints:
    {modal_config, volume, patterns}).

    Each file is first copied into results/_upload_staging (so a checkpoint rewritten mid-upload cannot
    corrupt it), then `modal volume put --force` runs in a subprocess whose only Modal config is the one
    given, on one background thread. A failed upload never stops training.
    """

    def __init__(self, spec: dict, run_name: str, results_dir: Path, relative_weights_dir: str,
                 default_volume: str):
        import fnmatch
        import queue

        self.spec, self.run = spec, run_name
        self.volume = spec.get("volume") or default_volume
        self.relative = relative_weights_dir
        self.patterns = spec.get("patterns", ["best.pt", "last.pt"])
        self.config_path = str(Path(os.path.expanduser(resolve_upload_modal_config(spec["modal_config"]))))
        self.stage = Path(results_dir) / "_upload_staging"
        self.stage.mkdir(parents=True, exist_ok=True)
        self._match = lambda name: any(fnmatch.fnmatch(name, pattern) for pattern in self.patterns)
        self.queue, self.pending, self.lock = queue.Queue(), {}, threading.Lock()
        self.done = self.failed = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"  ☁️  checkpoint upload ON: {', '.join(self.patterns)} -> Modal volume "
              f"'{self.volume}':/{self.run}/{self.relative}/ ({self.config_path})", flush=True)

    def offer(self, path: Path) -> None:
        import shutil

        path = Path(path)
        if not path.exists() or not self._match(path.name):
            return
        staged = self.stage / path.name
        try:
            shutil.copy2(path, staged)
        except OSError as error:
            print(f"  ⚠️  upload staging failed for {path.name}: {error}", flush=True)
            return
        with self.lock:
            fresh = path.name not in self.pending
            self.pending[path.name] = staged
        if fresh:
            self.queue.put(path.name)

    def _loop(self):
        import subprocess

        while True:
            name = self.queue.get()
            if name is None:
                return
            with self.lock:
                staged = self.pending.pop(name, None)
            if staged is None or not staged.exists():
                continue
            size = staged.stat().st_size
            remote = f"/{self.run}/{self.relative}/{name}"
            environment = dict(os.environ, MODAL_CONFIG_PATH=self.config_path, PYTHONUTF8="1")
            start = time.time()
            result = subprocess.run([sys.executable, "-m", "modal", "volume", "put", "--force", self.volume,
                                     str(staged), remote], capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", env=environment)
            seconds = time.time() - start
            if result.returncode == 0:
                self.done += 1
                print(f"  ☁️  uploaded {name} ({size / 1e6:.1f} MB) -> {self.volume}:{remote} in {seconds:.0f}s",
                      flush=True)
            else:
                self.failed += 1
                tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["?"]
                print(f"  ⚠️  upload FAILED for {name} after {seconds:.0f}s: {tail[0][:200]}", flush=True)

    def join(self) -> None:
        """Wait for every queued upload, then stop the thread."""
        self.queue.put(None)
        self.thread.join()
        print(f"  ☁️  checkpoint uploads finished: {self.done} uploaded, {self.failed} failed", flush=True)


# =============================================================================
# Ultralytics: dataset with a configurable mask path, trainer subclass
# =============================================================================
def make_dataset_class(config: dict):
    """The semantic dataset class used for training, validation and the test evaluation.

    Ultralytics' SemanticDataset plus: masks found with the dataset YAML's image_dir_name / masks_dir rule,
    validation and test frames at the training scale (data.validate_at_training_scale), and one label cache
    file per image list. Returns the class (Ultralytics builds it through data.build.SemanticDataset).
    """
    from ultralytics.data.dataset import SemanticDataset

    class MaskPathSemanticDataset(SemanticDataset):
        def get_label_files(self) -> list[str]:
            image_dir_name = self.data.get("image_dir_name")
            if not image_dir_name:
                return super().get_label_files()
            masks_dir = str(self.data.get("masks_dir", "masks")).replace("/", os.sep).strip(os.sep)
            marker, replacement = f"{os.sep}{image_dir_name}{os.sep}", f"{os.sep}{masks_dir}{os.sep}"
            mask_files = []
            for image_file in self.im_files:
                path = os.path.normpath(os.fspath(image_file))
                if marker not in path:
                    raise ValueError(f"image path has no '{marker}' folder to map to its mask: {path}")
                head, tail = path.rsplit(marker, 1)
                mask_files.append(head + replacement + os.path.splitext(tail)[0] + ".png")
            self.mask_files = mask_files
            return mask_files

        def _validation_at_training_scale(self) -> bool:
            return (not self.augment) and bool(config_get(config, "data.validate_at_training_scale", True))

        def load_image(self, i, rect_mode=True):
            # Ultralytics 8.4.208 sizes TRAINING frames by their SHORT side (imgsz 352 -> 626x352) but VALIDATION
            # frames by their LONG side (352x198): objects then look about 1.8x smaller in validation than in
            # training. With data.validate_at_training_scale (default true), validation frames are sized by the
            # short side too (626x352), the same scale the model trains on.
            if self._validation_at_training_scale():
                return super(SemanticDataset, self).load_image(i, rect_mode=rect_mode, resize_short=True)
            return super().load_image(i, rect_mode=rect_mode)

        def set_rectangle(self):
            # validation batches: Ultralytics sets each batch shape from the LONG side (352 x 224 for 16:9);
            # with validation at the training scale the SHORT side is imgsz and the long side is rounded up to
            # the stride (626x352 frames -> 640x352 batches; the 14 padding columns are labeled 255 = ignored)
            super().set_rectangle()
            if not self._validation_at_training_scale():
                return
            shapes = []
            for batch_number in range(int(self.batch[-1]) + 1):
                ratios = np.array([label["ratio_hw"] for label, index in zip(self.labels, self.batch)
                                   if index == batch_number])
                if ratios.max() < 1:                   # landscape: height = imgsz, width = imgsz / (h / w)
                    shapes.append([1.0, 1.0 / ratios.min()])
                elif ratios.min() > 1:                 # portrait: width = imgsz
                    shapes.append([ratios.max(), 1.0])
                else:
                    shapes.append([max(1.0, ratios.max()), max(1.0, 1.0 / ratios.min())])
            self.batch_shapes = (np.ceil(np.array(shapes) * self.imgsz / self.stride).astype(int) * self.stride)

        def get_labels(self):
            labels = super().get_labels()
            for label in labels:                       # kept for set_rectangle (Ultralytics pops "shape")
                height, width = label["shape"]
                label["ratio_hw"] = height / width
            return labels

        def _load_or_scan_cache(self, cache_path, cache_hash):
            # Ultralytics names the label cache after the folder of the FIRST mask (masks/train.cache). Our
            # validation walks also live under the official "train" folder, so train and val wrote the same
            # file and rebuilt it on every launch. Name it after the image list file instead (train.cache,
            # val.cache, test_human.cache). The hash check inside Ultralytics is unchanged.
            image_list = self.img_path if isinstance(self.img_path, (str, Path)) else None
            if image_list is not None and str(image_list).endswith(".txt"):
                cache_path = Path(cache_path).with_name(f"{Path(image_list).stem}.cache")
            return super()._load_or_scan_cache(cache_path, cache_hash)

    return MaskPathSemanticDataset


def make_trainer_class(monitor):
    """An Ultralytics semantic trainer that (1) finds masks with the dataset YAML's own path rule and
    (2) reports every optimizer step (gradient norm, update-to-weight ratio) to the monitor.

    Mask paths: Ultralytics finds a mask by replacing the LAST "/images/" folder of the image path with
    "/<masks_dir>/". Our frames live under "frames/", not "images/", so when the dataset YAML has an
    `image_dir_name` key, the LAST "/<image_dir_name>/" is replaced with "/<masks_dir>/" instead (masks_dir
    may contain several folders, e.g. "views/semantic_v1/masks"). Without that key, Ultralytics' own rule is
    used unchanged.
    """
    from ultralytics.data import build as ultralytics_build
    from ultralytics.data.dataset import SemanticDataset
    from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer

    MaskPathSemanticDataset = make_dataset_class(monitor.config)

    class BlindGuideSemanticTrainer(SemanticSegmentationTrainer):
        def __init__(self, *arguments, **keyword_arguments):
            super().__init__(*arguments, **keyword_arguments)
            # remove the outside tracking integrations listed in logging.disable_integrations; their setup
            # runs at on_pretrain_routine_end, which comes after this, so nothing of theirs has run yet
            disabled = [name.lower() for name in (monitor.config["logging"].get("disable_integrations") or [])]
            removed = set()
            for event, functions in self.callbacks.items():
                kept = []
                for function in functions:
                    module = (getattr(function, "__module__", "") or "").lower()
                    match = next((name for name in disabled if module.endswith("callbacks." + name)), None)
                    if match:
                        removed.add(match)
                    else:
                        kept.append(function)
                self.callbacks[event] = kept
            if removed:
                print(f"[setup] Ultralytics integrations switched off for this run: {', '.join(sorted(removed))}")

        def build_dataset(self, img_path, mode="train", batch=None):
            # build_yolo_dataset looks the dataset class up by name when it runs, so swapping the name
            # for the duration of this one call makes Ultralytics build our subclass with all its own
            # arguments (augmentation, image size, cache ...) unchanged
            original = ultralytics_build.SemanticDataset
            ultralytics_build.SemanticDataset = MaskPathSemanticDataset
            try:
                return super().build_dataset(img_path, mode=mode, batch=batch)
            finally:
                ultralytics_build.SemanticDataset = original

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            # Same as Ultralytics' get_dataloader + build_dataloader (single GPU), with the settings that
            # Ultralytics fixes internally taken from the dataloader: section of the config instead
            from ultralytics.data.build import InfiniteDataLoader, seed_worker

            settings = monitor.config.get("dataloader") or {}
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            training = mode == "train"
            shuffle = training
            if getattr(dataset, "rect", False) and shuffle:
                shuffle = False                       # rectangular batches cannot be shuffled
            batch = min(batch_size, len(dataset))
            batches = math.ceil(len(dataset) / batch) if batch else 0
            workers = int(self.args.workers if training else (settings.get("val_workers") or self.args.workers))
            if settings.get("cap_workers_at_cpu_count"):
                workers = min(workers, os.cpu_count() or workers)
            workers = min(workers, 0 if batches <= 1 else batches)   # never more workers than batches
            prefetch = int(settings.get("prefetch_factor", 4) if training else settings.get("val_prefetch_factor", 2))
            pin_memory = bool(settings.get("pin_memory", True)) and self.device.type == "cuda"
            drop_last = bool(settings.get("drop_last", False)) and training and len(dataset) > batch
            seed = torch.initial_seed()
            generator = torch.Generator()
            generator.manual_seed((6148914691236517205 - 1 + seed) % (1 << 64))   # Ultralytics' seeding, RANK -1
            print(f"[data] {mode} loader: {len(dataset):,} images, batch {batch}, {batches} batches, workers {workers}, "
                  f"prefetch {prefetch if workers else '-'}, pin_memory {pin_memory}, drop_last {drop_last}, "
                  f"shuffle {shuffle}, cache {self.args.cache}")
            if not training and getattr(dataset, "batch_shapes", None) is not None:
                shapes = sorted({f"{int(width)}x{int(height)}" for height, width in dataset.batch_shapes})
                print(f"[data] {mode} batch sizes (width x height): {', '.join(shapes)}"
                      f"{' (training scale)' if config_get(monitor.config, 'data.validate_at_training_scale', True) else ''}")
            view_size = (self.data or {}).get("image_size")   # set by make_resized_view_modal.py
            if view_size and int(view_size) != int(self.args.imgsz):
                print(f"  ⚠️  the dataset view was resized for imgsz {view_size} but training uses imgsz "
                      f"{self.args.imgsz}: Ultralytics resizes the frames a second time (use a view made "
                      f"for imgsz {self.args.imgsz})")
            return InfiniteDataLoader(dataset=dataset, batch_size=batch, shuffle=shuffle, num_workers=workers,
                                      prefetch_factor=prefetch if workers > 0 else None, pin_memory=pin_memory,
                                      collate_fn=getattr(dataset, "collate_fn", None), worker_init_fn=seed_worker,
                                      generator=generator, drop_last=drop_last)

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            monitor.note_batch(self, batch)
            return batch

        def optimizer_step(self):
            # the gradient norm as Ultralytics' own clipping (max_norm=10) sees it: read before its step,
            # on the scaled gradients, then divided by the loss scale
            gradient_norm = None
            try:
                gradients = [parameter.grad.detach().float() for parameter in self.model.parameters()
                             if parameter.grad is not None]
                if gradients:
                    scale = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
                    norms = torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
                    gradient_norm = float(torch.linalg.vector_norm(norms)) / max(scale, 1e-12)
            except Exception:
                gradient_norm = None
            monitor.before_optimizer_step(self)
            super().optimizer_step()
            monitor.after_optimizer_step(self, gradient_norm)

    return BlindGuideSemanticTrainer


# =============================================================================
# the monitor: Ultralytics callbacks -> console lines, CSV logs, summary, Modal flushes
# =============================================================================
def group_scores(matrix, names: dict, class_groups: dict) -> dict:
    """IoU of groups of classes (e.g. the SANPO paper's safe / not safe / obstacle), from a confusion matrix.

    matrix: Ultralytics' semantic confusion matrix [nc, nc], rows = true class, columns = predicted class
    (ignored pixels are not in it). The classes of each group are merged (rows and columns summed), then
    IoU = merged diagonal / (row sum + column sum - diagonal). mIoU = mean over the groups with true pixels.
    class_groups: {set name: {group name: [class names]}}; a class in no group of a set is left out of that set.
    Returns {"val_group_<set>_mIoU": float, "val_group_<set>_<group>": float, ...}.
    """
    if matrix is None or not class_groups:
        return {}
    is_tensor = torch is not None and torch.is_tensor(matrix)
    counts = matrix.detach().double().cpu().numpy() if is_tensor else np.asarray(matrix, dtype=float)
    index_of = {name: index for index, name in names.items()}
    scores = {}
    for set_name, groups in class_groups.items():
        unknown = [name for members in groups.values() for name in members if name not in index_of]
        if unknown:
            raise ValueError(f"class_groups.{set_name}: unknown class names {unknown}")
        group_indices = [[index_of[name] for name in members] for members in groups.values()]
        size = len(group_indices)
        merged = np.zeros((size, size))
        for row_group, rows in enumerate(group_indices):
            for column_group, columns in enumerate(group_indices):
                merged[row_group, column_group] = counts[np.ix_(rows, columns)].sum()
        intersection = np.diag(merged)
        union = merged.sum(1) + merged.sum(0) - intersection
        present = merged.sum(1) > 0
        for group_number, group_name in enumerate(groups):
            safe_name = str(group_name).replace(" ", "_")
            scores[f"val_group_{set_name}_{safe_name}"] = (round(float(intersection[group_number] / union[group_number]), 6)
                                                           if present[group_number] else "")
        iou = intersection[present] / union[present] if present.any() else np.array([0.0])
        scores[f"val_group_{set_name}_mIoU"] = round(float(iou.mean()), 6)
    return scores


class TrainingMonitor:
    """Everything printed and logged around Ultralytics' training loop.

    Ultralytics calls these methods at fixed points (its callbacks) and through the trainer subclass:
        on_pretrain_routine_end   data and model are ready: model and data summary
        on_train_start            just before the first epoch: steps, FLOPs probe setup
        on_train_epoch_start      epoch header
        on_train_batch_start      end of the data wait for this batch
        note_batch                a batch reached the model (its size and image shape)
        before / after_optimizer_step   gradient norm, update-to-weight ratio, one train_log.csv row
        on_train_batch_end        compute time of the batch; console lines every N steps
        on_fit_epoch_end          after validation: val_log.csv row, validation block, summary JSON
        on_model_save             checkpoint written: offer it to the uploader
        on_train_end              training over
    """

    def __init__(self, config: dict, results_dir: Path, run_id: str, persist=None, uploader=None):
        self.config, self.results_dir, self.run_id = config, Path(results_dir), run_id
        self.print_switches = config["logging"]["print"]
        self.unit = config["logging"].get("unit") or "img"
        self.persist = persist or AsyncPersist(None)
        self.uploader = uploader
        output = config["output"]
        self.train_log = CSVLogger(self.results_dir / output["train_log_csv"], run_id)
        self.val_log = CSVLogger(self.results_dir / output["val_log_csv"], run_id)
        self.ud_path = self.results_dir / "ud_ratio.csv"
        self.summary_path = self.results_dir / output["summary_json"]
        self.start_wall = time.time()
        self.step = 0                        # optimizer steps done by this launch (plus earlier epochs on resume)
        self.images_done = 0
        self.batch_start = None
        self.last_batch_end = None
        self.window = {"images": 0, "compute": 0.0, "data": 0.0, "wall_start": None}
        self.rolling = deque(maxlen=20)      # (images, wall seconds, data-wait seconds) of the last 20 steps
        self.batch_images = 0
        self.batch_shape = None
        self.flops_per_image = None
        self.peak = 0.0
        self.ud = None
        self.probe = None
        self.gradient_norms = deque(maxlen=50)
        self.clip_hits = 0                   # finite steps whose gradient norm was above Ultralytics' clip (10)
        self.finite_steps = 0
        self.overflow_steps = 0              # mixed precision: steps the gradient scaler skipped (inf / nan)
        self.previous_loss = None
        self.last_print = (None, None)
        self.history = []                    # one dict per validation
        self.best = None
        self.best_epoch = None
        self.steps_per_epoch = None
        self.total_steps = None
        self.pending_step = None             # set by after_optimizer_step, consumed by on_train_batch_end
        self.debug_saved_step = None         # last step whose batch was saved as a debug image

    # ---- registration --------------------------------------------------------------------------------
    def register(self, model):
        """Add every callback of this monitor to an Ultralytics model (YOLO object)."""
        for event in ("on_pretrain_routine_end", "on_train_start", "on_train_epoch_start", "on_train_batch_start",
                      "on_train_batch_end", "on_fit_epoch_end", "on_model_save", "on_train_end"):
            model.add_callback(event, getattr(self, event))

    # ---- setup -----------------------------------------------------------------------------------------
    def on_pretrain_routine_end(self, trainer):
        model = trainer.model
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        names = trainer.data.get("names", {})
        if self.print_switches.get("model", True):
            _banner("MODEL AND DATA", "\U0001F9E0")
            _kv("model", [
                ("weights", self.config["model"]["weights"]),
                ("class", type(model).__name__),
                ("classes (outputs)", len(names)),
                None,
                ("total params", f"{total:,}  ({total / 1e6:.2f} M)"),
                ("trainable", f"{trainable:,}  ({100 * trainable / max(total, 1):.1f}%)"),
                ("param memory", f"{sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.1f} MB"),
            ], "\U0001F9F1")
            rows = [("dataset YAML", trainer.args.data)]
            try:
                rows += [("train images", f"{len(trainer.train_loader.dataset):,}"),
                         ("validation images", f"{len(trainer.test_loader.dataset):,}")]
            except Exception:
                pass
            rows += [("masks folder", trainer.data.get("masks_dir")),
                     ("image folder replaced", trainer.data.get("image_dir_name", "images (Ultralytics default)")),
                     ("label mapping", f"{len(trainer.data.get('label_mapping') or {})} entries")]
            _kv("data", rows, "\U0001F5C2️")
            if names:
                _table("classes", ["id", "name"], [[index, name] for index, name in names.items()], "\U0001F3F7️")

    def on_train_start(self, trainer):
        self.start_wall = time.time()
        batches_per_epoch = len(trainer.train_loader)
        self.steps_per_epoch = max(1, -(-batches_per_epoch // max(int(trainer.accumulate), 1)))
        self.total_steps = self.steps_per_epoch * int(trainer.epochs)
        self.step = self.steps_per_epoch * int(trainer.start_epoch)       # a resumed run continues the count
        every = int(self.config["logging"].get("diagnostics_every") or 0)
        model = getattr(trainer.model, "module", trainer.model)
        self.ud = UpdateToWeightRatio(model, every, self.run_id)
        self.probe = OutputScaleProbe(model, every)
        if torch.cuda.is_available():
            self.peak = _peak_tflops(torch.cuda.get_device_name(0))
        print(f"\n[setup] batches/epoch={batches_per_epoch}  accumulate={trainer.accumulate}  "
              f"optimizer steps/epoch={self.steps_per_epoch}  total steps={self.total_steps}  "
              f"start epoch={int(trainer.start_epoch) + 1}/{trainer.epochs}")
        if every:
            print(f"[setup] update/weight ratio on {len(self.ud.parameters)} weight tensors and output scale probe "
                  f"on {self.probe.module_name}, every {every} steps")

    def on_train_epoch_start(self, trainer):
        print(f"\n{'═' * WIDTH}\n  📅 epoch {trainer.epoch + 1}/{trainer.epochs}"
              f"{'   (mosaic off from here)' if trainer.epoch >= trainer.epochs - trainer.args.close_mosaic else ''}"
              f"\n{'═' * WIDTH}")

    # ---- per batch / per step --------------------------------------------------------------------------
    def on_train_batch_start(self, trainer):
        now = time.perf_counter()
        if self.window["wall_start"] is None:
            self.window["wall_start"] = now
        if self.last_batch_end is not None:
            self.window["data"] += now - self.last_batch_end         # waited for the data loader
        self.batch_start = now
        if self.probe is not None:
            self.probe.arm(self.step + 1)

    def note_batch(self, trainer, batch):
        images = batch.get("img") if isinstance(batch, dict) else None
        if torch.is_tensor(images):
            self.batch_images = int(images.shape[0])
            self.batch_shape = (int(images.shape[-2]), int(images.shape[-1]))
            every = int(self.config["logging"].get("debug_images_every") or 0)
            upcoming_step = self.step + 1
            if every and (upcoming_step == 1 or upcoming_step % every == 0) and upcoming_step != self.debug_saved_step:
                self.debug_saved_step = upcoming_step
                try:
                    # save_debug_batch takes: the batch the model receives, trainer, step; returns: nothing
                    # (writes results/debug/<run_id>_step<N>.png and .json in a background thread)
                    self.save_debug_batch(batch, trainer, upcoming_step)
                except Exception as error:            # a debug image must never stop training
                    print(f"  ⚠️  debug image failed: {type(error).__name__}: {error}")

    def save_debug_batch(self, batch, trainer, step: int) -> None:
        """Save the first images of a training batch EXACTLY as the model receives them.

        Called from preprocess_batch AFTER Ultralytics' own preprocessing, so the tensor is the model input:
        every augmentation applied, already on the GPU, float in [0, 1], RGB. It is stored as PNG (lossless) after
        x * 255 rounded: the frames are 8-bit, so this recovers the exact values (the largest rounding error is
        written to the .json). One PNG per save, one row per image:
            model input | training mask in colour (black = 255 ignored) | the two blended
        The .json lists the source files, tensor shape / dtype / value range and the class colours.
        """
        count = int(self.config["logging"].get("debug_images_per_save") or 4)
        images = batch["img"][:count].detach().float().cpu()                     # [N, 3, H, W], 0..1, RGB
        masks = batch.get("semantic_mask")
        masks = masks[:count].detach().cpu().numpy() if torch.is_tensor(masks) else None
        files = [str(name) for name in (batch.get("im_file") or [])[:count]]
        names = dict(trainer.data.get("names") or {})
        epoch = int(trainer.epoch) + 1

        def write():
            scaled = images * 255.0
            rounding_error = float((scaled - scaled.round()).abs().max())
            pixels = scaled.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()   # N, H, W, 3 RGB
            palette = np.random.default_rng(0).integers(40, 256, size=(256, 3), dtype=np.uint8)
            palette[255] = 0                                                       # ignored pixels: black
            rows = []
            for index in range(pixels.shape[0]):
                image = pixels[index]
                panels = [image]
                if masks is not None:
                    mask = masks[index].astype(np.int64).clip(0, 255)
                    if mask.shape != image.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST).astype(np.int64)
                    coloured = palette[mask]
                    panels += [coloured, (0.5 * image + 0.5 * coloured).astype(np.uint8)]
                rows.append(np.concatenate(panels, axis=1))
            grid = np.concatenate(rows, axis=0)
            folder = self.results_dir / (self.config["output"].get("debug_dir") or "debug")
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"{self.run_id}_step{step:06d}_epoch{epoch:03d}"
            cv2.imwrite(str(folder / f"{stem}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            info = {"run_id": self.run_id, "step": step, "epoch": epoch, "source_files": files,
                    "tensor_shape": list(batch_shape), "tensor_dtype": tensor_dtype,
                    "value_range": [float(images.min()), float(images.max())],
                    "largest_rounding_error_x255": rounding_error,
                    "mask_shape": list(masks.shape) if masks is not None else None,
                    "columns": ["model input (RGB)", "mask (black = ignored)", "blend"] if masks is not None
                               else ["model input (RGB)"],
                    "class_colours_rgb": {str(class_id): palette[int(class_id)].tolist() for class_id in names}}
            (folder / f"{stem}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
            # published to the runs volume by the regular commits (output.commit_every_steps) and at the end

        batch_shape, tensor_dtype = tuple(batch["img"].shape), str(batch["img"].dtype)
        threading.Thread(target=write, daemon=True).start()

    def before_optimizer_step(self, trainer):
        if self.ud is not None and self.ud.due(self.step + 1):
            self.ud.before()

    def after_optimizer_step(self, trainer, gradient_norm):
        self.step += 1
        readings = self.ud.after(trainer.epoch + 1, self.step) if self.ud is not None else None
        if gradient_norm is not None:
            if math.isfinite(gradient_norm):
                self.gradient_norms.append(gradient_norm)
                self.finite_steps += 1
                if gradient_norm > 10.0:              # Ultralytics clips at max_norm=10
                    self.clip_hits += 1
            else:
                # an inf / nan gradient under mixed precision: the gradient scaler skips this step and lowers
                # its scale (normal for a few steps at the start)
                self.overflow_steps += 1
        self.pending_step = {"gradient_norm": gradient_norm, "ud": readings}

    def on_train_batch_end(self, trainer):
        now = time.perf_counter()
        if self.batch_start is not None:
            self.window["compute"] += now - self.batch_start
        self.window["images"] += self.batch_images
        self.images_done += self.batch_images
        self.last_batch_end = now
        if self.pending_step is None:                 # gradient accumulation: no optimizer step this batch
            return
        step_info, self.pending_step = self.pending_step, None
        self._log_step(trainer, step_info, now)

    def _log_step(self, trainer, step_info, now):
        window = self.window
        wall = max(now - (window["wall_start"] or now), 1e-9)
        self.rolling.append((window["images"], wall, window["data"]))
        compute_rate = window["images"] / max(window["compute"], 1e-9)
        rolling_images = sum(item[0] for item in self.rolling)
        rolling_wall = max(sum(item[1] for item in self.rolling), 1e-9)
        real_rate = rolling_images / rolling_wall
        wait_percent = 100.0 * sum(item[2] for item in self.rolling) / rolling_wall
        elapsed = time.time() - self.start_wall
        overall_rate = self.images_done / max(elapsed, 1e-9)
        self.window = {"images": 0, "compute": 0.0, "data": 0.0, "wall_start": now}

        losses = {}
        try:
            for key, value in (trainer.loss_items or {}).items():
                losses[f"loss_{key}"] = round(float(value), 6)
        except Exception:
            pass
        # loss per image = the sum of the loss parts. (Ultralytics' own trainer.loss is that sum multiplied
        # by the batch size, which is what is back-propagated; logged as loss_batch_scaled.)
        total_loss = sum(losses.values()) if losses else None
        scaled_loss = float(trainer.loss.detach()) if torch.is_tensor(trainer.loss) else None
        # learning rate per optimizer parameter group, by Ultralytics' group name (weight, bn, bias, muon)
        group_rates = {}
        for group in trainer.optimizer.param_groups:
            group_rates.setdefault(str(group.get("param_group", f"group{len(group_rates)}")), float(group["lr"]))
        learning_rates = list(group_rates.values())
        main_rate = group_rates.get("weight", max(learning_rates))
        memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        steps_left = max(self.total_steps - self.step, 0) if self.total_steps else 0
        steps_this_launch = max(self.step - self.steps_per_epoch * int(trainer.start_epoch), 1)
        eta = elapsed / steps_this_launch * steps_left
        epoch_step = self.step - self.steps_per_epoch * trainer.epoch
        probe_reading = self.probe.read() if self.probe is not None else None
        gradient_norm = step_info["gradient_norm"]

        mfu = None
        sm_busy = memory_busy = None
        if torch.cuda.is_available() and self.print_switches.get("compute", True):
            if self.flops_per_image is None and self.batch_shape:
                self.flops_per_image = (measure_flops_per_image(getattr(trainer.model, "module", trainer.model),
                                                                 *self.batch_shape, trainer.device)
                                        if self.config["logging"].get("flops_probe", True) else 0.0)
            if self.flops_per_image and self.peak:
                # x3: the backward pass costs about twice the forward pass
                mfu = 100.0 * 3.0 * self.flops_per_image * compute_rate / 1e12 / self.peak

        self.train_log.log({
            "epoch": trainer.epoch + 1, "epoch_step": epoch_step, "global_step": self.step,
            "loss": round(total_loss, 6) if total_loss is not None else "", **losses,
            "loss_batch_scaled": round(scaled_loss, 6) if scaled_loss is not None else "",
            "lr": main_rate, **{f"lr_{name}": rate for name, rate in group_rates.items()},
            "gnorm": (round(gradient_norm, 4) if gradient_norm is not None and math.isfinite(gradient_norm)
                      else ("overflow" if gradient_norm is not None else "")),
            "units_s": round(compute_rate, 1), "real_units_s": round(real_rate, 1),
            "data_wait_pct": round(wait_percent, 1), "overall_units_s": round(overall_rate, 1),
            "units_done": self.images_done, "mfu_pct": round(mfu, 2) if mfu is not None else "",
            "gpu_mem_gb": round(memory, 2), "elapsed_s": round(elapsed, 1), **OutputScaleProbe.row(probe_reading),
            # median log10(update / weight) over the weight tensors, on the steps where it is measured
            # (logging.diagnostics_every); every tensor separately is in ud_ratio.csv
            "ud_log10_median": (round(float(np.median([value for _, value in step_info["ud"]])), 4)
                                if step_info.get("ud") else "")})
        commit_every = int(self.config["output"].get("commit_every_steps") or 0)
        if commit_every and self.step % commit_every == 0:
            self.persist()

        if step_info["ud"] and self.print_switches.get("ud_ratio", True):
            print(self.ud.summarize(step_info["ud"]))
            self.ud.dump(self.ud_path)
        if probe_reading and self.print_switches.get("scale_probe", True):
            print(self.probe.summarize(probe_reading))

        every = int(self.config["logging"].get("console_log_every") or 0)
        if not every or (self.step % every and epoch_step != 1):
            return
        if self.print_switches.get("step_line", True):
            loss_value = total_loss if total_loss is not None else float("nan")
            delta = "" if self.previous_loss is None else f" ({loss_value - self.previous_loss:+.4f})"
            self.previous_loss = loss_value
            if len(set(learning_rates)) == 1:
                learning_rate_text = f"lr={main_rate:.3e}"
            else:                                     # warmup: the bias group starts high and falls to lr0
                learning_rate_text = "lr[" + " ".join(f"{name}={rate:.2e}" for name, rate in group_rates.items()) + "]"
            recent = list(self.gradient_norms)
            gradient_text = ""
            if gradient_norm is not None:
                current = f"{gradient_norm:.2f}" if math.isfinite(gradient_norm) else "overflow (step skipped)"
                average = f"avg{len(recent)} {sum(recent) / len(recent):.2f}, " if recent else ""
                gradient_text = (f"  gnorm={current} ({average}above clip 10: "
                                 f"{100.0 * self.clip_hits / max(self.finite_steps, 1):.0f}%"
                                 + (f", {self.overflow_steps} overflow steps" if self.overflow_steps else "") + ")")
            since = ""
            if self.last_print[0] is not None and self.step > self.last_print[1]:
                since = f"  +{format_duration(elapsed - self.last_print[0])}/{self.step - self.last_print[1]}st"
            self.last_print = (elapsed, self.step)
            parts = "  ".join(f"{key[5:]}={value:.4f}" for key, value in losses.items())
            print("\n" + "─" * 70)             # the newline ends Ultralytics' progress bar line first
            print(f"  epoch {trainer.epoch + 1}/{trainer.epochs}  epoch_step: {epoch_step}/{self.steps_per_epoch}"
                  f"  global_step: {self.step}/{self.total_steps}")
            print(f"     loss={loss_value:.4f}{delta}  [{parts}]  {learning_rate_text}{gradient_text}\n")
            print(f"     elapsed={format_duration(elapsed)}  ETA={format_duration(eta)}"
                  + (f"  gpu_mem(peak)={memory:.2f}GB" if torch.cuda.is_available() else "") + since)
        if self.print_switches.get("throughput", True):
            print(f"     throughput: {compute_rate:.0f} {self.unit}/s compute | {real_rate:.0f} {self.unit}/s real "
                  f"({len(self.rolling)}-step) | data-wait {wait_percent:.0f}%")
            efficiency = 100.0 * overall_rate / max(real_rate, 1e-9)
            print(f"     ➤ END-TO-END: {overall_rate:.0f} {self.unit}/s over this launch ({efficiency:.0f}% of in-loop)"
                  f" - {self.images_done:,} {self.unit}s in {format_duration(elapsed)}")
        if torch.cuda.is_available() and self.print_switches.get("compute", True):
            sm_busy, memory_busy = _gpu_busy()
            bits = []
            if mfu is not None:
                achieved = 3.0 * self.flops_per_image * compute_rate / 1e12
                bits.append(f"MFU {mfu:.1f}% ({achieved:.1f} of {self.peak:.0f} TFLOP/s fp16)")
            if self.flops_per_image:
                height, width = self.batch_shape
                bits.append(f"{self.flops_per_image / 1e9:.2f} GFLOP/{self.unit} fwd at {width}x{height}")
            if sm_busy is not None:
                bits.append(f"SM busy {sm_busy:.0f}%")
            if memory_busy is not None:
                bits.append(f"mem ctrl busy {memory_busy:.0f}%")
            if bits:
                print("     compute:    " + " | ".join(bits) + "\n", flush=True)

    # ---- validation, checkpoints, end ------------------------------------------------------------------
    def on_fit_epoch_end(self, trainer):
        metrics = dict(trainer.metrics or {})
        if not metrics:
            return
        # Ultralytics calls this once more after training, for the final validation of best.pt, with
        # trainer.epoch temporarily one past the last epoch: logged as trigger "final_best"
        final = trainer.epoch + 1 > trainer.epochs
        row = {"epoch": min(trainer.epoch + 1, trainer.epochs), "step": self.step,
               "trigger": "final_best" if final else "epoch"}
        for key, value in metrics.items():
            try:
                row["val_" + key.split("/")[-1]] = round(float(value), 6)
            except (TypeError, ValueError):
                pass
        per_class = {}
        try:
            validator_metrics = trainer.validator.metrics
            names = validator_metrics.names
            present = set(validator_metrics.ap_class_index)
            for index, value in enumerate(validator_metrics.per_class_iou):
                name = names.get(index, str(index))
                per_class[name] = round(float(value), 6) if index in present else None
                row[f"val_iou_{name}"] = per_class[name] if per_class[name] is not None else ""
        except Exception:
            pass
        try:
            # group_scores takes: the validation confusion matrix, class names, logging.class_groups;
            # returns: {"val_group_<set>_mIoU": ..., "val_group_<set>_<group>": IoU, ...}
            row.update(group_scores(trainer.validator.metrics.matrix, trainer.validator.metrics.names,
                                    self.config["logging"].get("class_groups") or {}))
        except Exception as error:
            print(f"  ⚠️  group scores failed: {type(error).__name__}: {error}")
        fitness = row.get("val_mIoU") if final else (float(trainer.fitness) if trainer.fitness is not None else None)
        row["fitness"] = fitness
        row["epoch_time_s"] = round(float(getattr(trainer, "epoch_time", 0.0) or 0.0), 1)
        row["elapsed_s"] = round(time.time() - self.start_wall, 1)
        if not final and fitness is not None and (self.best is None or fitness > self.best):
            self.best, self.best_epoch = fitness, trainer.epoch + 1
        row["best_fitness"] = self.best
        previous = self.history[-1] if self.history else None
        self.history.append(row)
        self.val_log.log(row)

        def delta(key):
            if not previous or not isinstance(previous.get(key), (int, float)) or not isinstance(row.get(key), (int, float)):
                return ""
            return f" ({row[key] - previous[key]:+.4f})"

        print("\n" + "-" * 70)
        title = "final validation of best.pt" if final else "validation"
        print(f"  \U0001F50E [{title}] epoch={row['epoch']}/{trainer.epochs} step={self.step}  "
              f"elapsed={format_duration(time.time() - self.start_wall)}")
        scalar_keys = [key for key in row if key.startswith("val_") and not key.startswith("val_iou_")]
        for start in range(0, len(scalar_keys), 4):
            print("     📊 " + "  ".join(f"{key[4:]}={row[key]:.4f}{delta(key)}" for key in scalar_keys[start:start + 4]
                                       if isinstance(row[key], (int, float))))
        current = row.get("val_mIoU")
        if self.best is None or not isinstance(current, (int, float)):
            print("")
        elif final:
            print(f"     🏁 best.pt (epoch {self.best_epoch}) re-validated: mIoU {current:.4f}")
        elif self.best_epoch == row["epoch"]:
            print(f"     ⭐ NEW BEST mIoU {current:.4f} at this epoch ({row['epoch']}) -> saved as best.pt")
        else:
            since = row["epoch"] - self.best_epoch
            print(f"     ⚠️  NO IMPROVEMENT: mIoU {current:.4f} is {self.best - current:.4f} below the best "
                  f"{self.best:.4f} from epoch {self.best_epoch} ({since} epoch{'s' if since != 1 else ''} ago); "
                  f"best.pt stays epoch {self.best_epoch}")
        if per_class and self.print_switches.get("per_class_eval", True):
            ordered = sorted(per_class.items(), key=lambda item: -1 if item[1] is None else item[1], reverse=True)
            for start in range(0, len(ordered), 3):
                print("        " + "   ".join(f"{name[:22]:22s} {'  n/a ' if value is None else f'{value:.4f}'}"
                                            for name, value in ordered[start:start + 3]))
        self.write_summary(trainer, status="running")
        self.persist()

    def on_model_save(self, trainer):
        if self.uploader is not None:
            for path in (trainer.last, trainer.best):
                self.uploader.offer(path)

    def on_train_end(self, trainer):
        self.write_summary(trainer, status="finished")

    def model_cost(self, trainer) -> dict:
        """Parameters, forward GFLOPs per image, peak GPU memory and best.pt size, for the paper's cost table."""
        model = getattr(trainer.model, "module", trainer.model)
        best = Path(str(getattr(trainer, "best", "") or ""))
        height, width = self.batch_shape or (None, None)
        return {
            "parameters": sum(parameter.numel() for parameter in model.parameters()) if model is not None else None,
            "forward_gflops_per_image": round(self.flops_per_image / 1e9, 4) if self.flops_per_image else None,
            "gflops_input_size": f"{width}x{height}" if width else None,
            "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                                   if torch is not None and torch.cuda.is_available() else None),
            "best_pt_megabytes": round(best.stat().st_size / 1e6, 2) if best.is_file() else None,
        }

    def write_summary(self, trainer, status: str):
        """training_summary.json: settings, hardware, best score, last validation, files."""
        try:
            summary = {
                "experiment": self.config["experiment"]["name"], "run_id": self.run_id, "status": status,
                "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "monitor": "val_mIoU", "mode": "max", "best_val_mIoU": self.best, "best_epoch": self.best_epoch,
                "epochs_done": len(self.history), "steps": self.step, "images_seen": self.images_done,
                "elapsed": clock(time.time() - self.start_wall),
                "last_validation": self.history[-1] if self.history else None,
                "weights": {"best": str(getattr(trainer, "best", "")), "last": str(getattr(trainer, "last", ""))},
                "model_cost": self.model_cost(trainer),
                "hardware": hardware_info(self.config), "config": self.config,
            }
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        except Exception as error:
            print(f"  [warn] could not write the summary: {type(error).__name__}: {error}")

    def close(self):
        self.train_log.close()
        self.val_log.close()
        if self.ud is not None:
            self.ud.dump(self.ud_path)


# =============================================================================
# the entry point: run_training
# =============================================================================
def run_training(config: dict, run_dir, extra_callbacks: dict | None = None, persist_function=None):
    """Train one run: capture the console, print everything, train with Ultralytics, log, plot, flush.

    config            load_config(...) output (defaults are filled in if missing)
    run_dir           the run folder; everything lands in run_dir/<output.run_dir>
    extra_callbacks   {Ultralytics event name: function(trainer) or [functions]} added by the run's train.py
    persist_function  called to publish written files (on Modal: the runs volume's commit)
    Returns the best validation mIoU (None if no validation ran).
    """
    config = with_defaults(config)
    run_dir = Path(run_dir)
    results_dir = run_dir / config["output"]["run_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    experiment["name"] = experiment.get("name") or run_dir.name
    # normally allocated on the launching PC and passed in; only a bare local run gets here without one
    experiment["run_id"] = experiment.get("run_id") or allocate_run_id(results_dir)
    with ConsoleCapture(results_dir / config["output"]["console_capture_dir"], experiment["run_id"]):
        return _run_training(config, run_dir, results_dir, extra_callbacks or {}, persist_function)


def _previous_config(capture_dir: Path, current_run_id: str) -> dict | None:
    """The config saved by the most recent EARLIER launch of this run folder, or None."""
    candidates = [path for path in capture_dir.glob("run*_config.yaml") if run_id_of(path.name) != current_run_id]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        return yaml.safe_load(latest.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resume_state(config: dict, results_dir: Path, capture_dir: Path) -> tuple[str, Path]:
    """Decide what this launch does with Ultralytics' last.pt: ("fresh" | "resume" | "finished", last.pt path).

    resume: auto   last.pt exists and the run is unfinished -> continue it (after the guard below);
                   last.pt exists and the run finished -> nothing to train, stop
    resume: false  last.pt exists -> refuse (a fresh start would overwrite the earlier training)

    The guard: a checkpoint outlives the config that produced it. If a guarded setting (resume_guard)
    changed since the earlier launch, continuing would mix two settings in one run, so this refuses
    unless resume_force is true.
    """
    last = results_dir / config["output"]["ultralytics_dir"] / "weights" / "last.pt"
    if not last.exists():
        return "fresh", last
    try:
        epoch = torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)
    except Exception as error:
        raise SystemExit(f"  ❌ {last} exists but cannot be read ({type(error).__name__}: {error}); "
                         f"move it away to start fresh") from error
    if epoch == -1:                         # Ultralytics strips the optimizer and sets epoch -1 at the end
        return "finished", last
    if str(config.get("resume")).lower() in ("false", "none", "0"):
        raise SystemExit(f"  ❌ {last} holds an unfinished training (epoch {epoch + 1}) and resume is false. "
                         f"Set resume: auto to continue it, or move results/{config['output']['ultralytics_dir']} "
                         f"away to start fresh.")
    previous = _previous_config(capture_dir, config["experiment"]["run_id"])
    if previous is not None:
        changed = [(key, config_get(previous, key), config_get(config, key)) for key in config.get("resume_guard", [])
                   if config_get(previous, key) != config_get(config, key)]
        if changed:
            _table("guarded settings changed since the last launch", ["setting", "before", "now"],
                   [[key, before, now] for key, before, now in changed], "🛑")
            if not config.get("resume_force"):
                raise SystemExit("  ❌ refusing to resume under changed settings (set resume_force: true to "
                                 "continue anyway, or start a new run folder)")
            print("  ⚠️  resume_force is true: continuing under the changed settings")
    return "resume", last


def _run_training(config: dict, run_dir: Path, results_dir: Path, extra_callbacks: dict, persist_function):
    from ultralytics import YOLO

    print_switches = config["logging"]["print"]
    run_id = config["experiment"]["run_id"]
    print("#" * 70)
    print(f"# 🚀 run_training: {config['experiment']['name']}")
    print(f"# 🆔 run id={run_id}  side={runtime_side()}  weights={config['model']['weights']}")
    print("#" * 70)
    seed = int(config_get(config, "ultralytics.seed", 42))
    set_seed(seed, bool(config_get(config, "ultralytics.deterministic", False)))
    print(f"[set_seed] seed={seed}  deterministic={config_get(config, 'ultralytics.deterministic')}")
    if print_switches.get("environment", True):
        print_environment(config)
    if print_switches.get("config", True):
        print_config(config)

    capture_dir = results_dir / config["output"]["console_capture_dir"]
    state, last = _resume_state(config, results_dir, capture_dir)
    save_config(config, capture_dir / f"{run_id}_config.yaml")
    print(f"[paths] results -> {results_dir}")
    print(f"[paths] Ultralytics folder -> {results_dir / config['output']['ultralytics_dir']}")
    if state == "finished":
        print(f"\n  ✅ this run already finished ({last} is final). Nothing to train; "
              f"start a new run folder for a new training.")
        _safe("make_plots", make_plots, config, results_dir, config["output"].get("plots_dir", "plots"))
        return None
    print(f"[resume] {'continuing the unfinished training in ' + str(last) if state == 'resume' else 'fresh start'}")

    persist = AsyncPersist(persist_function)
    uploader = None
    if config["output"].get("upload_checkpoints") and runtime_side() == "local":
        output = config["output"]
        uploader = CheckpointUploader(output["upload_checkpoints"], config["experiment"]["name"], results_dir,
                                      f"{output['run_dir']}/{output['ultralytics_dir']}/weights",
                                      config["modal"]["runs_volume"])
    monitor = TrainingMonitor(config, results_dir, run_id, persist, uploader)

    weights_dir = config_get(config, "model.weights_dir")
    previous_directory = os.getcwd()
    if weights_dir and state == "fresh":          # Ultralytics downloads official weights into the current folder
        Path(weights_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(weights_dir)
    try:
        model = YOLO(str(last) if state == "resume" else config["model"]["weights"], task=config["model"]["task"])
    finally:
        os.chdir(previous_directory)
    monitor.register(model)
    for event, functions in extra_callbacks.items():
        for function in (functions if isinstance(functions, (list, tuple)) else [functions]):
            model.add_callback(event, function)

    if state == "resume":
        # Ultralytics restores every setting from the checkpoint; only resume=True is passed
        arguments = {"resume": True}
    else:
        arguments = dict(config["ultralytics"])
        arguments.update(data=str(config["data"]["yaml"]), project=str(results_dir),
                         name=config["output"]["ultralytics_dir"], exist_ok=True)
    _kv("Ultralytics train arguments", sorted(arguments.items()), "\U0001F3CB️")

    status = "crashed"
    start = time.time()
    try:
        model.train(trainer=make_trainer_class(monitor), **arguments)
        status = "finished"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        monitor.close()
        _print_final_summary(config, monitor, results_dir, status, time.time() - start)
        if config["output"].get("make_plots", True):
            _safe("make_plots", make_plots, config, results_dir, config["output"].get("plots_dir", "plots"))
        write_run_report(config, results_dir)
        persist(blocking=True)                     # the run is over: this flush has to land
        if persist.skipped:
            print(f"  ℹ️  {persist.done} volume flushes, {persist.skipped} skipped as redundant")
        if persist.last_error:
            print(f"  ⚠️  last volume flush error: {persist.last_error}")
        if uploader is not None:
            uploader.join()
    return monitor.best


def _print_final_summary(config: dict, monitor: TrainingMonitor, results_dir: Path, status: str, seconds: float):
    if not config["logging"]["print"].get("final_summary", True):
        return
    print("\n" + "=" * 70)
    print(f"\U0001F3C1 {status.upper()} - {config['experiment']['name']}  ({monitor.run_id})")
    print("=" * 70)
    print(f"  ⏱️  this launch       : {format_duration(seconds)}  ({monitor.step} steps, "
          f"{len(monitor.history)} validations, {monitor.images_done:,} {monitor.unit}s)")
    if monitor.best is not None:
        print(f"  ⭐ best val mIoU      : {monitor.best:.4f}  @ epoch {monitor.best_epoch}")
        best_row = next((row for row in monitor.history if row.get("trigger") == "epoch"
                         and row.get("epoch") == monitor.best_epoch), None)
        if best_row:
            print(f"  \U0001F4CF pixel accuracy   : {best_row.get('val_pixel_acc')}")
            classes = [(key[8:], value) for key, value in best_row.items()
                       if key.startswith("val_iou_") and isinstance(value, (int, float))]
            classes.sort(key=lambda item: -item[1])
            _table(f"IoU per class at the best validation (epoch {monitor.best_epoch})", ["class", "IoU", ""],
                   [[name, f"{value:.4f}", "█" * int(value * 40)] for name, value in classes], "\U0001F50D")
    else:
        print("  ⭐ no validation ran")
    print(f"  \U0001F4BE weights           : {results_dir / config['output']['ultralytics_dir'] / 'weights'}")
    print(f"  \U0001F4DD logs              : {results_dir / config['output']['train_log_csv']}")
    print(f"                        {results_dir / config['output']['val_log_csv']}")
    print("=" * 70)


# =============================================================================
# modal glue
# =============================================================================
def modal_image(config: dict, training_dir, extra_pip=()):
    """Container image: system packages, Python packages, then the training/ folder (code only).

    Pip and apt steps go BEFORE the local files: Modal refuses build steps after add_local_* (it would
    rebuild on every code change). training/ lands at /root/training; results/ and weights are excluded.
    """
    import modal

    settings = config["modal"]
    image = modal.Image.debian_slim(python_version=str(settings.get("python_version", "3.12")))
    packages = list(settings.get("apt") or [])
    if packages:
        image = image.apt_install(*packages)
    image = image.pip_install(*(list(settings.get("pip") or []) + list(settings.get("extra_pip") or [])
                                + list(extra_pip)))
    image = image.add_local_dir(str(Path(training_dir)), "/root/training",
                                ignore=["**/results", "**/__pycache__", "**/*.pt", "**/*.pth", "**/*.npz",
                                        "**/*.png", "**/*.jpg", "**/*.ckpt", "**/*.safetensors"])
    return image


def modal_volumes(config: dict) -> dict:
    """{mount path: Volume} for the runs volume and the data volume."""
    import modal

    settings = config["modal"]
    volumes = {settings["runs_mount"]: modal.Volume.from_name(settings["runs_volume"], create_if_missing=True)}
    if settings.get("data_volume"):
        volumes[settings["data_mount"]] = modal.Volume.from_name(settings["data_volume"])
    return volumes


def modal_secrets(config: dict) -> list:
    import modal

    return [modal.Secret.from_name(name) for name in (config["modal"].get("secrets") or [])]


def modal_resources(config: dict) -> dict:
    """gpu / timeout always; cpu and memory only when the config asks for them."""
    settings = config["modal"]
    resources = {"gpu": settings["gpu"], "timeout": int(float(settings.get("timeout_hours", 8)) * 3600),
                 # release the GPU the moment training returns (Modal's default keeps it billed 60 s)
                 "scaledown_window": int(settings.get("scaledown_window", 2))}
    if settings.get("cpu"):
        resources["cpu"] = float(settings["cpu"])
    if settings.get("memory_gb"):
        resources["memory"] = int(float(settings["memory_gb"]) * 1024)   # config is GB, Modal wants MiB
    return resources


def remote_config(config: dict) -> dict:
    """Copy of the config with local paths swapped for the container's (modal.path_remap)."""
    remote = json.loads(json.dumps(config))        # cheap deep copy, leaves the local config alone
    for key, value in (config["modal"].get("path_remap") or {}).items():
        config_set(remote, key, value)
    return remote


def remote_run_ids(config: dict) -> set:
    """runNNN ids already on the runs volume for this run folder (no container needed)."""
    folder = f"{config['experiment']['name']}/{config['output']['run_dir']}/{config['output']['console_capture_dir']}"
    files = volume_files(config["modal"]["runs_volume"], folder) or {}
    return {run_id for name in files if (run_id := run_id_of(name.split("/")[-1]))}


def launch_run_id(config: dict, run_dir) -> str:
    """Allocate this launch's runNNN here on the launching PC, also checking the volume, so a number
    already used remotely is never handed out again."""
    known = set()
    try:
        known = remote_run_ids(config)
    except Exception as error:                     # offline: local state still works
        print(f"  ⚠️  could not read run ids off the volume ({type(error).__name__}: {error}) - "
              f"numbering from local state only")
    run_id = allocate_run_id(Path(run_dir) / config["output"]["run_dir"], known)
    print(f"  🆔 allocated {run_id} for {config['experiment']['name']}"
          + (f"  (volume already had {len(known)})" if known else ""))
    return run_id


def copy_view_to_local_disk(data_yaml, local_root, quiet: bool = False) -> str:
    """Copy a dataset view from the Modal Volume to the container's local disk; return the local data.yaml.

    A Modal Volume reads many small files slowly (Ultralytics warned "read: 1.3 MB/s"; Modal advises fewer than
    50,000 files per volume) but big files fast. make_resized_view_modal.py packs each walk's frames and masks
    into <view>/tars/<walk>.tar; here every archive is extracted (several in parallel) into
    <local_root>/<view name>/, and the image lists (*.txt) and data.yaml are copied next to them. The lists
    hold paths relative to their own folder, so the same data.yaml works there unchanged.
    Without a tars/ folder nothing is copied and the volume path is returned.
    """
    view_dir = Path(data_yaml).parent
    archives = sorted((view_dir / "tars").glob("*.tar")) if (view_dir / "tars").is_dir() else []
    if not archives:
        print(f"  ⚠️  {view_dir} has no tars/ folder: training reads the frames straight from the volume (slow)")
        return str(data_yaml)
    target = Path(local_root) / view_dir.name
    done_marker = target / ".copy_complete"
    if done_marker.exists():
        return str(target / Path(data_yaml).name)
    target.mkdir(parents=True, exist_ok=True)
    start, total_bytes = time.time(), sum(archive.stat().st_size for archive in archives)
    if not quiet:
        print(f"\n  📦 copying {len(archives)} archives ({total_bytes / 1e9:.1f} GB) of view '{view_dir.name}' "
              f"to local disk {target} ...", flush=True)

    def extract(archive):
        # several threads extract at once: create every folder first (exist_ok), because tarfile's own
        # os.makedirs fails when two threads create the same folder at the same moment
        with tarfile.open(archive, "r") as tar:
            members = tar.getmembers()
            for folder in {str(Path(member.name).parent) for member in members}:
                (target / folder).mkdir(parents=True, exist_ok=True)
            for attempt in range(3):
                try:
                    tar.extractall(target, members=members, filter="data")
                    break
                except FileExistsError:                 # a folder race despite the above: just extract again
                    if attempt == 2:
                        raise
        return archive.stat().st_size

    copied = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, size in enumerate(pool.map(extract, archives), 1):
            copied += size
            if not quiet and (number % 20 == 0 or number == len(archives)):
                elapsed = time.time() - start
                print(f"    [{number}/{len(archives)}] {copied / 1e9:.1f} GB | {copied / 1e6 / max(elapsed, 1e-6):.0f} MB/s"
                      f" | elapsed {format_duration(elapsed)}", flush=True)
    for small_file in list(view_dir.glob("*.txt")) + list(view_dir.glob("*.yaml")):
        shutil.copy2(small_file, target / small_file.name)
    done_marker.write_text("ok")
    if not quiet:
        print(f"  ✅ local copy ready in {format_duration(time.time() - start)}: {target / Path(data_yaml).name}", flush=True)
    return str(target / Path(data_yaml).name)


def modal_train(config: dict, run_dir, run_id: str = "", volumes: dict | None = None,
                extra_callbacks: dict | None = None):
    """The body of a run's Modal GPU function: remap paths, train, always commit the runs volume."""
    remote = remote_config(config)
    if run_id:
        remote["experiment"]["run_id"] = run_id    # allocated on the launching PC
    runs_volume = (volumes or {}).get(remote["modal"]["runs_mount"])
    remote_run_dir = Path(remote["modal"]["runs_mount"]) / remote["experiment"]["name"]
    if "model.weights_dir" not in (remote["modal"].get("path_remap") or {}):
        # the local weights folder (a Windows path) means nothing in the container: official weights are
        # downloaded once into the runs volume instead
        config_set(remote, "model.weights_dir", str(Path(remote["modal"]["runs_mount"]) / "_weights"))
    if remote["modal"].get("local_data_copy", True):
        # copy_view_to_local_disk takes: the data.yaml path on the volume, the local folder;
        # returns: the local data.yaml path (or the same path when the view has no tars/ folder)
        config_set(remote, "data.yaml", copy_view_to_local_disk(remote["data"]["yaml"],
                                                                remote["modal"].get("local_data_dir", "/tmp/local_data")))
    try:
        return run_training(remote, remote_run_dir, extra_callbacks,
                            persist_function=(runs_volume.commit if runs_volume is not None else None))
    finally:
        if runs_volume is not None:
            runs_volume.commit()                   # always publish the results, even after a crash


_SYNCED = False


def sync_back(config: dict, run_dir, status: str = "", once: bool = True) -> None:
    """Pull this run's results home, then draw the plots. Fires on success, on crash and on Ctrl+C."""
    global _SYNCED
    if once and _SYNCED:
        return
    _SYNCED = True
    try:
        print(f"\n  \U0001F504 syncing results back ({status}) ...")
        sync_results(config, run_dir, direction="down")
    except Exception as error:                     # a failed sync must never mask a training error
        print(f"  ⚠️  results sync failed: {type(error).__name__}: {error}")
    plot_after_run(config, run_dir, status)


def write_run_report(config: dict, results_dir) -> Path | None:
    """results/summary.md: a short, complete report of the whole run, made from the merged files.

    Reads training_summary.json, val_log.csv and train_log.csv (all launches of the run), so it is right
    after resumes too. Written at the end of every local run and after every Modal sync. Never raises.
    Returns the path of summary.md, or None when there is nothing to report yet.
    """
    try:
        results_dir = Path(results_dir)
        output = config["output"]
        validation_path = results_dir / output["val_log_csv"]
        train_path = results_dir / output["train_log_csv"]
        if not validation_path.exists():
            return None
        validation = pd.read_csv(validation_path)
        training = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
        summary_path = results_dir / output["summary_json"]
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        epochs = validation[validation["trigger"] == "epoch"].drop_duplicates("epoch", keep="last").sort_values("epoch")
        final = validation[validation["trigger"] == "final_best"]
        if epochs.empty:
            return None

        def number(value, digits=4):
            try:
                value = float(value)
                return "" if math.isnan(value) else f"{value:.{digits}f}"
            except (TypeError, ValueError):
                return ""

        best_row = epochs.loc[epochs["val_mIoU"].idxmax()]
        best_epoch = int(best_row["epoch"])
        planned_epochs = config_get(config, "ultralytics.epochs")
        launches = sorted(set(training["run_id"])) if "run_id" in training else sorted(set(validation["run_id"]))
        training_seconds = (float(training.groupby("run_id")["elapsed_s"].max().sum())
                            if {"run_id", "elapsed_s"} <= set(training.columns) else None)
        hardware = summary.get("hardware") or {}
        modal_settings = config.get("modal") or {}
        prices = modal_settings.get("prices_per_hour") or {}
        gpu = str(hardware.get("gpu_requested") or modal_settings.get("gpu") or "")
        cpu_cores = hardware.get("cpu_requested", modal_settings.get("cpu"))      # what the last launch used
        memory_gb = hardware.get("memory_gb_requested", modal_settings.get("memory_gb"))
        hourly = None
        if hardware.get("side") == "modal-container" and gpu in (prices.get("gpu") or {}):
            hourly = (prices["gpu"][gpu] + float(cpu_cores or 0) * prices.get("cpu_core", 0)
                      + float(memory_gb or 0) * prices.get("memory_gb", 0))
        group_columns = [column for column in epochs.columns if column.startswith("val_group_")]
        group_means = [column for column in group_columns if column.endswith("_mIoU")]
        status = summary.get("status", "unknown")
        done = int(epochs["epoch"].max())

        lines = [f"# {config['experiment']['name']}", ""]
        lines += [f"**{status}** · {done}/{planned_epochs} epochs · launches {', '.join(launches)} · "
                  f"updated {summary.get('updated', '')}", ""]
        lines += ["## Result", "", "| Score | Value |", "|---|---|",
                  f"| **Best val mIoU** | **{number(best_row['val_mIoU'])}** (epoch {best_epoch}) |"]
        if not final.empty:
            lines.append(f"| best.pt re-validated | {number(final.iloc[-1]['val_mIoU'])} |")
        lines.append(f"| Pixel accuracy (best epoch) | {number(best_row.get('val_pixel_acc'))} |")
        for column in group_means:
            lines.append(f"| {column[len('val_group_'):-len('_mIoU')]} mIoU (best epoch) | {number(best_row[column])} |")
        lines.append(f"| Last epoch mIoU | {number(epochs.iloc[-1]['val_mIoU'])} (epoch {done}) |")
        lines += ["", "> Validation set (data.yaml 'val') at Ultralytics' validation size. The paper score comes "
                      "from the separate test evaluation (human-labeled frames, full resolution).", ""]

        settings = [("weights", config_get(config, "model.weights")), ("data", Path(str(config_get(config, "data.yaml"))).parent.name)]
        for key in ("epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "warmup_epochs", "mosaic", "close_mosaic",
                    "scale", "degrees", "fliplr", "seed", "deterministic", "amp"):
            settings.append((key, config_get(config, f"ultralytics.{key}")))
        lines += ["## Setup", "", "| Setting | Value |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in settings]
        lines += ["", "Full config: `train_config/<launch>_config.yaml`, `training_summary.json`.", ""]

        cost = summary.get("model_cost") or {}
        lines += ["## Model and cost", "", "| Item | Value |", "|---|---|"]
        if cost.get("parameters"):
            lines.append(f"| Parameters | {cost['parameters']:,} ({cost['parameters'] / 1e6:.2f} M) |")
        if cost.get("forward_gflops_per_image"):
            lines.append(f"| Forward GFLOPs per image | {cost['forward_gflops_per_image']} at {cost.get('gflops_input_size')} |")
        if cost.get("best_pt_megabytes"):
            lines.append(f"| best.pt size | {cost['best_pt_megabytes']} MB |")
        if cost.get("peak_gpu_memory_gb") or hardware.get("gpu_peak_gb"):
            lines.append(f"| Peak GPU memory | {cost.get('peak_gpu_memory_gb') or hardware.get('gpu_peak_gb')} GB |")
        lines.append(f"| Hardware | {hardware.get('gpu', '?')}, {cpu_cores} CPU cores, "
                     f"{memory_gb} GB ({hardware.get('side', '?')}) |")
        if training_seconds:
            lines.append(f"| Training time | {format_duration(training_seconds)} (training loop, all launches; start-up "
                         f"of a few minutes per launch not included) |")
            if hourly:
                lines.append(f"| Estimated cost | ${hourly * training_seconds / 3600:.2f} at ${hourly:.2f}/h |")
        if "real_units_s" in training:
            lines.append(f"| Speed (median) | {number(training['real_units_s'].median(), 0)} img/s, data wait "
                         f"{number(training['data_wait_pct'].median(), 0)}% |")
        lines.append("")

        table_columns = ["epoch", "val_mIoU", "val_pixel_acc", "val_ce_loss", "val_dice_loss"] + group_means
        headers = ["Epoch", "mIoU", "Pixel acc", "CE loss", "Dice loss"] + [
            column[len("val_group_"):-len("_mIoU")] + " mIoU" for column in group_means]
        if not training.empty and "loss" in training:
            train_loss = training.groupby("epoch")["loss"].mean()
            headers.insert(1, "Train loss")
        lines += ["## Validation per epoch", "", "| " + " | ".join(headers) + " |",
                  "|" + "---|" * len(headers)]
        running_best = None
        for _, row in epochs.iterrows():
            epoch = int(row["epoch"])
            marker = ""
            if running_best is None or row["val_mIoU"] > running_best:
                running_best, marker = row["val_mIoU"], " ⭐"
            cells = [f"{epoch}{marker}"]
            if "Train loss" in headers:
                cells.append(number(train_loss.get(epoch)))
            cells += [number(row.get(column)) for column in table_columns[1:]]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "⭐ = new best at that epoch.", ""]

        class_columns = [column for column in epochs.columns if column.startswith("val_iou_")]
        if class_columns:
            values = sorted(((column[len("val_iou_"):], best_row[column]) for column in class_columns),
                            key=lambda item: -(item[1] if isinstance(item[1], (int, float)) and not math.isnan(item[1]) else -1))
            lines += [f"## IoU per class (best epoch {best_epoch})", "", "| Class | IoU | Class | IoU |",
                      "|---|---|---|---|"]
            half = (len(values) + 1) // 2
            for left, right in zip(values[:half], values[half:] + [("", "")]):
                left_value = number(left[1]) or "absent"
                right_value = (number(right[1]) or "absent") if right[0] else ""
                lines.append(f"| {left[0]} | {left_value} | {right[0]} | {right_value} |")
            lines += ["", "absent = no pixels of that class in the validation set (left out of the mIoU).", ""]
        if group_columns:
            lines += ["## Class groups (best epoch)", "", "| Group | IoU |", "|---|---|"]
            lines += [f"| {column[len('val_group_'):]} | {number(best_row[column])} |"
                      for column in group_columns if column not in group_means]
            lines.append("")

        if not training.empty:
            gradient = pd.to_numeric(training.get("gnorm"), errors="coerce") if "gnorm" in training else None
            overflow = int((training["gnorm"].astype(str) == "overflow").sum()) if "gnorm" in training else 0
            lines += ["## Training health", "", "| Item | Value |", "|---|---|",
                      f"| Train loss, first / last logged | {number(training['loss'].iloc[0])} / {number(training['loss'].iloc[-1])} |"]
            if gradient is not None and gradient.notna().any():
                lines.append(f"| Gradient norm, median | {number(gradient.median(), 2)} (clipped at 10; "
                             f"{100.0 * (gradient > 10).mean():.0f}% of steps above) |")
            lines.append(f"| Mixed-precision overflow steps | {overflow} |")
            if "ud_log10_median" in training and pd.to_numeric(training["ud_log10_median"], errors="coerce").notna().any():
                ud = pd.to_numeric(training["ud_log10_median"], errors="coerce").dropna()
                lines.append(f"| Update/weight ratio, log10 median, first / last | {number(ud.iloc[0], 2)} / "
                             f"{number(ud.iloc[-1], 2)} (about -3 is healthy) |")
            lines.append("")

        plots_dir = results_dir / output.get("plots_dir", "plots")
        files = [f"[{path.stem}]({plots_dir.name}/{path.name})" for path in sorted(plots_dir.glob("*.png"))]
        debug_dir = results_dir / (output.get("debug_dir") or "debug")
        lines += ["## Files", "", "- Plots: " + (", ".join(files) if files else "none yet"),
                  f"- Debug images (exact model input): `{debug_dir.name}/` "
                  f"({len(list(debug_dir.glob('*.png'))) if debug_dir.is_dir() else 0} saved)",
                  "- Logs: `train_log.csv` (every step), `val_log.csv` (every validation), `ud_ratio.csv`",
                  f"- Ultralytics: `{output.get('ultralytics_dir', 'ultralytics')}/` (weights/best.pt, last.pt, results.csv)", ""]
        report = results_dir / "summary.md"
        report.write_text("\n".join(lines), encoding="utf-8")
        print(f"  📝 report -> {report}")
        return report
    except Exception as error:
        print(f"  ⚠️  summary.md not written: {type(error).__name__}: {error}")
        return None


def plot_after_run(config: dict, run_dir, status: str = "") -> None:
    """Draw the training curves on this PC from the merged CSVs, whatever the status (a partial run's
    curves are useful too). A plotting error never masks the run's own result."""
    try:
        results_dir = Path(run_dir) / config["output"].get("run_dir", "results")
        if not (results_dir / config["output"]["train_log_csv"]).exists():
            print(f"  ⚠️  no local CSVs in {results_dir} - skipping the plots")
            return
        make_plots(config, results_dir, config["output"].get("plots_dir", "plots"))
    except Exception as error:
        print(f"  ⚠️  plotting failed: {type(error).__name__}: {error}")
    # write_run_report takes: config, results folder; returns: the path of summary.md (or None)
    write_run_report(config, Path(run_dir) / config["output"].get("run_dir", "results"))


def modal_launch(config: dict, run_dir, remote_function) -> None:
    """The body of a run's @app.local_entrypoint: Ctrl+C-safe launch, then sync home."""
    sync_on_interrupt(config, run_dir)            # Ctrl+C stops the app first, then pulls the results
    run_id = launch_run_id(config, run_dir)
    status = "crashed"
    try:
        remote_function.remote(run_id)
        status = "finished"
    finally:
        # the container cannot reach this PC, so the pull happens here, finished or not
        sync_back(config, run_dir, status)


def stop_running_app(config: dict, timeout: int = 90) -> str | None:
    """Stop this run's Modal app, so the GPU stops billing BEFORE time is spent syncing.

    The results are safe because they live on the runs VOLUME, which the container publishes every
    output.commit_every_steps steps and after every validation. Stopping first can lose at most what was
    written since the last publish. Found by NAME: modal.App is created as App(experiment name).
    """
    import subprocess

    name = config.get("experiment", {}).get("name")
    if not name:
        return None
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        result = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                                capture_output=True, timeout=timeout, env=environment)
        output = (result.stdout or b"").decode("utf-8", "replace")
        rows = json.loads(output[output.index("["):output.rindex("]") + 1])
    except Exception as error:
        print(f"     (could not list apps: {type(error).__name__}: {error})", flush=True)
        return None
    for row in rows:
        if row.get("Description") != name or row.get("Stopped at"):
            continue
        state = (row.get("State") or "").lower()
        if "ephemeral" not in state and "running" not in state:
            continue
        app_id = row.get("App ID")
        try:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", app_id, "-y"],
                           capture_output=True, timeout=timeout, env=environment)
            return app_id
        except Exception as error:
            print(f"     (stop failed: {type(error).__name__}: {error})", flush=True)
            return None
    return None


def sync_on_interrupt(config: dict, run_dir) -> None:
    """Sync on Ctrl+C too, by hooking the signal rather than trusting `finally`.

    `modal run` installs its own SIGINT handler and tears the local process down before the entrypoint's
    finally block runs. Ours runs first: stop the container (GPU released), pull the results, then hand over
    to Modal's handler as before.
    """
    import signal

    def mute_heartbeat_noise():
        """Drop Modal's heartbeat tracebacks against the app we just stopped on purpose."""
        class DropHeartbeat(_logging.Filter):
            def filter(self, record):
                try:
                    message = record.getMessage()
                except Exception:
                    return True
                noisy = "Loop attempt" in message and "heartbeat" in message
                if not noisy and record.exc_info:
                    noisy = "APP_STATE_STOPPED" in str(record.exc_info[1])
                return not noisy

        heartbeat_filter = DropHeartbeat()
        for logger_name in ("modal-client", "modal", "modal.runner", "modal._utils.async_utils"):
            _logging.getLogger(logger_name).addFilter(heartbeat_filter)
        for handler in _logging.getLogger().handlers:
            handler.addFilter(heartbeat_filter)

    def make(previous):
        def handler(signal_number, frame):
            print("\n  ⏹️  interrupted - stopping the container, then pulling the results "
                  "(Ctrl+C again to skip)", flush=True)
            mute_heartbeat_noise()
            app_id = stop_running_app(config)
            print(f"     container {'stopped: ' + app_id if app_id else 'not found (already down?)'} - "
                  f"GPU released, now pulling the results", flush=True)
            sync_back(config, run_dir, "interrupted")
            if callable(previous):
                return previous(signal_number, frame)      # let Modal stop the app as usual
            signal.signal(signal_number, signal.SIG_DFL)
            os.kill(os.getpid(), signal_number)
        return handler

    for signal_kind in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if signal_kind is None:
            continue
        try:
            signal.signal(signal_kind, make(signal.getsignal(signal_kind)))
        except Exception:                           # not the main thread, or no such signal
            pass


# =============================================================================
# modal: bring a run's results home (Python SDK, no CLI subprocesses)
# =============================================================================
# CSV logs merged row by row on these keys; every other file is copied to the side that lacks it
MERGE_CSVS = {"train_log.csv": ("run_id", "global_step"),
              "val_log.csv": ("run_id", "epoch", "trigger"),
              "ud_ratio.csv": ("run_id", "step", "param")}
# per-side "latest" files: the remote copy wins when the two differ
LATEST_ONLY = {"training_summary.json", "results.csv", "args.yaml"}


def _volume(name: str):
    import modal

    return modal.Volume.from_name(name)


def volume_files(volume: str, base: str) -> dict | None:
    """{path relative to base: size in bytes} for every file under base on a Modal volume.
    {} when base does not exist, None when the volume cannot be listed."""
    from modal.volume import FileEntryType

    base = base.strip("/")
    try:
        entries = _volume(volume).listdir(base, recursive=True)
    except Exception as error:
        message = str(error).lower()
        if "not found" in message or "no such" in message or "does not exist" in message:
            return {}
        print(f"  ⚠️  could not list {volume}:/{base} ({type(error).__name__}: {error})")
        return None
    files = {}
    for entry in entries:
        if entry.type != FileEntryType.FILE:
            continue
        path = entry.path.strip("/")
        relative = path[len(base):].strip("/") if path.startswith(base + "/") or path == base else path
        files[relative] = int(entry.size)
    return files


def pull_file(volume: str, remote: str, destination) -> Path:
    """Download one file from a Modal volume, atomically (temporary file, then replace)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with open(temporary, "wb") as handle:
        for chunk in _volume(volume).read_file(remote.strip("/")):
            handle.write(chunk)
    os.replace(temporary, destination)
    return destination


def push_files(volume: str, pairs: list, force: bool = True) -> None:
    """Upload [(local path, remote path)] to a Modal volume in one batch."""
    if not pairs:
        return
    with _volume(volume).batch_upload(force=force) as batch:
        for local, remote in pairs:
            batch.put_file(str(local), "/" + str(remote).strip("/"))


def read_csv(path: Path):
    """(frame, note). frame=None means unreadable: that side must never be trusted.
    Everything is read as text: a merge moves rows around, it never reformats the numbers inside them."""
    try:
        if not path.exists():
            return pd.DataFrame(), "missing"       # absent is fine; None means CORRUPT
        if path.stat().st_size == 0:
            return pd.DataFrame(), "empty"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
        return frame, f"{len(frame)} rows"
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "header only"
    except Exception as error:
        return None, f"CORRUPT ({type(error).__name__})"


def canonical(frame: pd.DataFrame, keys, columns=None) -> pd.DataFrame:
    """Align columns, drop duplicate keys (the fullest row wins), sort by run then step."""
    frame = frame.copy()
    if columns is not None:
        columns = ["run_id"] + [column for column in columns if column != "run_id"]
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[list(columns)]
    frame = frame.fillna("")
    if "run_id" in frame.columns:
        frame["run_id"] = frame["run_id"].replace("", "legacy")
    if frame.empty:
        return frame
    step_column = next((column for column in ("global_step", "step", "epoch") if column in frame.columns), None)
    if step_column:
        # a killed process leaves half a line behind; a real row always has a step
        frame = frame[pd.to_numeric(frame[step_column], errors="coerce").notna()]
    key_columns = [column for column in keys if column in frame.columns] or list(frame.columns)
    filled = (frame != "").sum(axis=1)
    frame = (frame.assign(_filled=filled).sort_values("_filled", ascending=False, kind="stable")
             .drop_duplicates(subset=key_columns, keep="first").drop(columns="_filled"))
    order = [column for column in ("run_id",) if column in frame.columns]
    if step_column:
        frame = frame.assign(_number=pd.to_numeric(frame[step_column], errors="coerce").fillna(-1))
        order.append("_number")
    if order:
        frame = frame.sort_values(order, kind="stable")
    return frame.drop(columns=[column for column in ("_number",) if column in frame.columns]).reset_index(drop=True)


def _same_rows(first: pd.DataFrame, second: pd.DataFrame, keys) -> bool:
    """Same rows regardless of order; numbers compared at 10 significant digits."""
    if len(first) != len(second) or list(first.columns) != list(second.columns):
        return False
    key_columns = [column for column in keys if column in first.columns] or list(first.columns)

    def normalized(frame):
        def number_text(value):
            try:
                return f"{float(value):.10g}"
            except (TypeError, ValueError):
                return value
        return frame.map(number_text).sort_values(key_columns, kind="stable").reset_index(drop=True)
    return normalized(first).equals(normalized(second))


def atomic_write(frame: pd.DataFrame, path: Path):
    """Temporary file, then replace, so a killed sync can never leave a half-written log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def sync_results(config: dict, run_dir, direction: str = "down", dry: bool = False,
                 include_checkpoints: bool | None = None) -> dict:
    """Make <run_dir>/results match <runs volume>:/<run name>/results.

    CSV logs are MERGED row by row (union, keyed on run_id + step), so a run resumed across several
    containers ends up with one complete history on both sides. Other files are copied to the side that
    lacks them; "latest" files (summary, Ultralytics' results.csv) take the remote copy when they differ.
    Checkpoints (Ultralytics' weights folder) are skipped unless asked for. direction: down | up | both.
    """
    import tempfile

    name = config["experiment"]["name"]
    volume = config["modal"]["runs_volume"]
    results_name = config["output"]["run_dir"]
    local_base, remote_base = Path(run_dir) / results_name, f"{name}/{results_name}"
    weights_prefix = f"{config['output']['ultralytics_dir']}/weights/"
    include = config["modal"].get("sync_checkpoints") if include_checkpoints is None else include_checkpoints
    skip_top = {".run_ids", "plots", "__pycache__", "_upload_staging"}

    def skipped(relative: str) -> bool:
        return (relative.split("/")[0] in skip_top or relative.endswith((".tmp", ".part", ".migrating"))
                or (not include and relative.startswith(weights_prefix)))

    print(f"\n  🔄 sync {volume}:/{remote_base}  <->  {local_base}   ({direction}{', dry run' if dry else ''})")
    remote = volume_files(volume, remote_base)
    if remote is None:
        return {"ok": False}
    remote = {relative: size for relative, size in remote.items() if not skipped(relative)}
    local = {}
    if local_base.is_dir():
        for path in local_base.rglob("*"):
            if path.is_file():
                relative = path.relative_to(local_base).as_posix()
                if not skipped(relative):
                    local[relative] = path.stat().st_size
    rows, gets, puts = [], [], []
    for csv_name, keys in MERGE_CSVS.items():
        if csv_name not in remote and csv_name not in local:
            continue
        local_frame, local_note = read_csv(local_base / csv_name)
        remote_frame, remote_note = pd.DataFrame(), "missing"
        if csv_name in remote:
            temporary = Path(tempfile.mkdtemp()) / csv_name
            try:
                pull_file(volume, f"{remote_base}/{csv_name}", temporary)
                remote_frame, remote_note = read_csv(temporary)
            except Exception as error:
                remote_frame, remote_note = None, f"download failed ({type(error).__name__})"
        if local_frame is None or remote_frame is None:
            rows.append([csv_name, "skip", f"local {local_note} / remote {remote_note}"])
            continue
        columns = list(local_frame.columns) + [column for column in remote_frame.columns
                                               if column not in local_frame.columns]
        merged = canonical(pd.concat([local_frame, remote_frame], ignore_index=True, sort=False), keys, columns)
        if direction in ("down", "both") and not _same_rows(merged, canonical(local_frame, keys, columns), keys):
            rows.append([csv_name, f"⬇ merge {len(local_frame)}→{len(merged)}", remote_note])
            if not dry:
                atomic_write(merged, local_base / csv_name)
        if direction in ("up", "both") and not _same_rows(merged, canonical(remote_frame, keys, columns), keys):
            rows.append([csv_name, f"⬆ merge {len(remote_frame)}→{len(merged)}", local_note])
            if not dry:
                temporary = Path(tempfile.mkdtemp()) / csv_name
                atomic_write(merged, temporary)
                puts.append((temporary, f"{remote_base}/{csv_name}"))
    for relative in sorted(set(local) | set(remote)):
        if relative in MERGE_CSVS:
            continue
        in_local, in_remote = relative in local, relative in remote
        latest = relative.split("/")[-1] in LATEST_ONLY
        if in_remote and (not in_local or (latest and local[relative] != remote[relative])):
            if direction in ("down", "both"):
                rows.append([relative, "⬇ download", f"{remote[relative]:,} B"])
                gets.append(relative)
        elif in_local and not in_remote and direction in ("up", "both"):
            rows.append([relative, "⬆ upload", f"{local[relative]:,} B"])
            puts.append((local_base / relative, f"{remote_base}/{relative}"))
    if not dry:
        start = time.time()
        failed = []
        for relative in gets:
            try:
                pull_file(volume, f"{remote_base}/{relative}", local_base / relative)
            except Exception as error:
                failed.append((relative, f"{type(error).__name__}: {error}"))
        try:
            push_files(volume, puts)
        except Exception as error:
            failed.append(("upload batch", f"{type(error).__name__}: {error}"))
        if gets or puts:
            print(f"    ⚡ {len(gets)} down, {len(puts)} up in {time.time() - start:.1f}s")
        for relative, error in failed:
            print(f"    ⚠️  {relative}: {error[:160]}")
    _table(f"{name} - {'DRY RUN, nothing written' if dry else 'synced'}", ["file", "action", "detail"], rows, "🔄")
    if not rows:
        print("    ✔ already in sync")
    return {"ok": True, "down": len(gets), "up": len(puts)}


def pull_checkpoint(config: dict, run_dir, name: str = "best.pt") -> Path:
    """Download one of Ultralytics' checkpoints of this run (best.pt / last.pt) from the runs volume."""
    output = config["output"]
    relative = f"{output['run_dir']}/{output['ultralytics_dir']}/weights/{name}"
    destination = Path(run_dir) / relative
    start = time.time()
    pull_file(config["modal"]["runs_volume"], f"{config['experiment']['name']}/{relative}", destination)
    print(f"  ⬇ {name} ({destination.stat().st_size / 1e6:.1f} MB) in {format_duration(time.time() - start)} "
          f"-> {destination}")
    return destination


# =============================================================================
# plots: PNGs from the CSV logs, CPU only
# =============================================================================
def _smooth(series, window: int):
    """Trailing rolling mean; min_periods=1 so the start of the curve survives."""
    if window and window > 1:
        return pd.Series(series).rolling(window, min_periods=1).mean()
    return pd.Series(series)


def _epoch_boundary_steps(frame) -> list:
    """Steps where a new epoch starts, drawn as faint guides."""
    steps, previous = [], None
    if frame is None or "epoch" not in frame.columns:
        return steps
    for step, epoch in zip(frame["global_step"], frame["epoch"]):
        if previous is not None and epoch != previous:
            steps.append(step)
        previous = epoch
    return steps


def _vertical_lines(axis, steps):
    for step in steps:
        axis.axvline(step, color="0.85", lw=0.8, zorder=0)


def _mark_extreme(axis, x, y, mode: str = "max", color: str = "crimson"):
    """Star the best point of a curve and label it."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or np.all(np.isnan(y)):
        return
    index = int(np.nanargmax(y)) if mode == "max" else int(np.nanargmin(y))
    x_value = float(np.asarray(x)[index])
    axis.scatter([x_value], [y[index]], marker="*", s=200, color=color, zorder=6)
    axis.annotate(f"{y[index]:.4f}\n@ {int(x_value)}", (x_value, y[index]), textcoords="offset points",
                  xytext=(8, -14), fontsize=8, color=color)


def _runs_in(frame, column="run_id") -> list:
    """Run ids present, in first-seen order. Two launches are never blended into one line."""
    if frame is None or column not in frame.columns:
        return [None]
    ids = [run_id for run_id in dict.fromkeys(frame[column].astype(str).tolist()) if run_id and run_id != "nan"]
    return ids or [None]


def _run_slice(frame, run_id):
    if frame is None:
        return None
    if run_id is None or "run_id" not in frame.columns:
        return frame
    part = frame[frame["run_id"].astype(str) == run_id]
    return part if len(part) else None


def make_plots(config: dict, results_dir, out_subdir: str = "plots", smooth_window: int = 25, dpi: int = 130) -> list:
    """Training-curve PNGs from train_log.csv, val_log.csv and ud_ratio.csv -> results/plots/.

    Every plot is drawn on its own, so one failing plot never stops the others. Several launches (run ids)
    in one CSV are drawn as separate series, never averaged together.
    """
    import matplotlib
    matplotlib.use("Agg")                     # headless: write files, never open a window
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    config = with_defaults(config)
    output = config["output"]
    name = config.get("experiment", {}).get("name", "experiment")
    train_path, val_path = results_dir / output["train_log_csv"], results_dir / output["val_log_csv"]
    plots_dir = results_dir / out_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    _banner("PLOTS", "\U0001F4C8")
    print(f"  train log : {train_path}")
    print(f"  val   log : {val_path}")
    print(f"  output    : {plots_dir}")

    def read(path):
        try:
            return pd.read_csv(path) if path.exists() and path.stat().st_size else None
        except Exception as error:
            print(f"  ⚠️  could not read {path.name}: {type(error).__name__}: {error}")
            return None

    train = read(train_path)
    validation = read(val_path)
    if validation is not None and "trigger" in validation.columns:
        final_rows = validation[validation["trigger"] == "final_best"]
        validation = validation[validation["trigger"] != "final_best"]
    else:
        final_rows = None
    if train is None and validation is None:
        print("  ⚠️  no logs yet - nothing to plot")
        return []

    runs = _runs_in(train if train is not None else validation)
    for frame in (train, validation):
        if frame is None:
            continue
        for run_id in _runs_in(frame):
            if run_id not in runs:
                runs.append(run_id)
    colors = plt.get_cmap("tab10").colors
    color_of = {run_id: colors[index % len(colors)] for index, run_id in enumerate(runs)}
    boundaries = _epoch_boundary_steps(train) if (len(runs) == 1 and train is not None) else []
    print(f"  runs      : {', '.join(str(run_id) for run_id in runs)}")
    saved = []

    def save(figure, file_name):
        figure.tight_layout()
        figure.savefig(plots_dir / file_name, dpi=dpi)
        plt.close(figure)
        saved.append(file_name)
        print(f"  ✅ {file_name}")

    def label(run_id, extra=""):
        return f"{run_id or 'run'}{extra}"

    # 1) training loss: the total and each part, raw and smoothed
    def plot_loss():
        if train is None:
            return
        parts = ["loss"] + [column for column in train.columns if column.startswith("loss_")]
        figure, axes = plt.subplots(len(parts), 1, figsize=(10, 3.2 * len(parts)), sharex=True, squeeze=False)
        for axis, column in zip(axes[:, 0], parts):
            for run_id in runs:
                part = _run_slice(train, run_id)
                if part is None or column not in part.columns:
                    continue
                values = pd.to_numeric(part[column], errors="coerce")
                axis.plot(part["global_step"], values, color=color_of[run_id], alpha=0.2, lw=0.8)
                axis.plot(part["global_step"], _smooth(values.values, smooth_window), color=color_of[run_id],
                          lw=1.8, label=label(run_id))
            _vertical_lines(axis, boundaries)
            axis.set_ylabel(column)
            axis.grid(True, alpha=0.3)
        axes[0, 0].set_title(f"{name} - training loss (smoothed x{smooth_window}; grey lines = epochs)")
        axes[0, 0].legend(fontsize=8)
        axes[-1, 0].set_xlabel("global step")
        save(figure, "loss_curves.png")

    # 2) one validation metric per epoch, best point starred
    def plot_metric(column, title, file_name, mode="max"):
        if validation is None or column not in validation.columns:
            return
        figure, axis = plt.subplots(figsize=(9, 5))
        for run_id in runs:
            part = _run_slice(validation, run_id)
            if part is None:
                continue
            values = pd.to_numeric(part[column], errors="coerce")
            axis.plot(part["epoch"], values, color=color_of[run_id], marker="o", ms=4, lw=1.8, label=label(run_id))
            _mark_extreme(axis, part["epoch"].values, values.values, mode, color_of[run_id])
        if final_rows is not None and len(final_rows) and column in final_rows.columns:
            value = pd.to_numeric(final_rows[column], errors="coerce").iloc[-1]
            axis.axhline(value, color="0.4", ls=":", lw=1.2, label=f"final check of best.pt: {value:.4f}")
        axis.set_xlabel("epoch")
        axis.set_ylabel(column)
        axis.set_title(f"{name} - {title}")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
        save(figure, file_name)

    # 3) IoU of every class per epoch, small multiples
    def plot_per_class():
        if validation is None:
            return
        columns = [column for column in validation.columns if column.startswith("val_iou_")]
        if not columns:
            return
        column_count = 5
        row_count = int(math.ceil(len(columns) / column_count))
        figure, axes = plt.subplots(row_count, column_count, figsize=(3.0 * column_count, 2.3 * row_count),
                                    sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for axis, column in zip(axes, columns):
            for run_id in runs:
                part = _run_slice(validation, run_id)
                if part is not None:
                    axis.plot(part["epoch"], pd.to_numeric(part[column], errors="coerce"), color=color_of[run_id],
                              marker="o", ms=2.5, lw=1.3)
            axis.set_title(column[8:], fontsize=9)
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.3)
        for axis in axes[len(columns):]:
            axis.axis("off")
        figure.suptitle(f"{name} - validation IoU per class", fontsize=11)
        save(figure, "val_iou_per_class.png")

    # 4) per-class IoU at the best epoch, as a bar chart
    def plot_best_bars():
        if validation is None or "val_mIoU" not in validation.columns:
            return
        columns = [column for column in validation.columns if column.startswith("val_iou_")]
        if not columns:
            return
        scores = pd.to_numeric(validation["val_mIoU"], errors="coerce")
        best = validation.loc[scores.idxmax()]
        values = sorted(((column[8:], pd.to_numeric(best[column], errors="coerce")) for column in columns),
                        key=lambda item: -1 if pd.isna(item[1]) else item[1])
        figure, axis = plt.subplots(figsize=(9, max(4, 0.3 * len(values))))
        axis.barh([item[0] for item in values], [0 if pd.isna(item[1]) else item[1] for item in values],
                  color="steelblue")
        for index, (_, value) in enumerate(values):
            axis.text((0 if pd.isna(value) else value) + 0.01, index, "n/a" if pd.isna(value) else f"{value:.3f}",
                      va="center", fontsize=8)
        axis.set_xlim(0, 1.05)
        axis.set_xlabel("IoU")
        axis.set_title(f"{name} - IoU per class at the best epoch ({int(best['epoch'])}, mIoU "
                       f"{float(best['val_mIoU']):.4f})")
        axis.grid(True, axis="x", alpha=0.3)
        save(figure, "best_epoch_iou_per_class.png")

    # 5) learning rate per step
    def plot_learning_rate():
        if train is None or "lr" not in train.columns:
            return
        figure, axis = plt.subplots(figsize=(9, 4))
        for run_id in runs:
            part = _run_slice(train, run_id)
            if part is not None:
                axis.plot(part["global_step"], part["lr"], color=color_of[run_id], lw=1.6, label=label(run_id))
        _vertical_lines(axis, boundaries)
        axis.set_xlabel("global step")
        axis.set_ylabel("learning rate")
        axis.set_title(f"{name} - learning rate (warmup, then cosine per epoch)")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
        save(figure, "learning_rate.png")

    # 6) throughput, data wait, MFU, GPU memory, gradient norm
    def plot_diagnostics():
        if train is None:
            return
        unit = config["logging"].get("unit") or "img"
        panels = [("units_s", f"{unit}/s (compute)"), ("real_units_s", f"{unit}/s (real)"),
                  ("data_wait_pct", "data wait %"), ("mfu_pct", "MFU %"), ("gpu_mem_gb", "GPU memory peak (GB)"),
                  ("gnorm", "gradient norm")]
        panels = [panel for panel in panels if panel[0] in train.columns]
        figure, axes = plt.subplots(len(panels), 1, figsize=(9, 2.6 * len(panels)), sharex=True, squeeze=False)
        for axis, (column, y_label) in zip(axes[:, 0], panels):
            for run_id in runs:
                part = _run_slice(train, run_id)
                if part is None:
                    continue
                values = pd.to_numeric(part[column], errors="coerce")
                axis.plot(part["global_step"], values, color=color_of[run_id], alpha=0.25, lw=0.8)
                axis.plot(part["global_step"], _smooth(values.values, smooth_window), color=color_of[run_id],
                          lw=1.6, label=label(run_id))
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.3)
            _vertical_lines(axis, boundaries)
        axes[0, 0].set_title(f"{name} - throughput / compute / memory / gradients")
        axes[0, 0].legend(fontsize=8)
        axes[-1, 0].set_xlabel("global step")
        save(figure, "training_diagnostics.png")

    # 7) update-to-weight ratio over steps and its last profile across the network
    def plot_update_ratio():
        path = results_dir / "ud_ratio.csv"
        if not path.exists():
            return
        frame = pd.read_csv(path)
        if "run_id" in frame.columns and frame["run_id"].nunique() > 1:
            frame = frame[frame["run_id"] == frame["run_id"].iloc[-1]]   # the latest launch only
        if frame.empty:
            return

        def depth(parameter_name):
            return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", parameter_name)]

        names = sorted(frame["param"].unique(), key=depth)
        head_index = max((int(match.group(1)) for parameter_name in names
                          if (match := re.match(r"model\.(\d+)\.", parameter_name))), default=None)
        head = [n for n in names if head_index is not None and n.startswith(f"model.{head_index}.")]
        trunk = [n for n in names if n not in head]
        steps = sorted(frame["step"].unique())
        colormap = plt.get_cmap("viridis")
        figure, (axis, profile_axis) = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [2, 1]})
        style = dict(marker="o", ms=3, ls="none") if len(steps) < 2 else dict(lw=0.7)
        for index, parameter_name in enumerate(trunk):
            part = frame[frame["param"] == parameter_name].sort_values("step")
            axis.plot(part["step"], part["log10_ud"], color=colormap(index / max(len(trunk) - 1, 1)), alpha=0.55,
                      **style)
        for parameter_name in head:
            part = frame[frame["param"] == parameter_name].sort_values("step")
            axis.plot(part["step"], part["log10_ud"], color="crimson", zorder=5, **style)
        axis.plot([], [], color=colormap(0.5), lw=2, label=f"trunk ({len(trunk)} tensors)")
        axis.plot([], [], color="crimson", lw=2, label=f"head ({len(head)} tensors)")
        axis.axhline(-3, color="k", ls="--", lw=1.2, label="-3 (learning from scratch)")
        axis.axhspan(-6, -5, color="steelblue", alpha=0.10, label="-5..-6 (pretrained drift)")
        axis.set_ylabel("log10(update / weight)")
        axis.set_title(f"{name} - update-to-weight ratio")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.3)
        last = frame[frame["step"] == frame["step"].max()].set_index("param").reindex(trunk + head)
        positions = range(len(trunk) + len(head))
        profile_axis.plot(positions, last["log10_ud"].values, "-", lw=0.8, color="0.6")
        profile_axis.scatter(list(positions)[:len(trunk)], last["log10_ud"].values[:len(trunk)], s=12,
                             color="steelblue", label="trunk")
        profile_axis.scatter(list(positions)[len(trunk):], last["log10_ud"].values[len(trunk):], s=40,
                             color="crimson", marker="*", label="head")
        profile_axis.axhline(-3, color="k", ls="--", lw=1.2)
        profile_axis.set_xlabel("tensor, input side -> head")
        profile_axis.set_ylabel("log10(update / weight)")
        profile_axis.set_title(f"profile at step {int(frame['step'].max())}")
        profile_axis.legend(fontsize=8)
        profile_axis.grid(True, alpha=0.3)
        save(figure, "ud_ratio.png")

    _safe("plot:loss", plot_loss)
    for column, title, mode in (("val_mIoU", "validation mIoU (monitored)", "max"),
                                ("val_pixel_acc", "validation pixel accuracy", "max")):
        _safe(f"plot:{column}", plot_metric, column, title, f"{column}.png", mode)
    if validation is not None:
        for column in validation.columns:
            if column.startswith("val_") and "loss" in column:
                _safe(f"plot:{column}", plot_metric, column, f"validation {column[4:]}", f"{column}.png", "min")
    _safe("plot:per_class", plot_per_class)
    _safe("plot:best_bars", plot_best_bars)
    _safe("plot:learning_rate", plot_learning_rate)
    _safe("plot:diagnostics", plot_diagnostics)
    _safe("plot:ud_ratio", plot_update_ratio)
    print(f"  \U0001F4C8 wrote {len(saved)} plot(s) -> {plots_dir}")
    return saved


# =============================================================================
# test evaluation (runs/<run>/test.py) and runs/leaderboard.md
# =============================================================================
def checkpoint_file(config: dict, run_dir, checkpoint: str) -> Path:
    """Local path of a run's checkpoint: results/train/ultralytics/weights/<best|last>.pt."""
    output = config["output"]
    return Path(run_dir) / output["run_dir"] / output["ultralytics_dir"] / "weights" / f"{checkpoint}.pt"


def evaluate_checkpoint(config: dict, checkpoint: str, weights: bytes) -> dict:
    """Score one checkpoint on every split in test.splits (runs inside the Modal container, on a GPU).

    Same frames, masks and scale as the training validation (make_dataset_class, frames sized by the short
    side), Ultralytics' own validator. The run's dataset view is first copied to local disk (tars/).
    Ultralytics' own console output is switched off; every line printed here starts with [<checkpoint>], so the
    lines of the parallel containers stay readable.
    Returns {"run", "checkpoint", "evaluated", "settings", "splits": {split: {frames, mIoU, pixel_acc, per_class,
    groups, confusion_matrix (rows = true class, columns = predicted), seconds}}}.
    """
    import logging

    os.environ["YOLO_VERBOSE"] = "False"            # no Ultralytics banners, settings notes or progress bars
    import yaml
    from ultralytics import YOLO
    from ultralytics.data import build as ultralytics_build
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)
    tag = f"[{checkpoint}]"

    def say(message: str) -> None:
        print(f"  {tag:<7} {message}", flush=True)

    remote = remote_config(config)
    test = config["test"]
    image_size = int(config["ultralytics"]["imgsz"])
    start = time.time()
    # copy_view_to_local_disk takes: data.yaml on the volume, local folder, quiet; returns: the local data.yaml
    local_yaml = Path(copy_view_to_local_disk(remote["data"]["yaml"],
                                              remote["modal"].get("local_data_dir", "/tmp/local_data"), quiet=True))
    say(f"data        {Path(remote['data']['yaml']).parent.name} copied to local disk in "
        f"{format_duration(time.time() - start)}")
    weights_path = Path("/tmp") / f"{config['experiment']['name']}_{checkpoint}.pt"
    weights_path.write_bytes(weights)
    model = YOLO(str(weights_path), task="semantic")
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    say(f"model       {checkpoint}.pt | {parameters:,} parameters | imgsz {image_size} (training scale) | "
        f"batch {test['batch']}")
    # make_dataset_class takes: the config; returns: the training code's dataset class
    ultralytics_build.SemanticDataset = make_dataset_class(remote)
    data = yaml.safe_load(local_yaml.read_text(encoding="utf-8"))
    result = {"run": config["experiment"]["name"], "checkpoint": checkpoint,
              "evaluated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
              "settings": {"image_size": image_size, "data": remote["data"]["yaml"], "batch": test["batch"],
                           "validate_at_training_scale": config_get(config, "data.validate_at_training_scale", True),
                           "lr0": config["ultralytics"]["lr0"], "epochs": config["ultralytics"]["epochs"],
                           "weights_start": config["model"]["weights"], "gpu": test["gpu"], "parameters": parameters},
              "splits": {}}
    for split, list_name in test["splits"].items():
        list_path = local_yaml.parent / list_name
        if not list_path.exists() or not list_path.read_text(encoding="utf-8").strip():
            say(f"{split:<11} skipped ({list_name} missing or empty)")
            continue
        split_yaml = local_yaml.parent / f"data_{split}.yaml"          # the same dataset, val = this list
        split_yaml.write_text(yaml.safe_dump({**data, "val": str(list_path)}, sort_keys=False), encoding="utf-8")
        split_start = time.time()
        metrics = model.val(data=str(split_yaml), split="val", imgsz=image_size, batch=int(test["batch"]), rect=True,
                            workers=int(test["cpu"]) * 2, plots=False, verbose=False, project="/tmp/val",
                            name=f"{checkpoint}_{split}", exist_ok=True)
        names = metrics.names
        # group_scores takes: confusion matrix, class names, class groups; returns: {column: IoU}
        groups = group_scores(metrics.matrix, names, config["logging"].get("class_groups") or {})
        scores = {"frames": len(list_path.read_text(encoding="utf-8").split()),
                  "mIoU": round(float(metrics.miou), 4), "pixel_acc": round(float(metrics.pixel_accuracy), 4),
                  "per_class": {names[index]: (round(float(value), 4) if metrics.nt_per_class[index] > 0 else None)
                                for index, value in enumerate(metrics.per_class_iou)},
                  "groups": {key.replace("val_group_", ""): value for key, value in groups.items()},
                  "confusion_matrix": metrics.matrix.detach().cpu().long().tolist(),
                  "seconds": round(time.time() - split_start)}
        result["splits"][split] = scores
        group_text = " | ".join(f"{key[:-len('_mIoU')]} {value:.4f}" for key, value in scores["groups"].items()
                                if key.endswith("_mIoU") and isinstance(value, (int, float)))
        say(f"{split:<11} {scores['frames']:>6,} frames | mIoU {scores['mIoU']:.4f} | pixel acc "
            f"{scores['pixel_acc']:.4f} | {group_text} | {scores['seconds']} s")
    say(f"done in {format_duration(time.time() - start)}")
    return result


def write_test_results(config: dict, run_dir, result: dict, quiet: bool = False) -> Path:
    """results/test/<checkpoint>/metrics.json + summary.md of one checkpoint. Returns the folder."""
    folder = Path(run_dir) / "results" / "test" / result["checkpoint"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    splits, settings = result["splits"], result["settings"]
    group_sets = sorted({key[:-len("_mIoU")] for scores in splits.values() for key in scores["groups"]
                         if key.endswith("_mIoU")})
    show = lambda value: "absent" if value is None else (f"{value:.4f}" if isinstance(value, (int, float)) else "")
    lines = [f"# Test: {result['run']} / {result['checkpoint']}.pt", "",
             f"{result['evaluated']} · imgsz {settings['image_size']} (training scale) · trained from "
             f"{settings['weights_start']}, lr0 {settings['lr0']}", "",
             "| Split | Frames | mIoU | Pixel acc | " + " | ".join(f"{name} mIoU" for name in group_sets) + " |",
             "|---|---|---|---|" + "---|" * len(group_sets)]
    for split, scores in splits.items():
        lines.append(f"| {split} | {scores['frames']:,} | **{scores['mIoU']:.4f}** | {scores['pixel_acc']:.4f} | "
                     + " | ".join(show(scores["groups"].get(f"{name}_mIoU")) for name in group_sets) + " |")
    main_split = config["test"]["leaderboard_split"]
    if main_split in splits:
        ordered = sorted(splits[main_split]["per_class"].items(), key=lambda item: -(item[1] if item[1] is not None else -1))
        half = (len(ordered) + 1) // 2
        lines += ["", f"## IoU per class ({main_split})", "", "| Class | IoU | Class | IoU |", "|---|---|---|---|"]
        for left, right in zip(ordered[:half], ordered[half:] + [("", "")]):
            lines.append(f"| {left[0]} | {show(left[1])} | {right[0]} | {show(right[1]) if right[0] else ''} |")
    lines += ["", "test_human = the SANPO paper's protocol (human-labeled test frames); val = for choosing between "
                  "runs. Labels compared at the frames' training size. All numbers and confusion matrices: metrics.json.", ""]
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if not quiet:
        print(f"  📝 {folder / 'summary.md'}")
    return folder


def write_leaderboard(runs_dir, quiet: bool = False) -> Path:
    """runs/leaderboard.md: every tested checkpoint of every run, sorted by the test.leaderboard_split mIoU."""
    runs_dir = Path(runs_dir)
    rows, untested, excluded, main_split = [], [], [], "test_human"
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir() and (path / "config.yaml").exists()):
        if run_dir.name == "smoke_test_local":
            continue
        config = load_config(run_dir)
        if not config["test"].get("in_leaderboard", True):
            excluded.append(f"{run_dir.name} ({config['test'].get('leaderboard_note') or 'excluded in its config.yaml'})")
            continue
        wanted = config["test"].get("leaderboard_checkpoint", "best")
        tested = sorted((run_dir / "results" / "test").glob(f"{wanted}/metrics.json"))
        if not tested:
            untested.append(run_dir.name)
            continue
        main_split = config["test"]["leaderboard_split"]
        summary_path = run_dir / config["output"]["run_dir"] / config["output"]["summary_json"]
        cost = (json.loads(summary_path.read_text(encoding="utf-8")).get("model_cost") or {}) if summary_path.exists() else {}
        for path in tested:
            result = json.loads(path.read_text(encoding="utf-8"))
            splits = result["splits"]
            group = next((value for key, value in ((splits.get(main_split) or {}).get("groups") or {}).items()
                          if key.endswith("_mIoU")), None)
            rows.append({"run": run_dir.name, "checkpoint": result["checkpoint"], "group": group,
                         **{split: (splits.get(split) or {}).get("mIoU") for split in ("test_human", "test_all", "val")},
                         "main": (splits.get(main_split) or {}).get("mIoU"), "imgsz": result["settings"]["image_size"],
                         "lr0": result["settings"]["lr0"], "start": Path(str(result["settings"]["weights_start"])).stem,
                         "parameters": cost.get("parameters") or result["settings"].get("parameters")})
    rows.sort(key=lambda row: -(row["main"] if row["main"] is not None else -1))
    show = lambda value: "" if value is None else f"{value:.4f}"
    lines = ["# Leaderboard", "",
             f"Sorted by test mIoU on the human-labeled test frames (SANPO paper's protocol). Written by runs/<run>/test.py, "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.", "",
             "| # | Run | Ckpt | **Test mIoU** (human) | Test mIoU (all) | Test 3-class | Val mIoU | imgsz | lr0 | Start | Params |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for rank, row in enumerate(rows, 1):
        parameters = f"{row['parameters'] / 1e6:.2f} M" if row["parameters"] else ""
        lines.append(f"| {rank} | {row['run']} | {row['checkpoint']} | **{show(row['test_human'])}** | {show(row['test_all'])} | "
                     f"{show(row['group'])} | {show(row['val'])} | {row['imgsz']} | {row['lr0']} | {row['start']} | {parameters} |")
    if untested:
        lines += ["", "Not tested yet: " + ", ".join(untested)]
    if excluded:
        lines += ["", "Left out: " + "; ".join(excluded)]
    lines += ["", "Choose between runs with the **Val** column; the test columns are the reported scores.", ""]
    path = runs_dir / "leaderboard.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    if not quiet:
        print(f"  🏁 {path}")
    return path


def modal_test_resources(config: dict) -> dict:
    """gpu / cpu / memory / timeout of a test.py container (the test: section)."""
    test = config["test"]
    return {"gpu": test["gpu"], "cpu": float(test["cpu"]), "memory": int(float(test["memory_gb"]) * 1024),
            "timeout": 2 * 3600, "max_containers": 8}


def modal_test_launch(config: dict, run_dir, remote_function, checkpoints: str = "") -> None:
    """The body of a test.py local entrypoint: send the checkpoints, write the results, update the leaderboard."""
    names = [name.strip() for name in (checkpoints or ",".join(config["test"]["checkpoints"])).split(",") if name.strip()]
    _banner(f"TEST: {config['experiment']['name']}", "\U0001F9EA")
    jobs = []
    for checkpoint in names:
        path = checkpoint_file(config, run_dir, checkpoint)
        if not path.exists():
            print(f"  ⚠️  {path} not found (sync the run first), skipped")
            continue
        jobs.append((checkpoint, path.read_bytes()))
        print(f"  📤 sending {checkpoint}.pt ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"\n  splits: {', '.join(config['test']['splits'])} | {config['test']['gpu']}, one container per checkpoint\n",
          flush=True)
    results = list(remote_function.starmap(jobs, order_outputs=True))
    _banner(f"TEST RESULTS: {config['experiment']['name']}", "\U0001F3C1")
    group_sets = sorted({key[:-len("_mIoU")] for result in results for scores in result["splits"].values()
                         for key in scores["groups"] if key.endswith("_mIoU")})
    rows = []
    for result in results:
        for split, scores in result["splits"].items():
            rows.append([result["checkpoint"], split, f"{scores['frames']:,}", f"{scores['mIoU']:.4f}",
                         f"{scores['pixel_acc']:.4f}"]
                        + [f"{scores['groups'].get(name + '_mIoU', 0):.4f}" for name in group_sets])
    _table("scores (test_human = the paper score, val = for choosing between runs)",
           ["checkpoint", "split", "frames", "mIoU", "pixel acc"] + [f"{name} mIoU" for name in group_sets],
           rows, "\U0001F4CA")
    print("\n  files written:")
    for result in results:
        print(f"    {write_test_results(config, run_dir, result, quiet=True) / 'summary.md'}")
    print(f"    {write_leaderboard(Path(run_dir).parent, quiet=True)}\n", flush=True)


# =============================================================================
# leaderboard and command line
# =============================================================================
def leaderboard(runs_dir) -> pd.DataFrame:
    """Best validation mIoU of every run folder under runs_dir, printed as a table."""
    rows = []
    for run_dir in sorted(path for path in Path(runs_dir).iterdir() if path.is_dir()):
        log_path = next((path for path in (run_dir / "results" / "train" / "val_log.csv",
                                           run_dir / "results" / "val_log.csv") if path.exists()), None)
        if log_path is None:
            continue
        try:
            log = pd.read_csv(log_path)
            if "trigger" in log.columns:
                log = log[log["trigger"] != "final_best"]
            if "val_mIoU" not in log.columns or log["val_mIoU"].dropna().empty:
                continue
            index = log["val_mIoU"].idxmax()
            rows.append({"run": run_dir.name, "best": float(log.loc[index, "val_mIoU"]),
                         "epoch": int(log.loc[index, "epoch"]), "validations": len(log),
                         "launches": log["run_id"].nunique() if "run_id" in log.columns else 1})
        except Exception as error:
            print(f"  ⚠️  {run_dir.name}: {type(error).__name__}: {error}")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("best", ascending=False).reset_index(drop=True)
        _table(f"leaderboard ({runs_dir})", ["#", "run", "best val mIoU", "epoch", "validations", "launches"],
               [[index + 1, row.run, f"{row.best:.4f}", row.epoch, row.validations, row.launches]
                for index, row in frame.iterrows()], "🏁")
    else:
        print("  no run has validation results yet")
    return frame


def _command_line():
    import argparse

    parser = argparse.ArgumentParser(description="shared_training_code: plots, sync, pull, leaderboard, config")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("plot", help="draw the curves of one run from its CSVs")
    command.add_argument("run_dir")
    command = commands.add_parser("sync", help="bring a run's results home from its Modal runs volume")
    command.add_argument("run_dir")
    command.add_argument("--direction", choices=["down", "up", "both"], default="down")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--checkpoints", action="store_true", help="also sync Ultralytics' weights folder")
    command = commands.add_parser("pull", help="download one checkpoint of a run from Modal (best.pt / last.pt)")
    command.add_argument("run_dir")
    command.add_argument("--name", default="best.pt")
    command = commands.add_parser("leaderboard", help="best validation mIoU of every run under a runs/ folder")
    command.add_argument("runs_dir")
    command = commands.add_parser("config", help="print a run's merged config")
    command.add_argument("run_dir")
    command = commands.add_parser("report", help="write a run's results/summary.md from its logs")
    command.add_argument("run_dir")
    arguments = parser.parse_args()
    if arguments.command == "leaderboard":
        leaderboard(arguments.runs_dir)
        return
    config = load_config(arguments.run_dir)
    if arguments.command == "plot":
        make_plots(config, Path(arguments.run_dir) / config["output"]["run_dir"], config["output"]["plots_dir"])
    elif arguments.command == "sync":
        sync_results(config, arguments.run_dir, arguments.direction, arguments.dry_run, arguments.checkpoints or None)
    elif arguments.command == "pull":
        pull_checkpoint(config, arguments.run_dir, arguments.name)
    elif arguments.command == "config":
        print_config(config)
    elif arguments.command == "report":
        write_run_report(config, Path(arguments.run_dir) / config["output"]["run_dir"])


if __name__ == "__main__":
    _command_line()
