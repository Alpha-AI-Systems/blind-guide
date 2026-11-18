"""train_all.py - exp6_synthetic_then_real_lr_decay: trains the 4 stages IN ORDER in ONE Modal container (one command).

  1. stage1_synthetic_head    <- yolo26n-sem-ade20k.pt
  2. stage2_real_head         <- stage 1 best.pt
  3. stage3_synthetic_chest   <- stage 2 best.pt
  4. stage4_real_chest        <- stage 3 best.pt   (the final model; only this stage has test.py)

Each stage is an ordinary run in its own folder (config.yaml, results/, logs, plots, summary.md, best.pt); the
experiment folder's config.yaml holds what the stages share. Before a stage: a "STAGE n of 4 STARTS" block and where
its weights come from. After it: a "STAGE n of 4 FINISHED" block (status, best.pt = epoch, step, val mIoU; last
validation; steps, time) and which best.pt the next stage starts from. At the end: a summary of all 4 stages.
RESUMABLE: run the same command again after a crash or Ctrl+C. Finished stages are skipped, an unfinished stage
continues from its last.pt, and a stage never starts before the stage it starts from has finished.
After the container ends (finished, crashed or Ctrl+C), every stage's results are synced home and plotted.
Then score the final model:  python -m modal run .../exp6_synthetic_then_real_lr_decay/stage4_real_chest/test.py

Modal (cloud GPU):  python -m modal run fast_path/object_detection/training/runs/exp6_synthetic_then_real_lr_decay/train_all.py
"""

import sys
from pathlib import Path

EXPERIMENT_NAME = "exp6_synthetic_then_real_lr_decay"      # MUST equal this folder's name (copying? change it)
STAGE_NAMES = [                                   # trained in this order; each starts from the one before it
    "stage1_synthetic_head",
    "stage2_real_head",
    "stage3_synthetic_chest",
    "stage4_real_chest",
]


def _training_root() -> Path:
    """The training/ folder holding shared_training_code.py: /root/training on Modal, ../.. locally
    (this file is training/runs/<experiment>/)."""
    for candidate in (Path("/root/training"), Path(__file__).resolve().parent.parent.parent):
        if (candidate / "shared_training_code.py").exists():
            return candidate
    raise FileNotFoundError("shared_training_code.py not found 2 folders above this experiment folder")


TRAINING_ROOT = _training_root()
EXPERIMENT_DIR = TRAINING_ROOT / "runs" / EXPERIMENT_NAME
STAGE_DIRS = [EXPERIMENT_DIR / name for name in STAGE_NAMES]
sys.path.insert(0, str(TRAINING_ROOT))
import shared_training_code as shared  # noqa: E402

# shared.load_config takes: the experiment folder; returns: DEFAULTS < shared_config.yaml < ./config.yaml
# (its modal: section sets the container: GPU, memory, timeout; every stage reads its own config.yaml on top)
experiment_config = shared.load_config(EXPERIMENT_DIR)

import modal  # noqa: E402

app = modal.App(shared.app_name(experiment_config))        # "exp6_synthetic_then_real_lr_decay"
volumes = shared.modal_volumes(experiment_config)          # runs volume + data volume (sanpo)


@app.function(image=shared.modal_image(experiment_config, TRAINING_ROOT), volumes=volumes,
              secrets=shared.modal_secrets(experiment_config), **shared.modal_resources(experiment_config))
def train_all_remote(run_ids: list):
    """Runs inside the Modal container: the 4 stages one after another, on the GPU named in ./config.yaml."""
    # shared.modal_train_stages takes: the stage folders, one run id per stage, the volumes;
    # trains each stage (skips finished ones), prints the STARTS / FINISHED blocks; returns: each stage's result
    return shared.modal_train_stages(STAGE_DIRS, run_ids, volumes)


@app.local_entrypoint()                            # what `modal run train_all.py` calls
def main():
    # shared.modal_launch_stages takes: the experiment config, the stage folders, the remote function;
    # allocates the run ids, runs the container (Ctrl+C safe), then syncs every stage's results home
    shared.modal_launch_stages(experiment_config, STAGE_DIRS, train_all_remote)
