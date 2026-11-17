"""test.py - exp6_synthetic_then_real stage 4 of 4, REAL CHEST, the final model (from stage 3's best.pt): score this stage's checkpoints on the validation and test splits.

Only the final stage has test.py: stages 1 to 3 are judged on their own validation sets. The leaderboard shows this
stage once, under the experiment's name (exp6_synthetic_then_real), not the stage's.
Writes results/test/<checkpoint>/metrics.json + summary.md, then updates runs/leaderboard.md.
Settings (splits, checkpoints, GPU): the test: section of the config (shared_training_code.py DEFAULTS_YAML).
The checkpoints are read from results/train/ultralytics/weights/ on this PC (synced after training).

Modal (cloud GPU):  python -m modal run fast_path/object_detection/training/runs/exp6_synthetic_then_real/stage4_real_chest/test.py
                    (add --checkpoints best to score only best.pt)
"""

import sys
from pathlib import Path

RUN_NAME = "exp6_synthetic_then_real/stage4_real_chest"      # MUST equal <experiment folder>/<this folder> (copying? change it)


def _training_root() -> Path:
    """The training/ folder holding shared_training_code.py: /root/training on Modal, ../../.. locally
    (this file is training/runs/<experiment>/<stage>/)."""
    for candidate in (Path("/root/training"), Path(__file__).resolve().parent.parent.parent.parent):
        if (candidate / "shared_training_code.py").exists():
            return candidate
    raise FileNotFoundError("shared_training_code.py not found 3 folders above this stage folder")

TRAINING_ROOT = _training_root()
RUN_DIR = TRAINING_ROOT / "runs" / RUN_NAME
sys.path.insert(0, str(TRAINING_ROOT))
import shared_training_code as shared  # noqa: E402

# shared.load_config takes: the stage folder; returns: DEFAULTS < shared_config.yaml < ../config.yaml < ./config.yaml
config = shared.load_config(RUN_DIR)

import modal  # noqa: E402

app = modal.App(f"{shared.app_name(config)}-test")
volumes = {config["modal"]["data_mount"]: modal.Volume.from_name(config["modal"]["data_volume"])}


@app.function(image=shared.modal_image(config, TRAINING_ROOT), volumes=volumes, **shared.modal_test_resources(config))
def test_remote(checkpoint: str, weights: bytes) -> dict:
    """Runs inside the Modal container: scores one checkpoint on every test split."""
    # shared.evaluate_checkpoint takes: config, checkpoint name, its bytes; returns: the scores of every split
    return shared.evaluate_checkpoint(config, checkpoint, weights)


@app.local_entrypoint()
def main(checkpoints: str = ""):
    # shared.modal_test_launch: sends the checkpoints, writes results/test/, updates runs/leaderboard.md
    shared.modal_test_launch(config, RUN_DIR, test_remote, checkpoints)
