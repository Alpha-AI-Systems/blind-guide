"""test.py - exp4_frames_10fps_yolo26s: score this run's checkpoints on the validation and test splits.

Writes results/test/<checkpoint>/metrics.json + summary.md, then updates ../leaderboard.md.
Settings (splits, checkpoints, GPU): the test: section of the config (shared_training_code.py DEFAULTS_YAML).
The checkpoints are read from results/train/ultralytics/weights/ on this PC (synced after training).

Modal (cloud GPU):  python -m modal run fast_path/object_detection/training/runs/exp4_frames_10fps_yolo26s/test.py
                    (add --checkpoints best to score only best.pt)
"""

import sys
from pathlib import Path

RUN_NAME = "exp4_frames_10fps_yolo26s"      # MUST equal this folder's name (copying a run? change it)


def _training_root() -> Path:
    """The training/ folder holding shared_training_code.py: /root/training on Modal, ../.. locally."""
    for candidate in (Path("/root/training"), Path(__file__).resolve().parent.parent.parent):
        if (candidate / "shared_training_code.py").exists():
            return candidate
    raise FileNotFoundError("shared_training_code.py not found next to the runs/ folder")


TRAINING_ROOT = _training_root()
RUN_DIR = TRAINING_ROOT / "runs" / RUN_NAME
sys.path.insert(0, str(TRAINING_ROOT))
import shared_training_code as shared  # noqa: E402

# shared.load_config takes: the run folder; returns: DEFAULTS < shared_config.yaml < this folder's config.yaml
config = shared.load_config(RUN_DIR)

import modal  # noqa: E402

app = modal.App(f"{config['experiment']['name']}-test")
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
