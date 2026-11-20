"""train.py - final_yolo26s_640x352 stage 2 of 2, MIXED CHEST, the final model (16 real + 16 synthetic per batch, from stage 1's best.pt): trains THIS stage alone.

The normal way to run the experiment is ../train_all.py (both stages in order, one container). Use this file to
(re)train one stage alone, e.g. after changing its config. A stage with model.weights_from refuses to start until
that earlier stage has finished (its best.pt is on the runs volume).
Settings: ../config.yaml (shared by the stages) and ./config.yaml (this stage).

Local (this PC's GPU):  python fast_path/object_detection/training/runs/final_yolo26s_640x352/stage2_mix_chest/train.py
Modal (cloud GPU):      python -m modal run fast_path/object_detection/training/runs/final_yolo26s_640x352/stage2_mix_chest/train.py
"""

import sys
from pathlib import Path

RUN_NAME = "final_yolo26s_640x352/stage2_mix_chest"      # MUST equal <experiment folder>/<this folder> (copying? change it)


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

# Extra Ultralytics callbacks for this stage only: {event name: function(trainer)}, e.g.
# {"on_fit_epoch_end": my_function}. Event names: docs.ultralytics.com/usage/callbacks
EXTRA_CALLBACKS = {}


def run_local():
    """Train on this PC's GPU."""
    # shared.run_training takes: config, stage folder, extra callbacks; returns: the best validation mIoU
    shared.run_training(config, RUN_DIR, EXTRA_CALLBACKS)


try:
    # everything below is skipped when modal is not installed, so local runs still work
    import modal

    app = modal.App(shared.app_name(config))      # "final_yolo26s_640x352-stage2_mix_chest"
    volumes = shared.modal_volumes(config)

    @app.function(image=shared.modal_image(config, TRAINING_ROOT), volumes=volumes,
                  secrets=shared.modal_secrets(config), **shared.modal_resources(config))
    def train_remote(run_id: str = ""):
        """Runs inside the Modal container, on the GPU named in the config."""
        # shared.modal_train takes: config, stage folder, run id, volumes, callbacks; trains and commits the volume
        shared.modal_train(config, RUN_DIR, run_id=run_id, volumes=volumes, extra_callbacks=EXTRA_CALLBACKS)

    @app.local_entrypoint()                        # what `modal run train.py` calls
    def modal_main():
        # shared.modal_launch takes: config, stage folder, the remote function; Ctrl+C safe, syncs the results home
        shared.modal_launch(config, RUN_DIR, train_remote)

except ImportError:
    pass


if __name__ == "__main__":                         # required on Windows (data loader workers spawn)
    run_local()
