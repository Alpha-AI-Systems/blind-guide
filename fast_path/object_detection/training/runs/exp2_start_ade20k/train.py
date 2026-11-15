"""train.py - exp2_start_ade20k: what is specific to this run; shared_training_code.py does the rest.

Local (this PC's GPU):  python fast_path/object_detection/training/runs/exp2_start_ade20k/train.py
Modal (cloud GPU):      python -m modal run fast_path/object_detection/training/runs/exp2_start_ade20k/train.py
"""

import sys
from pathlib import Path

RUN_NAME = "exp2_start_ade20k"      # MUST equal this folder's name (copying a run? change it)


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

# Extra Ultralytics callbacks for this run only: {event name: function(trainer)}, e.g.
# {"on_fit_epoch_end": my_function}. Event names: docs.ultralytics.com/usage/callbacks
EXTRA_CALLBACKS = {}


def run_local():
    """Train on this PC's GPU."""
    # shared.run_training takes: config, run folder, extra callbacks; returns: the best validation mIoU
    shared.run_training(config, RUN_DIR, EXTRA_CALLBACKS)


try:
    # everything below is skipped when modal is not installed, so local runs still work
    import modal

    app = modal.App(config["experiment"]["name"])
    volumes = shared.modal_volumes(config)

    @app.function(image=shared.modal_image(config, TRAINING_ROOT), volumes=volumes,
                  secrets=shared.modal_secrets(config), **shared.modal_resources(config))
    def train_remote(run_id: str = ""):
        """Runs inside the Modal container, on the GPU named in the config."""
        shared.modal_train(config, RUN_DIR, run_id=run_id, volumes=volumes, extra_callbacks=EXTRA_CALLBACKS)

    @app.local_entrypoint()                        # what `modal run train.py` calls
    def modal_main():
        shared.modal_launch(config, RUN_DIR, train_remote)   # Ctrl+C safe; syncs the results home

except ImportError:
    pass


if __name__ == "__main__":                         # required on Windows (data loader workers spawn)
    run_local()
