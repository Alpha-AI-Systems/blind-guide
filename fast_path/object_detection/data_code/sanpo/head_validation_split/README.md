# Head dataset analysis and validation split (local 5 FPS)

This directory is new and does not modify the teammate's `validation_split/` for Chest.

1. Verify branch is `feat/local-sanpo-workflow`.
2. Extract the ZIP **only on that branch**.
3. In VSCode, open `head_validation_split.ipynb`; choose your `.venv` Python kernel.
4. Confirm `REPO` in the Settings cell is `C:\Projects\blind-guide`.
5. Run cells top-to-bottom. Profiling reads ~13,821 Head JPG/PNG pairs and fetches small provenance/description JSON; it takes CPU time and is resumable via per-session CSVs under `results/profiles/`.
6. Selection proposes Head validation walks and compares class distributions, weather, environment, blur and provenance. It does **not** publish a training split unless `SAVE_SPLIT = True`.
7. If no candidate passes the original Chest-like constraints, inspect the Head class statistics and consciously adjust rules; DO NOT simply use a random split.
8. Share the summary outputs (and any warnings) before preparing Head semantic/resized training views or beginning training.

Requirements in venv: `python -m pip install numpy pandas matplotlib opencv-python ipykernel` (most likely already installed).

**Safety:** the notebook writes only its own new `head_validation_split/results/` directory. It never touches original downloaded JPG/PNG, Chest files, existing team notebooks, Git branches, or experiment weights. It reads the Chest validation list and Chest index to check session overlap. An already published Head validation list is not overwritten.

**Design notes:**
- This adaptation retains the friend's selection concept: official train only, whole walks, class coverage, class pixels, class frames, environments, weather, 5th percentile sharpness, human/machine source, crowd proportion, normalized similarity and iterative walk swaps.
- Head has 73 official train sessions (11,218 frames) and 18 official test sessions (2,603 frames) according to the user's downloader verification. Expected validation walks about 7; actual selected frames depend on the data.
- `bus stop` remains in the original taxonomy for auditing. If any Head bus stop pixels are present, discuss whether to extend the current 29-class mapping before training.
- The notebook does not train; converting Head to its own semantic view and setting the trainer's paths must be done **after** an approved split.
