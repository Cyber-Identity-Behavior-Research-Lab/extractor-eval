# Feature extractor evaluation (MuSES)

## Purpose
Compare candidate models for AU, gaze, and head pose extraction on a
shared set of test clips before building the preprocessing pipeline.

## Structure
- models/      cloned third-party repos. Do not edit their code.
- envs/        one uv env file per model. Never install into base.
- data/        shared clips + ground truth. Read only.
- src/         shared helpers (io, schema, plots)
- notebooks/   one notebook per model, from 00_template.ipynb
- outputs/     <model>/<clip_id>.parquet in the common schema

## Rules
- Work one step at a time and stop for review after each step.
- Ask before installing packages, creating envs, or downloading weights.
- Every model output must follow src/schema.py:
  clip_id, frame, timestamp_ms, face_conf, <model outputs>
- Record repo URL, commit hash, and license in each notebook's model card.
- Do not claim a model supports an output without checking its repo/paper.
- No participant data in this folder; test clips are researcher recordings only.

# Creating a model environment

Follow these steps when asked to set up an environment for a new model.
Stop after each numbered step and report before continuing.

## Context
- Each model is cloned into `models/<model>/` and gets its own env in `envs/<model>/`.
- Environments are managed with uv. Do not use conda, pip directly, or global installs
  unless I approve it.
- `models/` contains third-party code. Do not edit files there. If a fix is needed,
  describe it and ask first.

## Steps

1. **Inspect the repo**
   - Read the README, requirements files, setup.py/pyproject.toml, and any install docs.
   - Report: required Python version, PyTorch and CUDA version, non-Python dependencies
     (dlib, cmake, ffmpeg, system libraries), whether weights must be downloaded
     separately, license, and the current commit hash (`git rev-parse HEAD`).
   - If the model needs system libraries pip cannot provide, stop and ask whether
     to use conda for this model.

2. **Create the env**
   - `cd envs/<model>`
   - `uv init --bare --python <version from step 1>`

3. **Configure PyTorch** (only if the model uses it)
   - Check CUDA with `nvidia-smi`.
   - Add the matching PyTorch index to `pyproject.toml` (`[[tool.uv.index]]` with
     `explicit = true`, plus `[tool.uv.sources]` for torch and torchvision).
   - Pin torch to the version the repo requires, if it states one.

4. **Install dependencies**
   - `uv add -r ../../models/<model>/requirements.txt` (or the repo's equivalent).
   - If the repo is installable: `uv add --editable ../../models/<model>`.
   - Shared helpers: `uv add --editable ../..`.
   - If a version conflict occurs, report it and propose a resolution. Do not
     silently loosen or change pins.

5. **Jupyter kernel**
   - `uv add --dev ipykernel`
   - `uv run python -m ipykernel install --user --name <model> --display-name "<model>"`

6. **Verify**
   - `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   - Import the model's main module.
   - Download weights to the location the repo expects and confirm they load.
   - Run the repo's demo or a single-image inference if one exists.

7. **Report**
   - Python version, torch/CUDA versions, commit hash, license, weights location.
   - Every error hit during setup and how it was fixed.
   - Format it so I can paste it into the "Model card" and "Setup log" sections of
     the model's notebook.

## Rules
- Commit `pyproject.toml` and `uv.lock`; never commit `.venv/` or model weights.
- Do not create the model's notebook unless I ask.
- Do not run models on files in `test_data/` during setup; use the repo's own
  demo inputs.