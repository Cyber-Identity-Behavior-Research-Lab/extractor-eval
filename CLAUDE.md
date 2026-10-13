# Feature extractor evaluation (MuSES)

## Purpose
Compare candidate models for AU, gaze, and head pose extraction on a shared
set of test clips before building the preprocessing pipeline.

## Structure
- `models/`              cloned third-party repos. Do not edit their code.
- `envs/<model>/`        one uv project per model (pyproject.toml, uv.lock, .venv/).
- `data/`                shared clips, clips_manifest.csv, ground truth. Read only.
- `src/`                 shared helpers (io, schema, plots), installable via the
                         root pyproject.toml.
- `notebooks/`           one notebook per model, copied from 00_template.ipynb,
                         plus 99_comparison.ipynb.
- `outputs/<model>/`     <clip_id>.parquet in the common schema.
- `model_registry.csv`   one summary row per model.

## General rules
- Work one step at a time and stop for review after each step.
- Ask before installing packages, creating envs, or downloading weights,
  unless I have asked for that specific action.
- Every model output must follow src/schema.py:
  clip_id, frame, timestamp_ms, face_conf, <model outputs>
- Record repo URL, commit hash, and license in each notebook's model card.
- Do not claim a model supports an output without checking its repo or paper.
- No participant data in this folder; test clips are researcher recordings only.
- Commit pyproject.toml and uv.lock; never commit .venv/, model weights, data/,
  or outputs/.

# Creating a model environment

Follow these steps when asked to set up an environment for a model.
A request to set up a model counts as approval to create its env; still ask
before downloading weights.

## Context
- The repo is cloned into `models/<model>/`; its env lives in `envs/<model>/`.
- Use uv only. Do not use conda, pip directly, or global installs unless I
  approve it.
- If the repo needs a fix to run, describe it and ask first. Do not edit it.

## Steps

1. **Inspect the repo**
   - Read the README, requirements files, setup.py/pyproject.toml, and install docs.
   - Report: required Python version, PyTorch and CUDA version, non-Python
     dependencies (dlib, cmake, ffmpeg, system libraries), whether weights are
     downloaded separately (source, size, license of weights), the repo license,
     the outputs the model actually produces (with units), and the commit hash
     (`git rev-parse HEAD`).
   - If it needs system libraries pip cannot provide, stop and ask whether to
     use conda for this model.

2. **Create the env**
   - `cd envs/<model>`
   - `uv init --bare --python <version from step 1>`

3. **Configure PyTorch** (only if the model uses it)
   - Check CUDA with `nvidia-smi`.
   - Add the matching PyTorch index to pyproject.toml (`[[tool.uv.index]]` with
     `explicit = true`, plus `[tool.uv.sources]` for torch and torchvision).
   - Pin torch to the version the repo requires, if it states one.

4. **Install dependencies**
   - `uv add -r ../../models/<model>/requirements.txt` (or the repo's equivalent).
   - If the repo is installable: `uv add --editable ../../models/<model>`.
   - Shared helpers: `uv add --editable ../..`.
   - On a version conflict, report it and propose a resolution. Do not silently
     loosen or change pins.

5. **Jupyter kernel**
   - `uv add --dev ipykernel`
   - `uv run python -m ipykernel install --user --name <model> --display-name "<model>"`

6. **Verify**
   - `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   - Import the model's main module and `src`.
   - Ask, then download weights to the location the repo expects; confirm they load.
   - Run the repo's demo or a single-image inference using the repo's own demo
     inputs, not files in data/.

7. **Report**
   - Python version, torch/CUDA versions, commit hash, licenses (code and
     weights), weights location, and confirmed outputs.
   - Every error hit during setup and how it was fixed.
   - Format it to paste into the "Model card" and "Setup log" sections of the
     model's notebook, plus a draft row for model_registry.csv.

## Setup-specific rules
- Do not create the model's notebook unless I ask.