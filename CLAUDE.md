# Feature extractor evaluation (MuSES)

## Purpose
Compare candidate models for AU, gaze, and head pose extraction on a
shared set of test clips before building the preprocessing pipeline.

## Structure
- models/      cloned third-party repos. Do not edit their code.
- envs/        one conda env file per model. Never install into base.
- data/   shared clips + ground truth. Read only.
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