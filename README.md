# Feature extractor evaluation (MuSES)

This repo compares candidate models for extracting action units (AU), gaze, and head pose. Every model runs on the same shared set of test clips. The goal is to choose extractors before building the preprocessing pipeline.

## Layout

| Folder | Contents |
|---|---|
| `models/` | Cloned third-party repos. Their code is not edited. These are git-ignored. |
| `envs/` | One conda env file per model. |
| `data/` | Shared test clips and ground truth. Treat as read-only. Video files are git-ignored. |
| `src/` | Shared helpers: `io.py`, `schema.py`, `plots.py`. |
| `notebooks/` | One notebook per model, copied from `00_template.ipynb`. |
| `outputs/` | `<model>/<clip_id>.parquet` in the common schema. These are git-ignored. |

## Adding a model

1. Clone the model's repo into `models/`.
2. Create `envs/<model>.yml` and its conda env. Do not install into `base`.
3. Copy `notebooks/00_template.ipynb` to `notebooks/NN_<model>.ipynb`.
4. Work through the notebook's sections in order:
   1. model card
   2. setup log
   3. run on clips
   4. save outputs
   5. quick checks
   6. verdict

## Common output schema

Every model's output must follow `src/schema.py`. The columns are:

```
clip_id, frame, timestamp_ms, face_conf, <model outputs>
```

## Data policy

No participant data goes in this folder. The test clips are researcher recordings only.
