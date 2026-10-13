"""Common output schema shared by every model notebook.

One parquet file per (model, clip): ``outputs/<model>/<clip_id>.parquet``.
One row per video frame, including frames where no face was found.

Required columns
----------------
clip_id       str      matches clip_id in data/clips_manifest.csv
frame         int64    0-based frame index in the video
timestamp_ms  float64  presentation timestamp read from the video, in ms
face_conf     float64  face detection confidence in [0, 1].
                       0 means no face was found; all model outputs on that row are NaN.
                       If a model only reports found / not found, use 1.0 / 0.0.

Model output columns (include only the families the model supports)
-------------------------------------------------------------------
Action units, for each AU the model reports (NN is two digits, e.g. AU01, AU12):
    AUNN_int   float64  intensity on a 0-5 scale (rescale if the model uses another range)
    AUNN_occ   float64  occurrence: 1.0 present, 0.0 absent
  Both columns are always written. If the model gives only one of them, the
  other is all NaN. AUs the model does not report are left out.

Gaze, in degrees:
    gaze_yaw_deg     positive = subject looks to their left
    gaze_pitch_deg   positive = subject looks up

Head pose, in degrees:
    pose_yaw_deg     positive = subject turns head to their left
    pose_pitch_deg   positive = subject tilts head up
    pose_roll_deg    positive = subject tilts head toward their left shoulder

  Convert each model's native convention to these signs in its notebook, and
  record the conversion in the model card. Zero for all angles = facing the camera.

Model-specific extras that are worth keeping but aren't part of the comparison
must be prefixed with ``extra_`` (e.g. ``extra_emotion``).

Assumes one face per frame: if a model returns several, keep the most confident.
"""

import re

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "clip_id": "object",
    "frame": "int64",
    "timestamp_ms": "float64",
    "face_conf": "float64",
}

GAZE_COLUMNS = ["gaze_yaw_deg", "gaze_pitch_deg"]
POSE_COLUMNS = ["pose_yaw_deg", "pose_pitch_deg", "pose_roll_deg"]

AU_PATTERN = re.compile(r"^AU(\d{2})_(int|occ)$")
EXTRA_PREFIX = "extra_"

AU_INT_RANGE = (0.0, 5.0)
# Loose sanity limits; values outside these almost always mean radians or a sign/axis bug.
GAZE_LIMIT_DEG = 90.0
POSE_LIMIT_DEG = 180.0


class SchemaError(ValueError):
    pass


def au_columns(df):
    """Sorted AU column names in df."""
    return sorted(c for c in df.columns if AU_PATTERN.match(c))


def au_ids(df):
    """Sorted AU ids present in df, e.g. ['AU01', 'AU12']."""
    return sorted({c.split("_")[0] for c in au_columns(df)})


def output_columns(df):
    """All model output columns (AUs, gaze, pose) present in df."""
    return au_columns(df) + [c for c in GAZE_COLUMNS + POSE_COLUMNS if c in df.columns]


def add_missing_au_pairs(df):
    """For every AU with only _int or only _occ, add the other column as NaN."""
    df = df.copy()
    for au in au_ids(df):
        for kind in ("int", "occ"):
            col = f"{au}_{kind}"
            if col not in df.columns:
                df[col] = np.nan
    return df


def validate(df, clip_id=None):
    """Raise SchemaError if df does not follow the schema. Returns df unchanged."""
    errors = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    unknown = [
        c for c in df.columns
        if c not in REQUIRED_COLUMNS
        and c not in GAZE_COLUMNS + POSE_COLUMNS
        and not AU_PATTERN.match(c)
        and not c.startswith(EXTRA_PREFIX)
    ]
    if unknown:
        errors.append(f"unknown columns (rename or prefix with '{EXTRA_PREFIX}'): {unknown}")

    if not output_columns(df):
        errors.append("no model output columns (AU, gaze, or pose)")

    # Required column types and values
    if df["clip_id"].nunique() != 1:
        errors.append("clip_id must have exactly one value per file")
    elif clip_id is not None and df["clip_id"].iloc[0] != clip_id:
        errors.append(f"clip_id is {df['clip_id'].iloc[0]!r}, expected {clip_id!r}")

    if not pd.api.types.is_integer_dtype(df["frame"]):
        errors.append(f"frame must be integer, got {df['frame'].dtype}")
    elif not df["frame"].is_monotonic_increasing or df["frame"].duplicated().any():
        errors.append("frame must be strictly increasing")

    for col in ("timestamp_ms", "face_conf"):
        if not pd.api.types.is_float_dtype(df[col]):
            errors.append(f"{col} must be float, got {df[col].dtype}")

    if df["timestamp_ms"].isna().any():
        errors.append("timestamp_ms has NaN values")
    elif not df["timestamp_ms"].is_monotonic_increasing:
        errors.append("timestamp_ms must be non-decreasing")

    if df["face_conf"].isna().any() or not df["face_conf"].between(0, 1).all():
        errors.append("face_conf must be in [0, 1] with no NaN (use 0 for no face)")

    # Paired AU columns
    for au in au_ids(df):
        for kind in ("int", "occ"):
            if f"{au}_{kind}" not in df.columns:
                errors.append(f"{au}_{kind} missing (call add_missing_au_pairs)")

    # Output types and plausible ranges
    for col in output_columns(df):
        if not pd.api.types.is_float_dtype(df[col]):
            errors.append(f"{col} must be float, got {df[col].dtype}")
            continue
        values = df[col].dropna()
        if col.endswith("_int") and not values.between(*AU_INT_RANGE).all():
            errors.append(f"{col} outside {AU_INT_RANGE}; rescale to 0-5")
        elif col.endswith("_occ") and not values.isin([0.0, 1.0]).all():
            errors.append(f"{col} must be 0.0 / 1.0 (threshold probabilities first)")
        elif col in GAZE_COLUMNS and (values.abs() > GAZE_LIMIT_DEG).any():
            errors.append(f"{col} has |value| > {GAZE_LIMIT_DEG} deg; radians?")
        elif col in POSE_COLUMNS and (values.abs() > POSE_LIMIT_DEG).any():
            errors.append(f"{col} has |value| > {POSE_LIMIT_DEG} deg; radians?")

    # No-face rows must not carry outputs
    no_face = df["face_conf"] == 0
    if no_face.any() and df.loc[no_face, output_columns(df)].notna().any().any():
        errors.append("rows with face_conf == 0 must have NaN outputs")

    if errors:
        raise SchemaError("schema check failed:\n  - " + "\n  - ".join(errors))
    return df
