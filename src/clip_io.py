"""Reading test clips and reading/writing model outputs.

Notebooks should only touch data/ and outputs/ through these functions, so every
model sees the same frames and timestamps and every output passes the schema check.
"""

from pathlib import Path

import cv2
import pandas as pd

from src.schema import validate

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MANIFEST_PATH = DATA_DIR / "clips_manifest.csv"

# Columns the manifest must have. video_path is relative to data/.
MANIFEST_COLUMNS = ["clip_id", "video_path"]


def load_manifest(path=MANIFEST_PATH):
    manifest = pd.read_csv(path, dtype={"clip_id": str})
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if manifest["clip_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate clip_id values")
    return manifest


def clip_path(row):
    """Absolute video path for a manifest row (Series or namedtuple)."""
    return DATA_DIR / row.video_path


def video_info(video_path):
    """Container metadata. n_frames comes from the file header and can be wrong;
    count the frames from iter_frames for the real number."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(video_path, rgb=False):
    """Yield (frame, timestamp_ms, image) for every frame in the video.

    timestamp_ms is the presentation timestamp from the container, so it stays
    correct for variable-frame-rate clips. It does not necessarily start at 0.
    image is BGR uint8 unless rgb=True.

    Frames the decoder flushes at the end of a stream can have no timestamp
    (OpenCV reports 0.0). Those get the previous timestamp plus the previous
    frame interval, and a warning is printed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    try:
        frame = 0
        prev_ts, prev_step, n_filled = None, None, 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if prev_ts is not None and timestamp_ms <= prev_ts:
                if prev_step is None:
                    raise ValueError(f"{video_path}: no usable timestamp at frame {frame}")
                timestamp_ms = prev_ts + prev_step
                n_filled += 1
            elif prev_ts is not None:
                prev_step = timestamp_ms - prev_ts
            prev_ts = timestamp_ms
            if rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            yield frame, timestamp_ms, image
            frame += 1
    finally:
        cap.release()
    if n_filled:
        print(f"warning: {video_path}: {n_filled} frame(s) had no timestamp; extrapolated")


def output_path(model, clip_id):
    return OUTPUTS_DIR / model / f"{clip_id}.parquet"


def save_output(df, model, clip_id):
    """Validate df against the schema, then write outputs/<model>/<clip_id>.parquet."""
    validate(df, clip_id=clip_id)
    path = output_path(model, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_output(model, clip_id):
    return pd.read_parquet(output_path(model, clip_id))


def load_outputs(model):
    """All of a model's outputs concatenated into one DataFrame."""
    paths = sorted((OUTPUTS_DIR / model).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no outputs for model {model!r} in {OUTPUTS_DIR / model}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
