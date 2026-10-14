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
# Each test set is a folder under data/ with its own manifest.csv.
# path is relative to that folder; any other columns are free-form notes.
MANIFEST_NAME = "manifest.csv"
MANIFEST_COLUMNS = ["clip_id", "path"]

# Still images are treated as one-frame clips (frame 0, timestamp 0 ms).
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"cannot read image {path}")
    return image


def load_manifest(test_set, categories=None):
    """Read data/<test_set>/manifest.csv. Adds a test_set column used by clip_path.

    categories: optional list of values in the manifest's category column
    (e.g. ["gaze", "emotion"]); only those rows are kept. None keeps every row.
    """
    path = DATA_DIR / test_set / MANIFEST_NAME
    manifest = pd.read_csv(path, dtype={"clip_id": str})
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if manifest["clip_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate clip_id values")
    if categories is not None:
        if "category" not in manifest.columns:
            raise ValueError(f"{path} has no category column to filter on")
        unknown = sorted(set(categories) - set(manifest["category"]))
        if unknown:
            raise ValueError(f"{path} has no rows with category {unknown}")
        manifest = manifest[manifest["category"].isin(categories)].reset_index(drop=True)
    manifest.insert(2, "test_set", test_set)
    missing_files = [p for p in manifest["path"] if not (DATA_DIR / test_set / p).exists()]
    if missing_files:
        raise FileNotFoundError(f"{path} lists files that don't exist: {missing_files}")
    return manifest


def clip_path(row):
    """Absolute video path for a manifest row (Series or namedtuple)."""
    return DATA_DIR / row.test_set / row.path


def video_info(video_path):
    """Container metadata. n_frames comes from the file header and can be wrong;
    count the frames from iter_frames for the real number."""
    if is_image(video_path):
        height, width = _read_image(video_path).shape[:2]
        return {"fps": None, "n_frames": 1, "width": width, "height": height}
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

    A still image yields a single frame: (0, 0.0, image).
    """
    if is_image(video_path):
        image = _read_image(video_path)
        yield 0, 0.0, cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if rgb else image
        return
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


def load_outputs(model, clip_ids=None):
    """A model's outputs concatenated into one DataFrame.

    clip_ids: optional list of clips to load (e.g. manifest["clip_id"]), so
    files left over from earlier runs on other clips are skipped. None loads all.
    """
    paths = sorted((OUTPUTS_DIR / model).glob("*.parquet"))
    if clip_ids is not None:
        wanted = set(clip_ids)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        raise FileNotFoundError(f"no outputs for model {model!r} in {OUTPUTS_DIR / model}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
