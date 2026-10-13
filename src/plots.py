"""Standard quick-check plots and metrics, identical for every model.

All plot functions take a DataFrame in the common schema (see src/schema.py)
and return the matplotlib Figure.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.schema import GAZE_COLUMNS, POSE_COLUMNS, au_ids

DEFAULT_CONF_THRESHOLD = 0.5


def detection_rate(df, threshold=DEFAULT_CONF_THRESHOLD):
    """Fraction of frames with face_conf >= threshold, per clip."""
    return (df["face_conf"] >= threshold).groupby(df["clip_id"]).mean().rename("detection_rate")


def failure_frames(df, threshold=DEFAULT_CONF_THRESHOLD):
    """Rows where the face was missed or detected with low confidence."""
    return df.loc[df["face_conf"] < threshold, ["clip_id", "frame", "timestamp_ms", "face_conf"]]


def fps_table(timings, df):
    """Frames per second per clip. timings: {clip_id: seconds of wall time}."""
    n_frames = df.groupby("clip_id").size()
    seconds = pd.Series(timings, name="seconds")
    return pd.DataFrame({"n_frames": n_frames, "seconds": seconds}).assign(
        fps=lambda t: t["n_frames"] / t["seconds"]
    )


def _time_s(df):
    return df["timestamp_ms"] / 1000.0


def plot_face_conf(df, threshold=DEFAULT_CONF_THRESHOLD):
    """face_conf over time for one clip, with the detection threshold."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(_time_s(df), df["face_conf"], lw=0.8)
    ax.axhline(threshold, color="grey", ls="--", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="face_conf", ylim=(-0.05, 1.05), title=df["clip_id"].iloc[0])
    fig.tight_layout()
    return fig


def plot_aus(df, kind="int"):
    """One row per AU over time for one clip. kind: 'int' or 'occ'."""
    aus = [au for au in au_ids(df) if df[f"{au}_{kind}"].notna().any()]
    if not aus:
        raise ValueError(f"no AU_{kind} values in this output")
    fig, axes = plt.subplots(len(aus), 1, figsize=(10, 0.9 * len(aus) + 1), sharex=True, squeeze=False)
    ylim = (-0.2, 5.2) if kind == "int" else (-0.1, 1.1)
    t = _time_s(df)
    for ax, au in zip(axes[:, 0], aus):
        ax.plot(t, df[f"{au}_{kind}"], lw=0.8)
        ax.set_ylim(ylim)
        ax.set_ylabel(au, rotation=0, ha="right", va="center")
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(f"{df['clip_id'].iloc[0]}: AU {kind}")
    fig.tight_layout()
    return fig


def _plot_angles(df, columns, title):
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError(f"no {title} columns in this output")
    fig, ax = plt.subplots(figsize=(10, 3))
    for col in columns:
        ax.plot(_time_s(df), df[col], lw=0.8, label=col)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set(xlabel="time (s)", ylabel="degrees", title=f"{df['clip_id'].iloc[0]}: {title}")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_gaze(df):
    return _plot_angles(df, GAZE_COLUMNS, "gaze")


def plot_pose(df):
    return _plot_angles(df, POSE_COLUMNS, "head pose")


def plot_detection_rate(df, threshold=DEFAULT_CONF_THRESHOLD):
    """Bar chart of detection rate per clip, for one model's combined outputs."""
    rates = detection_rate(df, threshold)
    fig, ax = plt.subplots(figsize=(max(4, 0.5 * len(rates) + 2), 3))
    ax.bar(rates.index.astype(str), rates.values)
    ax.set(ylabel=f"detection rate (conf >= {threshold})", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
