#!/usr/bin/env python3
"""Plot one LeRobot episode as a chest-camera keyframe strip over synced joint signals.

The episode is split into equal time intervals by dashed boundary markers on the
signal axes; the top row holds the chest-camera frame sampled at each interval's
midpoint. Below the strip, observation.state, observation.velocity and
observation.effort share a single time axis, so image n covers the span between
boundary n and boundary n + 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHEST_CAMERA_KEY = "observation.images.chest"

FIGURE_WIDTH_INCHES = 24.0

SIGNAL_COLUMNS = [
    ("observation.state", "joint position (rad)"),
    ("observation.velocity", "joint velocity (rad/s)"),
    ("observation.effort", "joint effort (Nm)"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--keyframe-count", type=int, default=18)
    parser.add_argument("-o", "--output", type=Path, required=True)
    arguments = parser.parse_args()

    episode_metadata = read_episode_metadata(arguments.dataset, arguments.episode)
    timestamps, signals, joint_names = read_episode_signals(
        arguments.dataset, arguments.episode
    )
    boundary_timestamps = select_boundary_timestamps(
        timestamps, arguments.keyframe_count
    )
    keyframe_timestamps = midpoints_between(boundary_timestamps)
    keyframe_images = decode_chest_camera_frames(
        arguments.dataset, episode_metadata, keyframe_timestamps
    )

    figure = draw_episode_overview(
        dataset_name=arguments.dataset.name,
        episode_index=arguments.episode,
        timestamps=timestamps,
        signals=signals,
        joint_names=joint_names,
        boundary_timestamps=boundary_timestamps,
        keyframe_timestamps=keyframe_timestamps,
        keyframe_images=keyframe_images,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=130)
    print(
        f"Saved: {arguments.output}  frames={len(timestamps)}  "
        f"duration={timestamps[-1]:.1f}s  keyframes={len(keyframe_timestamps)}"
    )


def read_episode_metadata(dataset_root: Path, episode_index: int) -> dict[str, object]:
    metadata_files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not metadata_files:
        raise SystemExit(f"no episode metadata under {dataset_root / 'meta' / 'episodes'}")

    wanted_columns = [
        "episode_index",
        "length",
        f"videos/{CHEST_CAMERA_KEY}/chunk_index",
        f"videos/{CHEST_CAMERA_KEY}/file_index",
        f"videos/{CHEST_CAMERA_KEY}/from_timestamp",
        f"videos/{CHEST_CAMERA_KEY}/to_timestamp",
    ]
    for metadata_file in metadata_files:
        table = pq.read_table(metadata_file, columns=wanted_columns)
        episode_rows = table.filter(pc.field("episode_index") == episode_index)
        if episode_rows.num_rows:
            return {name: episode_rows.column(name)[0].as_py() for name in wanted_columns}
    raise SystemExit(f"episode {episode_index} not found in {dataset_root}")


def read_episode_signals(
    dataset_root: Path, episode_index: int
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not data_files:
        raise SystemExit(f"no data parquet under {dataset_root / 'data'}")

    for data_file in data_files:
        table = pq.read_table(data_file)
        episode_rows = table.filter(pc.field("episode_index") == episode_index)
        if not episode_rows.num_rows:
            continue
        timestamps = np.asarray(
            episode_rows.column("timestamp").to_pylist(), dtype=np.float64
        )
        signals = {
            column: np.stack(episode_rows.column(column).to_pylist()).astype(np.float64)
            for column, _ in SIGNAL_COLUMNS
        }
        joint_names = read_joint_names(dataset_root, next(iter(signals.values())).shape[1])
        return timestamps - timestamps[0], signals, joint_names
    raise SystemExit(f"episode {episode_index} has no rows in {dataset_root / 'data'}")


def read_joint_names(dataset_root: Path, joint_count: int) -> list[str]:
    import json

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"].get("names") or []
    if len(names) != joint_count:
        return [f"joint{index}" for index in range(joint_count)]
    return list(names)


def select_boundary_timestamps(timestamps: np.ndarray, keyframe_count: int) -> np.ndarray:
    if keyframe_count < 1:
        raise SystemExit("--keyframe-count must be at least 1")
    return np.linspace(timestamps[0], timestamps[-1], keyframe_count + 1)


def midpoints_between(boundary_timestamps: np.ndarray) -> np.ndarray:
    return (boundary_timestamps[:-1] + boundary_timestamps[1:]) / 2.0


def decode_chest_camera_frames(
    dataset_root: Path,
    episode_metadata: dict[str, object],
    keyframe_timestamps: np.ndarray,
) -> list[np.ndarray]:
    video_path = (
        dataset_root
        / "videos"
        / CHEST_CAMERA_KEY
        / f"chunk-{episode_metadata[f'videos/{CHEST_CAMERA_KEY}/chunk_index']:03d}"
        / f"file-{episode_metadata[f'videos/{CHEST_CAMERA_KEY}/file_index']:03d}.mp4"
    )
    if not video_path.exists():
        raise SystemExit(f"chest camera video not found: {video_path}")

    episode_start = float(episode_metadata[f"videos/{CHEST_CAMERA_KEY}/from_timestamp"])
    images: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        video_stream = container.streams.video[0]
        for keyframe_timestamp in keyframe_timestamps:
            images.append(
                decode_frame_at(
                    container, video_stream, episode_start + float(keyframe_timestamp)
                )
            )
    return images


def decode_frame_at(container, video_stream, target_seconds: float) -> np.ndarray:
    container.seek(
        int(target_seconds / video_stream.time_base), stream=video_stream, backward=True
    )
    last_frame = None
    for frame in container.decode(video_stream):
        last_frame = frame
        if frame.time is not None and frame.time >= target_seconds:
            break
    if last_frame is None:
        raise SystemExit(f"no frame decodable at t={target_seconds:.3f}s")
    return last_frame.to_ndarray(format="rgb24")


def draw_episode_overview(
    dataset_name: str,
    episode_index: int,
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    joint_names: list[str],
    boundary_timestamps: np.ndarray,
    keyframe_timestamps: np.ndarray,
    keyframe_images: list[np.ndarray],
) -> plt.Figure:
    figure = plt.figure(figsize=(FIGURE_WIDTH_INCHES, 12.0))
    outer_grid = figure.add_gridspec(
        2, 1, height_ratios=[1.4, 5.0], hspace=0.12, top=0.93, bottom=0.06, left=0.06, right=0.98
    )

    draw_keyframe_strip(
        figure, outer_grid[0], keyframe_images, keyframe_timestamps
    )
    signal_axes = draw_signal_axes(
        figure, outer_grid[1], timestamps, signals, joint_names, boundary_timestamps
    )

    figure.suptitle(
        f"{dataset_name}   episode {episode_index}   "
        f"{len(timestamps)} frames, {timestamps[-1]:.1f}s",
        fontsize=14,
    )
    signal_axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=len(joint_names),
        fontsize=8,
        frameon=False,
    )
    return figure


def draw_keyframe_strip(
    figure: plt.Figure,
    grid_cell,
    keyframe_images: list[np.ndarray],
    keyframe_timestamps: np.ndarray,
) -> None:
    strip_grid = grid_cell.subgridspec(1, len(keyframe_images), wspace=0.03)
    for position, (image, keyframe_timestamp) in enumerate(
        zip(keyframe_images, keyframe_timestamps)
    ):
        axis = figure.add_subplot(strip_grid[0, position])
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{position + 1}   t={keyframe_timestamp:.1f}s", fontsize=8)
        for spine in axis.spines.values():
            spine.set_color("0.4")


def draw_signal_axes(
    figure: plt.Figure,
    grid_cell,
    timestamps: np.ndarray,
    signals: dict[str, np.ndarray],
    joint_names: list[str],
    boundary_timestamps: np.ndarray,
) -> list[plt.Axes]:
    signal_grid = grid_cell.subgridspec(len(SIGNAL_COLUMNS), 1, hspace=0.08)
    keyframe_timestamps = midpoints_between(boundary_timestamps)
    joint_colors = plt.get_cmap("tab10")(np.arange(len(joint_names)) % 10)

    axes: list[plt.Axes] = []
    for row, (column, axis_label) in enumerate(SIGNAL_COLUMNS):
        axis = figure.add_subplot(signal_grid[row, 0], sharex=axes[0] if axes else None)
        for joint_index, joint_name in enumerate(joint_names):
            axis.plot(
                timestamps,
                signals[column][:, joint_index],
                lw=1.0,
                color=joint_colors[joint_index],
                label=joint_name if row == 0 else None,
            )
        draw_interval_markers(
            axis, boundary_timestamps, keyframe_timestamps, annotate=row == 0
        )
        axis.set_ylabel(axis_label, fontsize=9)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=8)
        if row < len(SIGNAL_COLUMNS) - 1:
            axis.tick_params(labelbottom=False)
        axes.append(axis)

    axes[-1].set_xlabel("time (s)", fontsize=10)
    axes[-1].set_xlim(timestamps[0], timestamps[-1])
    return axes


def draw_interval_markers(
    axis: plt.Axes,
    boundary_timestamps: np.ndarray,
    keyframe_timestamps: np.ndarray,
    annotate: bool,
) -> None:
    for boundary_timestamp in boundary_timestamps:
        axis.axvline(boundary_timestamp, color="0.35", lw=0.8, ls="--", alpha=0.7)
    for position, keyframe_timestamp in enumerate(keyframe_timestamps):
        if annotate:
            axis.annotate(
                str(position + 1),
                xy=(keyframe_timestamp, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="0.25",
            )


if __name__ == "__main__":
    main()
