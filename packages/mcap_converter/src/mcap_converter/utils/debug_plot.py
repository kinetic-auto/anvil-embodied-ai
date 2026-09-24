"""Debug plots for converted LeRobot datasets.

Generates per-episode observation.state vs action comparison plots to visually
verify action_from_observation_n alignment after conversion.

Packed obs/action can differ (e.g. effort-only state vs pos+effort action).
Subplots are matched by ``info.json`` channel name, not by vector index.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional


def overlay_channels(
    obs_names: list[str],
    act_names: list[str],
    n_obs: int,
    n_act: int,
) -> list[tuple[str, int | None, int | None]]:
    """Build (title, obs_index, action_index) rows, matching packed names."""
    obs_ok = len(obs_names) == n_obs
    act_ok = len(act_names) == n_act
    if obs_ok and act_ok and (obs_names or act_names):
        obs_idx = {n: i for i, n in enumerate(obs_names)}
        act_idx = {n: i for i, n in enumerate(act_names)}
        ordered = list(dict.fromkeys([*act_names, *obs_names]))
        return [(n, obs_idx.get(n), act_idx.get(n)) for n in ordered]

    n_pair = min(n_obs, n_act)
    names = obs_names if obs_ok and obs_names else [f"joint_{i}" for i in range(n_pair)]
    channels: list[tuple[str, int | None, int | None]] = [
        (names[i] if i < len(names) else f"joint_{i}", i, i) for i in range(n_pair)
    ]
    if n_obs != n_act:
        for i in range(n_pair, n_obs):
            channels.append((f"observation.state[{i}]", i, None))
        for i in range(n_pair, n_act):
            channels.append((f"action[{i}]", None, i))
    return channels


def plot_conversion_debug(
    output_dir: str,
    n_episodes: int = 5,
    action_from_observation_n: Optional[int] = None,
) -> None:
    """Generate obs_state vs action debug plots for the first N episodes.

    Reads converted parquet data and meta/info.json from output_dir.
    Saves one PNG per episode to {output_dir}/debug_plots/.

    Args:
        output_dir: Path to the converted LeRobot dataset directory.
        n_episodes: Number of episodes to plot (default 5).
        action_from_observation_n: Frame offset used during conversion, shown
            in plot titles. If None, attempts to read from conversion_config.yaml.
    """
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    import numpy as np

    root = Path(output_dir)
    plots_dir = root / "debug_plots"
    plots_dir.mkdir(exist_ok=True)

    info_path = root / "meta" / "info.json"
    obs_names: list[str] = []
    act_names: list[str] = []
    features: dict = {}
    fps: int = 30
    if info_path.exists():
        info = json.loads(info_path.read_text())
        features = info.get("features", {})
        obs_names = list(features.get("observation.state", {}).get("names", []) or [])
        act_names = list(features.get("action", {}).get("names", []) or [])
        fps = info.get("fps", 30)
    has_obs_state = "observation.state" in features

    if action_from_observation_n is None:
        config_path = root / "conversion_config.yaml"
        if config_path.exists():
            try:
                import yaml
                cfg = yaml.safe_load(config_path.read_text())
                action_from_observation_n = cfg.get("action_from_observation_n", 10)
            except Exception:
                action_from_observation_n = 10
        else:
            action_from_observation_n = 10

    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        print(f"[debug_plot] No parquet files found in {root / 'data'}")
        return

    columns = ["episode_index", "frame_index", "action"]
    if has_obs_state:
        columns.insert(2, "observation.state")

    tables = []
    for f in data_files:
        tbl = pq.read_table(f, columns=columns)
        import pyarrow.compute as pc
        mask = pc.less(tbl["episode_index"], n_episodes)
        filtered = tbl.filter(mask)
        if filtered.num_rows > 0:
            tables.append(filtered)

    if not tables:
        print(f"[debug_plot] No data found for first {n_episodes} episodes")
        return

    import pyarrow as pa
    combined = pa.concat_tables(tables)
    ep_indices = combined["episode_index"].to_pylist()
    frame_indices = combined["frame_index"].to_pylist()
    obs_state_rows = (
        combined["observation.state"].to_pylist() if has_obs_state else [None] * len(ep_indices)
    )
    action_rows = combined["action"].to_pylist()

    episodes: dict[int, dict] = {}
    for ep, fr, obs, act in zip(ep_indices, frame_indices, obs_state_rows, action_rows):
        if ep not in episodes:
            episodes[ep] = {"frames": [], "obs": [], "act": []}
        episodes[ep]["frames"].append(fr)
        episodes[ep]["obs"].append(obs)
        episodes[ep]["act"].append(act)

    offset_sec = action_from_observation_n / fps
    offset_label = f"n={action_from_observation_n} ({offset_sec*1000:.0f}ms @ {fps}fps)"

    for ep_idx in sorted(episodes.keys())[:n_episodes]:
        ep_data = episodes[ep_idx]
        order = sorted(range(len(ep_data["frames"])), key=lambda i: ep_data["frames"][i])
        act_arr = np.array([ep_data["act"][i] for i in order], dtype=np.float32)
        if has_obs_state:
            obs_arr = np.array([ep_data["obs"][i] for i in order], dtype=np.float32)
        else:
            obs_arr = np.zeros((act_arr.shape[0], 0), dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[:, None]
        if act_arr.ndim == 1:
            act_arr = act_arr[:, None]

        channels = overlay_channels(
            obs_names, act_names, obs_arr.shape[1], act_arr.shape[1]
        )
        n_plots = len(channels)
        ncols = min(4, n_plots)
        nrows = math.ceil(n_plots / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        fig.suptitle(
            f"Episode {ep_idx} — obs.state (blue) vs action (orange) by name | offset {offset_label}",
            fontsize=11,
        )

        frames = np.arange(obs_arr.shape[0])
        legend_done = False
        for j, (name, obs_i, act_i) in enumerate(channels):
            row, col = divmod(j, ncols)
            ax = axes[row][col]
            if obs_i is not None:
                ax.plot(
                    frames,
                    obs_arr[:, obs_i],
                    color="steelblue",
                    linewidth=1.0,
                    label="obs.state",
                )
            if act_i is not None:
                ax.plot(
                    frames,
                    act_arr[:, act_i],
                    color="darkorange",
                    linewidth=1.0,
                    linestyle="--",
                    label="action",
                )
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("frame", fontsize=8)
            ax.tick_params(labelsize=7)
            if not legend_done and obs_i is not None and act_i is not None:
                ax.legend(fontsize=7)
                legend_done = True

        for j in range(n_plots, nrows * ncols):
            row, col = divmod(j, ncols)
            axes[row][col].set_visible(False)

        fig.tight_layout()
        out_path = plots_dir / f"episode_{ep_idx:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[debug_plot] Saved {out_path}")
