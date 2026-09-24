#!/usr/bin/env python3
"""Overlay measured vs predicted position and effort from an inference monitor CSV.

Usage:
    python scripts/plot_inference_overview.py monitor_output/inference_data.csv
    python scripts/plot_inference_overview.py monitor_output/inference_data.csv -o report.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODEL_NAMES = [
    "right_finger_joint1",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
]
# controller_joint_order → model_joint_order
# ctrl: j1,j2,j3,j4,j5,j6,finger  →  model: finger,j1,j2,j3,j4,j5,j6
CTRL_TO_MODEL = [6, 0, 1, 2, 3, 4, 5]


def _extract(rows: list[dict[str, str]], prefix: str) -> np.ndarray | None:
    keys = sorted(
        [k for k in rows[0] if k.startswith(prefix)],
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    if not keys:
        return None
    return np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)


def _to_model_order(values: np.ndarray) -> np.ndarray:
    if values.shape[1] != len(CTRL_TO_MODEL):
        return values
    return values[:, CTRL_TO_MODEL]


def load_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    if len(rows) < 2:
        raise SystemExit(f"too few rows ({len(rows)}) in {path}")

    ts = np.array([float(r["timestamp"]) for r in rows], dtype=np.float64)
    ts = ts - ts[0]
    obs = _extract(rows, "obs_state_")
    if obs is None:
        raise SystemExit(f"missing obs_state_* columns in {path}")
    effort = _extract(rows, "obs_effort_")
    raw = _extract(rows, "raw_output_")
    if raw is None or raw.shape[1] < 14:
        raise SystemExit(f"need 14-dim raw_output_* (pos|eff) in {path}")

    n = min(7, obs.shape[1], raw.shape[1] // 2)
    return (
        ts,
        _to_model_order(obs)[:, :n],
        None if effort is None else _to_model_order(effort)[:, :n],
        raw[:, :n],
        raw[:, n : 2 * n],
    )


def plot_overview(
    ts: np.ndarray,
    obs_pos: np.ndarray,
    obs_eff: np.ndarray | None,
    pred_pos: np.ndarray,
    pred_eff: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    n = obs_pos.shape[1]
    names = MODEL_NAMES[:n]
    colors = plt.get_cmap("tab10")(np.arange(n) % 10)

    fig, axes = plt.subplots(2, 1, figsize=(24.0, 8.5), sharex=True)
    fig.suptitle(title, fontsize=14)

    for j, name in enumerate(names):
        axes[0].plot(ts, obs_pos[:, j], lw=1.2, color=colors[j], label=name)
        axes[0].plot(ts, pred_pos[:, j], lw=1.2, ls="--", color=colors[j])
        if obs_eff is not None:
            axes[1].plot(ts, obs_eff[:, j], lw=1.2, color=colors[j])
        axes[1].plot(ts, pred_eff[:, j], lw=1.2, ls="--", color=colors[j])

    axes[0].set_ylabel("position (rad)")
    axes[1].set_ylabel("effort (Nm)")
    axes[1].set_xlabel("time (s)")
    if obs_eff is None:
        axes[1].set_title("measured effort missing — dashed = predicted only", fontsize=9, loc="left")

    style_handles = [
        Line2D([0], [0], color="0.25", lw=1.5, label="obs"),
        Line2D([0], [0], color="0.25", lw=1.5, ls="--", label="inference"),
    ]
    joint_handles, joint_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles=joint_handles + style_handles,
        labels=joint_labels + [h.get_label() for h in style_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=n + 2,
        fontsize=8,
        frameon=False,
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to inference_data.csv")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <csv_dir>/inference_overview.png)",
    )
    args = parser.parse_args()
    if not args.csv.exists():
        raise SystemExit(f"ERROR: {args.csv} not found")

    ts, obs_pos, obs_eff, pred_pos, pred_eff = load_csv(args.csv)
    output = args.output or args.csv.parent / "inference_overview.png"
    has_effort = obs_eff is not None
    title = (
        f"Inference overlay  ({len(ts)} steps, {ts[-1]:.1f}s)  "
        f"solid=obs  dashed=inference  "
        f"{'pos+effort measured' if has_effort else 'pos measured, effort predicted only'}"
    )
    plot_overview(ts, obs_pos, obs_eff, pred_pos, pred_eff, title, output)


if __name__ == "__main__":
    main()
