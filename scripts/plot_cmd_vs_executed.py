#!/usr/bin/env python3
"""Plot inference command vs robot-executed joint position from a monitor CSV.

obs_state  = joint positions at the moment the command was published
control_cmd = action sent to /follower_*_forward_position_controller/commands

Same-step error is cmd[t] - obs[t] (how far the arm was from the new target).
Lag-1 error is cmd[t] - obs[t+1] (how well the next sample tracked the command).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Matches inference_openyam_single.yaml controller_joint_order
_DEFAULT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "finger_joint1",
]


def _load(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    with csv_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            reader = csv.DictReader([line] + list(f))
            rows = list(reader)
            break
    if len(rows) < 2:
        raise SystemExit(f"too few rows in {csv_path}")

    def cols(prefix: str) -> np.ndarray:
        keys = sorted(
            [k for k in rows[0] if k.startswith(prefix)],
            key=lambda c: int(c.rsplit("_", 1)[-1]),
        )
        return np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)

    ts = np.array([float(r["timestamp"]) for r in rows], dtype=np.float64)
    return ts, cols("obs_state_"), cols("control_cmd_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()
    ts, obs, cmd = _load(args.csv)
    n = min(obs.shape[1], cmd.shape[1])
    obs, cmd = obs[:, :n], cmd[:, :n]
    names = _DEFAULT_NAMES[:n]
    t = ts - ts[0]
    err = cmd - obs
    err_lag = cmd[:-1] - obs[1:]
    t_lag = t[:-1]

    ncols = 4
    nrows = int(np.ceil(n / ncols)) * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows), squeeze=False)
    fig.suptitle(
        f"Commanded vs executed  ({len(t)} steps, {t[-1]:.1f}s)\n"
        f"MAE same-step={np.mean(np.abs(err)):.4f} rad   "
        f"MAE lag-1={np.mean(np.abs(err_lag)):.4f} rad",
        fontsize=12,
    )

    for j in range(n):
        row, col = (j // ncols) * 2, j % ncols
        ax = axes[row][col]
        ax.plot(t, obs[:, j], color="steelblue", lw=1.0, label="executed (obs)")
        ax.plot(t, cmd[:, j], color="crimson", lw=1.0, ls="--", label="commanded")
        ax.set_title(names[j], fontsize=9)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=6, loc="best")

        ax_e = axes[row + 1][col]
        ax_e.plot(t, err[:, j], color="0.35", lw=0.8, label="cmd−obs (same step)")
        ax_e.plot(t_lag, err_lag[:, j], color="darkorange", lw=0.8, label="cmd[t]−obs[t+1]")
        ax_e.axhline(0.0, color="0.7", lw=0.5)
        ax_e.set_ylabel("error rad", fontsize=7)
        ax_e.set_xlabel("t (s)", fontsize=7)
        ax_e.tick_params(labelsize=6)
        if j == 0:
            ax_e.legend(fontsize=5, loc="best")

    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")

    fig.tight_layout()
    out = args.output or args.csv.parent / "cmd_vs_executed.png"
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}")
    print("per-joint MAE same-step (rad):")
    for name, mae in zip(names, np.mean(np.abs(err), axis=0)):
        print(f"  {name:16s} {mae:.5f}")
    print("per-joint MAE lag-1 (rad):")
    for name, mae in zip(names, np.mean(np.abs(err_lag), axis=0)):
        print(f"  {name:16s} {mae:.5f}")


if __name__ == "__main__":
    main()
