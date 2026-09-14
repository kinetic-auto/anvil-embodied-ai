#!/usr/bin/env python3
"""Plot observation.velocity and observation.effort for one LeRobot episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


NAMES = [
    "right_finger_joint1",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    parquet = next((args.dataset / "data").rglob("*.parquet"))
    table = pq.read_table(parquet)
    ep = table.filter(pc.field("episode_index") == args.episode)
    if ep.num_rows == 0:
        raise SystemExit(f"episode {args.episode} not in {parquet}")

    ts = np.array(ep.column("timestamp").to_pylist(), dtype=np.float64)
    vel = np.stack(ep.column("observation.velocity").to_pylist()).astype(np.float64)
    eff = np.stack(ep.column("observation.effort").to_pylist()).astype(np.float64)
    n = vel.shape[1]
    names = NAMES[:n]
    t = ts - ts[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        f"{args.dataset.name}  episode {args.episode}  ({len(t)} frames, {t[-1]:.1f}s)",
        fontsize=12,
    )
    for j, name in enumerate(names):
        axes[0].plot(t, vel[:, j], lw=0.9, label=name)
        axes[1].plot(t, eff[:, j], lw=0.9, label=name)
    axes[0].set_ylabel("velocity")
    axes[1].set_ylabel("effort")
    axes[1].set_xlabel("t (s)")
    axes[0].legend(fontsize=7, ncol=4, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"Saved: {args.output}  frames={len(t)}")
    print("velocity  mean|abs|:", {n: float(np.mean(np.abs(vel[:, j]))) for j, n in enumerate(names)})
    print("effort    mean|abs|:", {n: float(np.mean(np.abs(eff[:, j]))) for j, n in enumerate(names)})


if __name__ == "__main__":
    main()
