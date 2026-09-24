#!/usr/bin/env python3
"""Write the ``io`` block into a checkpoint's ``anvil_config.json`` from its dataset names.

Usage:
    python scripts/annotate_checkpoint_io.py \
        --checkpoint model_zoo/<run>/checkpoints/last \
        --dataset /var/tmp/anvil-robot/data/datasets/<dataset>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from anvil_shared.state_observs import io_layout_from_dataset_features


def resolve_pretrained_dir(checkpoint: Path) -> Path:
    """Return the directory holding ``config.json``."""
    pretrained = checkpoint / "pretrained_model"
    if (pretrained / "config.json").exists():
        return pretrained
    if (checkpoint / "config.json").exists():
        return checkpoint
    raise FileNotFoundError(f"no config.json under {checkpoint} or its pretrained_model/")


def checkpoint_widths(pretrained_dir: Path) -> tuple[int, int]:
    """Return ``(state_dim, action_dim)`` from the policy ``config.json``."""
    config = json.loads((pretrained_dir / "config.json").read_text())
    state_shape = config["input_features"]["observation.state"]["shape"]
    action_shape = config["output_features"]["action"]["shape"]
    return int(state_shape[0]), int(action_shape[0])


def validate_layout_widths(
    io_layout: dict[str, list[str]], state_dim: int, action_dim: int
) -> None:
    """Check the dataset layout against the checkpoint widths."""
    n_joints = len(io_layout["joint_names"])
    expected_state = n_joints * len(io_layout["state_blocks"])
    expected_action = n_joints * len(io_layout["action_blocks"])
    if expected_state != state_dim or expected_action != action_dim:
        raise ValueError(
            f"dataset layout packs state {expected_state} / action {expected_action} but the "
            f"checkpoint has state {state_dim} / action {action_dim}"
        )


def annotate(checkpoint: Path, dataset: Path, dry_run: bool) -> dict[str, list[str]]:
    """Write the derived ``io`` block into ``anvil_config.json``."""
    pretrained_dir = resolve_pretrained_dir(checkpoint)
    info = json.loads((dataset / "meta" / "info.json").read_text())
    io_layout = io_layout_from_dataset_features(info["features"])
    validate_layout_widths(io_layout, *checkpoint_widths(pretrained_dir))

    anvil_config_path = pretrained_dir / "anvil_config.json"
    anvil_config = json.loads(anvil_config_path.read_text()) if anvil_config_path.exists() else {}
    anvil_config["io"] = io_layout
    if not dry_run:
        anvil_config_path.write_text(json.dumps(anvil_config, indent=2) + "\n")
    return io_layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="checkpoint dir (with or without pretrained_model/)",
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, help="LeRobot dataset root containing meta/info.json"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the io block without writing")
    args = parser.parse_args(argv)

    io_layout = annotate(args.checkpoint, args.dataset, args.dry_run)
    action = "would write" if args.dry_run else "wrote"
    print(f"{action} io block to {resolve_pretrained_dir(args.checkpoint) / 'anvil_config.json'}:")
    print(json.dumps(io_layout, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
