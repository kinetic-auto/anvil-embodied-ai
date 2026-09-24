"""annotate_checkpoint_io: dataset names → anvil_config.json io block, width-checked."""

from __future__ import annotations

import json
from pathlib import Path

import annotate_checkpoint_io as annotate_module
import pytest

JOINTS = ["right_finger_joint1", "right_joint1"]


def _write_dataset(root: Path, state_fields: list[str], action_fields: list[str]) -> None:
    def names(fields: list[str]) -> list[str]:
        return [f"{joint}.{field}" for field in fields for joint in JOINTS]

    (root / "meta").mkdir(parents=True)
    info = {
        "features": {
            "observation.state": {
                "shape": [len(JOINTS) * len(state_fields)],
                "names": names(state_fields),
            },
            "action": {"shape": [len(JOINTS) * len(action_fields)], "names": names(action_fields)},
        }
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))


def _write_checkpoint(
    root: Path, state_dim: int, action_dim: int, with_pretrained_dir: bool = True
) -> Path:
    target = root / "pretrained_model" if with_pretrained_dir else root
    target.mkdir(parents=True)
    config = {
        "input_features": {"observation.state": {"type": "STATE", "shape": [state_dim]}},
        "output_features": {"action": {"type": "ACTION", "shape": [action_dim]}},
    }
    (target / "config.json").write_text(json.dumps(config))
    (target / "anvil_config.json").write_text(json.dumps({"action_type": "absolute"}))
    return target


def test_annotate_merges_io_into_existing_anvil_config(tmp_path: Path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    _write_dataset(dataset, ["effort"], ["position", "effort"])
    pretrained = _write_checkpoint(checkpoint, state_dim=2, action_dim=4)

    io_layout = annotate_module.annotate(checkpoint, dataset, dry_run=False)

    written = json.loads((pretrained / "anvil_config.json").read_text())
    assert written["action_type"] == "absolute"
    assert (
        written["io"]
        == io_layout
        == {
            "joint_names": JOINTS,
            "state_blocks": ["effort"],
            "action_blocks": ["position", "effort"],
        }
    )


def test_dry_run_writes_nothing(tmp_path: Path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    _write_dataset(dataset, ["position"], ["position"])
    pretrained = _write_checkpoint(checkpoint, state_dim=2, action_dim=2, with_pretrained_dir=False)

    annotate_module.annotate(checkpoint, dataset, dry_run=True)

    assert "io" not in json.loads((pretrained / "anvil_config.json").read_text())


def test_width_mismatch_is_rejected(tmp_path: Path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    _write_dataset(dataset, ["position", "effort"], ["position", "effort"])
    _write_checkpoint(checkpoint, state_dim=2, action_dim=4)

    with pytest.raises(ValueError, match="checkpoint has state 2"):
        annotate_module.annotate(checkpoint, dataset, dry_run=True)


def test_missing_config_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        annotate_module.resolve_pretrained_dir(tmp_path)
