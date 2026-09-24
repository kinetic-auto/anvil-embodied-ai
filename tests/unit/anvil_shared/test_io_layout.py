"""io_block_layout / io_layout_from_dataset_features: names → (joints, fields)."""

from __future__ import annotations

import pytest
from anvil_shared.state_observs import io_block_layout, io_layout_from_dataset_features

JOINTS = ["right_finger_joint1", "right_joint1", "right_joint2"]


def _names(fields: list[str]) -> list[str]:
    return [f"{joint}.{field}" for field in fields for joint in JOINTS]


def test_two_block_layout():
    assert io_block_layout(_names(["position", "effort"])) == (JOINTS, ["position", "effort"])


def test_three_block_layout_keeps_field_order():
    assert io_block_layout(_names(["position", "velocity", "effort"])) == (
        JOINTS,
        ["position", "velocity", "effort"],
    )


def test_single_effort_block():
    assert io_block_layout(_names(["effort"])) == (JOINTS, ["effort"])


def test_legacy_unsuffixed_names_are_position():
    assert io_block_layout(JOINTS) == (JOINTS, ["position"])


def test_interleaved_names_rejected():
    interleaved = [f"{joint}.{field}" for joint in JOINTS for field in ("position", "effort")]
    with pytest.raises(ValueError, match="block order|contiguous"):
        io_block_layout(interleaved)


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="known field"):
        io_block_layout(["right_joint1.torque"])


def test_empty_rejected():
    with pytest.raises(ValueError):
        io_block_layout([])


def test_layout_from_dataset_features_flat_names():
    features = {
        "observation.state": {"shape": [3], "names": _names(["effort"])},
        "action": {"shape": [6], "names": _names(["position", "effort"])},
    }
    assert io_layout_from_dataset_features(features) == {
        "joint_names": JOINTS,
        "state_blocks": ["effort"],
        "action_blocks": ["position", "effort"],
    }


def test_layout_from_dataset_features_motor_name_groups():
    grouped = [{"motor_names": _names(["position"])}]
    features = {
        "observation.state": {"shape": [3], "names": grouped},
        "action": {"shape": [3], "names": grouped},
    }
    assert io_layout_from_dataset_features(features)["state_blocks"] == ["position"]


def test_layout_from_dataset_features_joint_mismatch_rejected():
    features = {
        "observation.state": {"shape": [3], "names": _names(["position"])},
        "action": {"shape": [2], "names": ["right_joint1.position", "right_joint2.position"]},
    }
    with pytest.raises(ValueError, match="differ"):
        io_layout_from_dataset_features(features)


def test_layout_from_dataset_features_missing_names_rejected():
    features = {"observation.state": {"shape": [3]}, "action": {"shape": [3], "names": JOINTS}}
    with pytest.raises(KeyError, match="no channel names"):
        io_layout_from_dataset_features(features)
