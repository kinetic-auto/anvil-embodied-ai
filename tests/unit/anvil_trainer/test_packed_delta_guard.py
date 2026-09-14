"""Reject --action-type=delta_* on a packed pos/vel/effort dataset."""

from __future__ import annotations

import json

import pytest

from anvil_trainer.config import TrainingConfig


def _write_info(tmp_path, action_names, state_names=None):
    if state_names is None:
        state_names = action_names
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    "action": {"names": action_names, "shape": [len(action_names)]},
                    "observation.state": {
                        "names": state_names,
                        "shape": [len(state_names)],
                    },
                }
            }
        )
    )


class TestRejectDeltaOnPacked:
    def test_raises_on_packed_names(self, tmp_path):
        names = [f"j{i}.position" for i in range(7)]
        names += [f"j{i}.velocity" for i in range(7)]
        names += [f"j{i}.effort" for i in range(7)]
        _write_info(tmp_path, names)
        cfg = TrainingConfig(action_type="delta_obs_t", dataset_root=str(tmp_path))
        with pytest.raises(ValueError, match="packed"):
            cfg.reject_delta_on_packed_features()

    def test_absolute_ok_on_packed(self, tmp_path):
        names = ["j1.position", "j1.velocity", "j1.effort"]
        _write_info(tmp_path, names)
        cfg = TrainingConfig(action_type="absolute", dataset_root=str(tmp_path))
        cfg.reject_delta_on_packed_features()

    def test_delta_ok_on_legacy_7dim(self, tmp_path):
        names = ["right_finger_joint1", "right_joint1"]
        _write_info(tmp_path, names)
        cfg = TrainingConfig(action_type="delta_obs_t", dataset_root=str(tmp_path))
        cfg.reject_delta_on_packed_features()
