"""Tests for anvil_shared.state_observs."""
from __future__ import annotations

import pytest

from anvil_shared.state_observs import (
    OBS_STATE,
    channel_groups,
    compose_observation_state,
    compose_packed_observation,
    compose_state_stats,
    is_packed_names,
    observation_key,
    packed_feature_names,
    packed_fields,
    position_channel_indices,
    state_features_to_suffixes,
    state_observ_keys,
)


class TestKeys:
    def test_state_suffix_is_observation_state(self):
        assert observation_key("state") == OBS_STATE

    def test_other_suffixes(self):
        assert observation_key("velocity") == "observation.velocity"
        assert observation_key("effort") == "observation.effort"

    def test_keys_preserve_order(self):
        assert state_observ_keys(["velocity", "effort"]) == [
            "observation.velocity",
            "observation.effort",
        ]


class TestComposeLists:
    def test_replaces_state_with_concat(self):
        item = {
            OBS_STATE: [0.1, 0.2],
            "observation.velocity": [1.0, 2.0],
            "observation.effort": [3.0, 4.0],
        }
        compose_observation_state(item, ["velocity", "effort"])
        assert item[OBS_STATE] == [1.0, 2.0, 3.0, 4.0]

    def test_includes_original_state_when_requested(self):
        item = {OBS_STATE: [0.1], "observation.velocity": [9.0]}
        compose_observation_state(item, ["state", "velocity"])
        assert item[OBS_STATE] == [0.1, 9.0]

    def test_missing_key_raises(self):
        with pytest.raises(KeyError, match="observation.effort"):
            compose_observation_state({OBS_STATE: [0.0]}, ["velocity", "effort"])


class TestComposeStats:
    def test_concatenates_mean_std(self):
        stats = {
            OBS_STATE: {"mean": [0.0, 1.0], "std": [0.1, 0.2], "min": [0, 0], "max": [1, 1], "count": 10},
            "observation.velocity": {
                "mean": [2.0, 3.0],
                "std": [0.3, 0.4],
                "min": [-1, -1],
                "max": [2, 2],
                "count": 10,
            },
        }
        out = compose_state_stats(stats, ["state", "velocity"])
        assert out["mean"] == [0.0, 1.0, 2.0, 3.0]
        assert out["std"] == [0.1, 0.2, 0.3, 0.4]
        assert out["count"] == 10


class TestPackedHelpers:
    def test_string_state_is_not_packed(self):
        assert packed_fields("position") is None

    def test_list_state_is_packed(self):
        assert packed_fields(["position", "velocity", "effort"]) == [
            "position",
            "velocity",
            "effort",
        ]

    def test_block_names(self):
        names = packed_feature_names(
            ["right_finger_joint1", "right_joint1"],
            ["position", "velocity", "effort"],
        )
        assert names == [
            "right_finger_joint1.position",
            "right_joint1.position",
            "right_finger_joint1.velocity",
            "right_joint1.velocity",
            "right_finger_joint1.effort",
            "right_joint1.effort",
        ]

    def test_is_packed_names(self):
        assert is_packed_names(["j1.position", "j1.velocity"]) is True
        assert is_packed_names(["j1", "j2"]) is False

    def test_position_indices_legacy(self):
        assert position_channel_indices(["j1", "j2"]) == [0, 1]

    def test_position_indices_packed(self):
        names = packed_feature_names(["j1", "j2"], ["position", "velocity", "effort"])
        assert position_channel_indices(names) == [0, 1]
        assert channel_groups(names)["velocity"] == [2, 3]
        assert channel_groups(names)["effort"] == [4, 5]

    def test_state_features_to_suffixes(self):
        assert state_features_to_suffixes(["position", "velocity", "effort"]) == [
            "state",
            "velocity",
            "effort",
        ]


class TestComposePackedObservation:
    def test_collapses_siblings_and_drops_them(self):
        item = {
            OBS_STATE: [0.1, 0.2],
            "observation.velocity": [1.0, 2.0],
            "observation.effort": [3.0, 4.0],
        }
        compose_packed_observation(item, ["position", "velocity", "effort"])
        assert item[OBS_STATE] == [0.1, 0.2, 1.0, 2.0, 3.0, 4.0]
        assert "observation.velocity" not in item
        assert "observation.effort" not in item

    def test_matches_compose_observation_state_block_order(self):
        siblings = {
            OBS_STATE: [0.1, 0.2],
            "observation.velocity": [1.0, 2.0],
            "observation.effort": [3.0, 4.0],
        }
        composed = dict(siblings)
        compose_observation_state(composed, ["state", "velocity", "effort"])
        packed = dict(siblings)
        compose_packed_observation(packed, ["position", "velocity", "effort"])
        assert packed[OBS_STATE] == composed[OBS_STATE]

    def test_torch_last_dim(self):
        torch = pytest.importorskip("torch")
        item = {
            OBS_STATE: torch.tensor([[0.1, 0.2]]),
            "observation.velocity": torch.tensor([[1.0, 2.0]]),
            "observation.effort": torch.tensor([[3.0, 4.0]]),
        }
        compose_packed_observation(item, ["position", "velocity", "effort"])
        assert item[OBS_STATE].shape == (1, 6)
        assert "observation.velocity" not in item
