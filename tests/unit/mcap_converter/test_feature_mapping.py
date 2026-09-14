"""Packed FeatureMapping.state (list form) + legacy string form."""

from __future__ import annotations

import inspect
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from mcap_converter.cli import convert as convert_cli
from mcap_converter.config.loader import ConfigLoader
from mcap_converter.config.schema import ActionTopicConfig, DataConfig, FeatureMapping
from mcap_converter.config.validators import (
    validate_config,
    validate_feature_mapping,
)
from mcap_converter.core.extractor import BufferedStreamExtractor
from mcap_converter.core.writer import LeRobotWriter

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "configs" / "mcap_converter"
OPENYAM_JOINTS = [
    "finger_joint1",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]


def _packed_mapping() -> FeatureMapping:
    return FeatureMapping(state=["position", "velocity", "effort"], others=[])


# =============================================================================
# Validator
# =============================================================================


class TestValidateFeatureMapping:
    def test_string_position_ok(self):
        assert validate_feature_mapping(FeatureMapping(state="position"), "obs") == []

    def test_list_ok(self):
        assert validate_feature_mapping(_packed_mapping(), "obs") == []

    def test_reject_unknown_list_entry(self):
        errors = validate_feature_mapping(
            FeatureMapping(state=["position", "foo"], others=[]), "obs"
        )
        assert any("foo" in e for e in errors)

    def test_reject_duplicates(self):
        errors = validate_feature_mapping(
            FeatureMapping(state=["position", "position"], others=[]), "obs"
        )
        assert any("duplicate" in e for e in errors)

    def test_reject_list_plus_others(self):
        errors = validate_feature_mapping(
            FeatureMapping(state=["position", "velocity"], others=["effort"]),
            "obs",
        )
        assert any("others must be empty" in e for e in errors)

    def test_empty_list_rejected(self):
        errors = validate_feature_mapping(FeatureMapping(state=[], others=[]), "obs")
        assert any("cannot be empty" in e for e in errors)

    def test_string_invalid_field(self):
        errors = validate_feature_mapping(FeatureMapping(state="torque"), "obs")
        assert any("torque" in e for e in errors)


class TestValidateExistingConfigs:
    @pytest.mark.parametrize(
        "name",
        [
            "openarm_single_quest.yaml",
            "openarm_single_quest_afo.yaml",
            "openyam_single_quest.yaml",
            "openarm_bimanual.yaml",
            "openarm_bimanual_quest.yaml",
            "openarm_bimanual_quest_16x9.yaml",
        ],
    )
    def test_shipped_yaml_validates(self, name):
        config = ConfigLoader.from_yaml(str(CONFIG_DIR / name))
        validate_config(config)

    def test_openyam_yaml_has_list_state(self):
        config = ConfigLoader.from_yaml(str(CONFIG_DIR / "openyam_single_quest.yaml"))
        assert config.observation_feature_mapping.state == [
            "position",
            "velocity",
            "effort",
        ]
        assert config.action_feature_mapping.state == ["position", "velocity", "effort"]
        assert config.action_from_observation is True
        assert config.action_from_observation_n == 10

    def test_validate_config_runs_from_convert(self):
        source = inspect.getsource(convert_cli.main)
        assert "validate_config(config)" in source


# =============================================================================
# Writer
# =============================================================================


class TestWriterPackedFeatures:
    def test_packed_shape_and_block_names(self):
        config = DataConfig(
            observation_feature_mapping=_packed_mapping(),
            action_feature_mapping=_packed_mapping(),
        )
        writer = LeRobotWriter(output_dir="/tmp/unused", repo_id="t", config=config)
        features = writer._define_features({"right": OPENYAM_JOINTS}, ["chest"])

        assert features["observation.state"]["shape"] == (21,)
        assert features["action"]["shape"] == (21,)
        assert "observation.velocity" not in features
        assert "observation.effort" not in features

        names = features["observation.state"]["names"]
        assert names[0] == "right_finger_joint1.position"
        assert names[6] == "right_joint6.position"
        assert names[7] == "right_finger_joint1.velocity"
        assert names[14] == "right_finger_joint1.effort"
        assert names[-1] == "right_joint6.effort"
        assert features["action"]["names"] == names

    def test_legacy_siblings(self):
        config = DataConfig(
            observation_feature_mapping=FeatureMapping(
                state="position", others=["velocity", "effort"]
            ),
            action_feature_mapping=FeatureMapping(state="position", others=[]),
        )
        writer = LeRobotWriter(output_dir="/tmp/unused", repo_id="t", config=config)
        features = writer._define_features({"right": OPENYAM_JOINTS}, ["chest"])

        assert features["observation.state"]["shape"] == (7,)
        assert features["action"]["shape"] == (7,)
        assert features["observation.velocity"]["shape"] == (7,)
        assert features["observation.effort"]["shape"] == (7,)
        assert features["observation.state"]["names"][0] == "right_finger_joint1"


# =============================================================================
# Extractor
# =============================================================================


def _buffer(rows):
    buf = deque()
    for ts, pos, vel, eff in rows:
        buf.append(
            (
                ts,
                np.array(pos, dtype=np.float32),
                np.array(vel, dtype=np.float32),
                np.array(eff, dtype=np.float32),
            )
        )
    return buf


class TestExtractorPacked:
    def test_block_order_and_afo_lookahead(self):
        config = DataConfig(
            action_topics={
                "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                    arm="right", joint_order=["joint1", "joint2"]
                ),
            },
            action_from_observation=True,
            observation_feature_mapping=_packed_mapping(),
            action_feature_mapping=_packed_mapping(),
        )
        extractor = BufferedStreamExtractor(config=config, buffer_seconds=5.0, fps=60, quiet=True)
        joint_buffers = {
            ("observation", "right"): {
                "buffer": _buffer(
                    [
                        (0.0, [1.0, 2.0], [0.1, 0.2], [10.0, 20.0]),
                        (1.0, [3.0, 4.0], [0.3, 0.4], [30.0, 40.0]),
                    ]
                )
            }
        }

        result = extractor._align_joint_states(joint_buffers, target_ts=0.0, action_ts=1.0)

        assert result is not None
        np.testing.assert_array_almost_equal(
            result["observation.state"], [1.0, 2.0, 0.1, 0.2, 10.0, 20.0]
        )
        np.testing.assert_array_almost_equal(
            result["action"], [3.0, 4.0, 0.3, 0.4, 30.0, 40.0]
        )
        assert "observation.velocity" not in result
        assert "observation.effort" not in result
        assert not np.allclose(result["action"], result["observation.state"])

    def test_afo_lookahead_ignores_preseeded_empty_action_buffers(self):
        """_init_joint_buffers pre-seeds ("action", robot) keys. AFO must still look ahead."""
        config = DataConfig(
            action_topics={
                "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                    arm="right", joint_order=["joint1", "joint2"]
                ),
            },
            action_from_observation=True,
            observation_feature_mapping=_packed_mapping(),
            action_feature_mapping=_packed_mapping(),
        )
        extractor = BufferedStreamExtractor(config=config, buffer_seconds=5.0, fps=60, quiet=True)
        joint_buffers = extractor._init_joint_buffers()
        joint_buffers[("observation", "right")] = {
            "buffer": _buffer(
                [
                    (0.0, [1.0, 2.0], [0.1, 0.2], [10.0, 20.0]),
                    (1.0, [3.0, 4.0], [0.3, 0.4], [30.0, 40.0]),
                ]
            )
        }

        result = extractor._align_joint_states(joint_buffers, target_ts=0.0, action_ts=1.0)

        assert result is not None
        np.testing.assert_array_almost_equal(
            result["action"], [3.0, 4.0, 0.3, 0.4, 30.0, 40.0]
        )
        np.testing.assert_array_almost_equal(
            result["observation.state"], [1.0, 2.0, 0.1, 0.2, 10.0, 20.0]
        )

    def test_missing_vel_effort_become_zeros(self):
        config = DataConfig(
            action_from_observation=True,
            observation_feature_mapping=_packed_mapping(),
            action_feature_mapping=_packed_mapping(),
        )
        extractor = BufferedStreamExtractor(config=config, buffer_seconds=5.0, fps=60, quiet=True)
        buf = deque()
        buf.append((0.0, np.array([1.0, 2.0], dtype=np.float32), np.array([]), np.array([])))
        buf.append((1.0, np.array([3.0, 4.0], dtype=np.float32), np.array([]), np.array([])))
        result = extractor._align_joint_states(
            {("observation", "right"): {"buffer": buf}},
            target_ts=0.0,
            action_ts=1.0,
        )
        np.testing.assert_array_almost_equal(
            result["observation.state"], [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            result["action"], [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_legacy_siblings_still_emitted(self):
        config = DataConfig(
            action_topics={
                "/follower_r_forward_position_controller/commands": ActionTopicConfig(
                    arm="right", joint_order=["joint1", "joint2"]
                ),
            },
            observation_feature_mapping=FeatureMapping(
                state="position", others=["velocity", "effort"]
            ),
            action_feature_mapping=FeatureMapping(state="position", others=[]),
        )
        extractor = BufferedStreamExtractor(config=config, buffer_seconds=5.0, fps=60, quiet=True)
        joint_buffers = {
            ("observation", "right"): {
                "buffer": _buffer([(0.0, [1.0, 2.0], [0.1, 0.2], [10.0, 20.0])])
            },
            ("action", "right"): {
                "buffer": _buffer([(0.0, [9.0, 8.0], [], [])])
            },
        }
        result = extractor._align_joint_states(joint_buffers, target_ts=0.0)

        np.testing.assert_array_almost_equal(result["observation.state"], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result["observation.velocity"], [0.1, 0.2])
        np.testing.assert_array_almost_equal(result["observation.effort"], [10.0, 20.0])
        np.testing.assert_array_almost_equal(result["action"], [9.0, 8.0])
