"""Tests for mapping a shorter observation.state onto action channels."""

import numpy as np

from anvil_eval.evaluator import align_obs_to_action


ACTION = [
    "right_finger_joint1.position",
    "right_joint1.position",
    "right_finger_joint1.effort",
    "right_joint1.effort",
]
STATE = [
    "right_finger_joint1.effort",
    "right_joint1.effort",
]


def test_same_width_is_passthrough():
    obs = np.arange(4, dtype=np.float32)
    out = align_obs_to_action(obs, ACTION, ACTION)
    np.testing.assert_array_equal(out, obs)
    assert out is not obs


def test_effort_only_state_maps_onto_effort_action_channels():
    obs = np.array([1.5, 2.5], dtype=np.float32)
    out = align_obs_to_action(obs, ACTION, STATE)
    assert out.shape == (4,)
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_array_equal(out[2:], obs)
