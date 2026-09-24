"""Live observation packing used by MultiProcessStrategy._build_observation."""

from __future__ import annotations

import pytest

from anvil_shared.state_observs import OBS_STATE, compose_packed_observation


def test_packed_layout_matches_compose_and_drops_siblings():
    torch = pytest.importorskip("torch")
    observation = {
        OBS_STATE: torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]),
        "observation.velocity": torch.tensor([[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]]),
        "observation.effort": torch.tensor([[2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]]),
    }
    compose_packed_observation(observation, ["position", "velocity", "effort"])
    assert observation[OBS_STATE].shape == (1, 21)
    assert observation[OBS_STATE][0, 0].item() == pytest.approx(0.1)
    assert observation[OBS_STATE][0, 7].item() == pytest.approx(1.0)
    assert observation[OBS_STATE][0, 14].item() == pytest.approx(2.0)
    assert "observation.velocity" not in observation
    assert "observation.effort" not in observation


def test_legacy_yaml_keeps_7dim_state_and_siblings():
    """Omit state_layout: packed — strategy leaves sibling keys as written."""
    torch = pytest.importorskip("torch")
    observation = {
        OBS_STATE: torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]),
    }
    assert observation[OBS_STATE].shape == (1, 7)
    assert "observation.velocity" not in observation
