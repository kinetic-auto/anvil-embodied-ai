"""Compose selected observation.* vectors into observation.state.

ACT only reads the key named ``observation.state``. Sibling keys such as
``observation.velocity`` / ``observation.effort`` are typed STATE by LeRobot
but never projected. Concatenating them into ``observation.state`` is how
those signals reach the encoder — same idea as conversion-time composite
state vectors, applied at train / eval / inference time.

Pure-Python: no torch/numpy imports at module load. Concat uses torch if
the values are tensors, otherwise list flatten.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

OBS_STATE = "observation.state"
VALID_JOINT_FIELDS = ("position", "velocity", "effort")
_FEATURE_TO_OBS_KEY = {
    "position": OBS_STATE,
    "velocity": "observation.velocity",
    "effort": "observation.effort",
}


def packed_fields(state: str | Sequence[str]) -> list[str] | None:
    """Return the field list when ``state`` is packed, else None (string form)."""
    if isinstance(state, str):
        return None
    return list(state)


def packed_feature_names(joint_names: Sequence[str], fields: Sequence[str]) -> list[str]:
    """Block-layout names: all joints for field 0, then field 1, ..."""
    return [f"{joint}.{field}" for field in fields for joint in joint_names]


def is_packed_names(names: Sequence[str]) -> bool:
    """True when names carry ``.velocity`` or ``.effort`` suffixes."""
    return any(n.endswith(".velocity") or n.endswith(".effort") for n in names)


def field_channel_indices(names: Sequence[str], field: str) -> list[int]:
    """Indices whose names end with ``.{field}`` (e.g. ``.position``)."""
    suffix = f".{field}"
    return [i for i, name in enumerate(names) if name.endswith(suffix)]


def position_channel_indices(names: Sequence[str]) -> list[int]:
    """Position channels for aggregate metrics.

    Packed names → indices ending in ``.position``. Legacy names with no
    field suffix → every channel (the whole vector is positions).
    """
    pos = field_channel_indices(names, "position")
    if pos:
        return pos
    if is_packed_names(names):
        skip = set(field_channel_indices(names, "velocity")) | set(
            field_channel_indices(names, "effort")
        )
        return [i for i in range(len(names)) if i not in skip]
    return list(range(len(names)))


def channel_groups(names: Sequence[str]) -> dict[str, list[int]]:
    """Map field name → indices. Legacy names (no suffix) are ``position`` only."""
    groups: dict[str, list[int]] = {}
    for field in VALID_JOINT_FIELDS:
        idx = field_channel_indices(names, field)
        if idx:
            groups[field] = idx
    if not groups:
        groups["position"] = list(range(len(names)))
    return groups


def state_features_to_suffixes(state_features: Sequence[str]) -> list[str]:
    """Map live YAML ``state_features`` entries to compose_observation_state suffixes.

    ``position`` is stored as ``observation.state``; other fields keep their name.
    """
    return ["state" if f == "position" else f for f in state_features]


def compose_packed_observation(item: dict[str, Any], state_features: Sequence[str]) -> dict[str, Any]:
    """Collapse sibling keys into ``observation.state`` and drop the siblings.

    Expects the live strategy's key mapping: position → ``observation.state``,
    velocity → ``observation.velocity``, effort → ``observation.effort``.
    """
    suffixes = state_features_to_suffixes(state_features)
    compose_observation_state(item, suffixes)
    for feature in state_features:
        key = _FEATURE_TO_OBS_KEY.get(feature)
        if key and key != OBS_STATE:
            item.pop(key, None)
    return item


def observation_key(suffix: str) -> str:
    """Map a --state-observs suffix to a dataset key."""
    return OBS_STATE if suffix == "state" else f"observation.{suffix}"


def state_observ_keys(suffixes: Sequence[str]) -> list[str]:
    return [observation_key(s) for s in suffixes]


def compose_state_stats(stats: Mapping[str, Mapping[str, Any]], suffixes: Sequence[str]) -> dict[str, Any]:
    """Concatenate per-key mean/std/min/max into one observation.state stats dict."""
    keys = state_observ_keys(suffixes)
    missing = [k for k in keys if k not in stats]
    if missing:
        raise KeyError(f"state_observs stats missing keys: {missing}")

    def _flat(value: Any) -> list:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (int, float)):
            return [value]
        return list(value)

    out: dict[str, Any] = {}
    for field in ("mean", "std", "min", "max"):
        concat: list = []
        for key in keys:
            if field not in stats[key]:
                raise KeyError(f"state_observs stats {key!r} has no {field!r}")
            concat.extend(_flat(stats[key][field]))
        out[field] = concat
    out["count"] = stats[keys[0]].get("count")
    return out


def compose_observation_state(item: dict[str, Any], suffixes: Sequence[str]) -> dict[str, Any]:
    """In-place: set item[observation.state] to the concat of selected keys (last dim)."""
    if not suffixes:
        return item
    keys = state_observ_keys(suffixes)
    missing = [k for k in keys if k not in item]
    if missing:
        raise KeyError(f"state_observs missing keys: {missing}")
    parts = [item[k] for k in keys]
    item[OBS_STATE] = _cat_last(parts)
    return item


def _cat_last(parts: Sequence[Any]) -> Any:
    first = parts[0]
    if hasattr(first, "dim"):
        import torch

        flat = [p.unsqueeze(0) if p.dim() == 0 else p for p in parts]
        return torch.cat(flat, dim=-1)
    out: list = []
    for part in parts:
        if hasattr(part, "tolist"):
            part = part.tolist()
        if isinstance(part, (int, float)):
            out.append(part)
        else:
            out.extend(list(part))
    return out
