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

from collections.abc import Mapping, Sequence
from typing import Any

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


def io_block_layout(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split block-layout channel names into ``(joint_names, fields)``.

    ``["a.position", "b.position", "a.effort", "b.effort"]`` →
    ``(["a", "b"], ["position", "effort"])``. Legacy names with no field
    suffix are a single ``position`` block. The last ``.`` separates joint
    from field, so joint names themselves must not contain a dot. Raises
    ``ValueError`` when the names are not a clean block layout.
    """
    names = list(names)
    if not names:
        raise ValueError("cannot derive an IO layout from an empty name list")
    if not any("." in name for name in names):
        return names, ["position"]

    fields: list[str] = []
    joints_per_field: dict[str, list[str]] = {}
    for name in names:
        joint, _, field = name.rpartition(".")
        if field not in VALID_JOINT_FIELDS or not joint:
            raise ValueError(f"channel {name!r} is not '<joint>.<field>' with a known field")
        if field not in fields:
            fields.append(field)
        joints_per_field.setdefault(field, []).append(joint)

    joint_names = joints_per_field[fields[0]]
    if any(joints_per_field[field] != joint_names for field in fields):
        raise ValueError(f"channels are not one contiguous block per field: {names}")
    if names != packed_feature_names(joint_names, fields):
        raise ValueError(f"channels are not in block order (all joints of field 0, then field 1, ...): {names}")
    return joint_names, fields


def io_layout_from_dataset_features(features: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """Build the checkpoint ``io`` block from a LeRobot ``info.json`` ``features`` map.

    Returns ``{"joint_names": [...], "state_blocks": [...], "action_blocks": [...]}``.
    Raises ``KeyError``/``ValueError`` when names are missing or not block-layout.
    """

    def _names(key: str) -> list[str]:
        names = features[key].get("names") or []
        if names and isinstance(names[0], dict):
            names = [n for group in names for n in group.get("motor_names", [])]
        if not names:
            raise KeyError(f"dataset feature {key!r} has no channel names")
        return list(names)

    state_joints, state_blocks = io_block_layout(_names(OBS_STATE))
    action_joints, action_blocks = io_block_layout(_names("action"))
    if state_joints != action_joints:
        raise ValueError(
            f"observation.state joints {state_joints} differ from action joints {action_joints}"
        )
    return {
        "joint_names": state_joints,
        "state_blocks": state_blocks,
        "action_blocks": action_blocks,
    }


def packed_state_features_for_checkpoint(
    state_dim: int,
    action_dim: int,
    n_joints: int,
    yaml_features: Sequence[str],
) -> list[str]:
    """Choose packed ``state_features`` that match checkpoint vector widths.

    Used when live YAML packing does not match ``observation.state`` in the
    checkpoint (e.g. effort-only YAML with a pos|eff ACT).
    """
    yaml_list = [f for f in yaml_features if f in VALID_JOINT_FIELDS]
    if n_joints <= 0 or state_dim % n_joints != 0:
        return yaml_list
    n_blocks = state_dim // n_joints
    if n_blocks == 3:
        return ["position", "velocity", "effort"]
    if n_blocks == 2:
        return ["position", "effort"]
    if n_blocks == 1:
        if len(yaml_list) == 1:
            return yaml_list
        # action [pos|eff] + 7-dim state → effort-only observation
        if action_dim == 2 * n_joints:
            return ["effort"]
        return ["position"]
    return yaml_list


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
