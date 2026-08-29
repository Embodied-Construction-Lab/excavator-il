"""Strict point and point-group catalog for fixed excavation Missions."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "excavation_dig_point_catalog.v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "frame_id",
        "dig_points",
        "default_dig_group",
        "dig_groups",
    }
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


@dataclass(frozen=True)
class DigPointCatalog:
    points: Mapping[str, tuple[float, float, float]]
    groups: Mapping[str, tuple[str, ...]]
    default_group_id: str


def load_dig_point_catalog(path: str | Path) -> DigPointCatalog:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load dig point catalog: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("dig point catalog fields are invalid")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported dig point catalog schema")
    if value["frame_id"] != "machine_root_ros":
        raise ValueError("dig point catalog frame must be machine_root_ros")
    points = _points(value["dig_points"])
    groups = _groups(value["dig_groups"], tuple(points))
    default_group_id = _identifier(
        value["default_dig_group"], "default_dig_group"
    )
    if default_group_id not in groups:
        raise ValueError("default_dig_group is not defined")
    return DigPointCatalog(
        points=MappingProxyType(points),
        groups=MappingProxyType(groups),
        default_group_id=default_group_id,
    )


def _points(value: Any) -> dict[str, tuple[float, float, float]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dig_points must be a non-empty object")
    result: dict[str, tuple[float, float, float]] = {}
    for raw_point_id, raw_position in value.items():
        point_id = _identifier(raw_point_id, "dig point id")
        if not isinstance(raw_position, list) or len(raw_position) != 3:
            raise ValueError("dig point position must contain three numbers")
        try:
            position = tuple(float(axis) for axis in raw_position)
        except (TypeError, ValueError) as exc:
            raise ValueError("dig point position must be numeric") from exc
        if not all(math.isfinite(axis) for axis in position):
            raise ValueError("dig point position must be finite")
        result[point_id] = position  # type: ignore[assignment]
    return result


def _groups(
    value: Any,
    point_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("dig_groups must be a non-empty object")
    known = frozenset(point_ids)
    result: dict[str, tuple[str, ...]] = {}
    for raw_group_id, raw_members in value.items():
        group_id = _identifier(raw_group_id, "dig group id")
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError("each dig group must be a non-empty list")
        members = tuple(
            _identifier(member, "dig group point id") for member in raw_members
        )
        if len(set(members)) != len(members):
            raise ValueError("dig group point ids must be unique")
        if not set(members) <= known:
            raise ValueError("dig group references an unknown point")
        result[group_id] = members
    if result.get("all") != point_ids:
        raise ValueError("dig_groups.all must exactly match dig_points order")
    return result


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} is invalid")
    return value
