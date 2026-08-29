"""Validated immutable point-group selection for resident fixed cycles."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def normalize_dig_groups(
    dig_target_ids: tuple[str, ...],
    dig_groups: Mapping[str, tuple[str, ...]] | None,
    default_group_id: str,
) -> tuple[Mapping[str, tuple[str, ...]], str]:
    if not dig_target_ids or len(set(dig_target_ids)) != len(dig_target_ids):
        raise ValueError("dig_target_ids must be a non-empty unique tuple")
    if dig_groups is None:
        normalized = {"all": dig_target_ids}
    else:
        normalized = {}
        known = frozenset(dig_target_ids)
        for group_id, members in dig_groups.items():
            _identifier(group_id, "dig group id")
            if not isinstance(members, tuple) or not members:
                raise ValueError("each dig group must be a non-empty tuple")
            if len(set(members)) != len(members) or not set(members) <= known:
                raise ValueError("dig group members are invalid")
            normalized[group_id] = tuple(members)
        if normalized.get("all") != dig_target_ids:
            raise ValueError("dig group all must exactly match dig_target_ids")
    _identifier(default_group_id, "default dig group id")
    if default_group_id not in normalized:
        raise ValueError("default dig group is not defined")
    return MappingProxyType(normalized), default_group_id


def select_cycle_targets(
    groups: Mapping[str, tuple[str, ...]],
    group_id: str,
    first_target_id: str,
    cycle_count: int,
) -> tuple[str, ...]:
    _identifier(group_id, "dig group id")
    if group_id not in groups:
        raise ValueError("unknown V3-A dig group")
    members = groups[group_id]
    if first_target_id not in members:
        raise ValueError("first dig target is outside the selected dig group")
    first_index = members.index(first_target_id)
    return tuple(
        members[(first_index + offset) % len(members)]
        for offset in range(cycle_count)
    )


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} is invalid")
    return value
