"""Authoritative DIG target loading and formal source provenance."""

from __future__ import annotations

import json
import math
from pathlib import PurePosixPath
from typing import Any, Mapping

from ._guided_episode_config import GuidedEpisodeConfig
from .collector.config import validate_target_source_provenance
from .experiment_run import capture_repository_state, fingerprint_path


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def load_rl_dig_targets(
    config: GuidedEpisodeConfig,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Load selectable DIG points without importing the ROS Mission runtime."""
    path = config.rl_demo_config
    if path is None:
        return ()
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load RL demo config {path}: {exc}") from exc
    root = _object(root, "RL demo config")
    if root.get("schema_version") != "excavation_demo.v1":
        raise ValueError("RL demo schema_version must be excavation_demo.v1")
    points = root.get("dig_points")
    if not isinstance(points, list) or not points:
        raise ValueError("RL demo dig_points must be a non-empty list")
    targets: list[tuple[str, tuple[float, float, float]]] = []
    seen: set[str] = set()
    for index, raw_point in enumerate(points):
        point = _object(raw_point, f"RL demo dig_points[{index}]")
        point_id = _text(point.get("point_id"), f"RL demo dig_points[{index}].point_id")
        if point_id in seen:
            raise ValueError(f"duplicate RL demo point_id: {point_id}")
        raw_position = point.get("position_m")
        if not isinstance(raw_position, list) or len(raw_position) != 3:
            raise ValueError(f"RL demo {point_id}.position_m must contain three numbers")
        try:
            position = tuple(float(value) for value in raw_position)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RL demo {point_id}.position_m must be numeric") from exc
        if any(not math.isfinite(value) for value in position):
            raise ValueError(f"RL demo {point_id}.position_m must be finite")
        seen.add(point_id)
        targets.append((point_id, position))
    return tuple(targets)


def resolve_rl_dig_target(
    config: GuidedEpisodeConfig,
    point_id: str,
) -> tuple[float, float, float]:
    """Resolve one selected DIG point from the authoritative Airy demo config."""
    targets = dict(load_rl_dig_targets(config))
    try:
        return targets[point_id]
    except KeyError as exc:
        raise ValueError(
            f"selected dig_point_id is not configured in the RL demo: {point_id}"
        ) from exc


def capture_target_source_provenance(
    config: GuidedEpisodeConfig,
    point_id: str,
    expected_target_m: tuple[float, float, float],
) -> Mapping[str, str | bool]:
    """Bind a formal Episode to the clean Airy config read on this PC."""
    source = config.rl_demo_config
    if source is None:
        raise ValueError("formal collection requires rl_preposition.demo_config")
    repository = config.rl_airy_repo.expanduser()
    try:
        repository = repository.resolve(strict=True)
        relative = source.resolve(strict=True).relative_to(repository)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "rl_preposition.demo_config must be a file inside the "
            "AiryLidar repository"
        ) from exc
    relative_path = relative.as_posix()
    if (
        PurePosixPath(relative_path).is_absolute()
        or ".." in PurePosixPath(relative_path).parts
        or not relative_path
    ):
        raise ValueError(
            "rl_preposition.demo_config must have a normalized "
            "repository-relative path"
        )

    before = capture_repository_state(repository)
    if before.dirty:
        raise ValueError("formal collection requires a clean AiryLidar repository")
    fingerprint = fingerprint_path(source)
    if fingerprint.object_type != "file":
        raise ValueError("rl_preposition.demo_config must be a regular file")
    observed_target = resolve_rl_dig_target(config, point_id)
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        for actual, expected in zip(observed_target, expected_target_m, strict=True)
    ):
        raise ValueError(
            "selected DIG target changed while source provenance was captured"
        )
    after = capture_repository_state(repository)
    if after.dirty or after.commit != before.commit:
        raise ValueError(
            "AiryLidar repository changed while target provenance was captured"
        )
    return validate_target_source_provenance(
        {
            "repository": "airylidar",
            "path": relative_path,
            "sha256": fingerprint.sha256,
            "commit": after.commit,
            "dirty": False,
        }
    )
