"""Validated PC configuration for guided demonstration collection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


GUIDED_EPISODE_CONFIG_SCHEMA_VERSION = "excavator_guided_episode_config.v3"
_SSH_HOST = re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+")
_NETWORK_HOST = re.compile(r"[A-Za-z0-9_.:-]+")


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class GuidedEpisodeConfig:
    teleop_config: Path
    orin_ssh_host: str
    orin_repo: PurePosixPath
    orin_executable: PurePosixPath
    orin_collection_config: PurePosixPath
    task: str
    operator_id: str
    dig_target_m: tuple[float, float, float]
    material_id: str
    collector_ready_timeout_s: int
    ack_timeout_s: int
    teleop_print_every: int
    log_dir: Path
    rl_airy_repo: Path
    rl_ros_setup: Path
    rl_workspace_setup: Path
    rl_mission_config: Path
    rl_phase: str
    rl_timeout_s: int
    rl_serial_port: PurePosixPath
    rl_serial_release_timeout_s: int
    rl_orin_repo: PurePosixPath
    rl_orin_python: PurePosixPath
    rl_edge_config: PurePosixPath
    rl_pc_host: str
    rl_ready_timeout_s: int
    rl_demo_config: Path | None = None
    failure_reason: str = "diagnostic_task_failed"
    zero_soak_duration_s: int = 30
    orin_experiment_run_config: PurePosixPath | None = None

    @classmethod
    def load(cls, path: str | Path) -> "GuidedEpisodeConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            root = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load guided Episode config {config_path}: {exc}") from exc
        root = _object(root, "config")
        if root.get("schema_version") != GUIDED_EPISODE_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GUIDED_EPISODE_CONFIG_SCHEMA_VERSION}"
            )
        orin = _object(root.get("orin"), "orin")
        rl_preposition = _object(root.get("rl_preposition"), "rl_preposition")
        episode = _object(root.get("episode"), "episode")
        runtime = _object(root.get("runtime"), "runtime")
        ssh_host = _text(orin.get("ssh_host"), "orin.ssh_host")
        if _SSH_HOST.fullmatch(ssh_host) is None:
            raise ValueError("orin.ssh_host must be user@host without shell syntax")
        target = episode.get("dig_target_m")
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError("episode.dig_target_m must contain three numbers")
        target_values = tuple(float(value) for value in target)
        if any(not math.isfinite(value) for value in target_values):
            raise ValueError("episode.dig_target_m must be finite")
        teleop_print_every = _positive_int(
            runtime.get("teleop_print_every"), "runtime.teleop_print_every"
        )
        if teleop_print_every != 1:
            raise ValueError(
                "runtime.teleop_print_every must be 1 for 20 Hz deadman edge detection"
            )
        base = config_path.parent
        airy_repo = (
            base / _text(rl_preposition.get("airy_repo"), "rl_preposition.airy_repo")
        ).resolve()
        ros_setup = Path(
            _text(rl_preposition.get("ros_setup"), "rl_preposition.ros_setup")
        ).expanduser()
        if not ros_setup.is_absolute():
            raise ValueError("rl_preposition.ros_setup must be an absolute path")
        workspace_setup = (
            airy_repo
            / _text(
                rl_preposition.get("workspace_setup"),
                "rl_preposition.workspace_setup",
            )
        ).resolve()
        mission_config = (
            airy_repo
            / _text(
                rl_preposition.get("mission_config"),
                "rl_preposition.mission_config",
            )
        ).resolve()
        demo_config_value = rl_preposition.get("demo_config")
        demo_config = None
        if demo_config_value is not None:
            demo_config = (
                airy_repo
                / _text(demo_config_value, "rl_preposition.demo_config")
            ).resolve()
        phase = _text(rl_preposition.get("phase"), "rl_preposition.phase")
        if phase != "dig":
            raise ValueError("rl_preposition.phase must be dig for Episode collection")
        serial_port = PurePosixPath(
            _text(rl_preposition.get("serial_port"), "rl_preposition.serial_port")
        )
        if not serial_port.is_absolute() or not str(serial_port).startswith("/dev/"):
            raise ValueError("rl_preposition.serial_port must be an absolute /dev path")
        rl_orin_repo = PurePosixPath(
            _text(rl_preposition.get("orin_repo"), "rl_preposition.orin_repo")
        )
        rl_orin_python = PurePosixPath(
            _text(rl_preposition.get("orin_python"), "rl_preposition.orin_python")
        )
        if not rl_orin_repo.is_absolute() or not rl_orin_python.is_absolute():
            raise ValueError(
                "rl_preposition.orin_repo and orin_python must be absolute paths"
            )
        rl_edge_config = PurePosixPath(
            _text(rl_preposition.get("edge_config"), "rl_preposition.edge_config")
        )
        if rl_edge_config.is_absolute() or ".." in rl_edge_config.parts:
            raise ValueError(
                "rl_preposition.edge_config must be a safe path relative to orin_repo"
            )
        rl_pc_host = _text(
            rl_preposition.get("pc_host"), "rl_preposition.pc_host"
        )
        if _NETWORK_HOST.fullmatch(rl_pc_host) is None:
            raise ValueError("rl_preposition.pc_host must not contain shell syntax")
        experiment_run_value = orin.get("experiment_run_config")
        experiment_run_config = None
        if experiment_run_value is not None:
            experiment_run_config = PurePosixPath(
                _text(experiment_run_value, "orin.experiment_run_config")
            )
            if (
                experiment_run_config.is_absolute()
                or ".." in experiment_run_config.parts
            ):
                raise ValueError(
                    "orin.experiment_run_config must be a safe path relative to orin.repo"
                )
        return cls(
            teleop_config=(base / _text(root.get("teleop_config"), "teleop_config")).resolve(),
            orin_ssh_host=ssh_host,
            orin_repo=PurePosixPath(_text(orin.get("repo"), "orin.repo")),
            orin_executable=PurePosixPath(
                _text(orin.get("executable"), "orin.executable")
            ),
            orin_collection_config=PurePosixPath(
                _text(orin.get("collection_config"), "orin.collection_config")
            ),
            task=_text(episode.get("task"), "episode.task"),
            operator_id=_text(episode.get("operator_id"), "episode.operator_id"),
            dig_target_m=target_values,
            material_id=_text(episode.get("material_id"), "episode.material_id"),
            collector_ready_timeout_s=_positive_int(
                runtime.get("collector_ready_timeout_s"),
                "runtime.collector_ready_timeout_s",
            ),
            ack_timeout_s=_positive_int(
                runtime.get("ack_timeout_s"), "runtime.ack_timeout_s"
            ),
            teleop_print_every=teleop_print_every,
            log_dir=(base / _text(runtime.get("log_dir"), "runtime.log_dir")).resolve(),
            rl_airy_repo=airy_repo,
            rl_ros_setup=ros_setup.resolve(),
            rl_workspace_setup=workspace_setup,
            rl_mission_config=mission_config,
            rl_phase=phase,
            rl_timeout_s=_positive_int(
                rl_preposition.get("timeout_s"), "rl_preposition.timeout_s"
            ),
            rl_serial_port=serial_port,
            rl_serial_release_timeout_s=_positive_int(
                rl_preposition.get("serial_release_timeout_s"),
                "rl_preposition.serial_release_timeout_s",
            ),
            rl_orin_repo=rl_orin_repo,
            rl_orin_python=rl_orin_python,
            rl_edge_config=rl_edge_config,
            rl_pc_host=rl_pc_host,
            rl_ready_timeout_s=_positive_int(
                rl_preposition.get("ready_timeout_s"),
                "rl_preposition.ready_timeout_s",
            ),
            rl_demo_config=demo_config,
            failure_reason=_text(
                episode.get("failure_reason", "diagnostic_task_failed"),
                "episode.failure_reason",
            ),
            zero_soak_duration_s=_positive_int(
                runtime.get("zero_soak_duration_s", 30),
                "runtime.zero_soak_duration_s",
            ),
            orin_experiment_run_config=experiment_run_config,
        )

