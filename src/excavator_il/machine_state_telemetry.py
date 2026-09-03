"""Read the PC-side ``machine_state_v1`` snapshot for WebUI display."""

from __future__ import annotations

import json
import math
import os
import shlex
import signal
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .guided_episode import GuidedEpisodeConfig
from .remote_runtime import LineProcess


_MAX_SNAPSHOT_BYTES = 1_048_576
_JOINT_NAMES = ("boom", "arm", "bucket", "swing")
_CYLINDER_NAMES = ("boom", "stick", "bucket")


def _object(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"machine-state {field} must be an object")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"machine-state {field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"machine-state {field} must be a finite number")
    return number


def _read_snapshot(path: Path) -> tuple[Mapping[str, Any], int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"machine-state snapshot unavailable: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("machine-state snapshot must be a regular file")
        if metadata.st_size > _MAX_SNAPSHOT_BYTES:
            raise RuntimeError("machine-state snapshot exceeds 1 MiB")
        payload = os.read(descriptor, _MAX_SNAPSHOT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > _MAX_SNAPSHOT_BYTES:
        raise RuntimeError("machine-state snapshot exceeds 1 MiB")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"machine-state snapshot is invalid JSON: {exc}") from exc
    return _object(value, "snapshot"), metadata.st_mtime_ns


class MachineStateTelemetryReader:
    """Translate one fresh state-bridge snapshot into the WebUI display contract."""

    def __init__(self, snapshot_path: str | Path, *, max_age_ms: int = 500) -> None:
        if (
            isinstance(max_age_ms, bool)
            or not isinstance(max_age_ms, int)
            or max_age_ms <= 0
        ):
            raise ValueError("max_age_ms must be a positive integer")
        self._snapshot_path = Path(snapshot_path).expanduser().resolve(strict=False)
        self._max_age_ms = max_age_ms

    def snapshot(self) -> dict[str, Any]:
        packet, modified_ns = _read_snapshot(self._snapshot_path)
        age_ms = max(0.0, (time.time_ns() - modified_ns) / 1_000_000.0)
        if age_ms > self._max_age_ms:
            raise RuntimeError(
                f"machine-state snapshot is stale: age_ms={age_ms:.1f}"
            )
        if (
            packet.get("type") != "machine_state_v1"
            or packet.get("schema_version") != "1.0"
        ):
            raise RuntimeError("machine-state snapshot schema is unsupported")
        sequence = packet.get("seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise RuntimeError("machine-state seq must be a non-negative integer")
        safety = _object(packet.get("safety"), "safety")
        positions = _object(
            _object(packet.get("joint_state"), "joint_state").get("position_rad"),
            "joint_state.position_rad",
        )
        actuators = _object(packet.get("actuator_state"), "actuator_state")
        angles = {
            name: math.degrees(
                _finite_number(positions.get(name), f"joint_state.position_rad.{name}")
            )
            for name in _JOINT_NAMES
        }
        cylinders = {
            name: 1_000.0
            * _finite_number(
                _object(actuators.get(name), f"actuator_state.{name}").get(
                    "position_m"
                ),
                f"actuator_state.{name}.position_m",
            )
            for name in _CYLINDER_NAMES
        }
        fault_flags = safety.get("fault_flags", [])
        if not isinstance(fault_flags, list) or any(
            not isinstance(item, str) or not item for item in fault_flags
        ):
            raise RuntimeError("machine-state safety.fault_flags must be a string list")
        return {
            "source": "machine_state_v1/udp:18081",
            "seq": sequence,
            "age_ms": age_ms,
            "sensor_valid": safety.get("sensor_valid") is True,
            "control_enabled": safety.get("control_enabled") is True,
            "fault_flags": list(fault_flags),
            "joint_angles_deg": angles,
            "cylinders_mm": cylinders,
        }


class MachineStateTelemetryService:
    """Own the PC UDP state bridge used by both WebUI and RViz."""

    def __init__(
        self,
        *,
        guided_config: GuidedEpisodeConfig,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        ready_timeout_s: int = 10,
        max_age_ms: int = 500,
    ) -> None:
        if ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive")
        self._config = guided_config
        self._factory = line_process_factory
        self._output = output
        self._ready_timeout_s = ready_timeout_s
        self._max_age_ms = max_age_ms
        self._process: Any | None = None
        self._reader: MachineStateTelemetryReader | None = None

    @property
    def snapshot_path(self) -> Path:
        return (
            Path(self._config.rl_airy_repo)
            / "runtime_bridge/exports/latest_state.json"
        )

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("PC machine-state bridge is already active")
        bridge = (
            Path(self._config.rl_airy_repo)
            / "runtime_bridge/apps/pc_runtime_bridge.py"
        )
        command = shlex.join(
            [
                "exec",
                "/usr/bin/python3",
                str(bridge),
                "--publish-joint-states",
                "--print-every",
                "100",
                "--write-every",
                "1",
            ]
        )
        shell_command = " && ".join(
            [
                f"source {shlex.quote(str(self._config.rl_ros_setup))}",
                f"source {shlex.quote(str(self._config.rl_workspace_setup))}",
                f"cd {shlex.quote(str(self._config.rl_airy_repo))}",
                command,
            ]
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = (
            Path(self._config.log_dir) / f"webui_machine_state_bridge_{stamp}.log"
        )
        process = self._factory(
            ["/bin/zsh", "-lc", shell_command],
            log_path=log_path,
            prefix="machine-state",
            output=self._output,
        )
        self._process = process
        try:
            process.wait_for(
                lambda line: "pc state bridge started:" in line,
                self._ready_timeout_s,
            )
        except BaseException:
            process.stop(signal.SIGINT, timeout_s=10.0)
            self._process = None
            raise
        self._reader = MachineStateTelemetryReader(
            self.snapshot_path,
            max_age_ms=self._max_age_ms,
        )

    def snapshot(self) -> dict[str, Any]:
        process = self._process
        if process is None or self._reader is None:
            raise RuntimeError("PC machine-state bridge is not active")
        if not process.running:
            raise RuntimeError("PC machine-state bridge exited unexpectedly")
        return self._reader.snapshot()

    def close(self) -> None:
        process = self._process
        self._process = None
        self._reader = None
        if process is not None:
            process.stop(signal.SIGINT, timeout_s=10.0)
