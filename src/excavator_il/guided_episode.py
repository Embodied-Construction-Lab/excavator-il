"""PC-side deadman-guided hardware Episode collection."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol

from .remote_runtime import LineProcess, LineWaitTimeout, SshRuntimeHost


GUIDED_EPISODE_CONFIG_SCHEMA_VERSION = "excavator_guided_episode_config.v3"
_SSH_HOST = re.compile(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+")
_NETWORK_HOST = re.compile(r"[A-Za-z0-9_.:-]+")
_EPISODE_NAME = re.compile(r"episode_\d{4,}")
_BRACKETED_PASTE_MARKER = re.compile(r"\x1b\[(?:200|201)~")
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/guided_episode.pc.json"


class PositioningMode(str, Enum):
    DIRECT = "direct"
    MANUAL = "manual"
    RL = "rl"


class GuidedEpisodeStage(str, Enum):
    PREFLIGHT = "preflight"
    RL_POSITIONING = "rl_positioning"
    COLLECTOR_STARTING = "collector_starting"
    MANUAL_POSITIONING = "manual_positioning"
    TELEOPERATION = "teleoperation"
    RECORDER_STANDBY = "recorder_standby"
    RECORDING = "recording"
    REVIEW = "review"
    FINALIZING = "finalizing"
    VALIDATING = "validating"
    COMPLETED = "completed"


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
        )


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


class GuidedEpisodeOperations(Protocol):
    def preflight(self) -> None: ...

    def start_rl_runtime(self) -> None: ...

    def run_rl_preposition(
        self, target_id: str | None = None
    ) -> tuple[float, float, float]: ...

    def stop_rl_runtime_and_wait_for_serial(self) -> None: ...

    def start_collector(self) -> None: ...

    def start_teleop(self) -> None: ...

    def wait_for_ack(self, timeout_s: int) -> None: ...

    def wait_for_deadman_pressed(self) -> None: ...

    def wait_for_deadman_released(self) -> None: ...

    def start_episode(
        self, dig_target_m: tuple[float, float, float] | None = None
    ) -> str: ...

    def seal_episode(self) -> str: ...

    def finalize_episode(
        self, episode_path: str, result: str, reason: str = ""
    ) -> str: ...

    def abort_episode(self, reason: str) -> str: ...

    def discard_episode(self, episode_path: str) -> None: ...

    def stop_teleop(self) -> None: ...

    def stop_collector(self) -> None: ...

    def build_and_validate(self, episode_path: str) -> None: ...


class SystemGuidedEpisodeOperations:
    """Real PC/SSH boundary used by the guided Episode script."""

    _ACK = re.compile(
        r"accepted_acks=(?P<accepted>\d+) rejected_acks=(?P<rejected>\d+).*"
        r"deadman=(?P<deadman>True|False)"
    )

    def __init__(
        self,
        config: GuidedEpisodeConfig,
        *,
        output: Callable[[str], None] = print,
        timestamp: str | None = None,
        line_process_factory: Callable[..., Any] = LineProcess,
    ) -> None:
        self._config = config
        self._remote_host: SshRuntimeHost | None = None
        self._output = output
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._line_process_factory = line_process_factory
        self._collector: Any | None = None
        self._teleop: Any | None = None
        self._collector_pid: int | None = None
        self._rl_runtime: Any | None = None
        self._rl_runtime_pid: int | None = None
        self._operator_preview: Any | None = None
        self._operator_preview_pid: int | None = None
        self._rl_hardware_start_gate: PurePosixPath | None = None
        self._teleop_cursor = -1
        self._started_episode_paths: tuple[str, ...] = ()
        self._discardable_episode_paths: tuple[str, ...] = ()

    @property
    def log_paths(self) -> tuple[Path, Path, Path]:
        stem = f"guided_episode_{self._timestamp}"
        return (
            self._config.log_dir / f"{stem}.collector.log",
            self._config.log_dir / f"{stem}.teleop.log",
            self._config.log_dir / f"{stem}.validation.log",
        )

    def _ssh_argv(self, remote_command: str) -> list[str]:
        return self._ssh_host().argv(remote_command)

    def _in_repo(self, argv: list[str]) -> str:
        return (
            f"cd {shlex.quote(str(self._config.orin_repo))} && "
            f"{shlex.join(argv)}"
        )

    def _in_remote_rl_repo(self, argv: list[str]) -> str:
        return (
            f"cd {shlex.quote(str(self._config.rl_orin_repo))} && "
            f"{shlex.join(argv)}"
        )

    def _run_ssh(
        self,
        remote_command: str,
        *,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> str:
        return self._ssh_host().run(
            remote_command, accepted_returncodes=accepted_returncodes
        )

    def _remote_cli(
        self,
        argv: list[str],
        *,
        accepted_returncodes: tuple[int, ...] = (0,),
    ) -> Mapping[str, Any]:
        executable = str(self._config.orin_executable)
        output = self._run_ssh(
            self._in_repo([executable, *argv]),
            accepted_returncodes=accepted_returncodes,
        )

        try:
            response = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid remote CLI JSON: {output!r}") from exc
        if not isinstance(response, Mapping):
            raise RuntimeError("remote CLI response must be an object")
        return response

    def _ssh_host(self) -> SshRuntimeHost:
        if self._remote_host is None:
            self._remote_host = SshRuntimeHost(
                self._config.orin_ssh_host, run_command=subprocess.run
            )
        return self._remote_host

    @staticmethod
    def _episode_path(response: Mapping[str, Any]) -> str:
        path = response.get("path")
        if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
            raise RuntimeError("remote Episode response did not contain an absolute path")
        return path

    def preflight(self) -> None:
        if not self._config.teleop_config.is_file():
            raise RuntimeError(
                f"teleop config does not exist: {self._config.teleop_config}"
            )
        remote_check = self._in_repo(
            [
                "test",
                "-x",
                str(self._config.orin_executable),
            ]
        )
        self._run_ssh(remote_check)
        self._reclaim_known_serial_owner()

    def _known_serial_owner_argv(self) -> tuple[tuple[str, ...], ...]:
        collector = (
            str(self._config.orin_executable),
            "collect",
            "--config",
            str(self._config.orin_collection_config),
        )
        rl_runtime = (
            str(self._config.rl_orin_python),
            "-u",
            "orin_state_sender.py",
            "--serial-port",
            str(self._config.rl_serial_port),
            "--control-enabled",
            "--pc-host",
            self._config.rl_pc_host,
            "--edge-config",
            str(self._config.rl_edge_config),
            "--edge-motion-authorization",
            "ALLOW_EDGE_MACHINE_MOTION",
            "--print-every",
            "100",
        )
        # A manually launched but otherwise identical Runtime may use the
        # environment's ``python`` command instead of the configured absolute
        # interpreter path.  Match the complete behavior argv from ``-u``
        # onward so workflow takeover remains exact without depending on how
        # that interpreter was named.
        rl_runtime_interpreter_independent = rl_runtime[1:]
        return collector, rl_runtime, rl_runtime_interpreter_independent

    def _operator_preview_argv(self) -> tuple[str, ...]:
        return (
            str(self._config.orin_executable),
            "camera-preview",
            "--config",
            str(self._config.orin_collection_config),
        )

    def _reclaim_known_camera_owner(self) -> None:
        result = self._ssh_host().reclaim_serial_owner(
            serial_path="/dev/video0",
            known_argv_suffixes=(self._operator_preview_argv(),),
            timeout_s=self._config.rl_serial_release_timeout_s,
            execute=self._run_ssh,
        )
        if result == "reclaimed":
            self._output("检测到并释放了上一次遗留的 Orin 相机预览进程。")

    def _reclaim_known_serial_owner(self) -> None:
        result = self._ssh_host().reclaim_serial_owner(
            serial_path=self._config.rl_serial_port,
            known_argv_suffixes=self._known_serial_owner_argv(),
            timeout_s=self._config.rl_serial_release_timeout_s,
            execute=self._run_ssh,
        )
        if result == "reclaimed":
            self._output("检测到并释放了上一次遗留的 Orin 串口 Runtime。")

    def _rl_target(self, phase: str = "dig") -> tuple[float, float, float]:
        if phase not in {"dig", "dump"}:
            raise ValueError("RL phase must be dig or dump")
        try:
            root = json.loads(self._config.rl_mission_config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot load RL Mission config {self._config.rl_mission_config}: {exc}"
            ) from exc
        root = _object(root, "RL Mission config")
        if root.get("schema_version") != "excavation_mission.v1":
            raise RuntimeError("RL Mission schema_version must be excavation_mission.v1")
        targets = _object(root.get("targets"), "RL Mission targets")
        target = _object(
            targets.get(phase),
            f"RL Mission targets.{phase}",
        ).get("position_m")
        if not isinstance(target, list) or len(target) != 3:
            raise RuntimeError(f"RL Mission {phase} position_m must contain three numbers")
        try:
            values = tuple(float(value) for value in target)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"RL Mission {phase} position_m must contain numbers") from exc
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(f"RL Mission {phase} position_m must be finite")
        return values

    def _rl_runtime_preflight(self, *, require_serial_free: bool) -> None:
        python = shlex.quote(str(self._config.rl_orin_python))
        edge_config = shlex.quote(str(self._config.rl_edge_config))
        pc_host = shlex.quote(self._config.rl_pc_host)
        serial = shlex.quote(str(self._config.rl_serial_port))
        validate_edge = (
            "import json,sys; "
            "actual=json.load(open(sys.argv[1], encoding='utf-8'))"
            ".get('remote_behavior',{}).get('allowed_client_host'); "
            "actual == sys.argv[2] or sys.exit("
            "f'allowed_client_host={actual!r}, expected {sys.argv[2]!r}')"
        )
        serial_check = ""
        if require_serial_free:
            serial_check = f"""if fuser -s {serial}; then
  echo "serial is already owned: {serial}" >&2
  exit 16
fi
"""
        preflight_script = f"""set -eu
command -v pgrep >/dev/null
command -v fuser >/dev/null
test -x {python}
test -f {edge_config}
{python} -c {shlex.quote(validate_edge)} {edge_config} {pc_host}
if pgrep -u "$(id -u)" -f '[p]ython[^ ]* .*[/ ]orin_state_sender\\.py([[:space:]]|$)' >/dev/null; then
  echo "orin_state_sender.py is already running" >&2
  exit 15
fi
{serial_check}
echo ready
"""
        preflight = self._run_ssh(
            self._in_remote_rl_repo(
                ["/bin/sh", "-c", preflight_script]
            )
        )
        if preflight.strip() != "ready":
            raise RuntimeError("RL Runtime preflight did not confirm readiness")

    def _rl_remote_command(
        self, *, hardware_start_gate: PurePosixPath | None
    ) -> str:
        argv = [
            str(self._config.rl_orin_python),
            "-u",
            "orin_state_sender.py",
            "--serial-port",
            str(self._config.rl_serial_port),
            "--control-enabled",
            "--pc-host",
            self._config.rl_pc_host,
            "--edge-config",
            str(self._config.rl_edge_config),
            "--edge-motion-authorization",
            "ALLOW_EDGE_MACHINE_MOTION",
            "--print-every",
            "100",
        ]
        if hardware_start_gate is not None:
            argv.extend(("--hardware-start-gate", str(hardware_start_gate)))
        remote_command = self._in_remote_rl_repo(argv)
        return remote_command.replace(
            "&& ", "&& echo GUIDED_RL_PID=$$ && exec ", 1
        )

    def _spawn_rl_runtime(
        self, *, hardware_start_gate: PurePosixPath | None
    ) -> None:
        rl_log = self._config.log_dir / f"guided_episode_{self._timestamp}.rl-runtime.log"
        remote_command = self._rl_remote_command(
            hardware_start_gate=hardware_start_gate
        )
        self._rl_runtime = self._line_process_factory(
            self._ssh_argv(remote_command),
            log_path=rl_log,
            prefix="rl-runtime",
            output=self._output,
        )
        try:
            _, pid_line = self._rl_runtime.wait_for(
                lambda line: line.startswith("GUIDED_RL_PID="),
                self._config.rl_ready_timeout_s,
            )
            self._rl_runtime_pid = int(pid_line.split("=", maxsplit=1)[1])
            if hardware_start_gate is not None:
                self._rl_runtime.wait_for(
                    lambda line: "RL prewarm ready:" in line,
                    self._config.rl_ready_timeout_s,
                )
                return
            self._wait_for_rl_runtime_ready()
        except BaseException:
            self.stop_rl_runtime_and_wait_for_serial()
            raise

    def _wait_for_rl_runtime_ready(self) -> None:
        if self._rl_runtime is None:
            raise RuntimeError("RL Runtime process is not available")
        self._rl_runtime.wait_for(
            lambda line: "REMOTE EDGE CONTROL ARMED IDLE" in line,
            self._config.rl_ready_timeout_s,
        )
        self._rl_runtime.wait_for(
            lambda line: "sent seq=" in line and "sensor_valid=True" in line,
            self._config.rl_ready_timeout_s,
        )

    def _confirm_rl_serial_release(self) -> None:
        serial = shlex.quote(str(self._config.rl_serial_port))
        output = self._run_ssh(
            "/bin/sh -c "
            + shlex.quote(
                f"set -eu; command -v fuser >/dev/null; "
                f"if fuser -s {serial}; then echo 'serial still owned' >&2; exit 14; fi; "
                "echo released"
            )
        )
        if output.strip() != "released":
            raise RuntimeError("RL handoff did not confirm serial release")

    def prewarm_rl_runtime(self, hardware_start_gate: str | PurePosixPath) -> None:
        """Load RL deployment assets without opening the STM32 serial port."""

        if self._rl_runtime is not None:
            raise RuntimeError("an RL Runtime is already active or prewarmed")
        gate = PurePosixPath(hardware_start_gate)
        allowed_parent = PurePosixPath("/tmp/excavator-rl-control")
        if (
            not gate.is_absolute()
            or gate.parent != allowed_parent
            or not gate.name.startswith("hybrid_")
            or not gate.name.endswith(".start")
        ):
            raise ValueError(
                "RL hardware start gate must be /tmp/excavator-rl-control/hybrid_*.start"
            )
        self._rl_runtime_preflight(require_serial_free=False)
        self._run_ssh(
            "/bin/sh -c "
            + shlex.quote(
                "set -eu; mkdir -p -- "
                + shlex.quote(str(gate.parent))
                + "; rm -f -- "
                + shlex.quote(str(gate))
            )
        )
        self._rl_hardware_start_gate = gate
        self._spawn_rl_runtime(hardware_start_gate=gate)

    def start_rl_runtime(self) -> None:
        if self._rl_runtime is not None:
            gate = self._rl_hardware_start_gate
            if gate is None:
                raise RuntimeError("RL Runtime is already active")
            try:
                self._confirm_rl_serial_release()
                self._run_ssh("touch -- " + shlex.quote(str(gate)))
                self._rl_hardware_start_gate = None
                self._wait_for_rl_runtime_ready()
            except BaseException:
                self.stop_rl_runtime_and_wait_for_serial()
                raise
            return

        self._reclaim_known_serial_owner()
        self._rl_runtime_preflight(require_serial_free=True)
        self._spawn_rl_runtime(hardware_start_gate=None)

    def start_operator_preview(self) -> None:
        """Serve the front camera while RL exclusively owns the STM32 serial port."""

        if self._operator_preview is not None:
            raise RuntimeError("operator camera preview is already active")
        self._reclaim_known_camera_owner()
        command = self._in_repo(list(self._operator_preview_argv()))
        command = command.replace(
            "&& ", "&& echo GUIDED_PREVIEW_PID=$$ && exec ", 1
        )
        log_path = (
            self._config.log_dir
            / f"guided_episode_{self._timestamp}.camera-preview.log"
        )
        self._operator_preview = self._line_process_factory(
            self._ssh_argv(command),
            log_path=log_path,
            prefix="camera-preview",
            output=self._output,
        )
        try:
            _, pid_line = self._operator_preview.wait_for(
                lambda line: line.startswith("GUIDED_PREVIEW_PID="),
                self._config.collector_ready_timeout_s,
            )
            self._operator_preview_pid = int(pid_line.split("=", maxsplit=1)[1])
            self._operator_preview.wait_for(
                lambda line: "camera preview ready:" in line,
                self._config.collector_ready_timeout_s,
            )
        except BaseException:
            self.stop_operator_preview_and_wait_for_camera()
            raise

    def stop_operator_preview_and_wait_for_camera(self) -> None:
        preview = self._operator_preview
        pid = self._operator_preview_pid
        if preview is None:
            return
        if pid is None:
            preview.stop(signal.SIGKILL, timeout_s=2.0)
            self._operator_preview = None
            raise RuntimeError(
                "camera preview PID was not observed; camera release is unknown"
            )
        try:
            self._ssh_host().stop_owned_process(
                pid=pid,
                identity_ere=r"[c]amera-preview",
                serial_path="/dev/video0",
                timeout_s=self._config.rl_serial_release_timeout_s,
                execute=self._run_ssh,
            )
            try:
                preview.wait(timeout_s=2.0)
            except subprocess.TimeoutExpired:
                preview.stop(signal.SIGKILL, timeout_s=2.0)
        finally:
            self._operator_preview = None
            self._operator_preview_pid = None

    def run_rl_preposition(
        self, target_id: str | None = None
    ) -> tuple[float, float, float]:
        return self.run_rl_follow("dig", target_id=target_id)

    def run_rl_follow(
        self, phase: str, *, target_id: str | None = None
    ) -> tuple[float, float, float]:
        if phase not in {"dig", "dump"}:
            raise ValueError("RL phase must be dig or dump")
        if phase != "dig" and target_id is not None:
            raise ValueError("a selected demo target is only valid for the dig phase")
        required_paths = (
            self._config.rl_airy_repo,
            self._config.rl_ros_setup,
            self._config.rl_workspace_setup,
            self._config.rl_mission_config,
        )
        if target_id is not None:
            if self._config.rl_demo_config is None:
                raise RuntimeError("RL demo config is required for a selected DIG point")
            required_paths = (*required_paths, self._config.rl_demo_config)
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"RL positioning path does not exist: {', '.join(missing)}")
        if target_id is None:
            target = self._rl_target(phase)
            target_args = ["--mission", str(self._config.rl_mission_config)]
        else:
            targets = dict(load_rl_dig_targets(self._config))
            try:
                target = targets[target_id]
            except KeyError as exc:
                raise RuntimeError(f"unknown RL DIG target: {target_id}") from exc
            target_args = [
                "--demo",
                str(self._config.rl_demo_config),
                "--dig-point",
                target_id,
            ]
        rl_log = (
            self._config.log_dir
            / f"guided_episode_{self._timestamp}.rl-{phase}.log"
        )
        shell_command = " && ".join(
            (
                f"source {shlex.quote(str(self._config.rl_ros_setup))}",
                f"source {shlex.quote(str(self._config.rl_workspace_setup))}",
                f"cd {shlex.quote(str(self._config.rl_airy_repo))}",
                shlex.join(
                    [
                        "exec",
                        "/usr/bin/python3",
                        "-m",
                        "mission.runtime_ros.run_plan_follow_live",
                        phase,
                        *target_args,
                        "--wait-s",
                        str(self._config.ack_timeout_s),
                    ]
                ),
            )
        )
        process = self._line_process_factory(
            ["/bin/zsh", "-lc", shell_command],
            log_path=rl_log,
            prefix="rl-position",
            output=self._output,
        )
        try:
            process.wait(timeout_s=self._config.rl_timeout_s)
        except subprocess.TimeoutExpired as exc:
            process.stop(signal.SIGINT, timeout_s=3.0)
            raise RuntimeError(
                f"RL positioning timed out after {self._config.rl_timeout_s}s"
            ) from exc
        if process.returncode != 0:
            raise RuntimeError(
                f"RL positioning failed with exit code {process.returncode}; "
                f"see {rl_log}"
            )
        return target

    def run_rl_fixed_action(self, behavior: str, *, behavior_port: int) -> None:
        if behavior not in {"ExecuteDig", "ExecuteDump"}:
            raise ValueError("behavior must be ExecuteDig or ExecuteDump")
        if (
            isinstance(behavior_port, bool)
            or not isinstance(behavior_port, int)
            or not 1 <= behavior_port <= 65535
        ):
            raise ValueError("behavior_port must be an integer in [1, 65535]")
        required_paths = (self._config.rl_airy_repo,)
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"RL fixed-action path does not exist: {', '.join(missing)}")
        _user, orin_host = self._config.orin_ssh_host.split("@", maxsplit=1)
        rl_log = (
            self._config.log_dir
            / f"guided_episode_{self._timestamp}.{behavior}.log"
        )
        shell_command = " && ".join(
            (
                f"cd {shlex.quote(str(self._config.rl_airy_repo))}",
                shlex.join(
                    [
                        "exec",
                        "/usr/bin/python3",
                        "-m",
                        "runtime_bridge.apps.run_orin_fixed_action",
                        behavior,
                        "--host",
                        orin_host,
                        "--port",
                        str(behavior_port),
                    ]
                ),
            )
        )
        process = self._line_process_factory(
            ["/bin/zsh", "-lc", shell_command],
            log_path=rl_log,
            prefix="rl-fixed-action",
            output=self._output,
        )
        try:
            process.wait(timeout_s=self._config.rl_timeout_s)
        except subprocess.TimeoutExpired as exc:
            process.stop(signal.SIGINT, timeout_s=3.0)
            raise RuntimeError(
                f"{behavior} timed out after {self._config.rl_timeout_s}s"
            ) from exc
        if process.returncode != 0:
            raise RuntimeError(
                f"{behavior} failed with exit code {process.returncode}; see {rl_log}"
            )

    def stop_rl_runtime_and_wait_for_serial(self) -> None:
        runtime = self._rl_runtime
        runtime_pid = self._rl_runtime_pid
        gate = self._rl_hardware_start_gate
        if runtime is None:
            return
        if runtime_pid is None:
            runtime.stop(signal.SIGKILL, timeout_s=2.0)
            self._rl_runtime = None
            self._rl_hardware_start_gate = None
            raise RuntimeError("RL Runtime PID was not observed; serial release is unknown")
        try:
            self._ssh_host().stop_owned_process(
                pid=runtime_pid,
                identity_ere=r"[o]rin_state_sender\.py",
                serial_path=self._config.rl_serial_port,
                timeout_s=self._config.rl_serial_release_timeout_s,
                require_serial_release=gate is None,
                cleanup_paths=(gate,) if gate is not None else (),
                execute=self._run_ssh,
            )
            try:
                runtime.wait(timeout_s=2.0)
            except subprocess.TimeoutExpired:
                runtime.stop(signal.SIGKILL, timeout_s=2.0)
        finally:
            self._rl_runtime = None
            self._rl_runtime_pid = None
            self._rl_hardware_start_gate = None

    def start_collector(self) -> None:
        collector_log, _, _ = self.log_paths
        executable = str(self._config.orin_executable)
        command = self._in_repo(
            [
                executable,
                "collect",
                "--config",
                str(self._config.orin_collection_config),
            ]
        )
        command = command.replace("&& ", "&& echo GUIDED_COLLECTOR_PID=$$ && exec ", 1)
        self._collector = self._line_process_factory(
            self._ssh_argv(command),
            log_path=collector_log,
            prefix="collector",
            output=self._output,
        )
        try:
            _, pid_line = self._collector.wait_for(
                lambda line: line.startswith("GUIDED_COLLECTOR_PID="),
                self._config.collector_ready_timeout_s,
            )
            self._collector_pid = int(pid_line.split("=", maxsplit=1)[1])
            self._collector.wait_for(
                lambda line: "collector ready:" in line,
                self._config.collector_ready_timeout_s,
            )
        except BaseException:
            self.stop_collector()
            raise

    def start_teleop(self) -> None:
        _, teleop_log, _ = self.log_paths
        self._teleop_cursor = -1
        self._teleop = self._line_process_factory(
            [
                sys.executable,
                "-u",
                "-m",
                "excavator_il.cli",
                "teleop",
                "--config",
                str(self._config.teleop_config),
                "--print-every",
                str(self._config.teleop_print_every),
            ],
            log_path=teleop_log,
            prefix="teleop",
            output=self._output,
            echo_output=False,
        )

    @classmethod
    def _accepted_safe_ack(cls, line: str) -> bool:
        match = cls._ACK.search(line)
        if match is None:
            return False
        rejected = int(match.group("rejected"))
        deadman = match.group("deadman") == "True"
        if rejected or deadman:
            raise RuntimeError(
                "teleop ACK gate failed: rejected ACK or deadman pressed before standby"
            )
        return int(match.group("accepted")) > 0

    def wait_for_ack(self, timeout_s: int) -> None:
        if self._teleop is None:
            raise RuntimeError("teleop is not running")
        self._teleop_cursor, _ = self._teleop.wait_for(
            self._accepted_safe_ack,
            timeout_s,
            after_index=self._teleop_cursor,
        )

    @classmethod
    def _deadman_is(cls, line: str, expected: bool) -> bool:
        match = cls._ACK.search(line)
        return match is not None and (match.group("deadman") == "True") is expected

    def _wait_for_deadman(self, expected: bool) -> None:
        if self._teleop is None:
            raise RuntimeError("teleop is not running")
        self._teleop_cursor, _ = self._teleop.wait_for(
            lambda line: self._deadman_is(line, expected),
            None,
            after_index=self._teleop_cursor,
        )

    def wait_for_deadman_pressed(self) -> None:
        self._wait_for_deadman(True)

    def wait_for_deadman_released(self) -> None:
        self._wait_for_deadman(False)

    def monitor_deadman_released(self, duration_s: int) -> None:
        """Require continuous safe ACK samples for one bounded soak interval."""
        if self._teleop is None:
            raise RuntimeError("teleop is not running")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        deadline = time.monotonic() + duration_s
        sample_count = 0
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            try:
                self._teleop_cursor, line = self._teleop.wait_for(
                    lambda candidate: self._ACK.search(candidate) is not None,
                    min(1.0, remaining_s),
                    after_index=self._teleop_cursor,
                )
            except LineWaitTimeout:
                if time.monotonic() >= deadline:
                    break
                raise
            match = self._ACK.search(line)
            assert match is not None
            if match.group("deadman") == "True":
                raise RuntimeError("zero-command soak failed: deadman was pressed")
            if int(match.group("rejected")):
                raise RuntimeError("zero-command soak failed: Collector rejected an ACK")
            sample_count += 1
        if sample_count < duration_s * 15:
            raise RuntimeError(
                f"zero-command soak received only {sample_count} teleop samples"
            )

    def start_episode(
        self, dig_target_m: tuple[float, float, float] | None = None
    ) -> str:
        target = self._config.dig_target_m if dig_target_m is None else dig_target_m
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "start",
                "--task",
                self._config.task,
                "--operator",
                self._config.operator_id,
                "--dig-target-m",
                *(str(value) for value in target),
                "--material-id",
                self._config.material_id,
            ]
        )
        path = self._episode_path(response)
        self._started_episode_paths = (*self._started_episode_paths, path)
        return path

    def seal_episode(self) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "seal",
            ]
        )
        path = self._episode_path(response)
        if response.get("status") != "pending_review":
            raise RuntimeError("sealed Episode was not confirmed pending_review")
        if path not in self._started_episode_paths:
            raise RuntimeError("sealed Episode was not started by this run")
        self._discardable_episode_paths = (*self._discardable_episode_paths, path)
        return path

    def finalize_episode(
        self, episode_path: str, result: str, reason: str = ""
    ) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "finalize",
                episode_path,
                "--result",
                result,
                "--failure-reason",
                reason,
            ]
        )
        path = self._episode_path(response)
        if path != episode_path:
            raise RuntimeError("Collector finalized an unexpected Episode path")
        self._discardable_episode_paths = tuple(
            candidate
            for candidate in self._discardable_episode_paths
            if candidate != path
        )
        return path

    def abort_episode(self, reason: str) -> str:
        response = self._remote_cli(
            [
                "episode",
                "--config",
                str(self._config.orin_collection_config),
                "abort",
                "--reason",
                reason,
            ]
        )
        path = self._episode_path(response)
        return path

    def discard_episode(self, episode_path: str) -> None:
        if episode_path not in self._discardable_episode_paths:
            raise RuntimeError("refusing to discard an unapproved Episode path")
        path = PurePosixPath(episode_path)
        if (
            not path.is_absolute()
            or _EPISODE_NAME.fullmatch(path.name) is None
            or ".." in path.parts
            or len(path.parts) < 4
        ):
            raise RuntimeError(f"refusing to discard unsafe Episode path: {path}")
        quoted_path = shlex.quote(str(path))
        self._run_ssh(
            f"test -d {quoted_path} && test ! -L {quoted_path} && "
            f"rm -rf -- {quoted_path} && test ! -e {quoted_path}"
        )
        self._discardable_episode_paths = tuple(
            candidate
            for candidate in self._discardable_episode_paths
            if candidate != episode_path
        )

    def stop_teleop(self) -> None:
        if self._teleop is not None:
            self._teleop.stop(signal.SIGINT)
            self._teleop = None

    def stop_collector(self) -> None:
        collector = self._collector
        collector_pid = self._collector_pid
        if (
            collector_pid is not None
            and collector is not None
            and collector.running
        ):
            self._run_ssh(f"kill -TERM -- -{collector_pid}")
        try:
            if collector is not None:
                collector.wait(timeout_s=2.0)
        except subprocess.TimeoutExpired as exc:
            if collector_pid is None:
                assert collector is not None
                collector.stop(signal.SIGKILL, timeout_s=2.0)
                raise RuntimeError(
                    "Collector SSH transport did not exit and its remote PID is unknown"
                ) from exc
            remote_state = self._run_ssh(
                "for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 "
                "18 19 20; do "
                f"if ! kill -0 -- -{collector_pid} 2>/dev/null; then "
                "echo exited; exit 0; fi; sleep 0.25; done; echo running"
            ).strip()
            assert collector is not None
            collector.stop(signal.SIGKILL, timeout_s=2.0)
            if remote_state != "exited":
                raise RuntimeError(
                    f"remote Collector process group {collector_pid} is still running "
                    "after TERM timeout"
                ) from exc
        finally:
            self._collector = None
            self._collector_pid = None

    def build_and_validate(self, episode_path: str) -> None:
        _, _, validation_log = self.log_paths
        outputs = []
        for argv in (
            ["build-steps", episode_path],
            ["validate", episode_path],
        ):
            response = self._remote_cli(argv)
            outputs.append(json.dumps(dict(response), ensure_ascii=False, indent=2))
        report_path = str(PurePosixPath(episode_path) / "quality_report.json")
        report_text = self._run_ssh(self._in_repo(["cat", report_path]))
        try:
            report = json.loads(report_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("quality_report.json is not valid JSON") from exc
        outputs.append(json.dumps(report, ensure_ascii=False, indent=2))
        validation_log.parent.mkdir(parents=True, exist_ok=True)
        with validation_log.open("a", encoding="utf-8") as log:
            log.write(f"=== {episode_path} ===\n")
            log.write("\n".join(outputs) + "\n")
        self._output(json.dumps(report, ensure_ascii=False, indent=2))

    def inspect_zero_soak(self, episode_path: str) -> Mapping[str, Any]:
        return self._remote_cli(
            ["inspect-zero-soak", episode_path],
            accepted_returncodes=(0, 3),
        )


def run_standalone_teleop(
    config: GuidedEpisodeConfig,
    operations: GuidedEpisodeOperations,
    *,
    wait_fn: Callable[[], None],
    output: Callable[[str], None] = print,
    stage_callback: Callable[[GuidedEpisodeStage], None] | None = None,
) -> None:
    """Run deadman-gated manual control without creating an Episode."""
    collector_started = False
    teleop_started = False
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    emit_stage = stage_callback or (lambda _stage: None)
    try:
        emit_stage(GuidedEpisodeStage.PREFLIGHT)
        operations.preflight()
        emit_stage(GuidedEpisodeStage.COLLECTOR_STARTING)
        operations.start_collector()
        collector_started = True
        operations.start_teleop()
        teleop_started = True
        operations.wait_for_ack(config.ack_timeout_s)
        emit_stage(GuidedEpisodeStage.TELEOPERATION)
        output(
            "仅遥操作已就绪：按住 deadman 后用双杆控制；释放 deadman 立即回零。"
            "按 Ctrl+C 或点击安全停止退出，不会创建 Episode。"
        )
        wait_fn()
    except BaseException as exc:
        failure = exc
    finally:
        if teleop_started:
            try:
                operations.stop_teleop()
            except Exception as exc:
                cleanup_errors.append(f"teleop cleanup failed: {exc}")
        if collector_started:
            try:
                operations.stop_collector()
            except Exception as exc:
                cleanup_errors.append(f"Collector cleanup failed: {exc}")
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if failure is not None:
                output(f"ERROR: {message}")
            else:
                failure = RuntimeError(message)
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    emit_stage(GuidedEpisodeStage.COMPLETED)


def run_guided_episode(
    config: GuidedEpisodeConfig,
    operations: GuidedEpisodeOperations,
    *,
    preposition: bool = False,
    positioning_mode: PositioningMode | str | None = None,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    stage_callback: Callable[[GuidedEpisodeStage], None] | None = None,
    rl_target_id: str | None = None,
) -> str:
    """Collect deadman-bounded attempts and validate them after motion I/O stops."""
    if positioning_mode is None:
        mode = PositioningMode.MANUAL if preposition else PositioningMode.DIRECT
    else:
        mode = PositioningMode(positioning_mode)
        if preposition and mode is not PositioningMode.MANUAL:
            raise ValueError("preposition=True conflicts with positioning_mode")
    collector_started = False
    rl_runtime_started = False
    teleop_started = False
    episode_active = False
    deadman_started = False
    pending_path: str | None = None
    completed_path: str | None = None
    retained_paths: tuple[str, ...] = ()
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    episode_target_m = config.dig_target_m
    emit_stage = stage_callback or (lambda _stage: None)
    try:
        emit_stage(GuidedEpisodeStage.PREFLIGHT)
        operations.preflight()
        if mode is PositioningMode.RL:
            emit_stage(GuidedEpisodeStage.RL_POSITIONING)
            output(
                "RL 定位阶段：将按 AiryLidar Mission 配置执行 Plan DIG → Follow。"
            )
            operations.start_rl_runtime()
            rl_runtime_started = True
            if rl_target_id is None:
                episode_target_m = operations.run_rl_preposition()
            else:
                episode_target_m = operations.run_rl_preposition(rl_target_id)
            operations.stop_rl_runtime_and_wait_for_serial()
            rl_runtime_started = False
            output(
                "RL Follow 已成功归零，RL Runtime 已退出并释放串口；"
                "开始切换到人工示教 Collector。"
            )
        emit_stage(GuidedEpisodeStage.COLLECTOR_STARTING)
        operations.start_collector()
        collector_started = True
        if mode is PositioningMode.MANUAL:
            operations.start_teleop()
            teleop_started = True
            operations.wait_for_ack(config.ack_timeout_s)
            emit_stage(GuidedEpisodeStage.MANUAL_POSITIONING)
            output(
                "预定位阶段（不记录 Episode）：按住 deadman，用双杆把挖掘机移动到 "
                "RL Follow 的交接位姿附近。"
            )
            _wait_for_preposition_complete(input_fn, output)
            operations.wait_for_deadman_released()
            operations.stop_teleop()
            teleop_started = False
            output(
                "预定位结束：已确认 deadman 释放并停止预定位 teleop。"
                "请保持双杆 X/Y/Z 全部回中，开始正式 Recorder 门禁。"
            )
        operations.start_episode(episode_target_m)
        episode_active = True
        operations.start_teleop()
        teleop_started = True
        operations.wait_for_ack(config.ack_timeout_s)
        while True:
            emit_stage(GuidedEpisodeStage.RECORDER_STANDBY)
            output(
                "Recorder 已进入待命。保持双杆 X/Y/Z 全部回中；按下 deadman 后可立即操纵 XY。"
            )
            operations.wait_for_deadman_pressed()
            deadman_started = True
            emit_stage(GuidedEpisodeStage.RECORDING)
            output(
                "记录已开始：按住 deadman 完成动作；记录阶段只执行 XY，完成后将 X/Y/Z 全部回中并松开 deadman。"
            )
            operations.wait_for_deadman_released()
            completed_path = operations.seal_episode()
            episode_active = False
            pending_path = completed_path
            output("检测到 deadman 松开，动作命令已回零，Episode 已自动保存。")
            emit_stage(GuidedEpisodeStage.REVIEW)
            outcome = _read_outcome(input_fn, output)
            emit_stage(GuidedEpisodeStage.FINALIZING)
            if outcome == "success":
                completed_path = operations.finalize_episode(
                    completed_path, "success"
                )
                pending_path = None
            elif outcome == "failure":
                completed_path = operations.finalize_episode(
                    completed_path, "failure", config.failure_reason
                )
                pending_path = None
            retained_paths = (*retained_paths, completed_path)
            if outcome != "retake":
                break
            operations.discard_episode(completed_path)
            pending_path = None
            retained_paths = tuple(
                path for path in retained_paths if path != completed_path
            )
            output(
                f"本次已删除：{completed_path}。双杆 X/Y/Z 全部回中后可再次按 deadman 重录，"
                "Episode 编号保持不变。"
            )
            operations.start_episode(episode_target_m)
            episode_active = True
            deadman_started = False
    except BaseException as exc:
        failure = exc
        if episode_active:
            try:
                completed_path = operations.abort_episode(
                    "guided_episode_interrupted"
                )
                if deadman_started:
                    retained_paths = (*retained_paths, completed_path)
            except Exception as abort_exc:
                output(f"ERROR: failed to abort active Episode: {abort_exc}")
            episode_active = False
        elif pending_path is not None:
            try:
                completed_path = operations.finalize_episode(
                    pending_path,
                    "aborted",
                    "guided_episode_interrupted",
                )
                retained_paths = (*retained_paths, completed_path)
            except Exception as finalize_exc:
                output(
                    "ERROR: failed to finalize sealed Episode after interruption: "
                    f"{finalize_exc}"
                )
            pending_path = None
    finally:
        if rl_runtime_started:
            try:
                operations.stop_rl_runtime_and_wait_for_serial()
            except Exception as exc:
                cleanup_errors.append(f"RL Runtime cleanup failed: {exc}")
        if teleop_started:
            try:
                operations.stop_teleop()
            except Exception as exc:
                cleanup_errors.append(f"teleop cleanup failed: {exc}")
        if collector_started:
            try:
                operations.stop_collector()
            except Exception as exc:
                cleanup_errors.append(f"Collector cleanup failed: {exc}")
        if cleanup_errors:
            message = "; ".join(cleanup_errors)
            if failure is not None:
                output(f"ERROR: {message}")
            else:
                failure = RuntimeError(message)
    emit_stage(GuidedEpisodeStage.VALIDATING)
    for episode_path in retained_paths:
        try:
            operations.build_and_validate(episode_path)
        except BaseException as build_exc:
            if failure is None:
                failure = build_exc
            else:
                output(
                    f"ERROR: failed to validate retained Episode "
                    f"{episode_path}: {build_exc}"
                )
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    assert completed_path is not None
    emit_stage(GuidedEpisodeStage.COMPLETED)
    return completed_path


def _read_outcome(
    input_fn: Callable[[str], str], output: Callable[[str], None]
) -> str:
    choices = {
        "成功": "success",
        "s": "success",
        "失败": "failure",
        "f": "failure",
        "重录": "retake",
        "r": "retake",
    }
    while True:
        raw_value = input_fn("请输入结果（成功/s、失败/f、重录/r）后按 Enter：")
        value = _BRACKETED_PASTE_MARKER.sub("", raw_value).strip().lower()
        outcome = choices.get(value)
        if outcome is not None:
            return outcome
        output("无法识别结果，请输入：成功、失败或重录。")


def _read_positioning_choice(
    input_fn: Callable[[str], str], output: Callable[[str], None]
) -> PositioningMode:
    choices = {
        "": PositioningMode.DIRECT,
        "rl定位": PositioningMode.RL,
        "rl": PositioningMode.RL,
        "l": PositioningMode.RL,
        "人工预定位": PositioningMode.MANUAL,
        "预定位": PositioningMode.MANUAL,
        "y": PositioningMode.MANUAL,
        "yes": PositioningMode.MANUAL,
        "直接采集": PositioningMode.DIRECT,
        "n": PositioningMode.DIRECT,
        "no": PositioningMode.DIRECT,
    }
    while True:
        value = input_fn(
            "选择采集前定位方式（RL定位/l、人工预定位/y、直接采集/n，默认 n）："
        ).strip().lower()
        choice = choices.get(value)
        if choice is not None:
            return choice
        output("无法识别选择，请输入：RL定位/l、人工预定位/y 或直接采集/n。")


def _wait_for_preposition_complete(
    input_fn: Callable[[str], str], output: Callable[[str], None]
) -> None:
    while True:
        value = input_fn(
            "预定位完成后，将双杆 X/Y/Z 全部回中并松开 deadman，再输入 完成/c："
        ).strip().lower()
        if value in {"完成", "c", "complete"}:
            return
        output("预定位仍在进行；完成后请输入：完成/c。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="collect and validate one guided diagnostic Episode"
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="guided PC workflow configuration",
    )
    args = parser.parse_args(argv)
    try:
        config = GuidedEpisodeConfig.load(args.config)
        operations = SystemGuidedEpisodeOperations(config)
        positioning_mode = _read_positioning_choice(input, print)
        path = run_guided_episode(
            config, operations, positioning_mode=positioning_mode
        )
    except KeyboardInterrupt:
        print("guided Episode aborted by operator", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    collector_log, teleop_log, validation_log = operations.log_paths
    print(f"Episode complete and validated: {path}")
    print(f"collector log: {collector_log}")
    print(f"teleop log: {teleop_log}")
    print(f"validation log: {validation_log}")
    return 0
