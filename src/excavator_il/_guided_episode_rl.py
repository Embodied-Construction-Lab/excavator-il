"""RL positioning and fixed-behavior orchestration for guided collection."""

from __future__ import annotations

import json
import math
import shlex
import signal
import subprocess
from pathlib import PurePosixPath
from typing import Any, Mapping

from ._guided_episode_targets import load_rl_dig_targets, resolve_rl_dig_target


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


class _GuidedEpisodeRlOperations:
    """Private RL process lifecycle mixed into the system operations Adapter."""

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
