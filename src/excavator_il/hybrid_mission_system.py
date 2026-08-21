"""System Adapter for staged RL/ACT Mission execution across PC and Orin."""

from __future__ import annotations

import hashlib
import re
import shlex
import signal
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

from .act_runtime_contract import REQUIRED_MOTION_AUTHORIZATION
from .guided_episode import (
    GuidedEpisodeConfig,
    SystemGuidedEpisodeOperations,
)
from .hybrid_mission import HybridMissionConfig
from .remote_runtime import LineProcess, SshRuntimeHost


class HybridRlOperations(Protocol):
    """Narrow RL-side Interface required by the hybrid Mission Adapter."""

    def prewarm_rl_runtime(self, hardware_start_gate: str | PurePosixPath) -> None: ...

    def start_rl_runtime(self) -> None: ...

    def run_rl_follow(self, phase: str, *, target_id: str | None = None) -> Any: ...

    def run_rl_fixed_action(self, behavior: str, *, behavior_port: int) -> None: ...

    def stop_rl_runtime_and_wait_for_serial(self) -> None: ...


class SystemHybridMissionOperations:
    """Overlap hardware-free startup while preserving one physical command owner."""

    def __init__(
        self,
        config: HybridMissionConfig,
        *,
        guided_config: GuidedEpisodeConfig | None = None,
        rl_operations: HybridRlOperations | None = None,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        timestamp: str | None = None,
    ) -> None:
        self._config = config
        self._guided = guided_config or GuidedEpisodeConfig.load(config.guided_config)
        self._remote_host = SshRuntimeHost(
            self._guided.orin_ssh_host, run_command=subprocess.run
        )
        self._output = output
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rl = rl_operations or SystemGuidedEpisodeOperations(
            self._guided,
            output=output,
            timestamp=f"hybrid_{self._timestamp}",
        )
        self._line_process_factory = line_process_factory
        self._act_process: Any | None = None
        self._act_remote_pid: int | None = None
        gate_token = hashlib.sha256(self._timestamp.encode("utf-8")).hexdigest()[:16]
        self._act_gate_name = f"hybrid_{gate_token}.start"
        self._act_gate_path = PurePosixPath(
            "/home/jetson16/workspace_excavator/act_inference/control"
        ) / self._act_gate_name
        self._rl_gate_path = PurePosixPath(
            "/tmp/excavator-rl-control"
        ) / self._act_gate_name
        self._rl_runtime_active = False
        self._rl_runtime_prepared = False

    def run_rl_to_dig(self, target_id: str) -> None:
        self._reclaim_stale_rl_prewarm()
        self._start_rl_runtime()
        try:
            self._start_act_prewarm(self._config.act_max_steps)
            self._rl.run_rl_follow("dig", target_id=target_id)
        except BaseException:
            self._stop_rl_runtime()
            self._stop_act_and_wait_for_serial()
            raise
        else:
            self._stop_rl_runtime()

    def run_rl_to_dump_and_dump(self) -> None:
        self._start_rl_runtime()
        try:
            self._rl.run_rl_follow("dump")
            self._rl.run_rl_fixed_action(
                "ExecuteDump",
                behavior_port=self._config.rl_behavior_port,
            )
        except BaseException:
            self._stop_rl_runtime()
            raise
        self._output(
            "RL Runtime 保持热启动并处于零动作待命；下一段返回将直接复用。"
        )

    def run_rl_return_to_dig(self, target_id: str) -> None:
        self._start_rl_runtime()
        try:
            self._rl.run_rl_follow("dig", target_id=target_id)
        finally:
            self._stop_rl_runtime()

    def prewarm_next_act(self, max_steps: int) -> None:
        """Load the next ACT policy while RL still owns the physical interface."""

        self._start_act_prewarm(max_steps)

    def _start_rl_runtime(self) -> None:
        if self._rl_runtime_active:
            return
        self._rl.start_rl_runtime()
        self._rl_runtime_prepared = False
        self._rl_runtime_active = True

    def _start_rl_prewarm(self) -> None:
        if self._rl_runtime_active or self._rl_runtime_prepared:
            return
        try:
            self._reclaim_stale_rl_prewarm()
            self._rl.prewarm_rl_runtime(self._rl_gate_path)
        except Exception as exc:
            self._rl_runtime_prepared = False
            self._output(
                "RL 预热失败，将在 ACT 完成后回退到安全冷启动："
                f"{type(exc).__name__}: {exc}"
            )
            return
        self._rl_runtime_prepared = True

    def _stop_rl_runtime(self) -> None:
        if not self._rl_runtime_active and not self._rl_runtime_prepared:
            return
        try:
            self._rl.stop_rl_runtime_and_wait_for_serial()
        finally:
            self._rl_runtime_active = False
            self._rl_runtime_prepared = False

    def _start_act_prewarm(self, max_steps: int) -> None:
        if self._act_process is not None:
            raise RuntimeError("an ACT prewarm is already active")
        self._reclaim_stale_act_prewarm()
        remote_command = self._act_remote_command(
            max_steps=max_steps,
            hardware_start_gate=self._act_gate_name,
        )
        log_path = (
            Path(self._guided.log_dir)
            / f"hybrid_mission_{self._timestamp}.act.log"
        )
        process = self._line_process_factory(
            self._ssh_argv(remote_command),
            log_path=log_path,
            prefix="act-prewarm",
            output=self._output,
        )
        self._act_process = process
        try:
            _, pid_line = process.wait_for(
                lambda line: line.startswith("HYBRID_ACT_PID="),
                self._config.act_ready_timeout_s,
            )
            self._act_remote_pid = int(pid_line.split("=", maxsplit=1)[1])
            process.wait_for(
                lambda line: "ACT 预热等待模式" in line,
                self._config.act_ready_timeout_s,
            )
        except BaseException:
            self._stop_act_and_wait_for_serial(require_serial_release=False)
            raise

    def _reclaim_stale_act_prewarm(self) -> None:
        result = self._remote_host.reclaim_hardware_gated_runtime(
            process_marker="act-runtime",
            gate_prefix="/opt/act-control/hybrid_",
            protected_devices=(str(self._guided.rl_serial_port), "/dev/video0"),
            timeout_s=self._guided.rl_serial_release_timeout_s,
            execute=self._run_remote,
        )
        if result == "reclaimed":
            self._output("已回收上一轮遗留的 ACT 预热进程。")

    def _reclaim_stale_rl_prewarm(self) -> None:
        result = self._remote_host.reclaim_hardware_gated_runtime(
            process_marker="orin_state_sender.py",
            gate_prefix="/tmp/excavator-rl-control/hybrid_",
            protected_devices=(str(self._guided.rl_serial_port),),
            timeout_s=self._guided.rl_serial_release_timeout_s,
            execute=self._run_remote,
        )
        if result == "reclaimed":
            self._output("已回收上一轮遗留的 RL 预热进程。")

    def _act_remote_command(
        self, *, max_steps: int, hardware_start_gate: str | None
    ) -> str:
        command = (
            f"cd {shlex.quote(str(self._guided.orin_repo))} && "
            "echo HYBRID_ACT_PID=$$ && exec "
            f"bash {shlex.quote(self._config.act_remote_script)} "
            f"--authorization {shlex.quote(REQUIRED_MOTION_AUTHORIZATION)} "
            f"--max-steps {max_steps}"
        )
        if hardware_start_gate is not None:
            command += (
                " --hardware-start-gate "
                f"{shlex.quote(hardware_start_gate)}"
            )
        return command

    def run_act_dig(self, max_steps: int) -> None:
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer")
        if self._act_process is not None:
            process = self._act_process
            prewarmed = True
            if max_steps != self._config.act_max_steps:
                raise ValueError(
                    "prewarmed ACT step budget does not match Mission config"
                )
        else:
            process = None
            prewarmed = False
        remote_command = self._act_remote_command(
            max_steps=max_steps,
            hardware_start_gate=None,
        )
        log_path = (
            Path(self._guided.log_dir)
            / f"hybrid_mission_{self._timestamp}.act.log"
        )
        if process is None:
            process = self._line_process_factory(
                self._ssh_argv(remote_command),
                log_path=log_path,
                prefix="act-dig",
                output=self._output,
            )
            self._act_process = process
        try:
            if not prewarmed:
                _, pid_line = process.wait_for(
                    lambda line: line.startswith("HYBRID_ACT_PID="),
                    self._config.act_ready_timeout_s,
                )
                self._act_remote_pid = int(pid_line.split("=", maxsplit=1)[1])
            else:
                process.wait_for(
                    lambda line: "ACT prewarm ready:" in line,
                    self._config.act_ready_timeout_s,
                )
                self._confirm_act_serial_release()
                self._run_remote(
                    "touch -- " + shlex.quote(str(self._act_gate_path))
                )
            process.wait_for(
                lambda line: "ACT hardware ready: mode=motion" in line,
                self._config.act_ready_timeout_s,
            )
            self._start_rl_prewarm()
            process.wait(timeout_s=self._config.act_run_timeout_s)
            if process.returncode != 0:
                raise RuntimeError(
                    f"bounded ACT dig exited with code {process.returncode}; see {log_path}"
                )
            self._confirm_act_serial_release()
        except BaseException:
            self._stop_act_and_wait_for_serial()
            raise
        else:
            self._act_process = None
            self._act_remote_pid = None

    def safe_stop(self) -> None:
        errors: list[str] = []
        try:
            self._stop_act_and_wait_for_serial(
                require_serial_release=not self._rl_runtime_active
            )
        except Exception as exc:
            errors.append(f"ACT stop: {exc}")
        try:
            self._stop_rl_runtime()
        except Exception as exc:
            errors.append(f"RL stop: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def _ssh_argv(self, command: str) -> list[str]:
        return self._remote_host.argv(command)

    def _run_remote(self, command: str) -> str:
        return self._remote_host.run(command)

    def _confirm_act_serial_release(self) -> None:
        serial = shlex.quote(str(self._guided.rl_serial_port))
        output = self._run_remote(
            "/bin/sh -c "
            + shlex.quote(
                f"set -eu; command -v fuser >/dev/null; "
                f"if fuser -s {serial}; then echo 'serial still owned' >&2; exit 14; fi; "
                "echo released"
            )
        )
        if output.strip() != "released":
            raise RuntimeError("ACT exit did not confirm serial release")

    def _stop_act_and_wait_for_serial(
        self, *, require_serial_release: bool = True
    ) -> None:
        process = self._act_process
        remote_pid = self._act_remote_pid
        if process is None:
            return
        if remote_pid is None:
            process.stop(signal.SIGTERM, timeout_s=2.0)
            self._act_process = None
            raise RuntimeError("ACT Runtime PID was not observed; serial release is unknown")
        act_identity_pattern = (
            "(act-runtime|run_act_motion\\.sh.*--hardware-start-gate[ =]+"
            + re.escape(self._act_gate_name)
            + ")"
        )
        try:
            self._remote_host.stop_owned_process(
                pid=remote_pid,
                identity_ere=act_identity_pattern,
                serial_path=self._guided.rl_serial_port,
                timeout_s=self._guided.rl_serial_release_timeout_s,
                require_serial_release=require_serial_release,
                cleanup_paths=(self._act_gate_path,),
                execute=self._run_remote,
            )
            try:
                process.wait(timeout_s=2.0)
            except Exception:
                process.stop(signal.SIGTERM, timeout_s=2.0)
        finally:
            self._act_process = None
            self._act_remote_pid = None
