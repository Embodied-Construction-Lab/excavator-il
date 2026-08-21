"""Lifecycle Adapter for the existing AiryLidar Operator/RViz launch."""

from __future__ import annotations

import shlex
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .guided_episode import GuidedEpisodeConfig
from .remote_runtime import LineProcess


@dataclass(frozen=True)
class AiryOperatorSnapshot:
    stage: str = "stopped"
    error: str = ""
    logs: tuple[str, ...] = ()


class AiryOperatorSupervisor:
    """Start the authoritative ROS2 launch without reimplementing its behavior."""

    def __init__(
        self,
        *,
        guided_config: GuidedEpisodeConfig,
        behavior_port: int,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        ready_timeout_s: int = 60,
    ) -> None:
        if not 1 <= behavior_port <= 65535:
            raise ValueError("behavior_port must be within [1, 65535]")
        if ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive")
        self._config = guided_config
        self._behavior_port = behavior_port
        self._factory = line_process_factory
        self._output = output
        self._ready_timeout_s = ready_timeout_s
        self._process: Any | None = None
        self._state = AiryOperatorSnapshot()

    def snapshot(self) -> AiryOperatorSnapshot:
        process = self._process
        if process is not None and self._state.stage == "ready" and not process.running:
            logs = tuple(process.lines)
            process.stop(signal.SIGINT, timeout_s=10.0)
            self._process = None
            self._state = AiryOperatorSnapshot(
                stage="failed",
                error="AiryLidar Operator 已意外退出",
                logs=logs,
            )
            process = None
        logs = tuple(process.lines) if process is not None else self._state.logs
        return AiryOperatorSnapshot(
            stage=self._state.stage,
            error=self._state.error,
            logs=logs,
        )

    def start(self) -> AiryOperatorSnapshot:
        stale_process = self._process
        if stale_process is not None:
            if stale_process.running:
                raise RuntimeError("AiryLidar Operator is already active")
            stale_process.stop(signal.SIGINT, timeout_s=10.0)
            self._process = None
        _user, orin_host = self._config.orin_ssh_host.split("@", maxsplit=1)
        launch = shlex.join(
            [
                "exec",
                "ros2",
                "launch",
                "airy_excavator_bringup",
                "operator.launch.py",
                "profile:=live_commissioning",
                "motion_authorization:=ALLOW_LIVE_MACHINE_MOTION",
                f"orin_host:={orin_host}",
                f"orin_port:={self._behavior_port}",
            ]
        )
        shell_command = " && ".join(
            [
                f"source {shlex.quote(str(self._config.rl_ros_setup))}",
                f"source {shlex.quote(str(self._config.rl_workspace_setup))}",
                f"cd {shlex.quote(str(self._config.rl_airy_repo))}",
                launch,
            ]
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(self._config.log_dir) / f"webui_airy_operator_{stamp}.log"
        self._state = AiryOperatorSnapshot(stage="starting")
        process = self._factory(
            ["/bin/zsh", "-lc", shell_command],
            log_path=log_path,
            prefix="airy-operator",
            output=self._output,
        )
        self._process = process
        try:
            process.wait_for(
                lambda line: "live Plan ready:" in line,
                self._ready_timeout_s,
            )
        except BaseException as exc:
            process.stop(signal.SIGINT, timeout_s=10.0)
            self._process = None
            self._state = AiryOperatorSnapshot(
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
                logs=tuple(process.lines),
            )
            raise
        self._state = AiryOperatorSnapshot(stage="ready")
        return self.snapshot()

    def stop(self) -> AiryOperatorSnapshot:
        process = self._process
        if process is None:
            self._process = None
            self._state = AiryOperatorSnapshot(stage="stopped")
            return self._state
        process.stop(signal.SIGINT, timeout_s=10.0)
        self._state = AiryOperatorSnapshot(
            stage="stopped", logs=tuple(process.lines)
        )
        self._process = None
        return self._state

    def close(self) -> None:
        self.stop()
