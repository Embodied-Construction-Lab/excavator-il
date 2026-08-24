"""Prepare one live DUMP trajectory while ACT is still finishing."""

from __future__ import annotations

import math
from pathlib import Path
import re
import shlex
import signal
import subprocess
from typing import Any, Callable

from .hybrid_mission_resident import PreparedDumpActivation
from .remote_runtime import LineProcess, LineWaitTimeout


_READY_MARKER = "prepared follow ready:"
_SAFE_FALLBACK_EXIT_CODE = 3
_SAFE_TIMESTAMP = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class SystemPreparedDumpAdapter:
    """Own one local AiryLidar prepared Plan→Follow child process."""

    def __init__(
        self,
        *,
        airy_repo: str | Path,
        ros_setup: str | Path,
        workspace_setup: str | Path,
        mission_config: str | Path,
        log_dir: str | Path,
        wait_s: int,
        ready_grace_ms: int,
        run_timeout_s: float,
        start_tolerance_m: float,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        timestamp: str,
    ) -> None:
        self._airy_repo = _absolute_path(airy_repo, "airy_repo")
        self._ros_setup = _absolute_path(ros_setup, "ros_setup")
        self._workspace_setup = _absolute_path(
            workspace_setup, "workspace_setup"
        )
        self._mission_config = _absolute_path(
            mission_config, "mission_config"
        )
        self._log_dir = _absolute_path(log_dir, "log_dir")
        self._wait_s = _bounded_int(wait_s, "wait_s", 1, 600)
        ready_ms = _bounded_int(
            ready_grace_ms, "ready_grace_ms", 10, 1000
        )
        self._ready_grace_s = ready_ms / 1000.0
        self._run_timeout_s = _positive_number(
            run_timeout_s, "run_timeout_s"
        )
        self._start_tolerance_m = _bounded_number(
            start_tolerance_m, "start_tolerance_m", 0.01, 0.5
        )
        if not callable(line_process_factory):
            raise ValueError("line_process_factory must be callable")
        if not callable(output):
            raise ValueError("output must be callable")
        if (
            not isinstance(timestamp, str)
            or _SAFE_TIMESTAMP.fullmatch(timestamp) is None
        ):
            raise ValueError("timestamp must be safe non-empty text")
        self._line_process_factory = line_process_factory
        self._output = output
        self._gate_path = (
            self._log_dir
            / f"hybrid_mission_{timestamp}.prepared-dump.start"
        )
        self._plan_gate_path = (
            self._log_dir
            / f"hybrid_mission_{timestamp}.prepared-dump.plan"
        )
        self._log_path = (
            self._log_dir
            / f"hybrid_mission_{timestamp}.prepared-dump.log"
        )
        self._process: Any | None = None
        self._plan_requested = False

    def start_prepare(self) -> None:
        """Spawn planning and return without waiting for the ready marker."""

        if self._process is not None:
            raise RuntimeError("prepared dump process is already owned")
        missing = [
            str(path)
            for path in (
                self._airy_repo,
                self._ros_setup,
                self._workspace_setup,
                self._mission_config,
            )
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "prepared dump path does not exist: " + ", ".join(missing)
            )
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._unlink_gates()
        self._plan_requested = False
        command = self._shell_command()
        self._process = self._line_process_factory(
            ["/bin/zsh", "-lc", command],
            log_path=self._log_path,
            prefix="prepared-dump",
            output=self._output,
        )

    def trigger_prepare(self) -> None:
        """Freeze fresh live inputs and request planning from the warm child."""

        process = self._process
        if process is None:
            raise RuntimeError("prepared dump process has not been started")
        if process.returncode is not None:
            raise RuntimeError("prepared dump process exited before planning")
        if self._plan_requested:
            raise RuntimeError("prepared dump planning was already requested")
        self._plan_gate_path.touch(exist_ok=False)
        self._plan_requested = True

    def activate_prepared(self) -> PreparedDumpActivation:
        process = self._process
        if process is None:
            raise RuntimeError("prepared dump process has not been started")
        if not self._plan_requested:
            raise RuntimeError("prepared dump planning has not been requested")
        try:
            try:
                process.wait_for(
                    lambda line: _READY_MARKER in line,
                    self._ready_grace_s,
                )
            except LineWaitTimeout:
                self._cancel_current()
                return PreparedDumpActivation.FALLBACK_SAFE
            except RuntimeError as exc:
                return self._finished_outcome(
                    process,
                    context="before the readiness marker",
                    failure=exc,
                )

            self._gate_path.touch(exist_ok=False)
            try:
                process.wait(timeout_s=self._run_timeout_s)
            except subprocess.TimeoutExpired as exc:
                self._cancel_current()
                raise RuntimeError("prepared dump Follow timed out") from exc
            return self._finished_outcome(
                process,
                context="after the start gate",
            )
        finally:
            self._unlink_gates()

    def cancel(self) -> None:
        self._cancel_current()

    def _finished_outcome(
        self,
        process: Any,
        *,
        context: str,
        failure: BaseException | None = None,
    ) -> PreparedDumpActivation:
        returncode = process.returncode
        if returncode == 0:
            self._process = None
            return PreparedDumpActivation.ACTIVATED
        if returncode == _SAFE_FALLBACK_EXIT_CODE:
            self._process = None
            return PreparedDumpActivation.FALLBACK_SAFE
        self._cancel_current()
        raise RuntimeError(
            "prepared dump process failed "
            f"{context} with return code {returncode}"
        ) from failure

    def _cancel_current(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None and process.returncode is None:
                process.stop(signal.SIGINT, timeout_s=3.0)
        finally:
            self._plan_requested = False
            self._unlink_gates()

    def _unlink_gates(self) -> None:
        self._plan_gate_path.unlink(missing_ok=True)
        self._gate_path.unlink(missing_ok=True)

    def _shell_command(self) -> str:
        command = shlex.join(
            [
                "exec",
                "/usr/bin/python3",
                "-m",
                "mission.runtime_ros.run_prepared_plan_follow_live",
                "dump",
                "--mission",
                str(self._mission_config),
                "--wait-s",
                str(self._wait_s),
                "--start-gate",
                str(self._gate_path),
                "--plan-gate",
                str(self._plan_gate_path),
                "--first-waypoint-distance-m",
                f"{self._start_tolerance_m:g}",
            ]
        )
        return " && ".join(
            (
                f"source {shlex.quote(str(self._ros_setup))}",
                f"source {shlex.quote(str(self._workspace_setup))}",
                f"cd {shlex.quote(str(self._airy_repo))}",
                command,
            )
        )


def _absolute_path(value: str | Path, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or "\x00" in str(path):
        raise ValueError(f"{field} must be an absolute NUL-free path")
    return path


def _bounded_int(value: Any, field: str, low: int, high: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")
    return value


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _bounded_number(
    value: Any,
    field: str,
    low: float,
    high: float,
) -> float:
    number = _positive_number(value, field)
    if not low <= number <= high:
        raise ValueError(f"{field} must be in [{low:g}, {high:g}]")
    return number
