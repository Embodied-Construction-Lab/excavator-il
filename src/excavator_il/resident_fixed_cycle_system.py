"""PC display/start/cancel adapter for the Orin-local V3-A fixed cycle."""

from __future__ import annotations

import json
import re
import shlex
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .guided_episode import GuidedEpisodeConfig
from .hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from .hybrid_mission_session import (
    MAX_HYBRID_CYCLE_COUNT,
    HybridMissionSnapshot,
)
from .hybrid_experiment_run import (
    HybridMissionEvidenceLifecycle,
    HybridMissionRunRequest,
)
from .remote_runtime import LineProcess, SshRuntimeHost

CONFIG_SCHEMA_VERSION = "excavator_resident_fixed_cycle_pc.v2"
COMMISSIONING_AUTHORIZATION = "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING"
CONTROL_SCHEMA_VERSION = "resident_fixed_cycle_control.v1"
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "guided_config",
        "fixed_cycle_plan",
        "runtime_root",
        "owner_script",
        "act_worker_script",
        "control_socket",
        "ready_timeout_s",
        "status_poll_ms",
        "act_max_steps",
        "commissioning_authorization",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "run_id",
        "stage",
        "requested_cycles",
        "completed_cycles",
        "current_dig_point_id",
        "terminal",
        "outcome",
        "reason_code",
    }
)
_TERMINAL_UI_STAGES = frozenset({"completed", "failed", "cancelled"})
_STAGE_TO_UI = {
    "IDLE": "idle",
    "FOLLOW_DIG": "running_rl_to_dig",
    "ACT_DIG": "running_act_dig",
    "FOLLOW_DUMP": "running_rl_to_dump_and_dump",
    "EXECUTE_DUMP": "running_rl_to_dump_and_dump",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
@dataclass(frozen=True)
class ResidentFixedCyclePcConfig:
    guided_config: Path
    fixed_cycle_plan: PurePosixPath
    runtime_root: PurePosixPath
    owner_script: str
    act_worker_script: str
    control_socket: PurePosixPath
    ready_timeout_s: float
    status_poll_s: float
    act_max_steps: int
    commissioning_authorization: str

    @classmethod
    def load(cls, path: str | Path) -> "ResidentFixedCyclePcConfig":
        config_path = Path(path).expanduser().resolve()
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load V3-A PC config: {exc}") from exc
        if not isinstance(value, Mapping) or set(value) != _CONFIG_FIELDS:
            raise ValueError("V3-A PC config fields are invalid")
        if value["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported V3-A PC config schema")
        plan = _absolute_posix(value["fixed_cycle_plan"], "fixed_cycle_plan")
        root = _absolute_posix(value["runtime_root"], "runtime_root")
        socket_path = _absolute_posix(value["control_socket"], "control_socket")
        if root not in socket_path.parents:
            raise ValueError("control_socket must be inside runtime_root")
        ready_timeout_s = _bounded_number(
            value["ready_timeout_s"], "ready_timeout_s", 1.0, 300.0
        )
        poll_ms = _bounded_number(
            value["status_poll_ms"], "status_poll_ms", 20.0, 1000.0
        )
        return cls(
            guided_config=(
                config_path.parent
                / _text(value["guided_config"], "guided_config")
            ).resolve(),
            fixed_cycle_plan=plan,
            runtime_root=root,
            owner_script=_relative_script(value["owner_script"], "owner_script"),
            act_worker_script=_relative_script(
                value["act_worker_script"], "act_worker_script"
            ),
            control_socket=socket_path,
            ready_timeout_s=ready_timeout_s,
            status_poll_s=poll_ms / 1000.0,
            act_max_steps=_bounded_integer(
                value["act_max_steps"], "act_max_steps", 1, 2000
            ),
            commissioning_authorization=_commissioning_authorization(
                value["commissioning_authorization"]
            ),
        )
@dataclass(frozen=True)
class ResidentFixedCycleRemoteStatus:
    run_id: str
    stage: str
    requested_cycles: int
    completed_cycles: int
    current_dig_point_id: str
    terminal: bool
    outcome: str
    reason_code: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ResidentFixedCycleRemoteStatus":
        if not isinstance(value, Mapping) or set(value) != _STATUS_FIELDS:
            raise ValueError("V3-A status fields are invalid")
        stage = _text(value["stage"], "status.stage")
        if stage not in _STAGE_TO_UI:
            raise ValueError("V3-A status stage is invalid")
        requested = _bounded_integer(
            value["requested_cycles"], "status.requested_cycles", 0, 9
        )
        completed = _bounded_integer(
            value["completed_cycles"], "status.completed_cycles", 0, 9
        )
        if completed > requested:
            raise ValueError("completed_cycles cannot exceed requested_cycles")
        terminal = value["terminal"]
        if not isinstance(terminal, bool):
            raise ValueError("status.terminal must be boolean")
        if terminal != (stage in {"COMPLETED", "FAILED", "CANCELLED"}):
            raise ValueError("V3-A terminal flag and stage disagree")
        return cls(
            run_id=_optional_identifier(value["run_id"], "status.run_id"),
            stage=stage,
            requested_cycles=requested,
            completed_cycles=completed,
            current_dig_point_id=_optional_identifier(
                value["current_dig_point_id"], "status.current_dig_point_id"
            ),
            terminal=terminal,
            outcome=_text(value["outcome"], "status.outcome", allow_empty=True),
            reason_code=_text(
                value["reason_code"], "status.reason_code", allow_empty=True
            ),
        )

class ResidentFixedCycleProcesses:
    """Own the V3-A resident owner and ACT worker remote processes."""

    def __init__(
        self,
        config: ResidentFixedCyclePcConfig,
        *,
        guided_config: GuidedEpisodeConfig,
        remote_host: Any | None = None,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        timestamp: str | None = None,
    ) -> None:
        self._config = config
        self._guided = guided_config
        self._remote_host = remote_host or SshRuntimeHost(guided_config.orin_ssh_host)
        self._line_process_factory = line_process_factory
        self._output = output
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._owner_process: Any | None = None
        self._owner_pid: int | None = None
        self._act_process: Any | None = None
        self._act_pid: int | None = None

    def start(self) -> None:
        if self._owner_process is not None or self._act_process is not None:
            self.require_running()
            return
        try:
            self._start_owner()
            self._start_act_worker()
        except BaseException:
            try:
                self.stop()
            except Exception as exc:
                self._output(f"V3-A startup cleanup failed: {exc}")
            raise

    def require_running(self) -> None:
        for name, process in (
            ("resident owner", self._owner_process),
            ("resident ACT worker", self._act_process),
        ):
            if process is None:
                raise RuntimeError(f"{name} is not started")
            if process.returncode is not None:
                raise RuntimeError(
                    f"{name} exited with return code {process.returncode}"
                )

    def stop(self, *, terminal_disarmed: bool = False) -> None:
        errors: list[str] = []
        for operation in (self._stop_owner, self._stop_act_worker):
            try:
                operation(allow_nonzero=terminal_disarmed)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def _start_owner(self) -> None:
        argv = [
            "env",
            f"RESIDENT_RUNTIME_ROOT={self._config.runtime_root}",
            f"RESIDENT_PYTHON={self._guided.rl_orin_python}",
            "bash",
            self._config.owner_script,
            "--authorization",
            REQUIRED_HYBRID_MOTION_AUTHORIZATION,
            "--pc-host",
            self._guided.rl_pc_host,
            "--serial-port",
            str(self._guided.rl_serial_port),
            "--fixed-cycle-plan",
            str(self._config.fixed_cycle_plan),
        ]
        if self._config.commissioning_authorization:
            argv.extend(
                [
                    "--commissioning-authorization",
                    self._config.commissioning_authorization,
                ]
            )
        command = (
            f"cd {shlex.quote(str(self._guided.rl_orin_repo))} && "
            "echo RESIDENT_OWNER_PID=$$ && exec " + shlex.join(argv)
        )
        process = self._spawn(command, "v3a-owner")
        self._owner_process = process
        _, line = process.wait_for(
            lambda item: item.startswith("RESIDENT_OWNER_PID="),
            self._config.ready_timeout_s,
        )
        self._owner_pid = _remote_pid(line, "RESIDENT_OWNER_PID")
        _, ready_line = process.wait_for(
            lambda item: "RESIDENT_FIXED_CYCLE_READY " in item,
            self._config.ready_timeout_s,
        )
        expected = (
            f"control_socket={self._config.control_socket} "
            f"act_socket={self._config.runtime_root / 'act.sock'}"
        )
        if expected not in ready_line:
            raise RuntimeError("V3-A owner announced an unexpected control socket")
        process.wait_for(
            lambda item: "RESIDENT_HARDWARE_READY sensor_valid=True" in item,
            self._config.ready_timeout_s,
        )
    def _start_act_worker(self) -> None:
        argv = [
            "env",
            f"RESIDENT_RUNTIME_ROOT={self._config.runtime_root}",
            "bash",
            self._config.act_worker_script,
            "--authorization",
            REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        ]
        command = (
            f"cd {shlex.quote(str(self._guided.orin_repo))} && "
            "echo RESIDENT_ACT_PID=$$ && exec " + shlex.join(argv)
        )
        process = self._spawn(command, "v3a-act")
        self._act_process = process
        _, line = process.wait_for(
            lambda item: item.startswith("RESIDENT_ACT_PID="),
            self._config.ready_timeout_s,
        )
        self._act_pid = _remote_pid(line, "RESIDENT_ACT_PID")
        process.wait_for(
            lambda item: "ACT resident worker ready:" in item,
            self._config.ready_timeout_s,
        )
    def _spawn(self, command: str, prefix: str) -> Any:
        log_path = Path(self._guided.log_dir) / (
            f"resident_fixed_cycle_{self._timestamp}.{prefix}.log"
        )
        return self._line_process_factory(
            self._remote_host.argv(command),
            log_path=log_path,
            prefix=prefix,
            output=self._output,
        )

    def _stop_owner(self, *, allow_nonzero: bool) -> None:
        process, pid = self._owner_process, self._owner_pid
        if process is None:
            return
        try:
            if pid is not None and process.returncode is None:
                self._remote_host.stop_owned_process(
                    pid=pid,
                    identity_ere=r"[o]rin_state_sender\.py.*--resident-fixed-cycle-plan",
                    serial_path=self._guided.rl_serial_port,
                    timeout_s=self._guided.rl_serial_release_timeout_s,
                    require_serial_release=True,
                    cleanup_paths=(
                        self._config.control_socket,
                        self._config.runtime_root / "act.sock",
                    ),
                )
            _wait_process(process, allow_nonzero=allow_nonzero)
        finally:
            self._owner_process = None
            self._owner_pid = None

    def _stop_act_worker(self, *, allow_nonzero: bool) -> None:
        process, pid = self._act_process, self._act_pid
        if process is None:
            return
        try:
            if pid is not None and process.returncode is None:
                self._remote_host.stop_owned_process(
                    pid=pid,
                    identity_ere=r"([d]ocker.*resident_act_runtime|[r]un_act_resident\.sh)",
                    serial_path=PurePosixPath("/dev/video0"),
                    timeout_s=self._guided.rl_serial_release_timeout_s,
                    require_serial_release=True,
                )
            _wait_process(process, allow_nonzero=allow_nonzero)
        finally:
            self._act_process = None
            self._act_pid = None


class SshResidentFixedCycleOperations:
    """Translate three PC intentions into the strict Orin-local control API."""

    def __init__(
        self,
        config: ResidentFixedCyclePcConfig,
        *,
        guided_config: GuidedEpisodeConfig,
        processes: Any | None = None,
        remote_host: Any | None = None,
        output: Callable[[str], None] = print,
    ) -> None:
        self._config = config
        self._guided = guided_config
        self._remote_host = remote_host or SshRuntimeHost(guided_config.orin_ssh_host)
        self._processes = processes or ResidentFixedCycleProcesses(
            config,
            guided_config=guided_config,
            remote_host=self._remote_host,
            output=output,
        )

    def start(
        self,
        *,
        run_id: str,
        requested_cycles: int,
        first_dig_point_id: str,
    ) -> ResidentFixedCycleRemoteStatus:
        self._processes.start()
        return self._request(
            "start",
            "--run-id",
            run_id,
            "--cycles",
            str(requested_cycles),
            "--first-dig-point-id",
            first_dig_point_id,
        )

    def status(self) -> ResidentFixedCycleRemoteStatus:
        return self._request("heartbeat")

    def cancel(self) -> ResidentFixedCycleRemoteStatus:
        return self._request("cancel")

    def release(self, *, terminal_disarmed: bool) -> None:
        self._processes.stop(terminal_disarmed=terminal_disarmed)

    def _request(self, command: str, *arguments: str) -> ResidentFixedCycleRemoteStatus:
        argv = [
            str(self._guided.rl_orin_python),
            "-m",
            "edge_runtime.resident_fixed_cycle_control",
            "--socket",
            str(self._config.control_socket),
            *arguments,
            command,
        ]
        remote = (
            f"cd {shlex.quote(str(self._guided.rl_orin_repo))} && "
            + shlex.join(argv)
        )
        output = self._remote_host.run(remote)
        return _parse_control_response(output, command)


class ResidentFixedCycleSupervisor:
    """Expose one Orin-local fixed cycle through the existing WebUI Interface."""

    def __init__(
        self,
        *,
        operations: Any,
        dig_target_ids: tuple[str, ...],
        poll_interval_s: float,
        log_capacity: int = 400,
        config_path: str | Path | None = None,
        evidence_run_factory: Callable[[HybridMissionRunRequest], Any] | None = None,
    ) -> None:
        if not dig_target_ids or len(set(dig_target_ids)) != len(dig_target_ids):
            raise ValueError("dig_target_ids must be a non-empty unique tuple")
        if not 0.02 <= poll_interval_s <= 1.0:
            raise ValueError("poll_interval_s must be within [0.02, 1.0]")
        self._operations = operations
        self._dig_target_ids = dig_target_ids
        self._poll_interval_s = poll_interval_s
        self._lock = threading.RLock()
        self._logs: deque[str] = deque(maxlen=log_capacity)
        self._state = HybridMissionSnapshot()
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._config_path = (
            None if config_path is None else Path(config_path).expanduser().resolve()
        )
        if evidence_run_factory is not None and self._config_path is None:
            raise ValueError("config_path is required with an evidence factory")
        self._evidence_run_factory = evidence_run_factory
        self._evidence = HybridMissionEvidenceLifecycle(None)

    def snapshot(self) -> HybridMissionSnapshot:
        with self._lock:
            thread = self._thread
            return replace(
                self._state,
                logs=tuple(self._logs),
                can_stop=bool(thread is not None and thread.is_alive()),
            )

    def append_external_log(self, message: str) -> None:
        """Forward resident owner/ACT output into the existing WebUI log."""

        self._append_log(_text(message, "resident log"))

    def start(
        self,
        dig_target_id: str,
        *,
        automatic: bool,
        motion_authorization: str | None,
        cycle_count: int = 1,
    ) -> None:
        if automatic is not True:
            raise ValueError("V3-A supports automatic local cycles only")
        if motion_authorization != REQUIRED_HYBRID_MOTION_AUTHORIZATION:
            raise ValueError("V3-A requires exact motion authorization")
        if dig_target_id not in self._dig_target_ids:
            raise ValueError("unknown V3-A dig target")
        _bounded_integer(cycle_count, "cycle_count", 1, MAX_HYBRID_CYCLE_COUNT)
        with self._lock:
            if self._evidence.finalization_pending:
                raise RuntimeError(
                    "previous V3-A evidence finalization is pending"
                )
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a V3-A Mission is already active")
            evidence_run = None
            if self._evidence_run_factory is not None:
                evidence_run = self._evidence_run_factory(
                    HybridMissionRunRequest(
                        config_path=self._config_path,
                        dig_target_id=dig_target_id,
                        automatic=True,
                        requested_cycles=cycle_count,
                    )
                )
                run_id = getattr(evidence_run, "run_id", None)
                if not isinstance(run_id, str) or not run_id:
                    raise ValueError("V3-A evidence run must expose a run_id")
            else:
                run_id = "v3a-" + uuid.uuid4().hex
            first_index = self._dig_target_ids.index(dig_target_id)
            cycle_targets = tuple(
                self._dig_target_ids[
                    (first_index + offset) % len(self._dig_target_ids)
                ]
                for offset in range(cycle_count)
            )
            self._evidence = HybridMissionEvidenceLifecycle(
                evidence_run,
                cycle_targets=cycle_targets,
            )
            self._logs.clear()
            self._stop_requested = False
            self._state = HybridMissionSnapshot(
                stage="starting",
                dig_target_id=dig_target_id,
                automatic=True,
                requested_cycles=cycle_count,
                run_id=run_id,
            )
            self._evidence.start_mission(
                automatic=True,
                requested_cycles=cycle_count,
                dig_target_id=dig_target_id,
            )
            if self._evidence.error:
                message = (
                    "initial V3-A evidence write failed: " + self._evidence.error
                )
                self._state = replace(
                    self._state,
                    stage="failed",
                    error=message,
                    evidence_error=self._evidence.error,
                )
                self._finish_evidence(stage="failed")
                raise RuntimeError(message)
            thread = threading.Thread(
                target=self._run,
                args=(run_id, cycle_count, dig_target_id),
                name=f"resident-fixed-cycle-{run_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def advance(self, *, motion_authorization: str | None) -> None:
        raise RuntimeError("V3-A does not support segmented PC advance")

    def retry_evidence_finalization(self) -> None:
        with self._lock:
            if not self._evidence.finalization_pending:
                raise RuntimeError("no V3-A evidence finalization is pending")
            self._evidence.retry_finalize()
            self._state = replace(
                self._state,
                evidence_error=self._evidence.error,
            )
            if self._evidence.finalization_pending:
                raise RuntimeError(
                    "cannot finalize V3-A evidence: " + self._evidence.error
                )

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                raise RuntimeError("no V3-A Mission is active")
            self._stop_requested = True
            self._state = replace(self._state, stage="stopping")
        try:
            status = self._operations.cancel()
            self._apply_status(status)
        except Exception as exc:
            message = f"V3-A cancel request failed: {exc}"
            self._append_log(message)
            with self._lock:
                self._state = replace(self._state, error=message)

    def close(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self.stop()
            except RuntimeError:
                pass
            thread.join(timeout=15.0)

    def _run(self, run_id: str, cycles: int, first_target: str) -> None:
        terminal_disarmed = False
        try:
            status = self._operations.start(
                run_id=run_id,
                requested_cycles=cycles,
                first_dig_point_id=first_target,
            )
            while True:
                self._apply_status(status)
                if status.terminal:
                    terminal_disarmed = True
                    return
                time.sleep(self._poll_interval_s)
                with self._lock:
                    if self._stop_requested:
                        terminal_disarmed = (
                            self._state.stage in _TERMINAL_UI_STAGES
                        )
                        if not terminal_disarmed:
                            self._state = replace(
                                self._state,
                                stage="failed",
                                next_segment="",
                                error=(
                                    "cancel was not acknowledged; forcing resident "
                                    "owner release"
                                ),
                            )
                            self._finish_evidence(stage="failed")
                        return
                status = self._operations.status()
        except Exception as exc:
            with self._lock:
                self._state = replace(
                    self._state,
                    stage="failed",
                    next_segment="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._append_log(f"V3-A failed: {type(exc).__name__}: {exc}")
            self._finish_evidence(stage="failed")
        finally:
            try:
                self._operations.release(terminal_disarmed=terminal_disarmed)
            except Exception as exc:
                self._append_log(f"V3-A resource release failed: {exc}")
                with self._lock:
                    if self._state.stage not in _TERMINAL_UI_STAGES:
                        self._state = replace(
                            self._state,
                            stage="failed",
                            error=f"resource release failed: {exc}",
                        )

    def _apply_status(self, status: ResidentFixedCycleRemoteStatus) -> None:
        stage = _STAGE_TO_UI[status.stage]
        if status.stage == "FOLLOW_DIG" and status.completed_cycles > 0:
            stage = "running_rl_return_to_dig"
        error = status.reason_code if stage == "failed" else ""
        with self._lock:
            self._state = replace(
                self._state,
                stage=stage,
                dig_target_id=status.current_dig_point_id or self._state.dig_target_id,
                run_completed_cycles=status.completed_cycles,
                requested_cycles=status.requested_cycles,
                error=error,
                evidence_error=self._evidence.error,
            )
            recorded = self._evidence.record(
                "resident_fixed_cycle_status",
                {
                    "stage": status.stage,
                    "requested_cycles": status.requested_cycles,
                    "completed_cycles": status.completed_cycles,
                    "dig_target_id": status.current_dig_point_id,
                    "terminal": status.terminal,
                    "outcome": status.outcome,
                    "reason_code": status.reason_code,
                },
            )
            if status.terminal:
                self._finish_evidence(stage=stage)
            elif not recorded:
                raise RuntimeError(
                    "V3-A evidence recording failed: " + self._evidence.error
                )
        self._append_log(
            "V3-A local status: "
            f"stage={status.stage} cycles={status.completed_cycles}/"
            f"{status.requested_cycles} target={status.current_dig_point_id}"
        )

    def _append_log(self, message: str) -> None:
        with self._lock:
            self._logs.append(message)

    def _finish_evidence(self, *, stage: str) -> None:
        self._evidence.finish(
            stage=stage,
            error=self._state.error,
            requested_cycles=self._state.requested_cycles,
            completed_cycles=self._state.run_completed_cycles,
            automatic=True,
        )
        self._state = replace(
            self._state,
            evidence_error=self._evidence.error,
        )


def _parse_control_response(
    payload: str, command: str
) -> ResidentFixedCycleRemoteStatus:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("V3-A control returned invalid JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "ok",
        "command",
        "status",
        "error",
    }:
        raise RuntimeError("V3-A control response fields are invalid")
    if (
        value["schema_version"] != CONTROL_SCHEMA_VERSION
        or value["command"] != command
        or value["ok"] is not True
        or value["error"] is not None
    ):
        raise RuntimeError(f"V3-A {command} command was rejected")
    return ResidentFixedCycleRemoteStatus.from_mapping(value["status"])


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be text")
    return value


def _optional_identifier(value: Any, field: str) -> str:
    text = _text(value, field, allow_empty=True)
    if text and _SAFE_ID.fullmatch(text) is None:
        raise ValueError(f"{field} must be an identifier")
    return text


def _absolute_posix(value: Any, field: str) -> PurePosixPath:
    path = PurePosixPath(_text(value, field))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be an absolute normalized path")
    return path


def _relative_script(value: Any, field: str) -> str:
    text = _text(value, field)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized relative path")
    return text


def _commissioning_authorization(value: Any) -> str:
    text = _text(value, "commissioning_authorization", allow_empty=True)
    if text not in {"", COMMISSIONING_AUTHORIZATION}:
        raise ValueError(
            "commissioning_authorization must be empty or the exact V3-A token"
        )
    return text


def _bounded_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} is outside its allowed range")
    return result


def _bounded_integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} is outside its allowed range")
    return value


def _remote_pid(line: str, name: str) -> int:
    match = re.fullmatch(re.escape(name) + r"=([1-9][0-9]*)", line)
    if match is None:
        raise RuntimeError(f"invalid {name} readiness line")
    return int(match.group(1))


def _wait_process(process: Any, *, allow_nonzero: bool) -> None:
    process.wait(timeout_s=10.0)
    if process.returncode not in (0, None) and not allow_nonzero:
        raise RuntimeError(f"remote process exited with return code {process.returncode}")
