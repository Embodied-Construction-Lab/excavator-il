"""PC display/start/cancel adapter for the Orin-local V3-A fixed cycle."""

from __future__ import annotations

import hashlib
import shlex
import signal
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .guided_episode import GuidedEpisodeConfig
from .hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from .hybrid_mission_session import MAX_HYBRID_CYCLE_COUNT, HybridMissionSnapshot
from .hybrid_experiment_run import (
    HybridMissionEvidenceLifecycle,
    HybridMissionRunRequest,
)
from .remote_runtime import LineProcess, SshRuntimeHost
from ._resident_fixed_cycle_config import (
    CONFIG_SCHEMA_VERSION,
    ResidentFixedCyclePcConfig,
)
from ._resident_fixed_cycle_support import (
    COMMISSIONING_AUTHORIZATION,
    CONTROL_SCHEMA_VERSION,
    bounded_integer as _bounded_integer,
    parse_control_response as _parse_control_response,
    parse_owner_readiness as _parse_owner_readiness,
    remote_pid as _remote_pid,
    text as _text,
    wait_process as _wait_process,
)
from ._resident_fixed_cycle_groups import (
    normalize_dig_groups,
    select_cycle_targets,
)
from .resident_fixed_cycle_visualization import (
    ResidentFixedCycleRemoteStatus,
    V3aTrajectoryFile,
    v3a_trajectory_path,
)

_TERMINAL_UI_STAGES = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_STAGE_TO_UI = {
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
}
_TRACKING_BEHAVIORS = frozenset({"onnx_rl_tracking", "cartesian_p_tracking"})
_DIG_BEHAVIORS = frozenset(
    {"act_dig_lift", "act_dig_transport_dump", "fixed_dig"}
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
        self._owner_act_worker_required: bool | None = None

    def start(self) -> None:
        if self._owner_process is not None or self._act_process is not None:
            self.require_running()
            return
        try:
            self._start_owner()
            if self._requires_act_worker():
                self._start_act_worker()
        except BaseException:
            try:
                self.stop()
            except Exception as exc:
                self._output(f"V3-A startup cleanup failed: {exc}")
            raise

    def require_running(self) -> None:
        checks = [("resident owner", self._owner_process)]
        if self._requires_act_worker():
            checks.append(("resident ACT worker", self._act_process))
        for name, process in checks:
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
        catalog_digest = self._dig_catalog_sha256()
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
            "--edge-config",
            str(self._config.edge_runtime_config),
            "--fixed-cycle-plan",
            str(self._config.fixed_cycle_plan),
        ]
        if catalog_digest is not None:
            argv.extend(["--expected-dig-catalog-sha256", catalog_digest])
        if self._config.commissioning_authorization:
            argv.extend(
                [
                    "--commissioning-authorization",
                    self._config.commissioning_authorization,
                ]
            )
        if self._config.trajectory_controller_commissioning_authorization:
            argv.extend(
                [
                    "--trajectory-controller-commissioning-authorization",
                    self._config.trajectory_controller_commissioning_authorization,
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
        readiness = _parse_owner_readiness(ready_line)
        if (
            readiness.control_socket != self._config.control_socket
            or readiness.act_socket != self._config.runtime_root / "act.sock"
        ):
            raise RuntimeError("V3-A owner announced an unexpected control socket")
        if (
            readiness.trajectory_controller_backend
            != self._config.trajectory_controller_backend
        ):
            raise RuntimeError(
                "V3-A owner trajectory controller backend does not match PC config"
            )
        if readiness.mission_id != self._config.expected_mission_id:
            raise RuntimeError("V3-A owner mission_id does not match PC config")
        if readiness.mission_sha256 != self._config.expected_mission_sha256:
            raise RuntimeError("V3-A owner mission_sha256 does not match PC config")
        if (
            readiness.act_worker_required
            != self._config.expected_act_worker_required
        ):
            raise RuntimeError(
                "V3-A owner act_worker_required does not match PC config/assets"
            )
        if readiness.act_worker_behavior_id != self._config.expected_act_behavior_id:
            raise RuntimeError("V3-A owner ACT behavior does not match PC config")
        if (
            readiness.act_worker_model_sha256
            != self._config.expected_act_model_sha256
        ):
            raise RuntimeError("V3-A owner ACT model does not match PC config")
        self._owner_act_worker_required = readiness.act_worker_required
        process.wait_for(
            lambda item: "RESIDENT_HARDWARE_READY sensor_valid=True" in item,
            self._config.ready_timeout_s,
        )

    def _dig_catalog_sha256(self) -> str | None:
        relative = self._config.dig_point_catalog
        if relative is None:
            return None
        path = self._guided.rl_airy_repo.joinpath(*relative.parts)
        try:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Dig Point Catalog must be a regular file")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise RuntimeError(f"cannot hash Dig Point Catalog: {exc}") from exc

    def _requires_act_worker(self) -> bool:
        if self._owner_act_worker_required is not None:
            return self._owner_act_worker_required
        return self._config.expected_act_worker_required

    def _start_act_worker(self) -> None:
        argv = [
            "env",
            f"RESIDENT_RUNTIME_ROOT={self._config.runtime_root}",
        ]
        if self._config.act_runtime_config is not None:
            argv.extend(
                [
                    f"ACT_RUNTIME_CONFIG_PATH={self._config.act_runtime_config}",
                    f"ACT_CHECKPOINT_HOST_PATH={self._config.act_checkpoint_host_path}",
                    f"ACT_DEPLOYMENT_HOST_PATH={self._config.act_deployment_host_path}",
                ]
            )
        argv.extend(
            [
                "bash",
                self._config.act_worker_script,
                "--authorization",
                REQUIRED_HYBRID_MOTION_AUTHORIZATION,
            ]
        )
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
                    identity_ere=(
                        r"[o]rin_state_sender\.py.*--resident-fixed-cycle-plan"
                    ),
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
            self._owner_act_worker_required = None

    def _stop_act_worker(self, *, allow_nonzero: bool) -> None:
        process, pid = self._act_process, self._act_pid
        if process is None:
            return
        try:
            if pid is not None and process.returncode is None:
                self._remote_host.stop_owned_process(
                    pid=pid,
                    identity_ere=(
                        r"([d]ocker.*resident_act_runtime|"
                        r"[r]un_act_resident\.sh)"
                    ),
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
        trajectory_file: V3aTrajectoryFile | None = None,
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
        self._trajectory_file = trajectory_file or V3aTrajectoryFile(
            v3a_trajectory_path(guided_config.log_dir)
        )
        self._trajectory_file.update(None)

    def start(
        self,
        *,
        run_id: str,
        requested_cycles: int,
        first_dig_point_id: str,
        dig_group_id: str = "all",
    ) -> ResidentFixedCycleRemoteStatus:
        self._trajectory_file.update(None)
        self._processes.start()
        return self._request(
            "start",
            "--run-id",
            run_id,
            "--cycles",
            str(requested_cycles),
            "--first-dig-point-id",
            first_dig_point_id,
            "--dig-group-id",
            dig_group_id,
        )

    def status(self) -> ResidentFixedCycleRemoteStatus:
        return self._request("heartbeat")

    def cancel(self) -> ResidentFixedCycleRemoteStatus:
        return self._request("cancel")

    def release(self, *, terminal_disarmed: bool) -> None:
        try:
            self._processes.stop(terminal_disarmed=terminal_disarmed)
        finally:
            self._trajectory_file.update(None)

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
        status = _parse_control_response(output, command)
        if status.mission_id != self._config.expected_mission_id:
            raise RuntimeError("V3-B owner mission_id does not match PC config")
        self._trajectory_file.update(status.active_trajectory)
        return status


class ResidentFixedCycleSupervisor:
    """Expose one Orin-local fixed cycle through the existing WebUI Interface."""

    def __init__(
        self,
        *,
        operations: Any,
        dig_target_ids: tuple[str, ...],
        dig_groups: Mapping[str, tuple[str, ...]] | None = None,
        default_dig_group_id: str = "all",
        poll_interval_s: float,
        log_capacity: int = 400,
        config_path: str | Path | None = None,
        evidence_run_factory: Callable[[HybridMissionRunRequest], Any] | None = None,
    ) -> None:
        groups, default_group = normalize_dig_groups(
            dig_target_ids,
            dig_groups,
            default_dig_group_id,
        )
        if not 0.02 <= poll_interval_s <= 1.0:
            raise ValueError("poll_interval_s must be within [0.02, 1.0]")
        self._operations = operations
        self._dig_target_ids = dig_target_ids
        self._dig_groups = groups
        self._default_dig_group_id = default_group
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

    def clear_logs(self) -> None:
        with self._lock:
            self._logs.clear()

    def append_external_log(self, message: str) -> None:
        self._append_log(_text(message, "resident log"))

    def start(
        self,
        dig_target_id: str,
        *,
        automatic: bool,
        motion_authorization: str | None,
        cycle_count: int = 1,
        dig_group_id: str | None = None,
    ) -> None:
        if automatic is not True:
            raise ValueError("V3-A supports automatic local cycles only")
        if motion_authorization != REQUIRED_HYBRID_MOTION_AUTHORIZATION:
            raise ValueError("V3-A requires exact motion authorization")
        if dig_target_id not in self._dig_target_ids:
            raise ValueError("unknown V3-A dig target")
        _bounded_integer(cycle_count, "cycle_count", 1, MAX_HYBRID_CYCLE_COUNT)
        selected_group_id = dig_group_id or self._default_dig_group_id
        cycle_targets = select_cycle_targets(
            self._dig_groups,
            selected_group_id,
            dig_target_id,
            cycle_count,
        )
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
            self._evidence = HybridMissionEvidenceLifecycle(
                evidence_run,
                cycle_targets=cycle_targets,
            )
            self._logs.clear()
            self._stop_requested = False
            self._state = HybridMissionSnapshot(
                stage="starting",
                dig_target_id=dig_target_id,
                dig_group_id=selected_group_id,
                automatic=True,
                requested_cycles=cycle_count,
                run_id=run_id,
            )
            self._evidence.start_mission(
                automatic=True,
                requested_cycles=cycle_count,
                dig_target_id=dig_target_id,
                dig_group_id=selected_group_id,
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
                args=(run_id, cycle_count, dig_target_id, selected_group_id),
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
                # Serialize the remote acknowledgement with the background
                # poller so it cannot release the owner as unacknowledged while
                # a successful cancel request is still in flight.
                status = self._operations.cancel()
                self._apply_status(status)
            except Exception as exc:
                message = f"V3-A cancel request failed: {exc}"
                self._append_log(message)
                self._state = replace(self._state, error=message)

    def close(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                self.stop()
            except RuntimeError:
                pass
            thread.join(timeout=15.0)

    def _run(
        self,
        run_id: str,
        cycles: int,
        first_target: str,
        dig_group_id: str,
    ) -> None:
        terminal_disarmed = False
        try:
            status = self._operations.start(
                run_id=run_id,
                requested_cycles=cycles,
                first_dig_point_id=first_target,
                dig_group_id=dig_group_id,
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
        stage = _ui_stage(status)
        error = status.reason_code if stage == "failed" else ""
        with self._lock:
            if self._stop_requested and not status.terminal:
                # A heartbeat fetched before the cancel acknowledgement may
                # arrive afterwards.  It is stale with respect to operator
                # cancellation and must not overwrite the terminal status.
                return
            self._state = replace(
                self._state,
                stage=stage,
                dig_target_id=status.current_dig_point_id or self._state.dig_target_id,
                dig_group_id=status.dig_group_id or self._state.dig_group_id,
                run_completed_cycles=status.completed_cycles,
                requested_cycles=status.requested_cycles,
                error=error,
                evidence_error=self._evidence.error,
            )
            recorded = self._evidence.record(
                "resident_fixed_cycle_status",
                {
                    "stage": status.stage,
                    "mission_id": status.mission_id,
                    "active_behavior_id": status.active_behavior_id,
                    "requested_cycles": status.requested_cycles,
                    "completed_cycles": status.completed_cycles,
                    "dig_target_id": status.current_dig_point_id,
                    "dig_group_id": status.dig_group_id,
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
            f"mission={status.mission_id} behavior={status.active_behavior_id} "
            f"stage={status.stage} "
            f"cycles={status.completed_cycles}/"
            f"{status.requested_cycles} target={status.current_dig_point_id}"
            f" group={status.dig_group_id}"
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


def _ui_stage(status: ResidentFixedCycleRemoteStatus) -> str:
    terminal = _TERMINAL_STAGE_TO_UI.get(status.stage)
    if terminal is not None:
        return terminal
    if status.stage == "IDLE":
        return "idle"
    behavior = status.active_behavior_id
    if behavior in _DIG_BEHAVIORS:
        return "running_act_dig"
    if behavior == "fixed_dump":
        return "running_rl_to_dump_and_dump"
    if behavior in _TRACKING_BEHAVIORS:
        target = status.active_trajectory.target_id if status.active_trajectory else ""
        if target == "dump":
            return "running_rl_to_dump_and_dump"
        if status.completed_cycles > 0:
            return "running_rl_return_to_dig"
        return "running_rl_to_dig"
    raise RuntimeError("V3-A status has no supported active behavior")
