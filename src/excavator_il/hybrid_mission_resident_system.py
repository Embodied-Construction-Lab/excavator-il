"""System composition for one Mission-scoped resident Orin runtime stack."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
from typing import Any, Callable

from .guided_episode import GuidedEpisodeConfig, SystemGuidedEpisodeOperations
from .hybrid_mission import (
    REQUIRED_HYBRID_MOTION_AUTHORIZATION,
    HybridMissionConfig,
    ResidentMissionConfig,
)
from .hybrid_mission_resident import (
    ExistingRlBehaviorAdapter,
    ResidentControlAdapter,
    ResidentControlStatus,
    ResidentHybridMissionOperations,
    SshResidentControlAdapter,
)
from .remote_runtime import LineProcess, SshRuntimeHost
from .resident_mission_lease import ResidentMissionLeaseHeartbeat
from .resident_prepared_follow import SystemPreparedDumpAdapter


class ResidentMissionProcesses:
    """Own the two long-lived remote processes, never the policy handoff itself."""

    def __init__(
        self,
        config: HybridMissionConfig,
        *,
        guided_config: GuidedEpisodeConfig,
        remote_host: Any | None = None,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        timestamp: str | None = None,
    ) -> None:
        resident = _resident_contract(config)
        self._resident = resident
        self._guided = guided_config
        self._remote_host = remote_host or SshRuntimeHost(
            guided_config.orin_ssh_host
        )
        self._line_process_factory = line_process_factory
        self._output = output
        self._timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._owner_process: Any | None = None
        self._owner_pid: int | None = None
        self._act_process: Any | None = None
        self._act_pid: int | None = None

    @property
    def started(self) -> bool:
        return self._owner_process is not None and self._act_process is not None

    @property
    def owner_started(self) -> bool:
        return self._owner_process is not None

    def require_owner_running(self) -> None:
        process = self._owner_process
        if process is None:
            raise RuntimeError("resident owner is not started")
        returncode = process.returncode
        if returncode is not None:
            raise RuntimeError(
                f"resident owner exited with return code {returncode}"
            )

    def require_running(self) -> None:
        """Fail immediately when either resident process has exited."""

        self.require_owner_running()
        process = self._act_process
        if process is None:
            raise RuntimeError("resident ACT worker is not started")
        returncode = process.returncode
        if returncode is not None:
            raise RuntimeError(
                f"resident ACT worker exited with return code {returncode}"
            )

    def start_owner(self) -> None:
        if self._owner_process is not None:
            self.require_owner_running()
            return
        if self._act_process is not None:
            raise RuntimeError("resident ACT worker exists without its owner")
        self._start_owner()

    def start_act_worker(self) -> None:
        self.require_owner_running()
        if self._act_process is not None:
            self.require_running()
            return
        self._start_act_worker()

    def wait_for_owner_hardware_ready(self) -> None:
        """Wait until the resident owner has observed one valid sensor state."""

        self.require_owner_running()
        process = self._owner_process
        assert process is not None
        process.wait_for(
            lambda line: "RESIDENT_HARDWARE_READY sensor_valid=True" in line,
            self._resident.ready_timeout_s,
        )

    def start(self) -> None:
        if self.started:
            self.require_running()
            return
        try:
            self.start_owner()
            self.start_act_worker()
        except BaseException:
            try:
                self.stop()
            except Exception as cleanup_exc:
                self._output(
                    "resident Mission 启动清理失败："
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise

    def stop(self, *, terminal_disarmed: bool = False) -> None:
        if not isinstance(terminal_disarmed, bool):
            raise ValueError("terminal_disarmed must be a boolean")
        errors: list[str] = []
        if terminal_disarmed:
            try:
                self._stop_owner()
            except Exception as exc:
                errors.append(f"resident owner: {exc}")
            try:
                self._stop_act_worker(allow_fail_closed_exit=True)
            except Exception as exc:
                errors.append(f"ACT worker: {exc}")
        else:
            try:
                self._stop_act_worker()
            except Exception as exc:
                errors.append(f"ACT worker: {exc}")
            try:
                self._stop_owner()
            except Exception as exc:
                errors.append(f"resident owner: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def _start_owner(self) -> None:
        command = " && ".join(
            (
                f"cd {shlex.quote(str(self._guided.rl_orin_repo))}",
                "echo RESIDENT_OWNER_PID=$$",
                "exec "
                + shlex.join(
                    [
                        "env",
                        f"RESIDENT_RUNTIME_ROOT={self._resident.runtime_root}",
                        f"RESIDENT_PYTHON={self._guided.rl_orin_python}",
                        "bash",
                        self._resident.owner_script,
                        "--authorization",
                        REQUIRED_HYBRID_MOTION_AUTHORIZATION,
                        "--pc-host",
                        self._guided.rl_pc_host,
                        "--serial-port",
                        str(self._guided.rl_serial_port),
                    ]
                ),
            )
        )
        process = self._spawn(
            command,
            prefix="resident-owner",
            log_name="resident-owner",
        )
        self._owner_process = process
        _, pid_line = process.wait_for(
            lambda line: line.startswith("RESIDENT_OWNER_PID="),
            self._resident.ready_timeout_s,
        )
        self._owner_pid = _remote_pid(pid_line, "RESIDENT_OWNER_PID")
        process.wait_for(
            lambda line: "RESIDENT_CONTROL_READY " in line,
            self._resident.ready_timeout_s,
        )

    def _start_act_worker(self) -> None:
        command = " && ".join(
            (
                f"cd {shlex.quote(str(self._guided.orin_repo))}",
                "echo RESIDENT_ACT_PID=$$",
                "exec "
                + shlex.join(
                    [
                        "env",
                        f"RESIDENT_RUNTIME_ROOT={self._resident.runtime_root}",
                        "bash",
                        self._resident.act_worker_script,
                        "--authorization",
                        REQUIRED_HYBRID_MOTION_AUTHORIZATION,
                    ]
                ),
            )
        )
        process = self._spawn(
            command,
            prefix="resident-act",
            log_name="resident-act",
        )
        self._act_process = process
        _, pid_line = process.wait_for(
            lambda line: line.startswith("RESIDENT_ACT_PID="),
            self._resident.ready_timeout_s,
        )
        self._act_pid = _remote_pid(pid_line, "RESIDENT_ACT_PID")
        process.wait_for(
            lambda line: "ACT resident worker ready:" in line,
            self._resident.ready_timeout_s,
        )

    def _spawn(self, command: str, *, prefix: str, log_name: str) -> Any:
        log_path = Path(self._guided.log_dir) / (
            f"hybrid_mission_{self._timestamp}.{log_name}.log"
        )
        return self._line_process_factory(
            self._remote_host.argv(command),
            log_path=log_path,
            prefix=prefix,
            output=self._output,
        )

    def _stop_act_worker(self, *, allow_fail_closed_exit: bool = False) -> None:
        process, pid = self._act_process, self._act_pid
        if process is None:
            return
        owner = self._owner_process
        owner_driven_fail_closed_exit = (
            process.returncode is not None
            and owner is not None
            and owner.returncode is not None
        )
        try:
            if pid is None:
                process.stop(signal.SIGTERM, timeout_s=2.0)
                raise RuntimeError("resident ACT PID was not observed")
            self._remote_host.stop_owned_process(
                pid=pid,
                identity_ere=r"([d]ocker.*resident_act_runtime|[r]un_act_resident\.sh)",
                serial_path=PurePosixPath("/dev/video0"),
                timeout_s=self._guided.rl_serial_release_timeout_s,
                require_serial_release=True,
            )
            _wait_for_local_process(
                process,
                allow_nonzero=(
                    allow_fail_closed_exit or owner_driven_fail_closed_exit
                ),
            )
        finally:
            self._act_process = None
            self._act_pid = None

    def _stop_owner(self) -> None:
        process, pid = self._owner_process, self._owner_pid
        if process is None:
            return
        try:
            if pid is None:
                process.stop(signal.SIGTERM, timeout_s=2.0)
                raise RuntimeError("resident owner PID was not observed")
            self._remote_host.stop_owned_process(
                pid=pid,
                identity_ere=r"[o]rin_state_sender\.py.*--resident-motion-core",
                serial_path=self._guided.rl_serial_port,
                timeout_s=self._guided.rl_serial_release_timeout_s,
                require_serial_release=True,
                cleanup_paths=(
                    self._resident.control_socket,
                    self._resident.act_socket,
                ),
            )
            _wait_for_local_process(process)
        finally:
            self._owner_process = None
            self._owner_pid = None


class ResidentProcessAwareControlAdapter:
    """Fail fast when the resident ACT worker disappears mid-Mission."""

    def __init__(
        self,
        *,
        delegate: ResidentControlAdapter,
        processes: ResidentMissionProcesses | Any,
    ) -> None:
        self._delegate = delegate
        self._processes = processes

    def ensure_ready(self) -> ResidentControlStatus:
        status = self._delegate.ensure_ready()
        return self._validate_running_status(status)

    def status(self) -> ResidentControlStatus:
        self._processes.require_running()
        status = self._delegate.status()
        return self._validate_running_status(status)

    def activate_rl(self) -> ResidentControlStatus:
        self._processes.require_running()
        status = self._delegate.activate_rl()
        return self._validate_running_status(status)

    def activate_act(self, max_steps: int) -> ResidentControlStatus:
        self._processes.require_running()
        status = self._delegate.activate_act(max_steps)
        return self._validate_running_status(status)

    def renew_lease(self) -> ResidentControlStatus:
        self._processes.require_owner_running()
        status = self._delegate.renew_lease()
        self._processes.require_owner_running()
        current = status
        if not current.mission_lease_active:
            raise RuntimeError("resident Mission lease did not become active")
        return current

    def terminal_disarm(self) -> ResidentControlStatus:
        # Cleanup must still be attempted after a worker crash.
        return self._delegate.terminal_disarm()

    def _validate_running_status(
        self, status: ResidentControlStatus
    ) -> ResidentControlStatus:
        self._processes.require_running()
        if not status.act_worker_ready:
            raise RuntimeError("resident ACT worker is not ready")
        return status


class SystemResidentHybridMissionOperations:
    """Adapt the existing Hybrid Mission API to one resident Orin stack."""

    def __init__(
        self,
        config: HybridMissionConfig,
        *,
        guided_config: GuidedEpisodeConfig | None = None,
        processes: ResidentMissionProcesses | Any | None = None,
        resident_operations: ResidentHybridMissionOperations | Any | None = None,
        lease_heartbeat: ResidentMissionLeaseHeartbeat | Any | None = None,
        prepared_dump_adapter: Any | None = None,
        rl_operations: Any | None = None,
        line_process_factory: Callable[..., Any] = LineProcess,
        output: Callable[[str], None] = print,
        timestamp: str | None = None,
    ) -> None:
        resident = _resident_contract(config)
        if processes is not None and resident_operations is not None:
            if lease_heartbeat is None:
                raise ValueError(
                    "lease_heartbeat is required with injected resident operations"
                )
            self._processes = processes
            self._operations = resident_operations
            self._lease_heartbeat = lease_heartbeat
            return

        guided = guided_config or GuidedEpisodeConfig.load(config.guided_config)
        mission_timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._processes = processes or ResidentMissionProcesses(
            config,
            guided_config=guided,
            line_process_factory=line_process_factory,
            output=output,
            timestamp=mission_timestamp,
        )
        behavior_operations = rl_operations or SystemGuidedEpisodeOperations(
            guided,
            output=output,
            timestamp=f"resident_{mission_timestamp}",
        )
        behavior = ExistingRlBehaviorAdapter(
            behavior_operations,
            behavior_port=config.rl_behavior_port,
        )
        control = ResidentProcessAwareControlAdapter(
            delegate=SshResidentControlAdapter(
                ssh_host=guided.orin_ssh_host,
                orin_repo=guided.rl_orin_repo,
                socket_path=resident.control_socket,
                python_executable=guided.rl_orin_python,
                ensure_services_ready=self._processes.start,
                command_timeout_s=max(30, resident.handoff_timeout_s + 5),
            ),
            processes=self._processes,
        )
        self._lease_heartbeat = (
            lease_heartbeat
            if lease_heartbeat is not None
            else ResidentMissionLeaseHeartbeat(
                control.renew_lease,
                interval_s=0.4,
            )
        )
        prepared_dump = prepared_dump_adapter
        if prepared_dump is None:
            prepared_dump = SystemPreparedDumpAdapter(
                airy_repo=guided.rl_airy_repo,
                ros_setup=guided.rl_ros_setup,
                workspace_setup=guided.rl_workspace_setup,
                mission_config=guided.rl_mission_config,
                log_dir=guided.log_dir,
                wait_s=guided.ack_timeout_s,
                ready_grace_ms=resident.prepared_ready_grace_ms,
                run_timeout_s=guided.rl_timeout_s,
                start_tolerance_m=resident.prepared_start_tolerance_m,
                line_process_factory=line_process_factory,
                output=output,
                timestamp=mission_timestamp,
            )
        self._operations = resident_operations or ResidentHybridMissionOperations(
            control=control,
            behavior=behavior,
            prepared_dump=prepared_dump,
            prepared_dump_lead_steps=resident.prepared_dump_lead_steps,
            act_run_timeout_s=config.act_run_timeout_s,
            handoff_timeout_s=resident.handoff_timeout_s,
            poll_interval_s=resident.poll_interval_ms / 1000.0,
        )

    def run_rl_to_dig(self, target_id: str) -> None:
        self._run(self._operations.run_rl_to_dig, target_id)

    def run_act_dig(self, max_steps: int) -> None:
        self._run(self._operations.run_act_dig, max_steps)

    def run_rl_to_dump_and_dump(self) -> None:
        self._run(self._operations.run_rl_to_dump_and_dump)

    def run_rl_return_to_dig(self, target_id: str) -> None:
        self._run(self._operations.run_rl_return_to_dig, target_id)

    def prewarm_next_act(self, max_steps: int) -> None:
        self._run(self._operations.prewarm_next_act, max_steps)

    def safe_stop(self) -> None:
        errors: list[str] = []
        terminal_disarmed = False
        try:
            self._lease_heartbeat.request_stop()
        except Exception as exc:
            errors.append(f"stop lease renewals: {exc}")
        if self._processes.owner_started:
            try:
                self._operations.safe_stop()
                terminal_disarmed = True
            except Exception as exc:
                errors.append(f"terminal disarm: {exc}")
        try:
            self._lease_heartbeat.stop()
        except Exception as exc:
            errors.append(f"lease heartbeat: {exc}")
        try:
            if self._processes.owner_started:
                self._processes.stop(terminal_disarmed=terminal_disarmed)
        except Exception as exc:
            errors.append(f"process release: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def _run(self, operation: Callable[..., None], *args: Any) -> None:
        try:
            self._processes.start_owner()
            starting = not self._lease_heartbeat.running
            if starting:
                self._lease_heartbeat.start()
            self._lease_heartbeat.require_healthy()
            self._processes.start_act_worker()
            if starting:
                self._processes.wait_for_owner_hardware_ready()
                self._lease_heartbeat.require_healthy()
            operation(*args)
            self._lease_heartbeat.require_healthy()
        except BaseException as exc:
            try:
                self.safe_stop()
            except Exception as cleanup_exc:
                raise RuntimeError(
                    "resident Mission operation failed and cleanup also failed: "
                    f"{cleanup_exc}"
                ) from exc
            raise


def _resident_contract(config: HybridMissionConfig) -> ResidentMissionConfig:
    if config.runtime_backend != "resident" or config.resident is None:
        raise ValueError("resident Mission operations require the resident backend")
    return config.resident


def _remote_pid(line: str, name: str) -> int:
    match = re.fullmatch(re.escape(name) + r"=([1-9][0-9]*)", line)
    if match is None:
        raise RuntimeError(f"invalid {name} readiness line")
    return int(match.group(1))


def _wait_for_local_process(
    process: Any,
    *,
    allow_nonzero: bool = False,
) -> None:
    try:
        process.wait(timeout_s=2.0)
    except Exception:
        process.stop(signal.SIGTERM, timeout_s=2.0)
    returncode = process.returncode
    if returncode not in (None, 0) and not allow_nonzero:
        raise RuntimeError(
            f"resident remote process exited with return code {returncode}"
        )
