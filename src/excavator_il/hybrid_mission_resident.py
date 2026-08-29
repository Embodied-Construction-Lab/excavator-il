"""PC Adapter for a resident RL/ACT Mission owner on Orin."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import os
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol

from .remote_runtime import RunCommand, SshRuntimeHost


RESIDENT_CONTROL_SCHEMA_VERSION = "resident_motion_control.v1"
_MAX_UNIX_PATH_BYTES = 107
_MAX_CLI_RESPONSE_BYTES = 4097
_RESPONSE_FIELDS = frozenset(
    {"schema_version", "ok", "command", "status", "error"}
)
_STATUS_FIELDS = frozenset(
    {
        "phase",
        "control_generation",
        "active",
        "target",
        "last_handoff_latency_ms",
        "rl_is_active",
        "act_is_active",
        "act_worker_ready",
        "act_segment_generation",
        "act_segment_max_steps",
        "act_segment_completed_steps",
        "act_segment_complete",
        "mission_lease_active",
        "is_operational",
    }
)
_PHASES = frozenset(
    {"idle", "terminal_zero_pending", "target_zero_pending", "active"}
)
_CONTROL_MODES = frozenset({"manual_action", "velocity_reference"})
_SSH_DESTINATION = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.:-]*"
)


@dataclass(frozen=True)
class ResidentPolicyBinding:
    source: str
    mode: str


_RL_BINDING = ResidentPolicyBinding(
    source="rl_follow", mode="velocity_reference"
)
_ACT_BINDING = ResidentPolicyBinding(
    source="act_dig", mode="manual_action"
)


@dataclass(frozen=True)
class ResidentControlStatus:
    phase: str
    control_generation: int
    active: ResidentPolicyBinding | None
    target: ResidentPolicyBinding | None
    last_handoff_latency_ms: float | None
    rl_is_active: bool
    act_is_active: bool
    act_worker_ready: bool
    act_segment_generation: int | None
    act_segment_max_steps: int | None
    act_segment_completed_steps: int
    act_segment_complete: bool
    mission_lease_active: bool
    is_operational: bool


class ResidentControlAdapter(Protocol):
    def ensure_ready(self) -> ResidentControlStatus: ...

    def status(self) -> ResidentControlStatus: ...

    def activate_rl(self) -> ResidentControlStatus: ...

    def activate_act(self, max_steps: int) -> ResidentControlStatus: ...

    def renew_lease(self) -> ResidentControlStatus: ...

    def terminal_disarm(self) -> ResidentControlStatus: ...


class PreparedDumpActivation(str, Enum):
    """Only the two safe outcomes exposed by prepared dump activation."""

    ACTIVATED = "activated"
    FALLBACK_SAFE = "fallback_safe"


class PreparedDumpAdapter(Protocol):
    """Non-blocking preparation followed by one explicit activation decision."""

    def start_prepare(self) -> None: ...

    def trigger_prepare(self) -> None: ...

    def trigger_refresh(self) -> None: ...

    def activate_prepared(self) -> PreparedDumpActivation: ...

    def cancel(self) -> None: ...


class SshResidentControlAdapter:
    """Issue one validated resident-control request per BatchMode SSH call."""

    def __init__(
        self,
        *,
        ssh_host: str,
        orin_repo: str | PurePosixPath,
        socket_path: str | PurePosixPath,
        ensure_services_ready: Callable[[], None],
        python_executable: str | PurePosixPath = "python3",
        run_command: RunCommand = subprocess.run,
        connect_timeout_s: int = 5,
        command_timeout_s: int = 30,
        lease_command_timeout_s: int = 1,
    ) -> None:
        if (
            not isinstance(ssh_host, str)
            or _SSH_DESTINATION.fullmatch(ssh_host) is None
        ):
            raise ValueError("ssh_host must be a safe user@host destination")
        repo = _absolute_posix_path(orin_repo, "orin_repo")
        socket_value = _absolute_posix_path(socket_path, "socket_path")
        if len(os.fsencode(socket_value)) > _MAX_UNIX_PATH_BYTES:
            raise ValueError("socket_path is too long for a Unix domain socket")
        python_value = str(python_executable)
        if not python_value or "\x00" in python_value:
            raise ValueError("python_executable must be non-empty NUL-free text")
        if not callable(ensure_services_ready):
            raise ValueError("ensure_services_ready must be callable")
        self._remote_host = SshRuntimeHost(
            ssh_host,
            run_command=run_command,
            connect_timeout_s=connect_timeout_s,
            command_timeout_s=command_timeout_s,
        )
        self._lease_remote_host = SshRuntimeHost(
            ssh_host,
            run_command=run_command,
            connect_timeout_s=connect_timeout_s,
            command_timeout_s=lease_command_timeout_s,
        )
        self._orin_repo = repo
        self._socket_path = socket_value
        self._python_executable = python_value
        self._ensure_services_ready = ensure_services_ready

    def ensure_ready(self) -> ResidentControlStatus:
        self._ensure_services_ready()
        status = self.status()
        if not status.is_operational:
            raise RuntimeError("resident motion core is not operational")
        if not status.act_worker_ready:
            raise RuntimeError("resident ACT worker is not ready")
        return status

    def status(self) -> ResidentControlStatus:
        return self._request("status")

    def activate_rl(self) -> ResidentControlStatus:
        return self._request("activate_rl")

    def activate_act(self, max_steps: int) -> ResidentControlStatus:
        steps = _act_step_budget(max_steps)
        return self._request("activate_act", max_steps=steps)

    def renew_lease(self) -> ResidentControlStatus:
        return self._request("renew_lease")

    def terminal_disarm(self) -> ResidentControlStatus:
        return self._request("terminal_disarm")

    def _request(
        self,
        command: str,
        *,
        max_steps: int | None = None,
    ) -> ResidentControlStatus:
        argv = [
            self._python_executable,
            "-m",
            "edge_runtime.resident_control",
            "--socket",
            self._socket_path,
            command,
        ]
        if max_steps is not None:
            argv.extend(("--max-steps", str(max_steps)))
        remote_command = (
            f"cd {shlex.quote(self._orin_repo)} && {shlex.join(argv)}"
        )
        remote_host = (
            self._lease_remote_host
            if command == "renew_lease"
            else self._remote_host
        )
        output = remote_host.run(remote_command)
        return _parse_success_response(output, expected_command=command)


class ResidentBehaviorAdapter(Protocol):
    def run_rl_to_dig(self, target_id: str) -> None: ...

    def run_rl_to_dump_and_dump(self) -> None: ...

    def run_dump_action(self) -> None: ...

    def run_rl_return_to_dig(self, target_id: str) -> None: ...


class ExistingRlOperations(Protocol):
    def run_rl_follow(self, phase: str, *, target_id: str | None = None) -> Any: ...

    def run_rl_fixed_action(self, behavior: str, *, behavior_port: int) -> None: ...


class ExistingRlBehaviorAdapter:
    """Reuse the established behaviors without touching resident lifecycles."""

    def __init__(
        self,
        operations: ExistingRlOperations,
        *,
        behavior_port: int,
    ) -> None:
        if (
            isinstance(behavior_port, bool)
            or not isinstance(behavior_port, int)
            or not 1 <= behavior_port <= 65535
        ):
            raise ValueError("behavior_port must be an integer in [1, 65535]")
        self._operations = operations
        self._behavior_port = behavior_port

    def run_rl_to_dig(self, target_id: str) -> None:
        self._operations.run_rl_follow("dig", target_id=target_id)

    def run_rl_to_dump_and_dump(self) -> None:
        self._operations.run_rl_follow("dump")
        self.run_dump_action()

    def run_dump_action(self) -> None:
        self._operations.run_rl_fixed_action(
            "ExecuteDump", behavior_port=self._behavior_port
        )

    def run_rl_return_to_dig(self, target_id: str) -> None:
        self._operations.run_rl_follow("dig", target_id=target_id)


class ResidentHybridMissionOperations:
    """Keep the Orin owner resident while preserving the Mission operations API."""

    def __init__(
        self,
        *,
        control: ResidentControlAdapter,
        behavior: ResidentBehaviorAdapter,
        prepared_dump: PreparedDumpAdapter | None = None,
        prepared_dump_lead_steps: int | None = None,
        prepared_dump_refresh_lead_steps: int | None = None,
        act_run_timeout_s: float,
        handoff_timeout_s: float = 10.0,
        poll_interval_s: float = 0.1,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        for name, value in (
            ("act_run_timeout_s", act_run_timeout_s),
            ("handoff_timeout_s", handoff_timeout_s),
            ("poll_interval_s", poll_interval_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if float(poll_interval_s) < 0.01:
            raise ValueError("poll_interval_s must be at least 0.01 seconds")
        self._control = control
        self._behavior = behavior
        if (prepared_dump is None) != (prepared_dump_lead_steps is None):
            raise ValueError(
                "prepared_dump and prepared_dump_lead_steps must be configured together"
            )
        if prepared_dump_lead_steps is not None:
            _act_step_budget(prepared_dump_lead_steps)
        if prepared_dump_refresh_lead_steps is not None:
            _act_step_budget(prepared_dump_refresh_lead_steps)
            if prepared_dump_lead_steps is None:
                raise ValueError(
                    "prepared_dump_refresh_lead_steps requires prepared dump"
                )
            if prepared_dump_refresh_lead_steps >= prepared_dump_lead_steps:
                raise ValueError(
                    "prepared dump refresh lead must be less than initial lead"
                )
        self._prepared_dump = prepared_dump
        self._prepared_dump_lead_steps = prepared_dump_lead_steps
        self._prepared_dump_refresh_lead_steps = (
            prepared_dump_refresh_lead_steps
        )
        self._prepared_dump_started = False
        self._prepared_dump_triggered = False
        self._prepared_dump_refreshed = False
        self._terminal_disarmed = False
        self._act_run_timeout_s = float(act_run_timeout_s)
        self._handoff_timeout_s = float(handoff_timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._monotonic = monotonic
        self._sleep = sleep

    def run_rl_to_dig(self, target_id: str) -> None:
        self._run_rl_behavior(
            lambda: self._behavior.run_rl_to_dig(target_id)
        )

    def run_rl_to_dump_and_dump(self) -> None:
        if (
            self._prepared_dump is None
            or not self._prepared_dump_started
            or not self._prepared_dump_triggered
        ):
            self._run_rl_behavior(self._behavior.run_rl_to_dump_and_dump)
            return
        try:
            outcome = self._prepared_dump.activate_prepared()
            self._prepared_dump_started = False
            self._prepared_dump_triggered = False
            self._prepared_dump_refreshed = False
            if outcome is not PreparedDumpActivation.ACTIVATED:
                if outcome is not PreparedDumpActivation.FALLBACK_SAFE:
                    raise RuntimeError(
                        "prepared dump returned an invalid activation outcome"
                    )
        except BaseException as exc:
            self._abort_after_failure(exc, segment="prepared dump")
            raise
        if outcome is PreparedDumpActivation.FALLBACK_SAFE:
            self._run_rl_behavior(self._behavior.run_rl_to_dump_and_dump)
            return
        try:
            self._behavior.run_dump_action()
        except BaseException as exc:
            self._abort_after_failure(exc, segment="prepared dump")
            raise

    def run_rl_return_to_dig(self, target_id: str) -> None:
        self._run_rl_behavior(
            lambda: self._behavior.run_rl_return_to_dig(target_id)
        )

    def run_act_dig(self, max_steps: int) -> None:
        steps = _act_step_budget(max_steps)
        lead_steps = self._prepared_dump_lead_steps
        refresh_lead_steps = self._prepared_dump_refresh_lead_steps
        if lead_steps is not None and lead_steps >= steps:
            raise ValueError("prepared_dump_lead_steps must be less than max_steps")
        if refresh_lead_steps is not None and refresh_lead_steps >= steps:
            raise ValueError(
                "prepared_dump_refresh_lead_steps must be less than max_steps"
            )
        try:
            self._cancel_prepared_dump()
            self._control.ensure_ready()
            status = self._control.activate_act(steps)
            generation = status.act_segment_generation
            if (
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 0
            ):
                raise RuntimeError(
                    "resident ACT activation did not return a segment generation"
                )
            if generation != status.control_generation:
                raise RuntimeError(
                    "resident ACT activation generation is inconsistent"
                )
            if self._prepared_dump is not None:
                self._prepared_dump.start_prepare()
                self._prepared_dump_started = True
                self._prepared_dump_triggered = False
                self._prepared_dump_refreshed = False
            deadline = self._monotonic() + self._act_run_timeout_s
            self._wait_for_act_completion(
                status,
                generation=generation,
                max_steps=steps,
                prepare_at_step=(
                    None if lead_steps is None else steps - lead_steps
                ),
                refresh_at_step=(
                    None
                    if refresh_lead_steps is None
                    else steps - refresh_lead_steps
                ),
                deadline=deadline,
            )
        except BaseException as exc:
            self._abort_after_failure(exc, segment="ACT")
            raise

    def safe_stop(self) -> None:
        errors: list[str] = []
        try:
            self._terminal_disarm_once()
        except Exception as exc:
            errors.append(f"terminal disarm: {exc}")
        try:
            self._cancel_prepared_dump()
        except Exception as exc:
            errors.append(f"prepared dump: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def prewarm_next_act(self, max_steps: int) -> None:
        _act_step_budget(max_steps)
        self._control.ensure_ready()

    def _activate_rl(self) -> None:
        self._control.ensure_ready()
        status = self._control.activate_rl()
        self._wait_for_rl_active(status)

    def _run_rl_behavior(self, behavior: Callable[[], None]) -> None:
        try:
            self._activate_rl()
            behavior()
        except BaseException as exc:
            self._terminal_disarm_after_failure(exc, segment="RL")
            raise

    def _terminal_disarm_after_failure(
        self,
        failure: BaseException,
        *,
        segment: str,
    ) -> None:
        try:
            self._terminal_disarm_once()
        except Exception:
            raise RuntimeError(
                f"resident {segment} failed and terminal disarm also failed"
            ) from failure

    def _abort_after_failure(
        self,
        failure: BaseException,
        *,
        segment: str,
    ) -> None:
        errors: list[str] = []
        try:
            self._terminal_disarm_once()
        except Exception as exc:
            errors.append(f"terminal disarm: {exc}")
        try:
            self._cancel_prepared_dump()
        except Exception as exc:
            errors.append(f"prepared dump: {exc}")
        if errors:
            raise RuntimeError(
                f"resident {segment} failed and cleanup also failed: "
                + "; ".join(errors)
            ) from failure

    def _terminal_disarm_once(self) -> None:
        if self._terminal_disarmed:
            return
        self._control.terminal_disarm()
        self._terminal_disarmed = True

    def _cancel_prepared_dump(self) -> None:
        prepared = self._prepared_dump
        self._prepared_dump_started = False
        self._prepared_dump_triggered = False
        self._prepared_dump_refreshed = False
        if prepared is not None:
            prepared.cancel()

    def _wait_for_rl_active(self, status: ResidentControlStatus) -> None:
        deadline = self._monotonic() + self._handoff_timeout_s
        current = status
        while True:
            if current.rl_is_active:
                if (
                    current.is_operational
                    and current.phase == "active"
                    and current.active == _RL_BINDING
                    and current.target is None
                    and not current.act_is_active
                ):
                    return
                raise RuntimeError(
                    "resident status lacks an unambiguous active RL binding"
                )
            if not current.is_operational:
                raise RuntimeError("resident motion core is not operational")
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                raise TimeoutError("resident RL activation timed out")
            self._sleep(min(self._poll_interval_s, remaining))
            current = self._control.status()

    def _wait_for_act_completion(
        self,
        status: ResidentControlStatus,
        *,
        generation: int,
        max_steps: int,
        prepare_at_step: int | None,
        refresh_at_step: int | None,
        deadline: float,
    ) -> None:
        current = status
        while True:
            if not current.is_operational:
                raise RuntimeError("resident motion core is not operational")
            if current.act_segment_generation != generation:
                raise RuntimeError("resident ACT segment generation changed")
            if current.act_segment_max_steps != max_steps:
                raise RuntimeError("resident ACT segment step budget changed")
            if not current.act_worker_ready:
                raise RuntimeError("resident ACT worker disconnected")
            if (
                prepare_at_step is not None
                and self._prepared_dump_started
                and not self._prepared_dump_triggered
                and current.act_segment_completed_steps >= prepare_at_step
            ):
                assert self._prepared_dump is not None
                self._prepared_dump.trigger_prepare()
                self._prepared_dump_triggered = True
            if (
                refresh_at_step is not None
                and self._prepared_dump_started
                and self._prepared_dump_triggered
                and not self._prepared_dump_refreshed
                and current.act_segment_completed_steps >= refresh_at_step
            ):
                assert self._prepared_dump is not None
                self._prepared_dump.trigger_refresh()
                self._prepared_dump_refreshed = True
            if current.act_segment_complete:
                if current.act_segment_completed_steps != max_steps:
                    raise RuntimeError(
                        "resident ACT completed step count is inconsistent"
                    )
                if current.rl_is_active:
                    if (
                        current.phase != "active"
                        or current.active != _RL_BINDING
                        or current.target is not None
                        or current.act_is_active
                    ):
                        raise RuntimeError(
                            "resident ACT completion lacks an active RL binding"
                        )
                    return
                rl_handoff_is_pending = (
                    current.target == _RL_BINDING
                    and (
                        (
                            current.phase == "terminal_zero_pending"
                            and current.active == _ACT_BINDING
                        )
                        or (
                            current.phase == "target_zero_pending"
                            and current.active is None
                        )
                    )
                )
                if not rl_handoff_is_pending:
                    raise RuntimeError(
                        "ACT-to-RL handoff was revoked: "
                        + _control_status_summary(current)
                    )
            elif not current.act_is_active:
                activation_is_pending = (
                    current.act_segment_completed_steps == 0
                    and current.phase
                    in {"terminal_zero_pending", "target_zero_pending"}
                    and current.target == _ACT_BINDING
                )
                if not activation_is_pending:
                    raise RuntimeError("resident ACT authority was revoked")
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                raise TimeoutError("resident ACT segment timed out")
            self._sleep(min(self._poll_interval_s, remaining))
            current = self._control.status()


def _act_step_budget(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2000:
        raise ValueError("max_steps must be an integer in [1, 2000]")
    return value


def _control_status_summary(status: ResidentControlStatus) -> str:
    segment_state = "complete" if status.act_segment_complete else "running"
    return (
        f"phase={status.phase} generation={status.control_generation} "
        f"active={_binding_summary(status.active)} "
        f"target={_binding_summary(status.target)} "
        f"act_segment={status.act_segment_generation}:"
        f"{status.act_segment_completed_steps}/"
        f"{status.act_segment_max_steps}:{segment_state} "
        f"rl_active={status.rl_is_active} act_active={status.act_is_active} "
        f"lease_active={status.mission_lease_active}"
    )


def _binding_summary(binding: ResidentPolicyBinding | None) -> str:
    if binding is None:
        return "none"
    return f"{binding.source}/{binding.mode}"


def _absolute_posix_path(value: str | PurePosixPath, field: str) -> str:
    text = str(value)
    if not text or "\x00" in text or not PurePosixPath(text).is_absolute():
        raise ValueError(f"{field} must be an absolute NUL-free POSIX path")
    return text


def _parse_success_response(
    output: str,
    *,
    expected_command: str,
) -> ResidentControlStatus:
    try:
        if (
            not isinstance(output, str)
            or not output
            or len(output.encode("utf-8")) > _MAX_CLI_RESPONSE_BYTES
        ):
            raise ValueError("response size is invalid")
        value = json.loads(
            output,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_number,
        )
        if not isinstance(value, Mapping) or set(value) != _RESPONSE_FIELDS:
            raise ValueError("response fields are invalid")
        if value["schema_version"] != RESIDENT_CONTROL_SCHEMA_VERSION:
            raise ValueError("response schema is invalid")
        if value["ok"] is not True:
            raise ValueError("response is not successful")
        if value["command"] != expected_command:
            raise ValueError("response command does not match request")
        if value["error"] is not None:
            raise ValueError("successful response contains an error")
        return _parse_status(value["status"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("resident control returned an invalid response") from exc


def _parse_status(value: Any) -> ResidentControlStatus:
    if not isinstance(value, Mapping) or set(value) != _STATUS_FIELDS:
        raise ValueError("status fields are invalid")
    phase = value["phase"]
    if not isinstance(phase, str) or phase not in _PHASES:
        raise ValueError("status phase is invalid")
    latency = value["last_handoff_latency_ms"]
    if latency is not None:
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) < 0.0
        ):
            raise ValueError("status handoff latency is invalid")
        latency = float(latency)
    segment_max_steps = _optional_segment_max_steps(
        value["act_segment_max_steps"]
    )
    completed_steps = _uint64(value["act_segment_completed_steps"])
    if segment_max_steps is not None and completed_steps > segment_max_steps:
        raise ValueError("ACT completed steps exceed the segment budget")
    return ResidentControlStatus(
        phase=phase,
        control_generation=_uint64(value["control_generation"]),
        active=_parse_binding(value["active"]),
        target=_parse_binding(value["target"]),
        last_handoff_latency_ms=latency,
        rl_is_active=_boolean(value["rl_is_active"]),
        act_is_active=_boolean(value["act_is_active"]),
        act_worker_ready=_boolean(value["act_worker_ready"]),
        act_segment_generation=_optional_uint64(
            value["act_segment_generation"]
        ),
        act_segment_max_steps=segment_max_steps,
        act_segment_completed_steps=completed_steps,
        act_segment_complete=_boolean(value["act_segment_complete"]),
        mission_lease_active=_boolean(value["mission_lease_active"]),
        is_operational=_boolean(value["is_operational"]),
    )


def _parse_binding(value: Any) -> ResidentPolicyBinding | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"source", "mode"}:
        raise ValueError("policy binding fields are invalid")
    source = value["source"]
    mode = value["mode"]
    if (
        not isinstance(source, str)
        or not source
        or source != source.strip()
        or not isinstance(mode, str)
        or mode not in _CONTROL_MODES
    ):
        raise ValueError("policy binding is invalid")
    return ResidentPolicyBinding(source=source, mode=mode)


def _uint64(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF
    ):
        raise ValueError("generation/count must be an unsigned 64-bit integer")
    return value


def _optional_uint64(value: Any) -> int | None:
    return None if value is None else _uint64(value)


def _optional_segment_max_steps(value: Any) -> int | None:
    if value is None:
        return None
    return _act_step_budget(value)


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("status flag must be boolean")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("resident control JSON contains duplicate fields")
        result[key] = value
    return result


def _invalid_number(value: str) -> None:
    raise ValueError(f"invalid JSON number: {value}")
