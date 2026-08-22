"""Strict ACT-worker side of the local Resident Mission Runtime protocol.

Its wire format mirrors ``excavator-orin-runtime/edge_runtime`` so either side
can reject malformed or semantically ambiguous frames before they reach a
policy or motion boundary.  The local Unix client carries only those frames;
this module deliberately has no model, camera, or serial imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import select
import socket
import threading
import time
from typing import Any


RESIDENT_STATE_SCHEMA_VERSION = "resident_act_state.v1"
CANDIDATE_SCHEMA_VERSION = "resident_policy_candidate.v1"
MAX_FRAME_BYTES = 4096
UINT32_MAX = 0xFFFFFFFF
UINT64_MAX = 0xFFFFFFFFFFFFFFFF

ACT_STATE_NAMES = (
    "boom_pos_m",
    "stick_pos_m",
    "bucket_pos_m",
    "boom_vel_mps",
    "stick_vel_mps",
    "bucket_vel_mps",
    "boom_angle_rad",
    "arm_angle_rad",
    "bucket_angle_rad",
    "swing_angle_rad",
    "swing_vel_radps",
)
ACTION_ORDER = ("boom", "stick", "bucket", "swing")
ACT_POLICY_SOURCE = "act_dig"
ACT_CONTROL_MODE = "manual_action"
_HEADER_BYTES = 4
_UNIX_PATH_MAX_BYTES = 107
_SEND_TIMEOUT_S = 0.1

_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "state_names",
        "state",
        "receive_monotonic_ns",
        "state_monotonic_ns",
        "control_seq",
        "sensor_seq",
        "sensor_is_new",
        "control_enabled",
        "estop",
        "rs485_ok",
        "dwj_ok",
        "imu_ok",
        "sensor_valid",
        "stm32_alive",
        "fault_flags",
        "control_generation",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "control_generation",
        "mode",
        "action_order",
        "action",
        "created_monotonic_ns",
        "valid_until_monotonic_ns",
    }
)


@dataclass(frozen=True)
class ResidentActState:
    """One immutable canonical ACT state plus current safety evidence."""

    state: tuple[float, ...]
    receive_monotonic_ns: int
    state_monotonic_ns: int
    control_seq: int
    sensor_seq: int
    sensor_is_new: bool
    control_enabled: bool
    estop: bool
    rs485_ok: bool
    dwj_ok: bool
    imu_ok: bool
    sensor_valid: bool
    stm32_alive: bool
    fault_flags: int
    control_generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, tuple) or len(self.state) != len(ACT_STATE_NAMES):
            raise ValueError("resident ACT state must be the canonical 11-value tuple")
        state = tuple(
            _finite_number(f"state[{index}]", value)
            for index, value in enumerate(self.state)
        )
        receive_ns = _uint64("receive_monotonic_ns", self.receive_monotonic_ns)
        state_ns = _uint64("state_monotonic_ns", self.state_monotonic_ns)
        if state_ns > receive_ns:
            raise ValueError("state_monotonic_ns must not exceed receive_monotonic_ns")
        _uint32("control_seq", self.control_seq)
        _uint32("sensor_seq", self.sensor_seq)
        _uint32("fault_flags", self.fault_flags)
        _uint64("control_generation", self.control_generation)
        for name in (
            "sensor_is_new",
            "control_enabled",
            "estop",
            "rs485_ok",
            "dwj_ok",
            "imu_ok",
            "sensor_valid",
            "stm32_alive",
        ):
            _boolean(name, getattr(self, name))
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class ResidentPolicyCandidate:
    """One policy output carrying its current motion-authority generation."""

    source: str
    control_generation: int
    mode: str
    action: tuple[float, float, float, float]
    created_monotonic_ns: int
    valid_until_monotonic_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("candidate source must be a non-empty string")
        if self.source.strip() != ACT_POLICY_SOURCE:
            raise ValueError(f"candidate source must be {ACT_POLICY_SOURCE}")
        if self.mode != ACT_CONTROL_MODE:
            raise ValueError(f"candidate mode must be {ACT_CONTROL_MODE}")
        _uint64("control_generation", self.control_generation)
        created_ns = _uint64("created_monotonic_ns", self.created_monotonic_ns)
        valid_until_ns = _uint64(
            "valid_until_monotonic_ns", self.valid_until_monotonic_ns
        )
        if valid_until_ns < created_ns:
            raise ValueError("candidate validity must not end before creation")
        if not isinstance(self.action, tuple) or len(self.action) != len(ACTION_ORDER):
            raise ValueError("candidate action must contain four values")
        action = tuple(
            _finite_number(f"action[{index}]", value)
            for index, value in enumerate(self.action)
        )
        object.__setattr__(self, "source", self.source.strip())
        object.__setattr__(self, "action", action)


def encode_resident_state(frame: ResidentActState) -> bytes:
    if not isinstance(frame, ResidentActState):
        raise ValueError("resident state must be a ResidentActState")
    payload = {
        "schema_version": RESIDENT_STATE_SCHEMA_VERSION,
        "state_names": list(ACT_STATE_NAMES),
        "state": list(frame.state),
        "receive_monotonic_ns": frame.receive_monotonic_ns,
        "state_monotonic_ns": frame.state_monotonic_ns,
        "control_seq": frame.control_seq,
        "sensor_seq": frame.sensor_seq,
        "sensor_is_new": frame.sensor_is_new,
        "control_enabled": frame.control_enabled,
        "estop": frame.estop,
        "rs485_ok": frame.rs485_ok,
        "dwj_ok": frame.dwj_ok,
        "imu_ok": frame.imu_ok,
        "sensor_valid": frame.sensor_valid,
        "stm32_alive": frame.stm32_alive,
        "fault_flags": frame.fault_flags,
        "control_generation": frame.control_generation,
    }
    return _encode_json(payload, kind="resident state")


def decode_resident_state(payload: bytes) -> ResidentActState:
    value = _decode_json(payload, kind="resident state")
    if set(value) != _STATE_FIELDS:
        raise ValueError("resident state fields are invalid")
    if value["schema_version"] != RESIDENT_STATE_SCHEMA_VERSION:
        raise ValueError("resident state schema_version is unsupported")
    if value["state_names"] != list(ACT_STATE_NAMES):
        raise ValueError("resident state_names must match the canonical ACT order")
    raw_state = value["state"]
    if not isinstance(raw_state, list) or len(raw_state) != len(ACT_STATE_NAMES):
        raise ValueError("resident ACT state must contain exactly 11 values")
    return ResidentActState(
        state=tuple(
            _finite_number(f"state[{index}]", item)
            for index, item in enumerate(raw_state)
        ),
        receive_monotonic_ns=_uint64(
            "receive_monotonic_ns", value["receive_monotonic_ns"]
        ),
        state_monotonic_ns=_uint64(
            "state_monotonic_ns", value["state_monotonic_ns"]
        ),
        control_seq=_uint32("control_seq", value["control_seq"]),
        sensor_seq=_uint32("sensor_seq", value["sensor_seq"]),
        sensor_is_new=_boolean("sensor_is_new", value["sensor_is_new"]),
        control_enabled=_boolean("control_enabled", value["control_enabled"]),
        estop=_boolean("estop", value["estop"]),
        rs485_ok=_boolean("rs485_ok", value["rs485_ok"]),
        dwj_ok=_boolean("dwj_ok", value["dwj_ok"]),
        imu_ok=_boolean("imu_ok", value["imu_ok"]),
        sensor_valid=_boolean("sensor_valid", value["sensor_valid"]),
        stm32_alive=_boolean("stm32_alive", value["stm32_alive"]),
        fault_flags=_uint32("fault_flags", value["fault_flags"]),
        control_generation=_uint64(
            "control_generation", value["control_generation"]
        ),
    )


def encode_policy_candidate(candidate: ResidentPolicyCandidate) -> bytes:
    if not isinstance(candidate, ResidentPolicyCandidate):
        raise ValueError("candidate must be a ResidentPolicyCandidate")
    payload = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "source": candidate.source,
        "control_generation": candidate.control_generation,
        "mode": candidate.mode,
        "action_order": list(ACTION_ORDER),
        "action": list(candidate.action),
        "created_monotonic_ns": candidate.created_monotonic_ns,
        "valid_until_monotonic_ns": candidate.valid_until_monotonic_ns,
    }
    return _encode_json(payload, kind="candidate")


def decode_policy_candidate(payload: bytes) -> ResidentPolicyCandidate:
    value = _decode_json(payload, kind="candidate")
    if set(value) != _CANDIDATE_FIELDS:
        raise ValueError("candidate payload fields are invalid")
    if value["schema_version"] != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("candidate schema_version is unsupported")
    if value["action_order"] != list(ACTION_ORDER):
        raise ValueError("candidate action_order must be canonical")
    raw_action = value["action"]
    if not isinstance(raw_action, list) or len(raw_action) != len(ACTION_ORDER):
        raise ValueError("candidate action must contain four values")
    return ResidentPolicyCandidate(
        source=_text("source", value["source"]),
        control_generation=_uint64(
            "control_generation", value["control_generation"]
        ),
        mode=_text("mode", value["mode"]),
        action=tuple(
            _finite_number(f"action[{index}]", item)
            for index, item in enumerate(raw_action)
        ),
        created_monotonic_ns=_uint64(
            "created_monotonic_ns", value["created_monotonic_ns"]
        ),
        valid_until_monotonic_ns=_uint64(
            "valid_until_monotonic_ns", value["valid_until_monotonic_ns"]
        ),
    )


class ResidentActDataClient:
    """Strict framed client for the single owner-side ACT data link."""

    def __init__(self, socket_path: str | os.PathLike[str]) -> None:
        path = Path(socket_path)
        if not path.is_absolute():
            raise ValueError("resident ACT socket path must be absolute")
        if len(os.fsencode(path)) > _UNIX_PATH_MAX_BYTES:
            raise ValueError("resident ACT socket path exceeds the Unix limit")
        self._path = path
        self._connection: socket.socket | None = None
        self._receive_buffer = bytearray()
        self._expected_payload_bytes: int | None = None
        self._send_lock = threading.Lock()
        self._receive_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def connect(self, *, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("resident ACT connect timeout must be positive")
        if self._connection is not None:
            return
        deadline = time.monotonic() + timeout_s
        last_error: OSError | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "resident ACT owner socket is unavailable"
                ) from last_error
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.settimeout(remaining)
                connection.connect(os.fspath(self._path))
                connection.setblocking(False)
            except OSError as exc:
                connection.close()
                last_error = exc
                if exc.errno not in (errno.ENOENT, errno.ECONNREFUSED):
                    raise ConnectionError("cannot connect to resident ACT owner") from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "resident ACT owner socket is unavailable"
                    ) from last_error
                time.sleep(min(0.02, remaining))
                continue
            self._connection = connection
            self._receive_buffer.clear()
            self._expected_payload_bytes = None
            return

    def receive_state(self, *, timeout_s: float) -> ResidentActState | None:
        if timeout_s <= 0:
            raise ValueError("resident ACT receive timeout must be positive")
        with self._receive_lock:
            connection = self._require_connection()
            deadline = time.monotonic() + timeout_s
            while True:
                payload = self._consume_complete_frame()
                if payload is not None:
                    return decode_resident_state(payload)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                readable, _, _ = select.select([connection], [], [], remaining)
                if not readable:
                    return None
                try:
                    chunk = connection.recv(MAX_FRAME_BYTES + _HEADER_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise ConnectionError("resident ACT owner closed the data link")
                self._receive_buffer.extend(chunk)

    def send_candidate(self, candidate: ResidentPolicyCandidate) -> None:
        payload = encode_policy_candidate(candidate)
        framed = len(payload).to_bytes(_HEADER_BYTES, "big") + payload
        with self._send_lock:
            connection = self._require_connection()
            deadline = time.monotonic() + _SEND_TIMEOUT_S
            offset = 0
            while offset < len(framed):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConnectionError("ACT candidate send timed out")
                try:
                    _, writable, _ = select.select([], [connection], [], remaining)
                    if not writable:
                        raise ConnectionError("ACT candidate send timed out")
                    sent = connection.send(memoryview(framed)[offset:])
                except BlockingIOError:
                    continue
                except OSError as exc:
                    raise ConnectionError(
                        "cannot send ACT candidate to resident owner"
                    ) from exc
                if sent <= 0:
                    raise ConnectionError("resident ACT owner closed the data link")
                offset += sent

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._receive_buffer.clear()
        self._expected_payload_bytes = None
        if connection is None:
            return
        try:
            connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        connection.close()

    def _require_connection(self) -> socket.socket:
        connection = self._connection
        if connection is None:
            raise RuntimeError("resident ACT data client is not connected")
        return connection

    def _consume_complete_frame(self) -> bytes | None:
        if self._expected_payload_bytes is None:
            if len(self._receive_buffer) < _HEADER_BYTES:
                return None
            self._expected_payload_bytes = int.from_bytes(
                self._receive_buffer[:_HEADER_BYTES], "big"
            )
            del self._receive_buffer[:_HEADER_BYTES]
            if not 0 < self._expected_payload_bytes <= MAX_FRAME_BYTES:
                raise ValueError("resident ACT state frame length is invalid")
        if len(self._receive_buffer) < self._expected_payload_bytes:
            return None
        payload = bytes(self._receive_buffer[: self._expected_payload_bytes])
        del self._receive_buffer[: self._expected_payload_bytes]
        self._expected_payload_bytes = None
        return payload


def _encode_json(value: dict[str, Any], *, kind: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{kind} cannot be encoded as finite JSON") from exc
    if not encoded or len(encoded) > MAX_FRAME_BYTES:
        raise ValueError(f"{kind} payload size is invalid")
    return encoded


def _decode_json(payload: bytes, *, kind: str) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError(f"{kind} payload size is invalid")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{kind} is not strict finite JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field is forbidden: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _boolean(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"resident state {name} must be boolean")
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate {name} must be a non-empty string")
    return value.strip()


def _uint32(name: str, value: Any) -> int:
    return _bounded_integer(name, value, maximum=UINT32_MAX)


def _uint64(name: str, value: Any) -> int:
    return _bounded_integer(name, value, maximum=UINT64_MAX)


def _bounded_integer(name: str, value: Any, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"resident state {name} is outside its integer range")
    return value


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resident state {name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"resident state {name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"resident state {name} must be a finite number")
    return number
