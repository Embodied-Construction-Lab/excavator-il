"""Versioned PC-to-Orin joystick protocol and expert-action mapping."""

from __future__ import annotations

import json
import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Any, Mapping


JOYSTICK_SCHEMA_VERSION = "excavator_joystick.v1"
AXIS_NAMES = ("X1", "Y1", "Z1", "X2", "Y2", "Z2")
AUTHENTICATION_FIELD = "authentication"


class JoystickProtocolError(ValueError):
    """Raised when a joystick datagram violates the public protocol."""


def authenticate_json_message(
    value: Mapping[str, Any], *, key: bytes, nonce: str
) -> bytes:
    """Bind a JSON message to one runtime nonce using HMAC-SHA256."""

    if len(key) < 32 or len(nonce) < 32:
        raise ValueError("HMAC key and runtime nonce are too short")
    unsigned = dict(value)
    unsigned.pop(AUTHENTICATION_FIELD, None)
    canonical = json.dumps(
        unsigned, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    signature = hmac.new(key, nonce.encode("ascii") + b"\0" + canonical, hashlib.sha256)
    authenticated = {
        **unsigned,
        AUTHENTICATION_FIELD: {
            "runtime_nonce": nonce,
            "hmac_sha256": signature.hexdigest(),
        },
    }
    return json.dumps(
        authenticated, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def verify_json_message(datagram: bytes, *, key: bytes, nonce: str) -> dict[str, Any]:
    """Verify HMAC and return the unsigned JSON object."""

    try:
        value = json.loads(datagram.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoystickProtocolError(f"invalid authenticated JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise JoystickProtocolError("authenticated message must be an object")
    authentication = value.pop(AUTHENTICATION_FIELD, None)
    if not isinstance(authentication, dict) or authentication.get("runtime_nonce") != nonce:
        raise JoystickProtocolError("runtime nonce mismatch")
    signature = authentication.get("hmac_sha256")
    if not isinstance(signature, str):
        raise JoystickProtocolError("HMAC is missing")
    expected = authenticate_json_message(value, key=key, nonce=nonce)
    expected_value = json.loads(expected)
    expected_signature = expected_value[AUTHENTICATION_FIELD]["hmac_sha256"]
    if not hmac.compare_digest(signature, expected_signature):
        raise JoystickProtocolError("HMAC verification failed")
    return value


@dataclass(frozen=True)
class ControllerIdentity:
    slot: int
    device_id: str
    name: str
    buttons: tuple[bool, ...]


@dataclass(frozen=True)
class JoystickPacket:
    session_id: str
    sample_seq: int
    pc_sample_monotonic_ns: int
    pc_sample_wall_ns: int
    axes: tuple[float, float, float, float, float, float]
    controllers: tuple[ControllerIdentity, ControllerIdentity]
    deadman_pressed: bool
    mapping_id: str
    calibration_id: str

    def axis(self, name: str) -> float:
        try:
            return self.axes[AXIS_NAMES.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc


@dataclass(frozen=True)
class ExpertAction:
    boom: float
    stick: float
    bucket: float
    swing: float
    valid: bool
    source_sample_seq: int
    mapping_id: str
    calibration_id: str

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.boom, self.stick, self.bucket, self.swing)


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise JoystickProtocolError(f"{key} must be an object")
    return child


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JoystickProtocolError(f"{field} must be a non-negative integer")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise JoystickProtocolError(f"{field} must be non-empty text")
    return value


def _axis(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JoystickProtocolError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not -1.0 <= number <= 1.0:
        raise JoystickProtocolError(f"{field} must be finite and within [-1, 1]")
    return number


def _controller(value: Any, expected_slot: int) -> ControllerIdentity:
    if not isinstance(value, Mapping):
        raise JoystickProtocolError("controllers entries must be objects")
    slot = _non_negative_int(value.get("slot"), "controllers.slot")
    if slot != expected_slot:
        raise JoystickProtocolError(f"controllers must contain slots 1 and 2 in order")
    raw_buttons = value.get("buttons")
    if not isinstance(raw_buttons, list) or any(not isinstance(item, bool) for item in raw_buttons):
        raise JoystickProtocolError("controllers.buttons must be a boolean list")
    return ControllerIdentity(
        slot=slot,
        device_id=_text(value.get("device_id"), "controllers.device_id"),
        name=_text(value.get("name"), "controllers.name"),
        buttons=tuple(raw_buttons),
    )


def decode_joystick_packet(datagram: bytes) -> JoystickPacket:
    """Decode one numeric joystick UDP datagram."""
    try:
        value = json.loads(datagram.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JoystickProtocolError(f"invalid joystick JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise JoystickProtocolError("joystick packet must be an object")
    if value.get("schema_version") != JOYSTICK_SCHEMA_VERSION:
        raise JoystickProtocolError(
            f"schema_version must be {JOYSTICK_SCHEMA_VERSION}"
        )

    axes = _required_mapping(value, "axes")
    raw_controllers = value.get("controllers")
    if not isinstance(raw_controllers, list) or len(raw_controllers) != 2:
        raise JoystickProtocolError("controllers must contain exactly two devices")
    deadman = value.get("deadman_pressed")
    if not isinstance(deadman, bool):
        raise JoystickProtocolError("deadman_pressed must be boolean")

    return JoystickPacket(
        session_id=_text(value.get("session_id"), "session_id"),
        sample_seq=_non_negative_int(value.get("sample_seq"), "sample_seq"),
        pc_sample_monotonic_ns=_non_negative_int(
            value.get("pc_sample_monotonic_ns"), "pc_sample_monotonic_ns"
        ),
        pc_sample_wall_ns=_non_negative_int(
            value.get("pc_sample_wall_ns"), "pc_sample_wall_ns"
        ),
        axes=tuple(_axis(axes.get(name), f"axes.{name}") for name in AXIS_NAMES),
        controllers=(
            _controller(raw_controllers[0], 1),
            _controller(raw_controllers[1], 2),
        ),
        deadman_pressed=deadman,
        mapping_id=_text(value.get("mapping_id"), "mapping_id"),
        calibration_id=_text(value.get("calibration_id"), "calibration_id"),
    )


def encode_joystick_packet(packet: JoystickPacket) -> bytes:
    """Encode a validated joystick packet without rounding axis values."""
    value = {
        "schema_version": JOYSTICK_SCHEMA_VERSION,
        "session_id": packet.session_id,
        "sample_seq": packet.sample_seq,
        "pc_sample_monotonic_ns": packet.pc_sample_monotonic_ns,
        "pc_sample_wall_ns": packet.pc_sample_wall_ns,
        "axes": dict(zip(AXIS_NAMES, packet.axes, strict=True)),
        "controllers": [
            {
                "slot": controller.slot,
                "device_id": controller.device_id,
                "name": controller.name,
                "buttons": list(controller.buttons),
            }
            for controller in packet.controllers
        ],
        "deadman_pressed": packet.deadman_pressed,
        "mapping_id": packet.mapping_id,
        "calibration_id": packet.calibration_id,
    }
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _apply_deadzone(value: float, deadzone: float) -> float:
    return 0.0 if abs(value) <= deadzone else value


def map_expert_action(packet: JoystickPacket, *, deadzone: float = 0.15) -> ExpertAction:
    """Map raw axes to authoritative ``[boom, stick, bucket, swing]`` order."""
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be within [0, 1)")
    values = tuple(
        _apply_deadzone(packet.axis(name), deadzone)
        for name in ("Y2", "Y1", "X2", "X1")
    )
    return ExpertAction(
        boom=values[0],
        stick=values[1],
        bucket=values[2],
        swing=values[3],
        valid=packet.deadman_pressed,
        source_sample_seq=packet.sample_seq,
        mapping_id=packet.mapping_id,
        calibration_id=packet.calibration_id,
    )
