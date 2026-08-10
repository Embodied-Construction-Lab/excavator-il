"""Pure collector decisions between UDP joystick packets and STM32 commands."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..joystick_protocol import (
    AXIS_NAMES,
    JoystickPacket,
    JoystickProtocolError,
    decode_joystick_packet,
    map_expert_action,
)
from .recorder import EpisodeRecorder
from ..stm32_protocol import (
    Stm32ProtocolError,
    Stm32TelemetryFrame,
    Stm32TelemetryParser,
)


@dataclass(frozen=True)
class CollectorDecision:
    accepted: bool
    reason: str
    sample_seq: int | None
    serial_payload: bytes | None
    ack_payload: bytes
    command_seq: int | None
    action_seq: int | None
    command_kind: str | None


class CollectorCore:
    """Validate, record and translate joystick packets without doing I/O."""

    def __init__(
        self,
        *,
        recorder: EpisodeRecorder,
        expected_device_ids: tuple[str, str],
        mapping_id: str,
        calibration_id: str,
        deadzone: float,
    ) -> None:
        self._recorder = recorder
        self._expected_device_ids = expected_device_ids
        self._mapping_id = mapping_id
        self._calibration_id = calibration_id
        self._deadzone = deadzone
        self._last_sequences: dict[tuple[str, str], int] = {}
        self._action_seq = 0
        self._command_seq = 0
        self._stm32_raw_frame_seq = 0
        self._stm32_parser = Stm32TelemetryParser()

    @staticmethod
    def _source_host(source_addr: str) -> str:
        return source_addr.rsplit(":", maxsplit=1)[0]

    @staticmethod
    def _ack(
        *, sample_seq: int | None, accepted: bool, reason: str, receive_monotonic_ns: int
    ) -> bytes:
        return json.dumps(
            {
                "schema_version": "excavator_joystick_ack.v1",
                "sample_seq": sample_seq,
                "accepted": accepted,
                "reason": reason,
                "orin_receive_monotonic_ns": receive_monotonic_ns,
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def _record_raw(
        self,
        *,
        datagram: bytes,
        source_addr: str,
        receive_monotonic_ns: int,
        receive_wall_ns: int,
        parse_ok: bool,
        parse_error: str,
        packet: JoystickPacket | None,
    ) -> None:
        self._recorder.record_json(
            "joystick_raw",
            {
                "episode_id": self._recorder.episode_id,
                "orin_receive_monotonic_ns": receive_monotonic_ns,
                "orin_receive_wall_ns": receive_wall_ns,
                "source_addr": source_addr,
                "raw_payload": datagram.decode("utf-8", errors="replace"),
                "parse_ok": parse_ok,
                "parse_error": parse_error,
                "session_id": None if packet is None else packet.session_id,
                "joystick_sample_seq": None if packet is None else packet.sample_seq,
            },
        )

    def _validate_contract(self, packet: JoystickPacket) -> None:
        device_ids = tuple(controller.device_id for controller in packet.controllers)
        if device_ids != self._expected_device_ids:
            raise JoystickProtocolError(
                f"controller identity mismatch: expected {self._expected_device_ids}, got {device_ids}"
            )
        if packet.mapping_id != self._mapping_id:
            raise JoystickProtocolError("mapping_id does not match collector configuration")
        if packet.calibration_id != self._calibration_id:
            raise JoystickProtocolError("calibration_id does not match collector configuration")

    def accept_joystick(
        self,
        datagram: bytes,
        *,
        source_addr: str,
        receive_monotonic_ns: int,
        receive_wall_ns: int,
    ) -> CollectorDecision:
        try:
            packet = decode_joystick_packet(datagram)
            self._validate_contract(packet)
        except JoystickProtocolError as exc:
            self._record_raw(
                datagram=datagram,
                source_addr=source_addr,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_wall_ns=receive_wall_ns,
                parse_ok=False,
                parse_error=str(exc),
                packet=None,
            )
            return CollectorDecision(
                accepted=False,
                reason="invalid_packet",
                sample_seq=None,
                serial_payload=None,
                ack_payload=self._ack(
                    sample_seq=None,
                    accepted=False,
                    reason="invalid_packet",
                    receive_monotonic_ns=receive_monotonic_ns,
                ),
                command_seq=None,
                action_seq=None,
                command_kind=None,
            )

        self._record_raw(
            datagram=datagram,
            source_addr=source_addr,
            receive_monotonic_ns=receive_monotonic_ns,
            receive_wall_ns=receive_wall_ns,
            parse_ok=True,
            parse_error="",
            packet=packet,
        )
        sequence_key = (self._source_host(source_addr), packet.session_id)
        previous = self._last_sequences.get(sequence_key)
        if previous is not None and packet.sample_seq <= previous:
            reason = "duplicate_or_out_of_order"
            return CollectorDecision(
                accepted=False,
                reason=reason,
                sample_seq=packet.sample_seq,
                serial_payload=None,
                ack_payload=self._ack(
                    sample_seq=packet.sample_seq,
                    accepted=False,
                    reason=reason,
                    receive_monotonic_ns=receive_monotonic_ns,
                ),
                command_seq=None,
                action_seq=None,
                command_kind=None,
            )
        self._last_sequences[sequence_key] = packet.sample_seq

        action = map_expert_action(packet, deadzone=self._deadzone)
        self._recorder.record_json(
            "expert_action",
            {
                "episode_id": self._recorder.episode_id,
                "action_seq": self._action_seq,
                "source_joystick_sample_seq": packet.sample_seq,
                "action_stamp_monotonic_ns": receive_monotonic_ns,
                "action_boom": action.boom,
                "action_stick": action.stick,
                "action_bucket": action.bucket,
                "action_swing": action.swing,
                "action_valid": action.valid,
                "mapping_id": action.mapping_id,
                "calibration_id": action.calibration_id,
            },
        )

        command_axes = packet.axes if packet.deadman_pressed else (0.0,) * 6
        command_seq = self._command_seq
        action_seq = self._action_seq
        command_kind = "manual" if packet.deadman_pressed else "safe_zero:deadman_released"
        command = {
            "schema_version": "stm32_manual_command.v1",
            **dict(zip(AXIS_NAMES, command_axes, strict=True)),
            "command_seq": command_seq,
            "command_source_stamp_ms": (receive_monotonic_ns // 1_000_000) & 0xFFFFFFFF,
        }
        serial_payload = (
            json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("ascii")
        self._action_seq += 1
        self._command_seq = (self._command_seq + 1) & 0xFFFFFFFF
        return CollectorDecision(
            accepted=True,
            reason="accepted",
            sample_seq=packet.sample_seq,
            serial_payload=serial_payload,
            ack_payload=self._ack(
                sample_seq=packet.sample_seq,
                accepted=True,
                reason="accepted",
                receive_monotonic_ns=receive_monotonic_ns,
            ),
            command_seq=command_seq,
            action_seq=action_seq,
            command_kind=command_kind,
        )

    def make_safe_zero(self, *, monotonic_ns: int, reason: str) -> CollectorDecision:
        if not reason:
            raise ValueError("safe-zero reason must be non-empty")
        command_seq = self._command_seq
        command = {
            "schema_version": "stm32_manual_command.v1",
            **dict.fromkeys(AXIS_NAMES, 0.0),
            "command_seq": command_seq,
            "command_source_stamp_ms": (monotonic_ns // 1_000_000) & 0xFFFFFFFF,
        }
        payload = (
            json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("ascii")
        self._command_seq = (self._command_seq + 1) & 0xFFFFFFFF
        return CollectorDecision(
            accepted=True,
            reason=reason,
            sample_seq=None,
            serial_payload=payload,
            ack_payload=b"",
            command_seq=command_seq,
            action_seq=None,
            command_kind=f"safe_zero:{reason}",
        )

    def record_command_result(
        self,
        decision: CollectorDecision,
        *,
        tx_monotonic_ns: int,
        write_ok: bool,
        write_error: str,
    ) -> None:
        if decision.serial_payload is None or decision.command_seq is None:
            raise ValueError("decision contains no serial command")
        self._recorder.record_json(
            "command_tx",
            {
                "episode_id": self._recorder.episode_id,
                "command_seq": decision.command_seq,
                "source_action_seq": decision.action_seq,
                "command_tx_monotonic_ns": int(tx_monotonic_ns),
                "command_kind": decision.command_kind,
                "raw_serial_payload": decision.serial_payload.decode("ascii").rstrip("\n"),
                "write_ok": bool(write_ok),
                "write_error": str(write_error),
            },
        )

    def accept_stm32(
        self,
        raw_line: bytes,
        *,
        receive_monotonic_ns: int,
        receive_wall_ns: int,
    ) -> Stm32TelemetryFrame | None:
        raw_frame_seq = self._stm32_raw_frame_seq
        self._stm32_raw_frame_seq += 1
        try:
            frame = self._stm32_parser.parse_line(
                raw_line, receive_monotonic_ns=receive_monotonic_ns
            )
        except Stm32ProtocolError as exc:
            self._recorder.record_json(
                "stm32_raw",
                {
                    "episode_id": self._recorder.episode_id,
                    "raw_frame_seq": raw_frame_seq,
                    "orin_receive_monotonic_ns": receive_monotonic_ns,
                    "orin_receive_wall_ns": receive_wall_ns,
                    "raw_payload": raw_line.decode("ascii", errors="replace").rstrip("\r\n"),
                    "parse_ok": False,
                    "parse_error": str(exc),
                    "telemetry": None,
                },
            )
            return None

        self._recorder.record_json(
            "stm32_raw",
            {
                "episode_id": self._recorder.episode_id,
                "raw_frame_seq": raw_frame_seq,
                "orin_receive_monotonic_ns": receive_monotonic_ns,
                "orin_receive_wall_ns": receive_wall_ns,
                "raw_payload": raw_line.decode("ascii").rstrip("\r\n"),
                "parse_ok": True,
                "parse_error": "",
                "telemetry": None if frame is None else dict(frame.values),
            },
        )
        if frame is not None:
            self._recorder.record_control(
                raw_frame_seq=raw_frame_seq,
                receive_monotonic_ns=receive_monotonic_ns,
                telemetry=frame.values,
            )
        return frame
