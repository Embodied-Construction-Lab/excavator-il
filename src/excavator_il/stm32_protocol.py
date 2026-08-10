"""Strict parser for the STM32 demonstration-control telemetry contract."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


STM32_TELEMETRY_SCHEMA_VERSION = "stm32_control_telemetry.v2"
STM32_TELEMETRY_FIELDS = (
    "schema_version",
    "control_seq",
    "control_stamp_ms",
    "sensor_seq",
    "sensor_stamp_ms",
    "sensor_is_new",
    "command_rx_seq",
    "command_source_stamp_ms",
    "command_received_stamp_ms",
    "command_age_ms",
    "command_action_boom",
    "command_action_stick",
    "command_action_bucket",
    "command_action_swing",
    "boom_pos_mm",
    "stick_pos_mm",
    "bucket_pos_mm",
    "boom_vel_mmps",
    "stick_vel_mmps",
    "bucket_vel_mmps",
    "boom_angle_deg",
    "arm_angle_deg",
    "bucket_angle_deg",
    "swing_angle_deg",
    "swing_vel_degps",
    "boom_v_ref_mmps",
    "stick_v_ref_mmps",
    "bucket_v_ref_mmps",
    "swing_v_ref_degps",
    "pid_out_boom",
    "pid_out_stick",
    "pid_out_bucket",
    "pid_out_swing",
    "valve_boom_deg",
    "valve_stick_deg",
    "valve_bucket_deg",
    "swing_percent",
    "pump_percent",
    "pwm_boom",
    "pwm_stick",
    "pwm_bucket",
    "pwm_swing",
    "pwm_pump",
    "control_mode",
    "homing_complete",
    "command_valid",
    "command_timed_out",
    "control_enabled",
    "estop",
    "limit_mask",
    "rs485_ok",
    "dwj_ok",
    "imu_ok",
    "fault_flags",
    "dropped_command_frames",
)

_INTEGER_FIELDS = frozenset(
    {
        "control_seq",
        "control_stamp_ms",
        "sensor_seq",
        "sensor_stamp_ms",
        "sensor_is_new",
        "command_rx_seq",
        "command_source_stamp_ms",
        "command_received_stamp_ms",
        "command_age_ms",
        "pwm_boom",
        "pwm_stick",
        "pwm_bucket",
        "pwm_swing",
        "pwm_pump",
        "control_mode",
        "homing_complete",
        "command_valid",
        "command_timed_out",
        "control_enabled",
        "estop",
        "limit_mask",
        "rs485_ok",
        "dwj_ok",
        "imu_ok",
        "fault_flags",
        "dropped_command_frames",
    }
)


class Stm32ProtocolError(ValueError):
    """Raised when STM32 telemetry cannot be accepted under the v2 schema."""


@dataclass(frozen=True)
class Stm32TelemetryFrame:
    receive_monotonic_ns: int
    values: Mapping[str, int | float | str]

    @property
    def control_seq(self) -> int:
        return int(self.values["control_seq"])

    @property
    def sensor_seq(self) -> int:
        return int(self.values["sensor_seq"])

    @property
    def sensor_is_new(self) -> bool:
        return bool(self.values["sensor_is_new"])

    @property
    def command_action(self) -> tuple[float, float, float, float]:
        return tuple(
            float(self.values[field])
            for field in (
                "command_action_boom",
                "command_action_stick",
                "command_action_bucket",
                "command_action_swing",
            )
        )

    @property
    def sensor_valid(self) -> bool:
        return all(bool(self.values[field]) for field in ("rs485_ok", "dwj_ok", "imu_ok"))


class Stm32TelemetryParser:
    """Parse the one-time CSV header followed by strict v2 telemetry rows."""

    def __init__(self) -> None:
        self._header_seen = False

    def parse_line(
        self, raw_line: bytes, *, receive_monotonic_ns: int
    ) -> Stm32TelemetryFrame | None:
        try:
            text = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise Stm32ProtocolError("STM32 telemetry must be ASCII") from exc
        if not text:
            raise Stm32ProtocolError("STM32 telemetry line is empty")
        try:
            fields = next(csv.reader([text]))
        except (csv.Error, StopIteration) as exc:
            raise Stm32ProtocolError(f"invalid STM32 CSV: {exc}") from exc

        if tuple(fields) == STM32_TELEMETRY_FIELDS:
            self._header_seen = True
            return None
        if len(fields) != len(STM32_TELEMETRY_FIELDS):
            raise Stm32ProtocolError(
                f"STM32 row has {len(fields)} fields; expected {len(STM32_TELEMETRY_FIELDS)}"
            )
        raw_values = dict(zip(STM32_TELEMETRY_FIELDS, fields, strict=True))
        if raw_values["schema_version"] != STM32_TELEMETRY_SCHEMA_VERSION:
            raise Stm32ProtocolError(
                f"unsupported STM32 schema: {raw_values['schema_version']}"
            )

        parsed: dict[str, int | float | str] = {
            "schema_version": raw_values["schema_version"]
        }
        for field in STM32_TELEMETRY_FIELDS[1:]:
            try:
                value: int | float
                if field in _INTEGER_FIELDS:
                    value = int(raw_values[field])
                    if value < 0:
                        raise ValueError
                else:
                    value = float(raw_values[field])
                    if not math.isfinite(value):
                        raise ValueError
            except ValueError as exc:
                raise Stm32ProtocolError(
                    f"invalid value for {field}: {raw_values[field]!r}"
                ) from exc
            parsed[field] = value
        return Stm32TelemetryFrame(
            receive_monotonic_ns=receive_monotonic_ns,
            values=MappingProxyType(parsed),
        )
