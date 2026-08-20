"""Read-only AiryLidar machine-state output from Collector-owned telemetry."""

from __future__ import annotations

import json
import math
import socket
from typing import Any

from ..stm32_protocol import Stm32TelemetryFrame


class MachineStateUdpPublisher:
    """Translate parsed STM32 v2 telemetry to AiryLidar ``machine_state_v1``."""

    def __init__(self, *, host: str, port: int, machine_id: str) -> None:
        self._endpoint = (host, port)
        self._machine_id = machine_id
        self._sequence = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(
        self, frame: Stm32TelemetryFrame, *, receive_wall_ns: int
    ) -> None:
        packet = build_machine_state_packet(
            frame,
            sequence=self._sequence,
            machine_id=self._machine_id,
            receive_wall_ns=receive_wall_ns,
        )
        payload = json.dumps(
            packet, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self._socket.sendto(payload, self._endpoint)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF

    def close(self) -> None:
        self._socket.close()


def build_machine_state_packet(
    frame: Stm32TelemetryFrame,
    *,
    sequence: int,
    machine_id: str,
    receive_wall_ns: int,
) -> dict[str, Any]:
    """Build the established PC runtime bridge packet without another serial owner."""

    values = frame.values
    hardware_faults = tuple(
        name
        for name, field in (
            ("rs485_invalid", "rs485_ok"),
            ("adc_invalid", "dwj_ok"),
            ("imu_invalid", "imu_ok"),
        )
        if not bool(int(values[field]))
    )
    raw_fault_flags = int(values["fault_flags"])
    fault_flags = list(hardware_faults)
    if raw_fault_flags:
        fault_flags.append(f"stm32_fault_flags:0x{raw_fault_flags:08x}")

    radians = math.radians
    return {
        "type": "machine_state_v1",
        "schema_version": "1.0",
        "seq": sequence,
        "stamp_ms": receive_wall_ns // 1_000_000,
        "stm32_stamp_ms": int(values["control_stamp_ms"]),
        "source": "orin",
        "machine_id": machine_id,
        "safety": {
            "estop": bool(int(values["estop"])),
            "stm32_alive": True,
            "sensor_valid": frame.sensor_valid,
            "control_enabled": bool(int(values["control_enabled"])),
            "fault_flags": fault_flags,
        },
        "actuator_state": {
            "boom": {
                "position_m": float(values["boom_pos_mm"]) / 1000.0,
                "velocity_mps": float(values["boom_vel_mmps"]) / 1000.0,
            },
            "stick": {
                "position_m": float(values["stick_pos_mm"]) / 1000.0,
                "velocity_mps": float(values["stick_vel_mmps"]) / 1000.0,
            },
            "bucket": {
                "position_m": float(values["bucket_pos_mm"]) / 1000.0,
                "velocity_mps": float(values["bucket_vel_mmps"]) / 1000.0,
            },
            "swing": {
                "position_rad": radians(float(values["swing_angle_deg"])),
                "velocity_rad_s": radians(float(values["swing_vel_degps"])),
            },
        },
        "joint_state": {
            "position_rad": {
                "swing": radians(float(values["swing_angle_deg"])),
                "boom": radians(float(values["boom_angle_deg"])),
                "arm": radians(float(values["arm_angle_deg"])),
                "bucket": radians(float(values["bucket_angle_deg"])),
            },
            "velocity_rad_s": {
                "swing": radians(float(values["swing_vel_degps"])),
                "boom": 0.0,
                "arm": 0.0,
                "bucket": 0.0,
            },
        },
    }
