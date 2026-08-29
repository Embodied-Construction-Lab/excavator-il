"""Causal camera/state joining and operator telemetry for resident ACT.

This private module keeps the observation-facing logic together while the
public resident runtime remains responsible for policy execution and worker
lifecycle.
"""

from __future__ import annotations

from collections import deque
import math
import threading

from .act_runtime import ActObservation
from .collector.camera import RgbCameraFrame
from .resident_protocol import ResidentActState


class ResidentCausalObservationBuffer:
    """Join the named resident state to the latest non-future camera frame."""

    def __init__(
        self,
        *,
        camera_roles: tuple[str, ...] = ("front",),
        capacity: int = 8,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("camera buffer capacity must be a positive integer")
        if (
            not isinstance(camera_roles, tuple)
            or not camera_roles
            or camera_roles[0] != "front"
            or len(set(camera_roles)) != len(camera_roles)
            or not set(camera_roles) <= {"front", "dump"}
        ):
            raise ValueError("resident camera roles must be front and optional dump")
        self._roles = camera_roles
        self._frames = {
            role: deque(maxlen=capacity) for role in camera_roles
        }
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def add_camera(self, frame: RgbCameraFrame, *, role: str = "front") -> None:
        if not isinstance(frame, RgbCameraFrame):
            raise ValueError("resident camera frame has the wrong type")
        if role not in self._frames:
            raise ValueError("resident camera role is not configured")
        with self._lock:
            frames = self._frames[role]
            if frames and (
                frame.capture_monotonic_ns
                <= frames[-1].capture_monotonic_ns
            ):
                raise ValueError(
                    f"{role} camera timestamps must be strictly increasing"
                )
            frames.append(frame)
            if all(self._frames[item] for item in self._roles):
                self._ready.set()

    def wait_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    def build(self, state: ResidentActState) -> ActObservation:
        with self._lock:
            frames_by_role = {
                role: tuple(frames) for role, frames in self._frames.items()
            }
        selected: dict[str, RgbCameraFrame] = {}
        for role in self._roles:
            camera = next(
                (
                    frame
                    for frame in reversed(frames_by_role[role])
                    if frame.capture_monotonic_ns <= state.state_monotonic_ns
                ),
                None,
            )
            if camera is None:
                raise ValueError(
                    f"no causal {role} camera frame is available for resident state"
                )
            selected[role] = camera
        front = selected["front"]
        extra = {role: selected[role] for role in self._roles if role != "front"}
        return ActObservation(
            state=state.state,
            front_rgb=front.rgb,
            state_monotonic_ns=state.state_monotonic_ns,
            camera_monotonic_ns=front.capture_monotonic_ns,
            extra_rgb_by_role={role: frame.rgb for role, frame in extra.items()},
            extra_camera_monotonic_ns_by_role={
                role: frame.capture_monotonic_ns for role, frame in extra.items()
            },
        )


def state_permits_act_motion(state: ResidentActState) -> bool:
    return (
        state.control_enabled
        and not state.estop
        and state.rs485_ok
        and state.dwj_ok
        and state.imu_ok
        and state.sensor_valid
        and state.stm32_alive
        and state.fault_flags == 0
    )


def safety_telemetry(state: ResidentActState) -> dict[str, int]:
    return {
        "control_enabled": int(state.control_enabled),
        "estop": int(state.estop),
        "fault_flags": state.fault_flags,
        "rs485_ok": int(state.rs485_ok),
        "dwj_ok": int(state.dwj_ok),
        "imu_ok": int(state.imu_ok),
    }


def operator_telemetry(
    state: ResidentActState,
) -> dict[str, int | float | bool]:
    values = state.state
    return {
        "control_seq": state.control_seq,
        "sensor_seq": state.sensor_seq,
        "sensor_is_new": state.sensor_is_new,
        "sensor_valid": state.sensor_valid,
        "control_enabled": state.control_enabled,
        "estop": state.estop,
        "command_timed_out": not state.stm32_alive,
        "fault_flags": state.fault_flags,
        "control_generation": state.control_generation,
        "rs485_ok": state.rs485_ok,
        "dwj_ok": state.dwj_ok,
        "imu_ok": state.imu_ok,
        "stm32_alive": state.stm32_alive,
        "boom_pos_mm": values[0] * 1_000.0,
        "stick_pos_mm": values[1] * 1_000.0,
        "bucket_pos_mm": values[2] * 1_000.0,
        "boom_vel_mmps": values[3] * 1_000.0,
        "stick_vel_mmps": values[4] * 1_000.0,
        "bucket_vel_mmps": values[5] * 1_000.0,
        "boom_angle_deg": math.degrees(values[6]),
        "arm_angle_deg": math.degrees(values[7]),
        "bucket_angle_deg": math.degrees(values[8]),
        "swing_angle_deg": math.degrees(values[9]),
        "swing_vel_degps": math.degrees(values[10]),
    }
