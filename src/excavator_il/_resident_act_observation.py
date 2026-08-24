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

    def __init__(self, *, capacity: int = 8) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("camera buffer capacity must be a positive integer")
        self._frames: deque[RgbCameraFrame] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def add_camera(self, frame: RgbCameraFrame) -> None:
        if not isinstance(frame, RgbCameraFrame):
            raise ValueError("resident camera frame has the wrong type")
        with self._lock:
            if self._frames and (
                frame.capture_monotonic_ns
                <= self._frames[-1].capture_monotonic_ns
            ):
                raise ValueError("camera timestamps must be strictly increasing")
            self._frames.append(frame)
            self._ready.set()

    def wait_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    def build(self, state: ResidentActState) -> ActObservation:
        with self._lock:
            frames = tuple(self._frames)
        camera = next(
            (
                frame
                for frame in reversed(frames)
                if frame.capture_monotonic_ns <= state.state_monotonic_ns
            ),
            None,
        )
        if camera is None:
            raise ValueError("no causal camera frame is available for resident state")
        return ActObservation(
            state=state.state,
            front_rgb=camera.rgb,
            state_monotonic_ns=state.state_monotonic_ns,
            camera_monotonic_ns=camera.capture_monotonic_ns,
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
