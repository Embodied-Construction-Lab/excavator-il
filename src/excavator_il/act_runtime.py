"""Fail-closed online ACT policy session for excavator observations."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import Enum
import math
import time
import threading
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .lerobot_conversion import STATE_FIELDS
from .raw_episode import ACTION_FIELDS
from .collector.camera import RgbCameraFrame
from .stm32_protocol import Stm32TelemetryFrame

REQUIRED_MOTION_AUTHORIZATION = "ALLOW_ACT_MACHINE_MOTION"
_ZERO_ACTION = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ActObservation:
    state: tuple[float, ...]
    front_rgb: np.ndarray
    state_monotonic_ns: int
    camera_monotonic_ns: int


class RuntimeMode(str, Enum):
    SHADOW = "shadow"
    MOTION = "motion"


@dataclass(frozen=True)
class ActRuntimeDecision:
    predicted_action: tuple[float, ...]
    commanded_action: tuple[float, ...]
    serial_axes: tuple[float, ...] | None
    reason: str


class CausalObservationBuffer:
    """Retain a bounded camera history and build Orin-clock causal samples."""

    def __init__(self, *, capacity: int = 8) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("camera buffer capacity must be a positive integer")
        self._camera_frames: deque[RgbCameraFrame] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def add_camera(self, frame: RgbCameraFrame) -> None:
        with self._lock:
            if self._camera_frames and (
                frame.capture_monotonic_ns <= self._camera_frames[-1].capture_monotonic_ns
            ):
                raise ValueError("camera timestamps must be strictly increasing")
            self._camera_frames.append(frame)
            self._ready.set()

    def wait_ready(self, timeout_s: float) -> bool:
        return self._ready.wait(timeout_s)

    def build(self, telemetry: Stm32TelemetryFrame) -> ActObservation:
        with self._lock:
            frames = tuple(self._camera_frames)
        camera = next(
            (
                frame
                for frame in reversed(frames)
                if frame.capture_monotonic_ns <= telemetry.receive_monotonic_ns
            ),
            None,
        )
        if camera is None:
            raise ValueError("no causal camera frame is available for STM32 state")
        return ActObservation(
            state=state_from_stm32_telemetry(telemetry.values),
            front_rgb=camera.rgb,
            state_monotonic_ns=telemetry.receive_monotonic_ns,
            camera_monotonic_ns=camera.capture_monotonic_ns,
        )


class ActRuntimeController:
    """Decide whether one ACT result may reach the serial command boundary."""

    def __init__(
        self,
        *,
        mode: RuntimeMode,
        motion_authorization: str | None = None,
        max_state_age_ms: float = 100.0,
        max_camera_age_ms: float = 120.0,
    ) -> None:
        ages_ms = (max_state_age_ms, max_camera_age_ms)
        if not all(math.isfinite(value) and value > 0 for value in ages_ms):
            raise ValueError("ACT runtime age limits must be finite and positive")
        self._mode = RuntimeMode(mode)
        self._motion_authorized = (
            motion_authorization == REQUIRED_MOTION_AUTHORIZATION
        )
        self._max_state_age_ns = int(max_state_age_ms * 1_000_000)
        self._max_camera_age_ns = int(max_camera_age_ms * 1_000_000)

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    def decide(
        self,
        *,
        predicted_action: tuple[float, ...],
        state_monotonic_ns: int,
        camera_monotonic_ns: int,
        now_monotonic_ns: int,
        telemetry: Mapping[str, int | float | str],
    ) -> ActRuntimeDecision:
        predicted = tuple(float(value) for value in predicted_action)
        if len(predicted) != len(ACTION_FIELDS) or not all(
            math.isfinite(value) and -1.000001 <= value <= 1.000001
            for value in predicted
        ):
            raise ValueError("ACT runtime predicted action is invalid")
        if self._mode is RuntimeMode.SHADOW:
            return ActRuntimeDecision(
                predicted_action=predicted,
                commanded_action=_ZERO_ACTION,
                serial_axes=None,
                reason="shadow_mode",
            )
        if not self._motion_authorized:
            return ActRuntimeDecision(predicted, _ZERO_ACTION, (0.0,) * 6, "motion_unauthorized")
        if now_monotonic_ns < max(state_monotonic_ns, camera_monotonic_ns):
            return ActRuntimeDecision(predicted, _ZERO_ACTION, (0.0,) * 6, "future_timestamp")
        if now_monotonic_ns - state_monotonic_ns > self._max_state_age_ns:
            return ActRuntimeDecision(predicted, _ZERO_ACTION, (0.0,) * 6, "state_stale")
        if now_monotonic_ns - camera_monotonic_ns > self._max_camera_age_ns:
            return ActRuntimeDecision(predicted, _ZERO_ACTION, (0.0,) * 6, "camera_stale")
        required_one = ("control_enabled", "rs485_ok", "dwj_ok", "imu_ok")
        if not all(int(telemetry.get(field, 0)) == 1 for field in required_one):
            return ActRuntimeDecision(predicted, _ZERO_ACTION, (0.0,) * 6, "safety_state_invalid")
        if int(telemetry.get("estop", 1)) != 0 or int(telemetry.get("fault_flags", 1)) != 0:
            return ActRuntimeDecision(predicted, _ZERO_ACTION, (0.0,) * 6, "safety_state_invalid")
        axes = (predicted[3], predicted[1], 0.0, predicted[2], predicted[0], 0.0)
        return ActRuntimeDecision(predicted, predicted, axes, "motion_allowed")


class ActRuntimeEngine:
    """Join LeRobot action selection to the fail-closed motion boundary."""

    def __init__(
        self,
        *,
        session: Any,
        controller: ActRuntimeController,
        max_inference_ms: float = 100.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not math.isfinite(max_inference_ms) or max_inference_ms <= 0:
            raise ValueError("ACT inference budget must be finite and positive")
        self._session = session
        self._controller = controller
        self._max_inference_ns = int(max_inference_ms * 1_000_000)
        self._clock = monotonic_ns

    def reset(self) -> None:
        self._session.reset()

    def warmup_live_observation(
        self, observation: ActObservation
    ) -> tuple[float, ...]:
        """Run one live observation through ACT, then clear any queued chunk state."""

        started_ns = self._clock()
        try:
            action = self._session.select_action(observation)
        finally:
            self._session.reset()
        completed_ns = self._clock()
        if completed_ns - started_ns > self._max_inference_ns:
            raise ValueError("live ACT warmup exceeded the inference budget")
        return action

    def step(
        self,
        *,
        observation: ActObservation,
        telemetry: Mapping[str, int | float | str],
    ) -> ActRuntimeDecision:
        started_ns = self._clock()
        try:
            predicted = self._session.select_action(observation)
        except (RuntimeError, ValueError):
            self._session.reset()
            return ActRuntimeDecision(
                predicted_action=_ZERO_ACTION,
                commanded_action=_ZERO_ACTION,
                serial_axes=(0.0,) * 6
                if self._controller.mode is RuntimeMode.MOTION
                else None,
                reason="policy_error",
            )
        completed_ns = self._clock()
        if completed_ns - started_ns > self._max_inference_ns:
            self._session.reset()
            return ActRuntimeDecision(
                predicted_action=predicted,
                commanded_action=_ZERO_ACTION,
                serial_axes=(0.0,) * 6
                if self._controller.mode is RuntimeMode.MOTION
                else None,
                reason="inference_budget_exceeded",
            )
        decision = self._controller.decide(
            predicted_action=predicted,
            state_monotonic_ns=observation.state_monotonic_ns,
            camera_monotonic_ns=observation.camera_monotonic_ns,
            now_monotonic_ns=completed_ns,
            telemetry=telemetry,
        )
        if decision.reason not in ("motion_allowed", "shadow_mode"):
            self._session.reset()
        return decision


def state_from_stm32_telemetry(
    telemetry: Mapping[str, int | float | str],
) -> tuple[float, ...]:
    linear_fields = (
        "boom_pos_mm",
        "stick_pos_mm",
        "bucket_pos_mm",
        "boom_vel_mmps",
        "stick_vel_mmps",
        "bucket_vel_mmps",
    )
    angular_fields = (
        "boom_angle_deg",
        "arm_angle_deg",
        "bucket_angle_deg",
        "swing_angle_deg",
        "swing_vel_degps",
    )
    try:
        values = tuple(float(telemetry[field]) / 1000.0 for field in linear_fields) + tuple(
            math.radians(float(telemetry[field])) for field in angular_fields
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("STM32 telemetry cannot form the ACT state") from exc
    if len(values) != len(STATE_FIELDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("STM32 telemetry produced a non-finite ACT state")
    return values


class ActPolicySession:
    """Adapt live observations to LeRobot while retaining its action queue."""

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
        postprocessor: Callable[[torch.Tensor], torch.Tensor],
        device: str,
    ) -> None:
        if policy.config.chunk_size != 20 or policy.config.n_action_steps != 10:
            raise ValueError("ACT runtime requires chunk_size=20 and n_action_steps=10")
        if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
            raise ValueError("ACT runtime does not support temporal ensemble checkpoints")
        if set(policy.config.input_features) != {
            "observation.state",
            "observation.images.front",
        }:
            raise ValueError("ACT runtime requires exactly one single front RGB input")
        if tuple(policy.config.input_features["observation.state"].shape) != (
            len(STATE_FIELDS),
        ):
            raise ValueError("ACT runtime checkpoint state contract is invalid")
        if tuple(policy.config.output_features["action"].shape) != (
            len(ACTION_FIELDS),
        ):
            raise ValueError("ACT runtime checkpoint action contract is invalid")
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._device = device
        self._image_shape = tuple(
            policy.config.input_features["observation.images.front"].shape
        )
        policy.eval()
        policy.reset()

    def reset(self) -> None:
        self._policy.reset()

    def select_action(self, observation: ActObservation) -> tuple[float, ...]:
        if len(observation.state) != len(STATE_FIELDS) or not all(
            math.isfinite(value) for value in observation.state
        ):
            raise ValueError("ACT runtime state must contain 11 finite values")
        image = np.asarray(observation.front_rgb)
        expected_hwc = (self._image_shape[1], self._image_shape[2], 3)
        if image.dtype != np.uint8 or image.shape != expected_hwc:
            raise ValueError(f"ACT runtime RGB must be uint8 with shape {expected_hwc}")
        batch = {
            "observation.state": torch.tensor(
                observation.state, dtype=torch.float32
            ).unsqueeze(0),
            "observation.images.front": torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
            .to(dtype=torch.float32)
            .div_(255.0)
            .unsqueeze(0),
        }
        processed = self._preprocessor(batch)
        with torch.no_grad():
            action = self._postprocessor(self._policy.select_action(processed))
        values = tuple(float(value) for value in action.detach().cpu().reshape(-1))
        if len(values) != len(ACTION_FIELDS) or not all(
            math.isfinite(value) and -1.000001 <= value <= 1.000001
            for value in values
        ):
            raise ValueError("ACT runtime produced an invalid normalized action")
        return values


def warmup_act_policy_session(session: Any) -> tuple[float, ...]:
    """Initialize CUDA kernels with a synthetic finite input, then clear ACT state."""

    observation = ActObservation(
        state=(0.0,) * len(STATE_FIELDS),
        front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        state_monotonic_ns=0,
        camera_monotonic_ns=0,
    )
    try:
        action = tuple(float(value) for value in session.select_action(observation))
        if len(action) != len(ACTION_FIELDS) or not all(
            math.isfinite(value) and -1.000001 <= value <= 1.000001
            for value in action
        ):
            raise ValueError("ACT policy warmup produced an invalid normalized action")
        return action
    finally:
        session.reset()
