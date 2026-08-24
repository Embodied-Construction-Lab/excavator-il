"""Long-lived ACT worker for the Orin Resident Mission Runtime.

The worker side never owns or imports the STM32 serial boundary.  It exchanges
strict state and policy-candidate frames with the resident owner over one local
Unix stream and keeps the policy and camera alive across policy handoffs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import os
import signal
import sys
import threading
import time
from typing import Any, Callable

from lerobot.policies import get_policy_class, make_pre_post_processors

from .act_deployment import verify_deployment_manifest
from .act_runtime import (
    ActPolicySession,
    ActRuntimeController,
    ActRuntimeEngine,
    RuntimeMode,
    warmup_act_policy_session,
)
from .act_runtime_config import load_act_runtime_config
from .act_runtime_contract import REQUIRED_MOTION_AUTHORIZATION
from .collector.camera import RgbCameraFrame, UvcCamera
from .collector.config import CameraConfig, load_collection_config
from .collector.preview import (
    LatestJpegFrame,
    LatestTelemetryFrame,
    MjpegPreviewServer,
)
from .dig_policy import DigPolicyFactory
from .resident_protocol import (
    ACT_CONTROL_MODE,
    ACT_POLICY_SOURCE,
    ResidentActDataClient,
    ResidentActOwnerClosed,
    ResidentActState,
    ResidentPolicyCandidate,
)
from ._resident_act_observation import (
    ResidentCausalObservationBuffer,
    operator_telemetry as _operator_telemetry,
    safety_telemetry as _safety_telemetry,
    state_permits_act_motion as _state_permits_act_motion,
)


_ZERO_ACTION = (0.0, 0.0, 0.0, 0.0)
_DEFAULT_CANDIDATE_TTL_MS = 300.0
_MIN_CANDIDATE_TTL_MS = 200.0
_MAX_CANDIDATE_TTL_MS = 300.0
LOGGER = logging.getLogger("excavator_il.resident_act_runtime")


def _emit_lifecycle(message: str) -> None:
    """Emit process-control markers independently of third-party logging setup."""

    print(message, flush=True)


def _load_commissioned_lerobot_act_session(config: Any) -> Any:
    """Load the commissioned LeRobot ACT Adapter with deployment rechecks."""

    provenance = {
        "manifest_path": config.deployment_manifest_path,
        "checkpoint_path": config.checkpoint_path,
        "machine_profile_path": config.machine_profile_path,
    }
    verify_deployment_manifest(**provenance)
    policy_class = get_policy_class("act")
    _emit_lifecycle("ACT resident build: policy load starting")
    policy = policy_class.from_pretrained(config.checkpoint_path)
    _emit_lifecycle("ACT resident build: policy load passed")
    policy.to(config.device)
    _emit_lifecycle("ACT resident build: CUDA transfer passed")
    policy.config.device = config.device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(config.checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
        postprocessor_overrides={"device_processor": {"device": config.device}},
    )
    _emit_lifecycle("ACT resident build: processors ready")
    session = ActPolicySession(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=config.device,
    )
    # Catch checkpoint replacement during the comparatively expensive load.
    verify_deployment_manifest(**provenance)
    _emit_lifecycle("ACT resident build: deployment recheck passed")
    return session


def _build_commissioned_dig_policy_factory(
    config: Any,
    *,
    commissioned_lerobot_act_loader: Callable[[Any], Any],
) -> DigPolicyFactory:
    return DigPolicyFactory(
        {
            "lerobot_act": lambda: commissioned_lerobot_act_loader(config),
        }
    )


@dataclass(frozen=True)
class ResidentActStep:
    """Auditable outcome of consuming one resident state frame."""

    state_monotonic_ns: int
    inference_completed_monotonic_ns: int
    control_generation: int
    commanded_action: tuple[float, float, float, float]
    reason: str
    candidate_sent: bool
    inference_performed: bool


@dataclass(frozen=True)
class ResidentActStatus:
    """Current per-activation progress without tying it to worker lifetime."""

    active_generation: int | None
    completed_steps: int
    total_completed_steps: int
    last_reason: str | None


class ResidentActRuntime:
    """Keep one ACT engine resident and publish typed manual-action candidates."""

    def __init__(
        self,
        *,
        transport: Any,
        engine: Any,
        candidate_ttl_ms: float = _DEFAULT_CANDIDATE_TTL_MS,
        camera_buffer_capacity: int = 8,
        status_callback: Callable[[ResidentActStatus], None] | None = None,
        telemetry_preview: LatestTelemetryFrame | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if (
            isinstance(candidate_ttl_ms, bool)
            or not isinstance(candidate_ttl_ms, (int, float))
            or not math.isfinite(float(candidate_ttl_ms))
            or not _MIN_CANDIDATE_TTL_MS
            <= float(candidate_ttl_ms)
            <= _MAX_CANDIDATE_TTL_MS
        ):
            raise ValueError(
                "candidate TTL must be finite and in "
                f"[{_MIN_CANDIDATE_TTL_MS:g}, {_MAX_CANDIDATE_TTL_MS:g}] ms"
            )
        if status_callback is not None and not callable(status_callback):
            raise ValueError("status callback must be callable")
        self._transport = transport
        self._engine = engine
        self._candidate_ttl_ns = int(float(candidate_ttl_ms) * 1_000_000)
        self._clock = monotonic_ns
        self._observations = ResidentCausalObservationBuffer(
            capacity=camera_buffer_capacity
        )
        self._last_control_generation: int | None = None
        self._last_sensor_sequence: int | None = None
        self._status_callback = status_callback
        self._telemetry_preview = telemetry_preview
        self._telemetry_lock = threading.Lock()
        self._last_telemetry_receive_ns = -1
        self._status_condition = threading.Condition()
        self._active_generation: int | None = None
        self._completed_steps = 0
        self._total_completed_steps = 0
        self._last_reason: str | None = None

    @property
    def status(self) -> ResidentActStatus:
        with self._status_condition:
            return self._status_locked()

    @property
    def transport(self) -> Any:
        return self._transport

    def add_camera_frame(self, frame: RgbCameraFrame) -> None:
        self._observations.add_camera(frame)

    def wait_camera_ready(self, timeout_s: float) -> bool:
        if timeout_s <= 0:
            raise ValueError("camera readiness timeout must be positive")
        return self._observations.wait_ready(timeout_s)

    def observe_state(self, state: ResidentActState) -> None:
        """Publish every received state to the optional operator telemetry seam."""

        if not isinstance(state, ResidentActState):
            raise ValueError("resident ACT runtime state has the wrong type")
        if self._telemetry_preview is None:
            return
        with self._telemetry_lock:
            if state.receive_monotonic_ns <= self._last_telemetry_receive_ns:
                return
            self._telemetry_preview.publish(
                _operator_telemetry(state),
                receive_monotonic_ns=state.receive_monotonic_ns,
            )
            self._last_telemetry_receive_ns = state.receive_monotonic_ns

    def handle_disconnect(self) -> None:
        """Discard queued policy state after the owner link disappears."""

        self._engine.reset()
        self._last_control_generation = None
        self._last_sensor_sequence = None
        self._begin_activation(0)

    def process_state(self, state: ResidentActState) -> ResidentActStep:
        """Consume one decoded state and emit at most one current-generation candidate."""

        if not isinstance(state, ResidentActState):
            raise ValueError("resident ACT runtime state has the wrong type")
        self.observe_state(state)
        previous_generation = self._last_control_generation
        previous_sequence = self._last_sensor_sequence
        generation_changed = state.control_generation != previous_generation
        sequence_gap = (
            not generation_changed
            and state.sensor_is_new
            and previous_sequence is not None
            and state.sensor_seq != ((previous_sequence + 1) & 0xFFFFFFFF)
        )
        self._last_control_generation = state.control_generation
        self._last_sensor_sequence = state.sensor_seq
        if generation_changed or sequence_gap:
            self._engine.reset()
        if generation_changed:
            self._begin_activation(state.control_generation)

        if state.control_generation == 0:
            return self._step_without_candidate(state, "inactive_generation")
        if not _state_permits_act_motion(state):
            if not generation_changed and not sequence_gap:
                self._engine.reset()
            if not state.sensor_is_new:
                return self._step_without_candidate(state, "safety_state_invalid")
            return self._send_zero(state, "safety_state_invalid")
        if not state.sensor_is_new:
            return self._step_without_candidate(state, "not_new_sensor_state")
        if sequence_gap:
            return self._send_zero(state, "state_sequence_gap")
        try:
            observation = self._observations.build(state)
        except ValueError:
            if not generation_changed:
                self._engine.reset()
            return self._send_zero(state, "observation_unavailable")
        decision = self._engine.step(
            observation=observation,
            telemetry=_safety_telemetry(state),
        )
        action = tuple(float(value) for value in decision.commanded_action)
        if len(action) != 4 or not all(
            math.isfinite(value) and -1.000001 <= value <= 1.000001
            for value in action
        ):
            self._engine.reset()
            return self._send_zero(state, "invalid_policy_decision")
        step = self._send_action(
            state,
            action,
            str(decision.reason),
            inference_performed=True,
        )
        self._record_completed_step(str(decision.reason))
        return step

    def _send_zero(self, state: ResidentActState, reason: str) -> ResidentActStep:
        return self._send_action(state, _ZERO_ACTION, reason)

    def _send_action(
        self,
        state: ResidentActState,
        action: tuple[float, float, float, float],
        reason: str,
        *,
        inference_performed: bool = False,
    ) -> ResidentActStep:
        completed_ns = self._clock()
        candidate = ResidentPolicyCandidate(
            source=ACT_POLICY_SOURCE,
            control_generation=state.control_generation,
            mode=ACT_CONTROL_MODE,
            action=action,
            created_monotonic_ns=completed_ns,
            valid_until_monotonic_ns=completed_ns + self._candidate_ttl_ns,
        )
        self._transport.send_candidate(candidate)
        step = ResidentActStep(
            state_monotonic_ns=state.state_monotonic_ns,
            inference_completed_monotonic_ns=completed_ns,
            control_generation=state.control_generation,
            commanded_action=action,
            reason=reason,
            candidate_sent=True,
            inference_performed=inference_performed,
        )
        if not inference_performed:
            self._record_reason(reason)
        return step

    def _step_without_candidate(
        self, state: ResidentActState, reason: str
    ) -> ResidentActStep:
        completed_ns = self._clock()
        step = ResidentActStep(
            state_monotonic_ns=state.state_monotonic_ns,
            inference_completed_monotonic_ns=completed_ns,
            control_generation=state.control_generation,
            commanded_action=_ZERO_ACTION,
            reason=reason,
            candidate_sent=False,
            inference_performed=False,
        )
        self._record_reason(reason)
        return step

    def _begin_activation(self, generation: int) -> None:
        with self._status_condition:
            self._active_generation = generation if generation > 0 else None
            self._completed_steps = 0
            self._last_reason = "activation_started" if generation > 0 else "inactive_generation"
            status = self._status_locked()
            self._status_condition.notify_all()
        self._notify_status(status)
        LOGGER.info("ACT resident activation: control_generation=%d", generation)

    def _record_completed_step(self, reason: str) -> None:
        with self._status_condition:
            self._completed_steps += 1
            self._total_completed_steps += 1
            self._last_reason = reason
            status = self._status_locked()
            self._status_condition.notify_all()
        self._notify_status(status)
        if status.completed_steps == 1 or status.completed_steps % 25 == 0:
            LOGGER.info(
                "ACT resident progress: control_generation=%s completed_steps=%d",
                status.active_generation,
                status.completed_steps,
            )

    def _record_reason(self, reason: str) -> None:
        with self._status_condition:
            self._last_reason = reason
            status = self._status_locked()
            self._status_condition.notify_all()
        self._notify_status(status)

    def _status_locked(self) -> ResidentActStatus:
        return ResidentActStatus(
            active_generation=self._active_generation,
            completed_steps=self._completed_steps,
            total_completed_steps=self._total_completed_steps,
            last_reason=self._last_reason,
        )

    def _notify_status(self, status: ResidentActStatus) -> None:
        if self._status_callback is not None:
            self._status_callback(status)


class _LatestResidentState:
    """One-slot mailbox so slow inference cannot replay a state backlog."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: ResidentActState | None = None
        self._revision = 0
        self._consumed_revision = 0
        self._closed = False

    def publish(self, state: ResidentActState) -> None:
        with self._condition:
            if self._closed:
                return
            self._latest = state
            self._revision += 1
            self._condition.notify_all()

    def receive(self, *, timeout_s: float) -> ResidentActState | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._closed and self._revision <= self._consumed_revision:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._closed or self._latest is None:
                return None
            self._consumed_revision = self._revision
            return self._latest

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class ResidentActWorker:
    """Run the warmed policy/camera once and remain alive across activations.

    A data-link failure is terminal for this worker instance: the policy queue
    is reset and the process exits rather than guessing whether its previous
    generation remains authoritative.  The owner-side watchdog and motion
    authority independently force zero at the only serial boundary.
    """

    def __init__(
        self,
        *,
        runtime: ResidentActRuntime,
        camera: Any,
        warmup: Callable[[], Any],
        camera_preview: LatestJpegFrame | None = None,
        preview_server: MjpegPreviewServer | None = None,
        connect_timeout_s: float = 5.0,
        camera_ready_timeout_s: float = 2.0,
    ) -> None:
        if not callable(warmup):
            raise ValueError("resident ACT warmup must be callable")
        if connect_timeout_s <= 0 or camera_ready_timeout_s <= 0:
            raise ValueError("resident ACT startup timeouts must be positive")
        self._runtime = runtime
        self._transport = runtime.transport
        self._camera = camera
        self._warmup = warmup
        self._camera_preview = camera_preview
        self._preview_server = preview_server
        self._connect_timeout_s = connect_timeout_s
        self._camera_ready_timeout_s = camera_ready_timeout_s
        self._mailbox = _LatestResidentState()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error_lock = threading.Lock()
        self._error: BaseException | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and not self._stop.is_set()

    @property
    def runtime(self) -> ResidentActRuntime:
        return self._runtime

    @property
    def error(self) -> BaseException | None:
        with self._error_lock:
            return self._error

    def wait_ready(self, *, timeout_s: float) -> bool:
        if timeout_s <= 0:
            raise ValueError("resident ACT readiness timeout must be positive")
        return self._ready.wait(timeout_s) and self.error is None

    def request_stop(self) -> None:
        self._stop.set()
        self._mailbox.close()
        self._transport.close()

    def run(self) -> None:
        camera_thread: threading.Thread | None = None
        reader_thread: threading.Thread | None = None
        preview_thread: threading.Thread | None = None
        raised: BaseException | None = None
        try:
            if self._preview_server is not None:
                preview_thread = threading.Thread(
                    target=self._guarded_worker,
                    args=(self._preview_server.serve_forever,),
                    name="resident-act-preview",
                    daemon=True,
                )
                preview_thread.start()
            _emit_lifecycle("ACT resident warmup starting")
            self._warmup()
            _emit_lifecycle("ACT resident warmup passed")
            if self.error is not None:
                raise RuntimeError("resident ACT preview failed") from self.error
            camera_thread = threading.Thread(
                target=self._guarded_worker,
                args=(self._camera_loop,),
                name="resident-act-camera",
                daemon=True,
            )
            camera_thread.start()
            if not self._runtime.wait_camera_ready(self._camera_ready_timeout_s):
                raise RuntimeError("resident ACT camera did not become ready")

            # Connected is the owner-visible readiness signal.  No state can be
            # requested before the model is warm and a real camera frame exists.
            self._transport.connect(timeout_s=self._connect_timeout_s)
            reader_thread = threading.Thread(
                target=self._guarded_worker,
                args=(self._state_reader_loop,),
                name="resident-act-state-reader",
                daemon=True,
            )
            reader_thread.start()
            self._ready.set()
            _emit_lifecycle("ACT resident worker ready: owner connected")

            while not self._stop.is_set():
                state = self._mailbox.receive(timeout_s=0.05)
                if state is not None:
                    self._runtime.process_state(state)
                error = self.error
                if error is not None:
                    raise RuntimeError("resident ACT worker failed") from error
            error = self.error
            if error is not None:
                raise RuntimeError("resident ACT worker failed") from error
        except ResidentActOwnerClosed:
            self._handle_owner_disconnect()
        except BaseException as exc:
            raised = exc
            self._record_error(exc)
        finally:
            self._ready.clear()
            self._stop.set()
            self._mailbox.close()
            self._transport.close()
            if self._preview_server is not None:
                self._preview_server.close()
            self._camera.close()
            for thread in (reader_thread, camera_thread, preview_thread):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            self._runtime.handle_disconnect()
            LOGGER.info("ACT resident worker stopped")
        if raised is not None:
            raise raised

    def _camera_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._camera.read_rgb()
            self._runtime.add_camera_frame(frame)
            if self._camera_preview is not None and frame.encoded_image is not None:
                self._camera_preview.publish(
                    frame.encoded_image,
                    capture_monotonic_ns=frame.capture_monotonic_ns,
                )

    def _state_reader_loop(self) -> None:
        while not self._stop.is_set():
            state = self._transport.receive_state(timeout_s=0.05)
            if state is not None:
                self._runtime.observe_state(state)
                self._mailbox.publish(state)

    def _guarded_worker(self, target: Callable[[], None]) -> None:
        try:
            target()
        except ResidentActOwnerClosed:
            self._handle_owner_disconnect()
        except BaseException as exc:
            if not self._stop.is_set():
                self._record_error(exc)
                self._stop.set()
                self._mailbox.close()

    def _handle_owner_disconnect(self) -> None:
        if not self._stop.is_set():
            _emit_lifecycle("ACT resident owner disconnected: stopping worker")
        self._stop.set()
        self._mailbox.close()

    def _record_error(self, error: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = error


def build_resident_act_worker(
    config_path: str | os.PathLike[str],
    *,
    socket_path: str | os.PathLike[str],
    status_callback: Callable[[ResidentActStatus], None] | None = None,
    operator_observation_config: str | os.PathLike[str] | None = None,
    camera_preview: LatestJpegFrame | None = None,
    telemetry_preview: LatestTelemetryFrame | None = None,
    preview_server: MjpegPreviewServer | None = None,
    dig_policy_provider: Callable[[Any], DigPolicyFactory] | None = None,
    commissioned_lerobot_act_loader: Callable[[Any], Any] | None = None,
) -> ResidentActWorker:
    """Load one verified ACT model and open one camera for resident operation."""

    config = load_act_runtime_config(config_path)
    observation_config = (
        None
        if operator_observation_config is None
        else load_collection_config(operator_observation_config)
    )
    if observation_config is not None:
        # The collection config names the physical host device (normally a
        # stable /dev/v4l/by-path path), while the launcher maps that device
        # into the ACT container under config.camera.device (normally
        # /dev/video0).  Device strings therefore belong to different device
        # namespaces and are not part of the observation contract.
        camera_format = (
            observation_config.camera.width,
            observation_config.camera.height,
            observation_config.camera.nominal_fps,
        )
        expected_format = (
            config.camera.width,
            config.camera.height,
            config.camera.nominal_fps,
        )
        if camera_format != expected_format:
            raise ValueError("ACT and operator observation camera formats must match")
        if observation_config.camera_preview is None:
            raise ValueError("operator observation config must enable camera preview")
        if any(item is not None for item in (camera_preview, telemetry_preview, preview_server)):
            raise ValueError("operator preview resources must have a single owner")
    backend_id = getattr(config, "dig_policy_backend", "lerobot_act")
    loader = (
        _load_commissioned_lerobot_act_session
        if commissioned_lerobot_act_loader is None
        else commissioned_lerobot_act_loader
    )
    provider = (
        (
            lambda loaded_config: _build_commissioned_dig_policy_factory(
                loaded_config,
                commissioned_lerobot_act_loader=loader,
            )
        )
        if dig_policy_provider is None
        else dig_policy_provider
    )
    policy_factory = provider(config)
    if not isinstance(policy_factory, DigPolicyFactory):
        raise ValueError("dig policy provider must return DigPolicyFactory")
    session = policy_factory.create(backend_id)
    controller = ActRuntimeController(
        mode=RuntimeMode.MOTION,
        motion_authorization=REQUIRED_MOTION_AUTHORIZATION,
        max_state_age_ms=config.max_inference_state_age_ms,
        max_camera_age_ms=config.max_camera_age_ms,
    )
    engine = ActRuntimeEngine(
        session=session,
        controller=controller,
        max_inference_ms=config.max_inference_ms,
    )
    transport = ResidentActDataClient(socket_path)
    camera = UvcCamera(
        CameraConfig(
            device=config.camera.device,
            width=config.camera.width,
            height=config.camera.height,
            nominal_fps=config.camera.nominal_fps,
            jpeg_quality=(
                95
                if observation_config is None
                else observation_config.camera.jpeg_quality
            ),
        )
    )
    if observation_config is not None:
        try:
            camera_preview = LatestJpegFrame()
            telemetry_preview = LatestTelemetryFrame()
            preview_server = MjpegPreviewServer(
                camera_preview,
                telemetry=telemetry_preview,
                bind_host=observation_config.camera_preview.bind_host,
                port=observation_config.camera_preview.port,
                allowed_client_host=observation_config.joystick.allowed_pc_host,
            )
        except BaseException:
            camera.close()
            raise
    runtime = ResidentActRuntime(
        transport=transport,
        engine=engine,
        status_callback=status_callback,
        telemetry_preview=telemetry_preview,
    )
    return ResidentActWorker(
        runtime=runtime,
        camera=camera,
        warmup=lambda: warmup_act_policy_session(session),
        camera_preview=camera_preview,
        preview_server=preview_server,
    )


def run_resident_act_worker(
    config_path: str | os.PathLike[str],
    *,
    socket_path: str | os.PathLike[str],
    operator_observation_config: str | os.PathLike[str] | None = None,
) -> None:
    """Script seam: construct the resident resources once, then run indefinitely."""

    worker = build_resident_act_worker(
        config_path,
        socket_path=socket_path,
        operator_observation_config=operator_observation_config,
    )
    worker.run()


def main(argv: list[str] | None = None) -> int:
    """Minimal module CLI; the existing project CLI can delegate here later."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Run the resident ACT policy worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--operator-observation-config")
    args = parser.parse_args(argv)
    worker = build_resident_act_worker(
        args.config,
        socket_path=args.socket_path,
        operator_observation_config=args.operator_observation_config,
    )
    previous: dict[int, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        worker.request_stop()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        worker.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
