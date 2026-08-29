"""Online Orin ACT I/O service with a no-write shadow boundary."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Callable

import torch

from ._act_runtime_control import (
    CommandWriteResult,
    LatestStateQueue,
    SensorSequenceTracker,
    Stm32CommandChannel,
)
from .act_runtime import (
    ActRuntimeController,
    ActRuntimeDecision,
    ActRuntimeEngine,
    CausalObservationBuffer,
    REQUIRED_MOTION_AUTHORIZATION,
    RuntimeMode,
    warmup_act_policy_session,
)
from .act_runtime_config import ActRuntimeConfig, load_act_runtime_config
from .act_policy_provider import build_commissioned_lerobot_act_factory
from .collector.camera import UvcCamera
from .collector.config import CameraConfig, load_collection_config
from .collector.machine_state import MachineStateUdpPublisher
from .collector.preview import LatestJpegFrame, LatestTelemetryFrame, MjpegPreviewServer
from .dig_policy import DigPolicyFactory
from .stm32_protocol import (
    Stm32ManualCommandEncoder,
    Stm32TelemetryFrame,
    Stm32TelemetryParser,
)


LOGGER = logging.getLogger("excavator_il.act_runtime")
RuntimeDigPolicyProvider = Callable[
    [ActRuntimeConfig, RuntimeMode],
    DigPolicyFactory,
]


class ActRuntimeStepProcessor:
    """Process one new 10 Hz state and cross the serial boundary at most once."""

    def __init__(
        self,
        *,
        observation_buffer: CausalObservationBuffer,
        engine: Any,
        command_channel: Stm32CommandChannel,
        record: Callable[[dict[str, Any]], None],
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._observations = observation_buffer
        self._engine = engine
        self._command_channel = command_channel
        self._record = record
        self._clock = monotonic_ns
        self._last_motion_epoch = command_channel.motion_epoch

    def process(
        self,
        telemetry: Stm32TelemetryFrame,
        *,
        state_generation: int | None = None,
        dropped_state_count: int = 0,
    ) -> ActRuntimeDecision:
        state_monotonic_ns = telemetry.receive_monotonic_ns
        camera_monotonic_ns: int | None = None
        epoch_before = self._command_channel.motion_epoch
        if self._last_motion_epoch != epoch_before:
            self._engine.reset()
        self._last_motion_epoch = epoch_before
        try:
            observation = self._observations.build(telemetry)
        except ValueError:
            self._engine.reset()
            decision = ActRuntimeDecision(
                predicted_action=(0.0,) * 4,
                commanded_action=(0.0,) * 4,
                serial_axes=(0.0,) * 6
                if self._command_channel.mode is RuntimeMode.MOTION
                else None,
                reason="observation_unavailable",
            )
            now_ns = self._clock()
        else:
            state_monotonic_ns = observation.state_monotonic_ns
            camera_monotonic_ns = observation.camera_monotonic_ns
            if dropped_state_count > 0:
                self._engine.reset()
                decision = ActRuntimeDecision(
                    predicted_action=(0.0,) * 4,
                    commanded_action=(0.0,) * 4,
                    serial_axes=(0.0,) * 6
                    if self._command_channel.mode is RuntimeMode.MOTION
                    else None,
                    reason="state_gap",
                )
                now_ns = self._clock()
            else:
                decision = self._engine.step(
                    observation=observation,
                    telemetry=telemetry.values,
                )
                now_ns = self._clock()
        write_attempted = decision.serial_axes is not None
        write_result: CommandWriteResult | None = None
        if write_attempted:
            write_result = self._command_channel.write_axes(
                decision.serial_axes,
                monotonic_ns=now_ns,
                state_generation=state_generation,
                motion_epoch=epoch_before,
            )
            if write_result.final_gate_reason != "accepted":
                self._engine.reset()
            self._last_motion_epoch = self._command_channel.motion_epoch
        self._record(
            {
                "schema_version": "excavator_act_runtime_step.v1",
                "state_monotonic_ns": state_monotonic_ns,
                "camera_monotonic_ns": camera_monotonic_ns,
                "decision_monotonic_ns": now_ns,
                "predicted_action": list(decision.predicted_action),
                "commanded_action": list(decision.commanded_action),
                "reason": decision.reason,
                "serial_write_attempted": write_attempted,
                "requested_serial_axes": None
                if write_result is None
                else list(write_result.requested_axes),
                "effective_serial_axes": None
                if write_result is None
                else list(write_result.effective_axes),
                "final_gate_reason": None
                if write_result is None
                else write_result.final_gate_reason,
                "command_seq": None
                if write_result is None
                else write_result.command_seq,
                "serial_write_performed": False
                if write_result is None
                else write_result.write_performed,
                "dropped_state_count": dropped_state_count,
            }
        )
        return decision

    def warmup_live(
        self,
        telemetry: Stm32TelemetryFrame,
        *,
        dropped_state_count: int = 0,
    ) -> tuple[float, ...]:
        if dropped_state_count < 0:
            raise ValueError("dropped_state_count must be non-negative")
        observation = self._observations.build(telemetry)
        return self._engine.warmup_live_observation(observation)


def _read_telemetry_until(
    *,
    serial_port: Any,
    parser: Stm32TelemetryParser,
    deadline: float,
    predicate: Callable[[Stm32TelemetryFrame], bool],
) -> Stm32TelemetryFrame | None:
    while time.monotonic() < deadline:
        line = serial_port.readline()
        if not line:
            continue
        frame = parser.parse_line(line, receive_monotonic_ns=time.monotonic_ns())
        if frame is not None and predicate(frame):
            return frame
    return None


def _startup_stm32(
    *,
    serial_port: Any,
    command_channel: Stm32CommandChannel,
    mode: RuntimeMode,
    timeout_s: float = 2.0,
) -> int:
    """Synchronize from fresh telemetry and prove a motion startup zero was accepted."""

    if timeout_s <= 0:
        raise ValueError("STM32 startup timeout must be positive")
    runtime_mode = RuntimeMode(mode)
    serial_port.reset_input_buffer()
    parser = Stm32TelemetryParser()
    frame = _read_telemetry_until(
        serial_port=serial_port,
        parser=parser,
        deadline=time.monotonic() + timeout_s,
        predicate=lambda _frame: True,
    )
    if frame is None:
        raise RuntimeError("cannot synchronize ACT runtime with fresh STM32 telemetry")
    next_sequence = command_channel.synchronize(frame)
    if runtime_mode is RuntimeMode.SHADOW:
        return next_sequence
    startup_sequence = command_channel.safe_zero(
        monotonic_ns=time.monotonic_ns(), reason="act_runtime_startup"
    )
    if startup_sequence is None:
        raise RuntimeError("motion startup zero command was not written")
    ack = _read_telemetry_until(
        serial_port=serial_port,
        parser=parser,
        deadline=time.monotonic() + timeout_s,
        predicate=lambda candidate: (
            int(candidate.values["command_valid"]) == 1
            and int(candidate.values["command_rx_seq"]) == startup_sequence
            and all(abs(value) <= 1e-9 for value in candidate.command_action)
        ),
    )
    if ack is None:
        raise RuntimeError("STM32 did not confirm the ACT startup zero-command ACK")
    return next_sequence


def _perform_live_warmup(
    *,
    states: LatestStateQueue,
    processor: ActRuntimeStepProcessor,
    timeout_s: float = 2.0,
) -> tuple[float, ...]:
    """Require one causal live observation before reporting runtime readiness."""

    if timeout_s <= 0:
        raise ValueError("live ACT warmup timeout must be positive")
    deadline = time.monotonic() + timeout_s
    last_error: ValueError | None = None
    while time.monotonic() < deadline:
        remaining_s = deadline - time.monotonic()
        try:
            item, dropped = states.get(timeout_s=min(0.05, remaining_s))
        except queue.Empty:
            continue
        frame, _generation = item
        try:
            return processor.warmup_live(frame, dropped_state_count=dropped)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError("ACT runtime did not complete live observation warmup") from last_error


class _JsonlLog:
    def __init__(self, root: Path, mode: RuntimeMode) -> None:
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = root / f"act_runtime_{mode.value}_{stamp}.jsonl"
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, separators=(",", ":")) + "\n"
        with self._lock:
            self._file.write(encoded)

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _route_telemetry_frame(
    *,
    frame: Stm32TelemetryFrame,
    command_channel: Stm32CommandChannel,
    states: LatestStateQueue,
    sensor_sequences: SensorSequenceTracker,
    telemetry_preview: LatestTelemetryFrame | None = None,
) -> None:
    """Update 20 Hz safety immediately; enqueue only new 10 Hz observations."""

    if telemetry_preview is not None:
        values = dict(frame.values)
        values["sensor_valid"] = frame.sensor_valid
        telemetry_preview.publish(
            values,
            receive_monotonic_ns=frame.receive_monotonic_ns,
        )
    generation = command_channel.update_state(frame)
    if not frame.sensor_is_new:
        return
    states.put(
        (frame, generation),
        external_gap_count=sensor_sequences.observe(frame.sensor_seq),
    )


class ActRuntimeService:
    def __init__(
        self,
        *,
        serial_port: Any,
        camera: UvcCamera,
        processor: ActRuntimeStepProcessor,
        command_channel: Stm32CommandChannel,
        max_steps: int | None = None,
        camera_preview: LatestJpegFrame | None = None,
        telemetry_preview: LatestTelemetryFrame | None = None,
        preview_server: MjpegPreviewServer | None = None,
        machine_state_publisher: MachineStateUdpPublisher | None = None,
    ) -> None:
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer when provided")
        self._serial = serial_port
        self._camera = camera
        self._processor = processor
        self._command_channel = command_channel
        self._parser = Stm32TelemetryParser()
        self._observations = processor._observations
        self._states = LatestStateQueue()
        self._sensor_sequences = SensorSequenceTracker()
        self._stop = threading.Event()
        self._inference_ready = threading.Event()
        self._live_warmup_action: tuple[float, ...] | None = None
        self._error: BaseException | None = None
        self._max_steps = max_steps
        self._completed_step_count = 0
        self._camera_preview = camera_preview
        self._telemetry_preview = telemetry_preview
        self._preview_server = preview_server
        self._machine_state_publisher = machine_state_publisher
        self._machine_state_send_failed = False

    @property
    def completed_step_count(self) -> int:
        return self._completed_step_count

    def request_stop(self) -> None:
        self._stop.set()

    def _worker(self, target: Callable[[], None]) -> None:
        try:
            target()
        except BaseException as exc:
            self._error = exc
            self._stop.set()

    def _camera_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._camera.read_rgb()
            self._observations.add_camera(frame)
            if self._camera_preview is not None and frame.encoded_image is not None:
                self._camera_preview.publish(
                    frame.encoded_image,
                    capture_monotonic_ns=frame.capture_monotonic_ns,
                )

    def _publish_machine_state(
        self, frame: Stm32TelemetryFrame, *, receive_wall_ns: int
    ) -> None:
        if self._machine_state_publisher is None:
            return
        try:
            self._machine_state_publisher.publish(
                frame, receive_wall_ns=receive_wall_ns
            )
        except OSError as exc:
            if not self._machine_state_send_failed:
                LOGGER.warning("AiryLidar machine-state UDP unavailable: %s", exc)
            self._machine_state_send_failed = True
        else:
            if self._machine_state_send_failed:
                LOGGER.info("AiryLidar machine-state UDP recovered")
            self._machine_state_send_failed = False

    def _serial_loop(self) -> None:
        while not self._stop.is_set():
            line = self._serial.readline()
            if not line:
                continue
            frame = self._parser.parse_line(
                line, receive_monotonic_ns=time.monotonic_ns()
            )
            if frame is not None:
                receive_wall_ns = time.time_ns()
                _route_telemetry_frame(
                    frame=frame,
                    command_channel=self._command_channel,
                    states=self._states,
                    sensor_sequences=self._sensor_sequences,
                    telemetry_preview=self._telemetry_preview,
                )
                self._publish_machine_state(
                    frame, receive_wall_ns=receive_wall_ns
                )

    def _inference_loop(self) -> None:
        self._live_warmup_action = _perform_live_warmup(
            states=self._states, processor=self._processor
        )
        self._inference_ready.set()
        while not self._stop.is_set():
            try:
                item, dropped = self._states.get(timeout_s=0.05)
            except queue.Empty:
                continue
            frame, generation = item
            self._processor.process(
                frame,
                state_generation=generation,
                dropped_state_count=dropped,
            )
            self._completed_step_count += 1
            if (
                self._max_steps is not None
                and self._completed_step_count >= self._max_steps
            ):
                LOGGER.info(
                    "ACT inference step budget reached: completed_steps=%d",
                    self._completed_step_count,
                )
                self._stop.set()

    def run(self) -> None:
        preview_worker: threading.Thread | None = None
        if self._preview_server is not None:
            preview_worker = threading.Thread(
                target=self._worker,
                args=(self._preview_server.serve_forever,),
                daemon=True,
            )
            preview_worker.start()
        camera_worker = threading.Thread(
            target=self._worker, args=(self._camera_loop,), daemon=True
        )
        camera_worker.start()
        workers = [camera_worker]
        if preview_worker is not None:
            workers.append(preview_worker)
        try:
            if not self._observations.wait_ready(2.0):
                raise RuntimeError("camera did not produce a live RGB frame")
            next_sequence = _startup_stm32(
                serial_port=self._serial,
                command_channel=self._command_channel,
                mode=self._command_channel.mode,
            )
            serial_worker = threading.Thread(
                target=self._worker, args=(self._serial_loop,), daemon=True
            )
            serial_worker.start()
            workers.append(serial_worker)
            inference_worker = threading.Thread(
                target=self._worker, args=(self._inference_loop,), daemon=True
            )
            inference_worker.start()
            workers.append(inference_worker)
            while not self._inference_ready.wait(0.05):
                if self._stop.is_set():
                    if self._error is not None:
                        raise RuntimeError(
                            f"ACT runtime worker failed: {self._error}"
                        ) from self._error
                    raise RuntimeError("ACT inference worker stopped before warmup")
            LOGGER.info(
                "ACT live warmup passed: action=%s", self._live_warmup_action
            )
            LOGGER.info(
                "ACT hardware ready: mode=%s initial_command_seq=%d",
                self._command_channel.mode.value,
                next_sequence,
            )
            while not self._stop.wait(0.05):
                watchdog_ns = time.monotonic_ns()
                self._command_channel.enforce_state_timeout(
                    monotonic_ns=watchdog_ns
                )
            if self._error is not None:
                raise RuntimeError(f"ACT runtime worker failed: {self._error}")
        finally:
            self._stop.set()
            if self._preview_server is not None:
                self._preview_server.close()
            self._command_channel.terminal_disarm(
                monotonic_ns=time.monotonic_ns(), reason="act_runtime_shutdown"
            )
            for thread in workers:
                thread.join(timeout=1.0)


def run_act_runtime(
    config_path: str | Path,
    *,
    motion_authorization: str | None = None,
    max_steps: int | None = None,
    hardware_start_gate: str | Path | None = None,
    operator_observation_config: str | Path | None = None,
    dig_policy_provider: RuntimeDigPolicyProvider | None = None,
) -> None:
    """Run one standard shadow/motion service with a selected digging Adapter.

    The default provider owns the commissioned LeRobot ACT provenance gates.
    Injected providers receive the selected Runtime mode and must perform the
    corresponding backend-specific provenance checks before returning their
    :class:`DigPolicyFactory`.
    """

    config = load_act_runtime_config(config_path)
    observation_config = (
        None
        if operator_observation_config is None
        else load_collection_config(operator_observation_config)
    )
    if observation_config is not None:
        if observation_config.serial.port != config.serial.port:
            raise ValueError("ACT and operator observation serial devices must match")
        if observation_config.serial.baudrate != config.serial.baudrate:
            raise ValueError("ACT and operator observation serial baudrates must match")
        # The collection config identifies the physical host camera with a
        # stable /dev/v4l/by-path name.  Launchers map that device into the
        # container under the ACT config name (normally /dev/video0), so the
        # two strings intentionally belong to different device namespaces.
        if (
            observation_config.camera.width,
            observation_config.camera.height,
            observation_config.camera.nominal_fps,
        ) != (
            config.camera.width,
            config.camera.height,
            config.camera.nominal_fps,
        ):
            raise ValueError("ACT and operator observation camera formats must match")
    mode = (
        RuntimeMode.MOTION
        if motion_authorization == REQUIRED_MOTION_AUTHORIZATION
        else RuntimeMode.SHADOW
    )
    provider = (
        (
            lambda loaded_config, selected_mode: build_commissioned_lerobot_act_factory(
                loaded_config,
                mode=selected_mode,
            )
        )
        if dig_policy_provider is None
        else dig_policy_provider
    )
    policy_factory = provider(config, mode)
    if not isinstance(policy_factory, DigPolicyFactory):
        raise ValueError("dig policy provider must return DigPolicyFactory")
    session = policy_factory.create(config.dig_policy_backend)
    LOGGER.info(
        "Dig policy selected: backend=%s implementation=%s",
        session.descriptor.backend_id,
        session.descriptor.implementation,
    )
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for online ACT runtime") from exc
    synthetic_warmup_action = warmup_act_policy_session(session)
    LOGGER.info("ACT synthetic CUDA warmup passed: action=%s", synthetic_warmup_action)
    if hardware_start_gate is not None:
        if mode is not RuntimeMode.MOTION:
            raise ValueError("hardware start gate is only valid in motion mode")
        gate_stop = threading.Event()
        previous_gate_handlers: dict[int, Any] = {}

        def stop_prewarm(signum: int, _frame: Any) -> None:
            LOGGER.info("received signal %d; stopping ACT prewarm", signum)
            gate_stop.set()

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_gate_handlers[signum] = signal.signal(
                    signum, stop_prewarm
                )
            if not _wait_for_hardware_start_gate(
                hardware_start_gate, stop_event=gate_stop
            ):
                return
        finally:
            for signum, handler in previous_gate_handlers.items():
                signal.signal(signum, handler)
    controller = ActRuntimeController(
        mode=mode,
        motion_authorization=motion_authorization,
        max_state_age_ms=config.max_inference_state_age_ms,
        max_camera_age_ms=config.max_camera_age_ms,
    )
    engine = ActRuntimeEngine(
        session=session,
        controller=controller,
        max_inference_ms=config.max_inference_ms,
    )
    observations = CausalObservationBuffer()
    log = _JsonlLog(config.log_root, mode)
    serial_port = serial.Serial(
        config.serial.port,
        config.serial.baudrate,
        timeout=0.1,
        write_timeout=0.1,
        exclusive=True,
    )
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
    camera_preview = None
    telemetry_preview = None
    preview_server = None
    machine_state_publisher = None
    if observation_config is not None:
        if observation_config.camera_preview is None:
            raise ValueError("operator observation config must enable camera preview")
        if observation_config.machine_state_udp is None:
            raise ValueError("operator observation config must enable machine-state UDP")
        camera_preview = LatestJpegFrame()
        telemetry_preview = LatestTelemetryFrame()
        preview_server = MjpegPreviewServer(
            camera_preview,
            telemetry=telemetry_preview,
            bind_host=observation_config.camera_preview.bind_host,
            port=observation_config.camera_preview.port,
            allowed_client_host=observation_config.joystick.allowed_pc_host,
        )
        machine_state_publisher = MachineStateUdpPublisher(
            host=observation_config.machine_state_udp.host,
            port=observation_config.machine_state_udp.port,
            machine_id=observation_config.machine_state_udp.machine_id,
        )
    encoder = Stm32ManualCommandEncoder()
    command_channel = Stm32CommandChannel(
        serial_port=serial_port,
        encoder=encoder,
        mode=mode,
        max_state_age_ms=config.state_silence_timeout_ms,
        record_command=log.write,
    )
    processor = ActRuntimeStepProcessor(
        observation_buffer=observations,
        engine=engine,
        command_channel=command_channel,
        record=log.write,
    )
    service = ActRuntimeService(
        serial_port=serial_port,
        camera=camera,
        processor=processor,
        command_channel=command_channel,
        max_steps=max_steps,
        camera_preview=camera_preview,
        telemetry_preview=telemetry_preview,
        preview_server=preview_server,
        machine_state_publisher=machine_state_publisher,
    )
    previous: dict[int, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal %d; stopping ACT runtime", signum)
        service.request_stop()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop)
        LOGGER.info("ACT runtime ready: mode=%s log=%s", mode.value, log.path)
        service.run()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        camera.close()
        serial_port.close()
        if machine_state_publisher is not None:
            machine_state_publisher.close()
        log.close()


def _wait_for_hardware_start_gate(
    path: str | Path,
    *,
    stop_event: threading.Event,
    poll_interval_s: float = 0.05,
) -> bool:
    """Wait after CUDA warmup without opening the serial port or camera."""

    gate = Path(path)
    if not gate.is_absolute():
        raise ValueError("hardware start gate must be an absolute path")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    LOGGER.info(
        "ACT prewarm ready: waiting for hardware start gate: %s", gate
    )
    while not stop_event.is_set():
        try:
            gate.unlink()
        except FileNotFoundError:
            if stop_event.wait(poll_interval_s):
                break
        else:
            LOGGER.info("ACT hardware start gate accepted: %s", gate)
            return True
    LOGGER.info("ACT prewarm cancelled before hardware acquisition")
    return False
