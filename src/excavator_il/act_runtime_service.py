"""Online Orin ACT I/O service with a no-write shadow boundary."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import queue
import signal
import threading
import time
from typing import Any, Callable

import torch
from lerobot.policies import get_policy_class, make_pre_post_processors

from .act_runtime import (
    ActPolicySession,
    ActRuntimeController,
    ActRuntimeDecision,
    ActRuntimeEngine,
    CausalObservationBuffer,
    REQUIRED_MOTION_AUTHORIZATION,
    RuntimeMode,
    warmup_act_policy_session,
)
from .act_runtime_config import ActRuntimeConfig, load_act_runtime_config
from .act_deployment import verify_deployment_manifest
from .collector.camera import UvcCamera
from .collector.config import CameraConfig
from .stm32_protocol import (
    Stm32ManualCommandEncoder,
    Stm32TelemetryFrame,
    Stm32TelemetryParser,
)


LOGGER = logging.getLogger("excavator_il.act_runtime")


@dataclass(frozen=True)
class CommandWriteResult:
    requested_axes: tuple[float, ...]
    effective_axes: tuple[float, ...]
    command_seq: int | None
    final_gate_reason: str
    write_performed: bool


class LatestStateQueue:
    """One-slot queue so GPU latency cannot replay a telemetry backlog."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._dropped_count = 0
        self._lock = threading.Lock()
        self._dropped_since_get = 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def put(self, frame: Any, *, external_gap_count: int = 0) -> None:
        if external_gap_count < 0:
            raise ValueError("external gap count must be non-negative")
        if external_gap_count:
            with self._lock:
                self._dropped_count += external_gap_count
                self._dropped_since_get += external_gap_count
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            self._dropped_count += 1
            self._dropped_since_get += 1
        self._queue.put_nowait(frame)

    def get(self, *, timeout_s: float) -> tuple[Any, int]:
        frame = self._queue.get(timeout=timeout_s)
        with self._lock:
            dropped = self._dropped_since_get
            self._dropped_since_get = 0
        return frame, dropped


class SensorSequenceTracker:
    """Detect missing or reset 10 Hz sensor states, including uint32 wrap."""

    def __init__(self) -> None:
        self._previous: int | None = None

    def observe(self, sensor_seq: int) -> int:
        sequence = int(sensor_seq) & 0xFFFFFFFF
        if self._previous is None:
            self._previous = sequence
            return 0
        expected = (self._previous + 1) & 0xFFFFFFFF
        self._previous = sequence
        return 0 if sequence == expected else 1


class Stm32CommandChannel:
    """Enforce the shadow no-write invariant at the physical serial boundary."""

    def __init__(
        self,
        *,
        serial_port: Any,
        encoder: Stm32ManualCommandEncoder,
        mode: RuntimeMode,
        max_state_age_ms: float = 100.0,
        record_command: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_state_age_ms <= 0:
            raise ValueError("runtime state timeout must be positive")
        self._serial = serial_port
        self._encoder = encoder
        self._mode = RuntimeMode(mode)
        self._max_state_age_ns = int(max_state_age_ms * 1_000_000)
        self._synchronized = False
        self._terminally_disarmed = False
        self._state_monotonic_ns: int | None = None
        self._state_timeout_zero_sent = False
        self._state_generation = 0
        self._state_safe = False
        self._motion_epoch = 0
        self._record_command = record_command
        self._lock = threading.Lock()

    @property
    def mode(self) -> RuntimeMode:
        return self._mode

    @property
    def synchronized(self) -> bool:
        return self._synchronized

    def synchronize(self, frame: Stm32TelemetryFrame) -> int:
        with self._lock:
            self._synchronized = True
            return self._encoder.synchronize(frame)

    @property
    def motion_epoch(self) -> int:
        with self._lock:
            return self._motion_epoch

    def _interrupt_motion_locked(self) -> None:
        self._motion_epoch += 1

    def _write_locked(
        self, axes: tuple[float, ...], monotonic_ns: int, *, reason: str
    ) -> int | None:
        if self._mode is RuntimeMode.SHADOW:
            return None
        if not self._synchronized:
            raise RuntimeError("STM32 command sequence is not synchronized")
        command_sequence = self._encoder.next_sequence
        payload = self._encoder.encode(axes=axes, monotonic_ns=monotonic_ns)
        written = self._serial.write(payload)
        if written != len(payload):
            raise OSError(f"short serial write: {written}/{len(payload)} bytes")
        self._serial.flush()
        if self._record_command is not None:
            self._record_command(
                {
                    "schema_version": "excavator_act_runtime_command.v1",
                    "command_monotonic_ns": monotonic_ns,
                    "command_seq": command_sequence,
                    "serial_axes": list(axes),
                    "reason": reason,
                    "serial_write_performed": True,
                }
            )
        return command_sequence

    def write_axes(
        self,
        axes: tuple[float, ...],
        *,
        monotonic_ns: int,
        state_generation: int | None = None,
        motion_epoch: int | None = None,
    ) -> CommandWriteResult:
        with self._lock:
            requested = axes
            if self._terminally_disarmed:
                return CommandWriteResult(
                    requested, (0.0,) * 6, None, "terminally_disarmed", False
                )
            nonzero = any(abs(value) > 1e-12 for value in axes)
            expected_motion_epoch = (
                self._motion_epoch if motion_epoch is None else motion_epoch
            )
            gate_reason = "accepted"
            if nonzero and expected_motion_epoch != self._motion_epoch:
                axes = (0.0,) * 6
                gate_reason = "motion_interrupted"
            state_fresh = (
                self._state_monotonic_ns is None
                or (
                    monotonic_ns >= self._state_monotonic_ns
                    and monotonic_ns - self._state_monotonic_ns
                    <= self._max_state_age_ns
                )
            )
            if nonzero and (
                state_generation is None
                or state_generation != self._state_generation
                or not state_fresh
                or not self._state_safe
                or self._state_timeout_zero_sent
            ):
                axes = (0.0,) * 6
                gate_reason = "state_not_fresh_or_current"
            sequence = self._write_locked(axes, monotonic_ns, reason=gate_reason)
            return CommandWriteResult(
                requested_axes=requested,
                effective_axes=axes,
                command_seq=sequence,
                final_gate_reason=(
                    "shadow_no_write"
                    if self._mode is RuntimeMode.SHADOW
                    else gate_reason
                ),
                write_performed=self._mode is RuntimeMode.MOTION,
            )

    def safe_zero(self, *, monotonic_ns: int, reason: str) -> int | None:
        if not reason:
            raise ValueError("safe-zero reason must be non-empty")
        with self._lock:
            return self._write_locked((0.0,) * 6, monotonic_ns, reason=reason)

    def update_state(self, frame: Stm32TelemetryFrame) -> int:
        with self._lock:
            was_safe = self._state_safe
            self._state_monotonic_ns = frame.receive_monotonic_ns
            values = frame.values
            self._state_safe = (
                all(
                    int(values.get(field, 0)) == 1
                    for field in ("control_enabled", "rs485_ok", "dwj_ok", "imu_ok")
                )
                and int(values.get("estop", 1)) == 0
                and int(values.get("fault_flags", 1)) == 0
            )
            safety_changed = self._state_safe != was_safe
            if frame.sensor_is_new or safety_changed:
                self._state_generation += 1
            self._state_timeout_zero_sent = False
            if was_safe and not self._state_safe and not self._terminally_disarmed:
                self._interrupt_motion_locked()
                self._write_locked(
                    (0.0,) * 6,
                    frame.receive_monotonic_ns,
                    reason="unsafe_telemetry",
                )
            return self._state_generation

    def enforce_state_timeout(self, *, monotonic_ns: int) -> bool:
        with self._lock:
            expired = (
                self._state_monotonic_ns is not None
                and monotonic_ns - self._state_monotonic_ns > self._max_state_age_ns
            )
            if not expired or self._state_timeout_zero_sent:
                return False
            self._state_timeout_zero_sent = True
            self._interrupt_motion_locked()
            self._write_locked((0.0,) * 6, monotonic_ns, reason="state_timeout")
            return True

    def terminal_disarm(self, *, monotonic_ns: int, reason: str) -> None:
        if not reason:
            raise ValueError("terminal disarm reason must be non-empty")
        with self._lock:
            if self._terminally_disarmed:
                return
            self._terminally_disarmed = True
            self._interrupt_motion_locked()
            if self._synchronized:
                self._write_locked((0.0,) * 6, monotonic_ns, reason=reason)


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


def _verify_checkpoint(config: ActRuntimeConfig) -> None:
    actual_names = {
        path.name for path in config.checkpoint_path.iterdir() if path.is_file()
    }
    expected_names = set(config.checkpoint_files_sha256)
    if actual_names != expected_names:
        raise ValueError("ACT checkpoint file set does not match runtime provenance")
    for name, expected in config.checkpoint_files_sha256.items():
        digest = hashlib.sha256((config.checkpoint_path / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"ACT checkpoint SHA-256 mismatch: {name}")


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
) -> None:
    """Update 20 Hz safety immediately; enqueue only new 10 Hz observations."""

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
    ) -> None:
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
            self._observations.add_camera(self._camera.read_rgb())

    def _serial_loop(self) -> None:
        while not self._stop.is_set():
            line = self._serial.readline()
            if not line:
                continue
            frame = self._parser.parse_line(
                line, receive_monotonic_ns=time.monotonic_ns()
            )
            if frame is not None:
                _route_telemetry_frame(
                    frame=frame,
                    command_channel=self._command_channel,
                    states=self._states,
                    sensor_sequences=self._sensor_sequences,
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

    def run(self) -> None:
        camera_worker = threading.Thread(
            target=self._worker, args=(self._camera_loop,), daemon=True
        )
        camera_worker.start()
        workers = [camera_worker]
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
            self._command_channel.terminal_disarm(
                monotonic_ns=time.monotonic_ns(), reason="act_runtime_shutdown"
            )
            for thread in workers:
                thread.join(timeout=1.0)


def run_act_runtime(
    config_path: str | Path, *, motion_authorization: str | None = None
) -> None:
    config = load_act_runtime_config(config_path)
    _verify_checkpoint(config)
    mode = (
        RuntimeMode.MOTION
        if motion_authorization == REQUIRED_MOTION_AUTHORIZATION
        else RuntimeMode.SHADOW
    )
    if mode is RuntimeMode.MOTION:
        verify_deployment_manifest(
            manifest_path=config.deployment_manifest_path,
            checkpoint_path=config.checkpoint_path,
            machine_profile_path=config.machine_profile_path,
        )
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for online ACT runtime") from exc
    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(config.checkpoint_path)
    policy.to(config.device)
    policy.config.device = config.device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(config.checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
        postprocessor_overrides={"device_processor": {"device": config.device}},
    )
    session = ActPolicySession(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=config.device,
    )
    synthetic_warmup_action = warmup_act_policy_session(session)
    LOGGER.info("ACT synthetic CUDA warmup passed: action=%s", synthetic_warmup_action)
    _verify_checkpoint(config)
    if mode is RuntimeMode.MOTION:
        verify_deployment_manifest(
            manifest_path=config.deployment_manifest_path,
            checkpoint_path=config.checkpoint_path,
            machine_profile_path=config.machine_profile_path,
        )
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
            jpeg_quality=95,
        )
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
        log.close()
