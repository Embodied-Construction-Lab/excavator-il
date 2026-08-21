"""Local, fail-closed diagnostics for dual-joystick identity and mapping."""

from __future__ import annotations

import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .teleop import (
    TeleopConfig,
    _load_pygame,
    _open_configured_devices,
)


RawJoystickSnapshot = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class AxisEndpointProbe:
    logical_axis: str
    expected_slot: int
    center: RawJoystickSnapshot
    first: RawJoystickSnapshot
    second: RawJoystickSnapshot
    returned_center: RawJoystickSnapshot


@dataclass(frozen=True)
class ButtonPress:
    slot: int
    button: int


@dataclass(frozen=True)
class AxisProbeResult:
    logical_axis: str
    valid: bool
    detected_slot: int | None
    detected_axis: int | None
    first_value: float | None
    second_value: float | None
    returned_value: float | None
    reason: str


@dataclass(frozen=True)
class JoystickMappingReport:
    axis_results: tuple[AxisProbeResult, ...]
    detected_axis_indices: tuple[tuple[int, ...], tuple[int, ...]]
    detected_deadman: tuple[int, int] | None
    matches_config: bool


@dataclass(frozen=True)
class _AxisPrompt:
    logical_axis: str
    slot: int
    first_instruction: str
    second_instruction: str


def _evaluate_axis_probe(
    probe: AxisEndpointProbe,
    *,
    min_excursion: float = 0.5,
    min_span: float = 1.0,
    center_tolerance: float = 0.15,
) -> AxisProbeResult:
    candidates: list[tuple[int, int]] = []
    for slot_index, center_axes in enumerate(probe.center):
        snapshots = (
            probe.first[slot_index],
            probe.second[slot_index],
            probe.returned_center[slot_index],
        )
        if any(len(axes) != len(center_axes) for axes in snapshots):
            return AxisProbeResult(
                probe.logical_axis, False, None, None, None, None, None, "axis count changed during probe"
            )
        for axis, center_value in enumerate(center_axes):
            first_value = probe.first[slot_index][axis]
            second_value = probe.second[slot_index][axis]
            if (
                max(abs(first_value - center_value), abs(second_value - center_value))
                >= min_excursion
                and abs(second_value - first_value) >= min_span
            ):
                candidates.append((slot_index + 1, axis))

    if len(candidates) != 1:
        reason = "no axis reached both endpoints" if not candidates else "ambiguous: multiple axes reached both endpoints"
        return AxisProbeResult(
            probe.logical_axis, False, None, None, None, None, None, reason
        )

    detected_slot, detected_axis = candidates[0]
    first_value = probe.first[detected_slot - 1][detected_axis]
    second_value = probe.second[detected_slot - 1][detected_axis]
    returned_value = probe.returned_center[detected_slot - 1][detected_axis]
    center_value = probe.center[detected_slot - 1][detected_axis]
    if detected_slot != probe.expected_slot:
        reason = f"moved slot {detected_slot}, expected slot {probe.expected_slot}"
        valid = False
    elif abs(returned_value - center_value) > center_tolerance:
        reason = "axis did not return to its starting center"
        valid = False
    else:
        reason = "ok"
        valid = True
    return AxisProbeResult(
        probe.logical_axis,
        valid,
        detected_slot,
        detected_axis,
        first_value,
        second_value,
        returned_value,
        reason,
    )


def evaluate_joystick_mapping(
    config: TeleopConfig,
    probes: Sequence[AxisEndpointProbe],
    button_presses: Sequence[ButtonPress],
) -> JoystickMappingReport:
    """Evaluate captured raw endpoints without applying inversion or scaling."""
    results = tuple(_evaluate_axis_probe(probe) for probe in probes)
    by_name = {
        result.logical_axis: result
        for result in results
        if result.valid and result.detected_axis is not None
    }
    ordered_names = (("X1", "Y1", "Z1"), ("X2", "Y2", "Z2"))
    detected_indices = tuple(
        tuple(by_name[name].detected_axis for name in names if name in by_name)
        for names in ordered_names
    )

    counts = Counter((press.slot, press.button) for press in button_presses)
    repeated = tuple(identity for identity, count in counts.items() if count >= 2)
    detected_deadman = repeated[0] if len(repeated) == 1 and len(counts) == 1 else None

    configured_axis_indices = config.axis_indices
    complete = all(len(indices) == 3 for indices in detected_indices)
    matches_config = (
        complete
        and detected_indices == configured_axis_indices
        and detected_deadman == (config.deadman_slot, config.deadman_button)
        and all(result.valid for result in results)
    )
    return JoystickMappingReport(
        axis_results=results,
        detected_axis_indices=detected_indices,
        detected_deadman=detected_deadman,
        matches_config=matches_config,
    )


def _capture_snapshot(pygame: object, devices: tuple[object, object]) -> RawJoystickSnapshot:
    samples: list[list[tuple[float, ...]]] = [[], []]
    for _ in range(7):
        pygame.event.pump()
        for slot_index, device in enumerate(devices):
            axes = tuple(
                float(device.get_axis(axis)) for axis in range(device.get_numaxes())
            )
            if any(not math.isfinite(value) for value in axes):
                raise RuntimeError("joystick reported a non-finite axis value")
            samples[slot_index].append(axes)
        time.sleep(0.01)

    medians: list[tuple[float, ...]] = []
    for slot_samples in samples:
        axis_count = len(slot_samples[0])
        if any(len(sample) != axis_count for sample in slot_samples):
            raise RuntimeError("joystick axis count changed during sampling")
        medians.append(
            tuple(
                statistics.median(sample[axis] for sample in slot_samples)
                for axis in range(axis_count)
            )
        )
    return medians[0], medians[1]


def _prompt_snapshot(
    pygame: object,
    devices: tuple[object, object],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    message: str,
) -> RawJoystickSnapshot:
    input_fn(f"{message}\n保持该位置并按 Enter 采样：")
    snapshot = _capture_snapshot(pygame, devices)
    rendered = tuple(tuple(round(value, 3) for value in axes) for axes in snapshot)
    output_fn(f"  raw={rendered}")
    return snapshot


def _capture_button_presses(
    pygame: object,
    devices: tuple[object, object],
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    *,
    duration_s: float = 6.0,
) -> tuple[ButtonPress, ...]:
    input_fn(
        "\n[DEADMAN] 松开所有手柄按钮。按 Enter 后，在 6 秒内只按放计划作为 "
        "deadman 的按钮 3 次："
    )
    pygame.event.get()
    instance_to_slot = {
        device.get_instance_id(): slot
        for slot, device in enumerate(devices, start=1)
    }
    presses: list[ButtonPress] = []
    deadline = time.monotonic() + duration_s
    next_notice = math.ceil(duration_s)
    while time.monotonic() < deadline:
        for event in pygame.event.get():
            if event.type != pygame.JOYBUTTONDOWN:
                continue
            slot = instance_to_slot.get(getattr(event, "instance_id", None))
            if slot is None:
                continue
            press = ButtonPress(slot, int(event.button))
            presses.append(press)
            output_fn(f"  detected: SLOT{press.slot} button {press.button} DOWN")
        remaining = deadline - time.monotonic()
        rounded = math.ceil(max(0.0, remaining))
        if rounded < next_notice:
            next_notice = rounded
            output_fn(f"  remaining: {rounded}s")
        time.sleep(0.01)
    return tuple(presses)


def run_joystick_diagnostic(
    config: TeleopConfig,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> JoystickMappingReport:
    """Interactively capture raw joystick mapping without opening any I/O link."""
    pygame = _load_pygame()
    pygame.init()
    pygame.joystick.init()
    devices: tuple[object, object] | None = None
    prompts = (
        _AxisPrompt("X1", 1, "左手柄主杆向左到底", "左手柄主杆向右到底"),
        _AxisPrompt("Y1", 1, "左手柄主杆向前到底", "左手柄主杆向后到底"),
        _AxisPrompt("Z1", 1, "左履带控制轴推到一端", "左履带控制轴推到另一端"),
        _AxisPrompt("X2", 2, "右手柄主杆向左到底", "右手柄主杆向右到底"),
        _AxisPrompt("Y2", 2, "右手柄主杆向前到底", "右手柄主杆向后到底"),
        _AxisPrompt("Z2", 2, "右履带控制轴推到一端", "右履带控制轴推到另一端"),
    )
    try:
        devices = _open_configured_devices(pygame, config)
        output_fn("LOCAL_ONLY: 不创建 socket，不发送 UDP，不访问 Orin/STM32。")
        for slot, (device, path) in enumerate(
            zip(devices, config.device_paths, strict=True), start=1
        ):
            output_fn(
                f"SLOT{slot}: {device.get_name()} guid={device.get_guid()} "
                f"axes={device.get_numaxes()} buttons={device.get_numbuttons()} path={path}"
            )

        probes: list[AxisEndpointProbe] = []
        for prompt in prompts:
            center = _prompt_snapshot(
                pygame,
                devices,
                input_fn,
                output_fn,
                f"\n[{prompt.logical_axis}] 两只手柄回到起始中位，其他控件不要动。",
            )
            first = _prompt_snapshot(
                pygame, devices, input_fn, output_fn, prompt.first_instruction
            )
            second = _prompt_snapshot(
                pygame, devices, input_fn, output_fn, prompt.second_instruction
            )
            returned_center = _prompt_snapshot(
                pygame,
                devices,
                input_fn,
                output_fn,
                "该控件回到本轮起始中位，其他控件保持不动。",
            )
            probes.append(
                AxisEndpointProbe(
                    prompt.logical_axis,
                    prompt.slot,
                    center,
                    first,
                    second,
                    returned_center,
                )
            )

        presses = _capture_button_presses(
            pygame, devices, input_fn, output_fn
        )
        report = evaluate_joystick_mapping(config, probes, presses)
        output_fn("\n诊断结果：")
        for result in report.axis_results:
            status = "PASS" if result.valid else "FAIL"
            output_fn(
                f"  {status} {result.logical_axis}: slot={result.detected_slot} "
                f"axis={result.detected_axis} first={result.first_value} "
                f"second={result.second_value} return={result.returned_value} "
                f"reason={result.reason}"
            )
        output_fn(f"  detected_axis_indices={report.detected_axis_indices}")
        output_fn(
            f"  configured_axis_indices={config.axis_indices}"
        )
        output_fn(f"  detected_deadman={report.detected_deadman}")
        output_fn(
            f"  configured_deadman={(config.deadman_slot, config.deadman_button)}"
        )
        output_fn(f"  matches_config={report.matches_config}")
        return report
    finally:
        if devices is not None:
            for device in devices:
                device.quit()
        pygame.joystick.quit()
        pygame.quit()
