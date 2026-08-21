from pathlib import Path
from types import SimpleNamespace

import excavator_il.joystick_diagnostic as joystick_diagnostic
from excavator_il.joystick_diagnostic import (
    AxisEndpointProbe,
    ButtonPress,
    evaluate_joystick_mapping,
    run_joystick_diagnostic,
)
from excavator_il.teleop import TeleopConfig


def _snapshot(
    slot1: tuple[float, ...], slot2: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return slot1, slot2


def test_mapping_diagnostic_detects_raw_axes_and_deadman_without_transforming_values():
    neutral = (0.0,) * 8
    probes = (
        AxisEndpointProbe("X1", 1, _snapshot(neutral, neutral), _snapshot((-0.98,) + neutral[1:], neutral), _snapshot((0.99,) + neutral[1:], neutral), _snapshot(neutral, neutral)),
        AxisEndpointProbe("Y1", 1, _snapshot(neutral, neutral), _snapshot((0.0, -0.97) + neutral[2:], neutral), _snapshot((0.0, 0.98) + neutral[2:], neutral), _snapshot(neutral, neutral)),
        AxisEndpointProbe("Z1", 1, _snapshot(neutral, neutral), _snapshot(neutral[:4] + (-0.96,) + neutral[5:], neutral), _snapshot(neutral[:4] + (0.97,) + neutral[5:], neutral), _snapshot(neutral, neutral)),
        AxisEndpointProbe("X2", 2, _snapshot(neutral, neutral), _snapshot(neutral, (-0.99,) + neutral[1:]), _snapshot(neutral, (0.98,) + neutral[1:]), _snapshot(neutral, neutral)),
        AxisEndpointProbe("Y2", 2, _snapshot(neutral, neutral), _snapshot(neutral, (0.0, -0.95) + neutral[2:]), _snapshot(neutral, (0.0, 0.96) + neutral[2:]), _snapshot(neutral, neutral)),
        AxisEndpointProbe("Z2", 2, _snapshot(neutral, neutral), _snapshot(neutral, neutral[:4] + (-0.94,) + neutral[5:]), _snapshot(neutral, neutral[:4] + (0.95,) + neutral[5:]), _snapshot(neutral, neutral)),
    )
    config = TeleopConfig(
        orin_host="192.168.31.10",
        orin_port=18090,
        rate_hz=20,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        device_ids=("same-guid", "same-guid"),
        device_paths=(__file__, __file__ + ".right"),
        axis_indices=((0, 1, 4), (0, 1, 4)),
        deadman_slot=1,
        deadman_button=22,
    )

    report = evaluate_joystick_mapping(
        config,
        probes,
        (ButtonPress(1, 22), ButtonPress(1, 22), ButtonPress(1, 22)),
    )

    assert report.detected_axis_indices == ((0, 1, 4), (0, 1, 4))
    assert report.detected_deadman == (1, 22)
    assert report.matches_config is True
    assert report.axis_results[0].first_value == -0.98
    assert report.axis_results[0].second_value == 0.99


def test_mapping_diagnostic_fails_closed_for_ambiguous_axis_or_button():
    neutral = (0.0,) * 8
    ambiguous = AxisEndpointProbe(
        "X1",
        1,
        _snapshot(neutral, neutral),
        _snapshot((-0.9, -0.8) + neutral[2:], neutral),
        _snapshot((0.9, 0.8) + neutral[2:], neutral),
        _snapshot(neutral, neutral),
    )
    config = TeleopConfig(
        orin_host="192.168.31.10",
        orin_port=18090,
        rate_hz=20,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        device_ids=("same-guid", "same-guid"),
        device_paths=(__file__, __file__ + ".right"),
        axis_indices=((0, 1, 4), (0, 1, 4)),
        deadman_slot=1,
        deadman_button=0,
    )

    report = evaluate_joystick_mapping(
        config,
        (ambiguous,),
        (ButtonPress(1, 0), ButtonPress(1, 1)),
    )

    assert report.matches_config is False
    assert report.axis_results[0].valid is False
    assert "ambiguous" in report.axis_results[0].reason
    assert report.detected_deadman is None


def test_interactive_diagnostic_captures_all_axes_and_releases_devices(monkeypatch):
    class FakeDevice:
        def __init__(self, slot):
            self.slot = slot
            self.axes = (0.0,) * 8
            self.closed = False

        def get_axis(self, axis):
            return self.axes[axis]

        def get_numaxes(self):
            return len(self.axes)

        def get_numbuttons(self):
            return 33

        def get_instance_id(self):
            return self.slot + 10

        def get_name(self):
            return f"stick-{self.slot}"

        def get_guid(self):
            return "same-guid"

        def quit(self):
            self.closed = True

    devices = (FakeDevice(1), FakeDevice(2))
    pygame = SimpleNamespace(
        init=lambda: None,
        quit=lambda: None,
        joystick=SimpleNamespace(init=lambda: None, quit=lambda: None),
        event=SimpleNamespace(pump=lambda: None),
    )
    neutral = (0.0,) * 8

    def endpoint(slot, axis, value):
        changed = neutral[:axis] + (value,) + neutral[axis + 1 :]
        return (changed, neutral) if slot == 1 else (neutral, changed)

    snapshots = []
    for slot, axis in ((1, 0), (1, 1), (1, 4), (2, 0), (2, 1), (2, 4)):
        snapshots.extend(
            [
                (neutral, neutral),
                endpoint(slot, axis, -0.95),
                endpoint(slot, axis, 0.96),
                (neutral, neutral),
            ]
        )
    snapshot_iter = iter(snapshots)

    def respond(unused_prompt):
        slot1, slot2 = next(snapshot_iter)
        devices[0].axes = slot1
        devices[1].axes = slot2
        return ""

    config = TeleopConfig(
        orin_host="192.168.31.10",
        orin_port=18090,
        rate_hz=20,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        device_ids=("same-guid", "same-guid"),
        device_paths=(Path("/dev/input/left"), Path("/dev/input/right")),
        axis_indices=((0, 1, 4), (0, 1, 4)),
        deadman_slot=1,
        deadman_button=22,
    )
    output = []
    monkeypatch.setattr(joystick_diagnostic, "_load_pygame", lambda: pygame)
    monkeypatch.setattr(
        joystick_diagnostic,
        "_open_configured_devices",
        lambda unused_pygame, unused_config: devices,
    )
    monkeypatch.setattr(joystick_diagnostic.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        joystick_diagnostic,
        "_capture_button_presses",
        lambda *args, **kwargs: (
            ButtonPress(1, 22),
            ButtonPress(1, 22),
            ButtonPress(1, 22),
        ),
    )

    report = run_joystick_diagnostic(
        config, input_fn=respond, output_fn=output.append
    )

    assert report.matches_config is True
    assert report.detected_axis_indices == ((0, 1, 4), (0, 1, 4))
    assert all(device.closed for device in devices)
    assert any("LOCAL_ONLY" in line for line in output)
