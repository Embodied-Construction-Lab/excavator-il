import json

import pytest

from excavator_il.teleop import DeviceSnapshot, TeleopConfig, build_joystick_packet


def test_teleop_packet_uses_unrounded_axes_and_stable_device_ids():
    packet = build_joystick_packet(
        sample_seq=3,
        session_id="pc-session-01",
        pc_sample_monotonic_ns=100,
        pc_sample_wall_ns=200,
        devices=(
            DeviceSnapshot("left-guid", "left", (0.123456, -0.2, 0.333333), (False, True)),
            DeviceSnapshot("right-guid", "right", (-0.444444, 0.555555, -0.6), (True, False)),
        ),
        deadman_pressed=True,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )

    assert packet.axes == (0.123456, -0.2, 0.333333, -0.444444, 0.555555, -0.6)
    assert packet.controllers[0].device_id == "left-guid"
    assert packet.controllers[1].device_id == "right-guid"


def test_teleop_config_requires_two_devices_and_fixed_20_hz(tmp_path):
    path = tmp_path / "teleop.json"
    value = {
        "schema_version": "excavator_teleop_config.v1",
        "orin_host": "192.168.0.55",
        "orin_port": 18090,
        "rate_hz": 20,
        "mapping_id": "dual_stick.v1",
        "calibration_id": "raw.v1",
        "devices": [
            {"device_id": "left-guid", "axis_indices": [0, 1, 2]},
            {"device_id": "right-guid", "axis_indices": [3, 4, 5]},
        ],
        "deadman": {"controller_slot": 1, "button_index": 0},
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    config = TeleopConfig.load(path)

    assert config.rate_hz == 20
    assert config.device_ids == ("left-guid", "right-guid")
    assert config.axis_indices[1] == (3, 4, 5)

    value["rate_hz"] = 10
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="rate_hz"):
        TeleopConfig.load(path)
