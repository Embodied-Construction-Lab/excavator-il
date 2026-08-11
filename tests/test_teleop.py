import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import excavator_il.teleop as teleop
from excavator_il.joystick_protocol import decode_joystick_packet
from excavator_il.teleop import DeviceSnapshot, TeleopConfig, build_joystick_packet


def test_teleop_packet_uses_unrounded_axes_and_stable_device_ids():
    packet = build_joystick_packet(
        sample_seq=3,
        session_id="pc-session-01",
        pc_sample_monotonic_ns=100,
        pc_sample_wall_ns=200,
        devices=(
            DeviceSnapshot("left-guid", "left", (0.123456, -0.2), (False, True)),
            DeviceSnapshot("right-guid", "right", (-0.444444, 0.555555), (True, False)),
        ),
        deadman_pressed=True,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
    )

    assert packet.axes == (0.123456, -0.2, 0.0, -0.444444, 0.555555, 0.0)
    assert packet.controllers[0].device_id == "left-guid"
    assert packet.controllers[1].device_id == "right-guid"


def test_teleop_config_requires_two_devices_and_fixed_20_hz(tmp_path):
    path = tmp_path / "teleop.json"
    value = {
        "schema_version": "excavator_teleop_config.v3",
        "orin_host": "192.168.0.55",
        "orin_port": 18090,
        "rate_hz": 20,
        "mapping_id": "dual_stick.v1",
        "calibration_id": "raw.v1",
        "devices": [
            {
                "device_id": "left-guid",
                "device_path": "/dev/input/by-id/left-event-joystick",
                "axis_indices": [0, 1],
            },
            {
                "device_id": "right-guid",
                "device_path": "/dev/input/by-id/right-event-joystick",
                "axis_indices": [3, 4],
            },
        ],
        "deadman": {"controller_slot": 1, "button_index": 0},
        "startup_gate": {
            "axis_abs_max": 0.15,
            "stable_samples": 10,
            "timeout_s": 5.0,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    config = TeleopConfig.load(path)

    assert config.rate_hz == 20
    assert config.device_ids == ("left-guid", "right-guid")
    assert config.axis_indices[1] == (3, 4)
    assert config.startup_axis_abs_max == 0.15
    assert config.startup_stable_samples == 10
    assert config.startup_timeout_s == 5.0

    value["rate_hz"] = 10
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="rate_hz"):
        TeleopConfig.load(path)

    value["rate_hz"] = 20
    value["schema_version"] = "excavator_teleop_config.v2"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        TeleopConfig.load(path)


def test_same_guid_devices_are_selected_by_stable_paths(tmp_path):
    left_event = tmp_path / "event9"
    right_event = tmp_path / "event10"
    left_event.touch()
    right_event.touch()
    left_path = tmp_path / "left-event-joystick"
    right_path = tmp_path / "right-event-joystick"
    left_path.symlink_to(left_event)
    right_path.symlink_to(right_event)
    guid = "same-guid"
    config_path = tmp_path / "teleop.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_teleop_config.v3",
                "orin_host": "192.168.0.55",
                "orin_port": 18090,
                "rate_hz": 20,
                "mapping_id": "dual_stick.v1",
                "calibration_id": "raw.v1",
                "devices": [
                    {
                        "device_id": guid,
                        "device_path": str(left_path),
                        "axis_indices": [0, 1],
                    },
                    {
                        "device_id": guid,
                        "device_path": str(right_path),
                        "axis_indices": [0, 1],
                    },
                ],
                "deadman": {"controller_slot": 1, "button_index": 0},
            }
        ),
        encoding="utf-8",
    )
    config = TeleopConfig.load(config_path)
    available = (
        teleop.ConnectedJoystick(0, guid, right_event, 20),
        teleop.ConnectedJoystick(1, guid, left_event, 10),
    )

    assert teleop.select_configured_device_indices(config, available) == (1, 0)


def test_teleop_config_rejects_duplicate_device_paths(tmp_path):
    device_path = "/dev/input/by-id/same-event-joystick"
    config_path = tmp_path / "teleop.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_teleop_config.v3",
                "orin_host": "192.168.0.55",
                "orin_port": 18090,
                "rate_hz": 20,
                "mapping_id": "dual_stick.v1",
                "calibration_id": "raw.v1",
                "devices": [
                    {
                        "device_id": "same-guid",
                        "device_path": device_path,
                        "axis_indices": [0, 1],
                    },
                    {
                        "device_id": "same-guid",
                        "device_path": device_path,
                        "axis_indices": [0, 1],
                    },
                ],
                "deadman": {"controller_slot": 1, "button_index": 0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="device_path.*distinct"):
        TeleopConfig.load(config_path)


def test_device_selection_rejects_two_paths_to_one_physical_device(tmp_path):
    event = tmp_path / "event9"
    event.touch()
    first_path = tmp_path / "first-event-joystick"
    second_path = tmp_path / "second-event-joystick"
    first_path.symlink_to(event)
    second_path.symlink_to(event)
    guid = "same-guid"
    config = TeleopConfig(
        orin_host="192.168.0.55",
        orin_port=18090,
        rate_hz=20,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        device_ids=(guid, guid),
        device_paths=(first_path, second_path),
        axis_indices=((0, 1), (0, 1)),
        deadman_slot=1,
        deadman_button=0,
    )
    available = (teleop.ConnectedJoystick(0, guid, event, 10),)

    with pytest.raises(RuntimeError, match="distinct physical devices"):
        teleop.select_configured_device_indices(config, available)


def test_list_joysticks_reports_sdl_device_paths(monkeypatch):
    class FakeJoystick:
        def __init__(self, index):
            self.index = index

        def init(self):
            return None

        def get_guid(self):
            return "same-guid"

        def get_instance_id(self):
            return self.index + 10

        def get_name(self):
            return f"stick-{self.index}"

        def get_numaxes(self):
            return 8

        def get_numbuttons(self):
            return 33

    joystick_module = SimpleNamespace(
        init=lambda: None,
        quit=lambda: None,
        get_count=lambda: 2,
        Joystick=FakeJoystick,
    )
    pygame = SimpleNamespace(
        init=lambda: None,
        quit=lambda: None,
        joystick=joystick_module,
    )
    paths = (b"/dev/input/event9", b"/dev/input/event10")
    monkeypatch.setattr(teleop, "_load_pygame", lambda: pygame)
    monkeypatch.setattr(
        teleop, "_sdl_joystick_path_function", lambda unused: paths.__getitem__
    )

    devices = teleop.list_pygame_devices()

    assert [device["device_path"] for device in devices] == [
        "/dev/input/event9",
        "/dev/input/event10",
    ]
    assert [device["device_id"] for device in devices] == ["same-guid", "same-guid"]


def test_list_joysticks_requires_sdl_2_24_or_newer(monkeypatch):
    pygame = SimpleNamespace(
        init=lambda: None,
        quit=lambda: None,
        get_sdl_version=lambda: (2, 23, 0),
        joystick=SimpleNamespace(
            init=lambda: None,
            quit=lambda: None,
            get_count=lambda: 1,
        ),
    )
    monkeypatch.setattr(teleop, "_load_pygame", lambda: pygame)

    with pytest.raises(RuntimeError, match=r"SDL 2\.24 or newer"):
        teleop.list_pygame_devices()


def test_hotplug_index_reuse_fails_before_udp_socket_creation(tmp_path, monkeypatch):
    left_event = tmp_path / "event9"
    right_event = tmp_path / "event10"
    left_event.touch()
    right_event.touch()
    config = TeleopConfig(
        orin_host="192.168.0.55",
        orin_port=18090,
        rate_hz=20,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        device_ids=("same-guid", "same-guid"),
        device_paths=(left_event, right_event),
        axis_indices=((0, 1), (0, 1)),
        deadman_slot=1,
        deadman_button=0,
    )
    snapshot = (
        teleop.ConnectedJoystick(0, "same-guid", left_event, 10),
        teleop.ConnectedJoystick(1, "same-guid", right_event, 20),
    )

    class ReplacedJoystick:
        def __init__(self, index):
            self.index = index

        def init(self):
            return None

        def get_guid(self):
            return "same-guid"

        def get_instance_id(self):
            return 99 if self.index == 0 else 20

    pygame = SimpleNamespace(
        init=lambda: None,
        quit=lambda: None,
        joystick=SimpleNamespace(
            init=lambda: None,
            quit=lambda: None,
            Joystick=ReplacedJoystick,
        ),
    )
    socket_created = False

    def create_socket(*args, **kwargs):
        nonlocal socket_created
        socket_created = True
        raise AssertionError("UDP socket must not be created")

    monkeypatch.setattr(teleop, "_load_pygame", lambda: pygame)
    monkeypatch.setattr(teleop, "_connected_pygame_devices", lambda unused: snapshot)
    monkeypatch.setattr(teleop.socket, "socket", create_socket)

    with pytest.raises(RuntimeError, match="changed during startup"):
        teleop.run_teleop(config)
    assert socket_created is False


def test_teleop_never_sends_captured_startup_transient(monkeypatch):
    """The FarmStick reports false axes/deadman briefly after SDL opens it."""

    class StopAfterFirstPacket(Exception):
        pass

    class FakeJoystick:
        def __init__(self, slot, pygame):
            self.slot = slot
            self.pygame = pygame

        def get_numaxes(self):
            return 8

        def get_numbuttons(self):
            return 33

        def get_axis(self, index):
            if self.slot == 2 and self.pygame.pump_count <= 2:
                return (-0.635162353515625, 0.823822021484375)[index]
            return 0.003 if index < 2 else 0.0

        def get_button(self, index):
            return bool(
                self.slot == 1
                and index == 22
                and self.pygame.pump_count <= 2
            )

        def get_guid(self):
            return "same-guid"

        def get_name(self):
            return f"slot-{self.slot}"

        def quit(self):
            return None

    class FakePygame:
        def __init__(self):
            self.pump_count = 0
            self.event = SimpleNamespace(pump=self.pump)
            self.joystick = SimpleNamespace(init=lambda: None, quit=lambda: None)

        def pump(self):
            self.pump_count += 1

        def init(self):
            return None

        def quit(self):
            return None

    pygame = FakePygame()
    devices = (FakeJoystick(1, pygame), FakeJoystick(2, pygame))
    config = TeleopConfig(
        orin_host="192.0.2.10",
        orin_port=18090,
        rate_hz=20,
        mapping_id="dual_stick.v1",
        calibration_id="raw.v1",
        device_ids=("same-guid", "same-guid"),
        device_paths=(Path("/dev/input/event9"), Path("/dev/input/event10")),
        axis_indices=((0, 1), (0, 1)),
        deadman_slot=1,
        deadman_button=22,
    )
    socket_created_at = None
    sent_packets = []

    class FakeSocket:
        def setblocking(self, unused):
            return None

        def sendto(self, payload, unused_destination):
            sent_packets.append(decode_joystick_packet(payload))
            raise StopAfterFirstPacket

        def close(self):
            return None

    def create_socket(*unused_args, **unused_kwargs):
        nonlocal socket_created_at
        socket_created_at = pygame.pump_count
        return FakeSocket()

    monkeypatch.setattr(teleop, "_load_pygame", lambda: pygame)
    monkeypatch.setattr(
        teleop, "_open_configured_devices", lambda unused_pygame, unused_config: devices
    )
    monkeypatch.setattr(teleop.socket, "socket", create_socket)
    monkeypatch.setattr(teleop.time, "sleep", lambda unused_seconds: None)

    with pytest.raises(StopAfterFirstPacket):
        teleop.run_teleop(config, print_every=0)

    assert socket_created_at is not None and socket_created_at >= 12
    assert len(sent_packets) == 1
    assert sent_packets[0].deadman_pressed is False
    assert max(abs(value) for value in sent_packets[0].axes) <= 0.15
