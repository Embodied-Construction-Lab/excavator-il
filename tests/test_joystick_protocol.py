import json
import pytest

from excavator_il.joystick_protocol import (
    authenticate_json_message,
    decode_joystick_packet,
    encode_joystick_packet,
    map_expert_action,
    verify_json_message,
)


def test_authenticated_message_binds_runtime_nonce_and_rejects_tampering():
    message = {"schema_version": "test.v1", "sample_seq": 7}
    key = b"k" * 32

    authenticated = authenticate_json_message(message, key=key, nonce="n" * 64)

    assert verify_json_message(authenticated, key=key, nonce="n" * 64)["sample_seq"] == 7
    tampered = json.loads(authenticated)
    tampered["sample_seq"] = 8
    with pytest.raises(ValueError, match="HMAC"):
        verify_json_message(json.dumps(tampered).encode(), key=key, nonce="n" * 64)


def test_numeric_packet_maps_to_canonical_expert_action_without_sign_change():
    payload = {
        "schema_version": "excavator_joystick.v1",
        "session_id": "pc-session-01",
        "sample_seq": 17,
        "pc_sample_monotonic_ns": 1_000_000_000,
        "pc_sample_wall_ns": 2_000_000_000,
        "axes": {
            "X1": -0.8,
            "Y1": 0.4,
            "Z1": 0.0,
            "X2": 0.2,
            "Y2": -0.6,
            "Z2": 0.0,
        },
        "controllers": [
            {"slot": 1, "device_id": "left-guid", "name": "left", "buttons": [False, True]},
            {"slot": 2, "device_id": "right-guid", "name": "right", "buttons": [True, False]},
        ],
        "deadman_pressed": True,
        "mapping_id": "dual_stick.v1",
        "calibration_id": "raw.v1",
    }

    packet = decode_joystick_packet(json.dumps(payload).encode("utf-8"))
    action = map_expert_action(packet, deadzone=0.15)

    assert action.as_tuple() == (-0.6, 0.4, 0.2, -0.8)
    assert action.valid is True
    assert action.source_sample_seq == 17
    assert packet.session_id == "pc-session-01"

    encoded = encode_joystick_packet(packet)
    assert decode_joystick_packet(encoded) == packet
