import json
import os
import socket
import threading

import pytest

import excavator_il.resident_protocol as resident_protocol_module
from excavator_il.resident_protocol import (
    ACTION_ORDER,
    ACT_STATE_NAMES,
    ResidentActDataClient,
    ResidentActOwnerClosed,
    ResidentActState,
    ResidentPolicyCandidate,
    decode_policy_candidate,
    decode_resident_state,
    encode_policy_candidate,
    encode_resident_state,
)


def _state(**overrides):
    values = {
        "state": (1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        "receive_monotonic_ns": 2_000,
        "state_monotonic_ns": 1_900,
        "control_seq": 7,
        "sensor_seq": 11,
        "sensor_is_new": True,
        "control_enabled": True,
        "estop": False,
        "rs485_ok": True,
        "dwj_ok": True,
        "imu_ok": True,
        "sensor_valid": True,
        "stm32_alive": True,
        "fault_flags": 0,
        "control_generation": 4,
    }
    values.update(overrides)
    return ResidentActState(**values)


def _recv_exact(connection, length):
    chunks = []
    while sum(map(len, chunks)) < length:
        chunk = connection.recv(length - sum(map(len, chunks)))
        if not chunk:
            raise EOFError
        chunks.append(chunk)
    return b"".join(chunks)


def test_data_client_uses_strict_bidirectional_length_framing(tmp_path):
    path = tmp_path / "runtime" / "act.sock"
    path.parent.mkdir()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.fspath(path))
    listener.listen(1)
    received = []

    def serve():
        connection, _ = listener.accept()
        with connection:
            payload = encode_resident_state(_state())
            framed = len(payload).to_bytes(4, "big") + payload
            for byte in framed:
                connection.sendall(bytes((byte,)))
            size = int.from_bytes(_recv_exact(connection, 4), "big")
            received.append(decode_policy_candidate(_recv_exact(connection, size)))

    server = threading.Thread(target=serve)
    server.start()
    client = ResidentActDataClient(path)
    try:
        client.connect(timeout_s=0.5)
        assert client.receive_state(timeout_s=0.5) == _state()
        candidate = ResidentPolicyCandidate(
            source="act_dig",
            control_generation=4,
            mode="manual_action",
            action=(0.1, -0.2, 0.3, 0.0),
            created_monotonic_ns=2_100,
            valid_until_monotonic_ns=2_200,
        )
        client.send_candidate(candidate)
    finally:
        client.close()
        server.join(timeout=1.0)
        listener.close()

    assert received == [candidate]


def test_data_client_distinguishes_owner_eof_from_protocol_failure():
    client_connection, owner_connection = socket.socketpair()
    client = ResidentActDataClient("/tmp/resident-act-owner-eof.sock")
    client._connection = client_connection
    owner_connection.close()
    try:
        with pytest.raises(ResidentActOwnerClosed, match="owner closed"):
            client.receive_state(timeout_s=0.1)
    finally:
        client.close()


def test_data_client_retries_nonblocking_would_block_and_partial_io(monkeypatch):
    state_payload = encode_resident_state(_state())
    inbound = len(state_payload).to_bytes(4, "big") + state_payload

    class _Socket:
        def __init__(self):
            self.receive_calls = 0
            self.inbound = bytearray(inbound)
            self.outbound = bytearray()
            self.send_calls = 0

        def recv(self, size):
            self.receive_calls += 1
            if self.receive_calls == 1:
                raise BlockingIOError
            chunk = bytes(self.inbound[:size])
            del self.inbound[:size]
            return chunk

        def send(self, payload):
            self.send_calls += 1
            if self.send_calls == 1:
                raise BlockingIOError
            count = min(7, len(payload))
            self.outbound.extend(payload[:count])
            return count

    connection = _Socket()
    client = ResidentActDataClient("/tmp/resident-act-would-block.sock")
    client._connection = connection
    monkeypatch.setattr(
        resident_protocol_module.select,
        "select",
        lambda readable, writable, exceptional, _timeout: (
            readable,
            writable,
            exceptional,
        ),
    )

    assert client.receive_state(timeout_s=0.1) == _state()
    candidate = ResidentPolicyCandidate(
        source="act_dig",
        control_generation=4,
        mode="manual_action",
        action=(0.1, -0.2, 0.3, 0.0),
        created_monotonic_ns=2_100,
        valid_until_monotonic_ns=2_200,
    )
    client.send_candidate(candidate)

    size = int.from_bytes(connection.outbound[:4], "big")
    assert decode_policy_candidate(bytes(connection.outbound[4:])) == candidate
    assert size == len(connection.outbound) - 4


def test_data_client_reports_its_connect_deadline_as_timeout(monkeypatch):
    class _Socket:
        def close(self):
            return None

    monotonic_values = iter((10.0, 10.2))
    monkeypatch.setattr(
        resident_protocol_module.socket,
        "socket",
        lambda *_args: _Socket(),
    )
    monkeypatch.setattr(
        resident_protocol_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    client = ResidentActDataClient("/tmp/resident-act-connect-timeout.sock")

    with pytest.raises(TimeoutError, match="owner socket is unavailable"):
        client.connect(timeout_s=0.1)


def test_resident_state_round_trip_preserves_the_canonical_act_contract():
    state = _state()

    encoded = encode_resident_state(state)

    assert decode_resident_state(encoded) == state
    wire = json.loads(encoded)
    assert wire["schema_version"] == "resident_act_state.v1"
    assert wire["state_names"] == list(ACT_STATE_NAMES)
    assert wire["control_generation"] == 4


def test_policy_candidate_round_trip_matches_the_orin_manual_action_contract():
    candidate = ResidentPolicyCandidate(
        source="act_dig",
        control_generation=4,
        mode="manual_action",
        action=(0.1, -0.2, 0.3, 0.0),
        created_monotonic_ns=2_100,
        valid_until_monotonic_ns=2_200,
    )

    encoded = encode_policy_candidate(candidate)

    assert decode_policy_candidate(encoded) == candidate
    wire = json.loads(encoded)
    assert wire == {
        "schema_version": "resident_policy_candidate.v2",
        "source": "act_dig",
        "control_generation": 4,
        "mode": "manual_action",
        "action_order": list(ACTION_ORDER),
        "action": [0.1, -0.2, 0.3, 0.0],
        "action_chunk": None,
        "created_monotonic_ns": 2_100,
        "valid_until_monotonic_ns": 2_200,
    }


def test_new_action_chunk_round_trip_is_exactly_ten_normalized_actions():
    chunk = tuple((0.01 * index, 0.0, -0.01 * index, 0.0) for index in range(10))
    candidate = ResidentPolicyCandidate(
        source="act_dig",
        control_generation=4,
        mode="manual_action",
        action=chunk[0],
        action_chunk=chunk,
        created_monotonic_ns=2_100,
        valid_until_monotonic_ns=2_200,
    )

    assert decode_policy_candidate(encode_policy_candidate(candidate)) == candidate
    assert json.loads(encode_policy_candidate(candidate))["action_chunk"] == [
        list(action) for action in chunk
    ]

    for invalid in (chunk[:9], chunk + (chunk[0],), ((0.0, 0.0, 0.0, 1.01),) * 10):
        with pytest.raises(ValueError):
            ResidentPolicyCandidate(
                source="act_dig",
                control_generation=4,
                mode="manual_action",
                action=(0.0, 0.0, 0.0, 0.0),
                action_chunk=invalid,
                created_monotonic_ns=2_100,
                valid_until_monotonic_ns=2_200,
            )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "unexpected": 1},
        lambda value: {**value, "state_names": list(reversed(ACT_STATE_NAMES))},
        lambda value: {**value, "state": [float("nan")] + value["state"][1:]},
        lambda value: {**value, "control_seq": True},
        lambda value: {**value, "state_monotonic_ns": 2_001},
    ],
)
def test_resident_state_decoder_rejects_ambiguous_or_invalid_frames(mutation):
    value = json.loads(encode_resident_state(_state()))

    with pytest.raises(ValueError):
        decode_resident_state(
            json.dumps(mutation(value), allow_nan=True, separators=(",", ":")).encode()
        )


def test_protocol_rejects_duplicate_json_fields():
    encoded = encode_resident_state(_state())
    duplicate = encoded[:-1] + b',"control_seq":8}'

    with pytest.raises(ValueError, match="strict finite JSON"):
        decode_resident_state(duplicate)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "unexpected": 1},
        lambda value: {**value, "action_order": list(reversed(ACTION_ORDER))},
        lambda value: {**value, "action": [float("inf"), 0.0, 0.0, 0.0]},
        lambda value: {**value, "control_generation": True},
        lambda value: {**value, "source": "act"},
        lambda value: {**value, "mode": "unknown"},
        lambda value: {**value, "valid_until_monotonic_ns": 2_099},
    ],
)
def test_policy_candidate_decoder_rejects_ambiguous_or_invalid_frames(mutation):
    original = ResidentPolicyCandidate(
        source="act_dig",
        control_generation=4,
        mode="manual_action",
        action=(0.1, -0.2, 0.3, 0.0),
        created_monotonic_ns=2_100,
        valid_until_monotonic_ns=2_200,
    )
    value = json.loads(encode_policy_candidate(original))

    with pytest.raises(ValueError):
        decode_policy_candidate(
            json.dumps(mutation(value), allow_nan=True, separators=(",", ":")).encode()
        )
