import json

import pytest

from excavator_il.collector.recorder import EpisodeRecorder, EpisodeStart


def test_episode_recorder_preserves_raw_streams_and_explicit_success_result(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="operator_01",
            dig_target_m=(0.8, 0.1, -0.2),
            material_id="dry_soil_01",
            provenance={"firmware_commit": "abc123"},
            camera_front={
                "device_id": "/dev/video0",
                "width": 32,
                "height": 24,
                "nominal_fps": 30,
                "pixel_format": "BGR8",
                "timestamp_clock": "CLOCK_MONOTONIC",
            },
        ),
        start_wall_ns=2_000,
        start_monotonic_ns=1_000,
    )
    recorder.record_json("joystick_raw", {"sample_seq": 1})
    recorder.record_json("expert_action", {"action_seq": 1})
    recorder.record_json("command_tx", {"command_seq": 1})
    recorder.record_json("stm32_raw", {"raw_frame_seq": 1})
    recorder.stop(
        success=True,
        failure_reason="",
        intervention=False,
        end_wall_ns=4_000,
        end_monotonic_ns=3_000,
    )

    assert episode.name == "episode_0001"
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["success"] is True
    assert metadata["start_monotonic_ns"] == 1_000
    assert metadata["end_monotonic_ns"] == 3_000
    for name in ("joystick_raw", "expert_action", "command_tx", "stm32_raw"):
        record = json.loads((episode / f"{name}.jsonl").read_text(encoding="utf-8"))
        assert record


def test_episode_recorder_writes_camera_frame_and_abort_status(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="operator_01",
            dig_target_m=(0.8, 0.1, -0.2),
            material_id="dry_soil_01",
            provenance={},
            camera_front={"device_id": "/dev/video0"},
        ),
        start_wall_ns=1,
        start_monotonic_ns=2,
    )

    relative_path = recorder.record_camera(
        encoded_image=b"jpeg-fixture",
        capture_monotonic_ns=123_456,
        extension="jpg",
    )
    recorder.stop(
        success=False,
        failure_reason="emergency_stop",
        intervention=True,
        end_wall_ns=3,
        end_monotonic_ns=4,
        aborted=True,
    )

    assert relative_path == "camera_front/000000.jpg"
    assert (episode / relative_path).read_bytes() == b"jpeg-fixture"
    timestamps = (episode / "camera_front_timestamps.csv").read_text(
        encoding="utf-8"
    )
    assert "0,123456,camera_front/000000.jpg" in timestamps
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "aborted"
    assert metadata["failure_reason"] == "emergency_stop"


def test_episode_recorder_seals_streams_before_operator_classification(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="operator_01",
            dig_target_m=(0.8, 0.0, -0.2),
            material_id="soil",
            provenance={},
            camera_front={"device_id": "/dev/video0"},
        ),
        start_wall_ns=10,
        start_monotonic_ns=20,
    )
    recorder.record_json("expert_action", {"action_seq": 1})

    sealed = recorder.seal(end_wall_ns=30, end_monotonic_ns=40)

    assert sealed == episode
    assert recorder.active is False
    pending = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert pending["status"] == "pending_review"
    assert pending["success"] is None
    assert pending["end_monotonic_ns"] == 40

    finalized = recorder.finalize_pending(
        episode,
        result="success",
        failure_reason="",
    )

    assert finalized == episode
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["success"] is True
    assert metadata["end_monotonic_ns"] == 40
    with pytest.raises(RuntimeError, match="pending_review"):
        recorder.finalize_pending(episode, result="success", failure_reason="")
