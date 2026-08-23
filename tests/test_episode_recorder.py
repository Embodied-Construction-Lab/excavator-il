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
    assert metadata["schema_version"] == "excavator_demo_raw.v1"
    assert metadata["camera_front"]["device_id"] == "/dev/video0"
    assert "cameras" not in metadata
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
            camera_dump={"device_id": "/dev/v4l/by-path/dump-camera"},
            task_variant="dig_transport_dump",
            soil_reset_block_id="block_01",
            dig_point_id="dig_01",
            target_source_provenance={
                "repository": "airylidar",
                "path": "mission/config/excavation_demo.json",
                "sha256": "a" * 64,
                "commit": "b" * 40,
                "dirty": False,
            },
        ),
        start_wall_ns=1,
        start_monotonic_ns=2,
    )

    relative_path = recorder.record_camera(
        camera_id="front",
        encoded_image=b"jpeg-fixture",
        capture_monotonic_ns=123_456,
        extension="jpg",
    )
    dump_path = recorder.record_camera(
        camera_id="dump",
        encoded_image=b"jpeg-dump",
        capture_monotonic_ns=123_556,
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
    assert dump_path == "camera_dump/000000.jpg"
    assert (episode / relative_path).read_bytes() == b"jpeg-fixture"
    assert (episode / dump_path).read_bytes() == b"jpeg-dump"
    timestamps = (episode / "camera_front_timestamps.csv").read_text(
        encoding="utf-8"
    )
    dump_timestamps = (episode / "camera_dump_timestamps.csv").read_text(
        encoding="utf-8"
    )
    assert "0,123456,camera_front/000000.jpg" in timestamps
    assert "0,123556,camera_dump/000000.jpg" in dump_timestamps
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "aborted"
    assert metadata["failure_reason"] == "emergency_stop"
    assert metadata["cameras"]["dump"]["device_id"] == (
        "/dev/v4l/by-path/dump-camera"
    )


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


def test_episode_recorder_v2_persists_two_semantic_rgb_streams_and_protocol(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    episode = recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="zhaoshuai",
            dig_target_m=(0.8, 0.0, -0.2),
            material_id="dry_soil_01",
            provenance={"firmware_commit": "abc123"},
            camera_front={"device_id": "/dev/v4l/by-path/front"},
            camera_dump={"device_id": "/dev/v4l/by-path/dump"},
            task_variant="dig_transport_dump",
            soil_reset_block_id="block_03",
            dig_point_id="dig_02",
            target_source_provenance={
                "repository": "airylidar",
                "path": "mission/config/excavation_demo.json",
                "sha256": "a" * 64,
                "commit": "b" * 40,
                "dirty": False,
            },
        ),
        start_wall_ns=10,
        start_monotonic_ns=20,
    )

    front_path = recorder.record_camera(
        camera_id="front",
        encoded_image=b"front-jpeg",
        capture_monotonic_ns=100,
        extension="jpg",
    )
    dump_path = recorder.record_camera(
        camera_id="dump",
        encoded_image=b"dump-jpeg",
        capture_monotonic_ns=110,
        extension="jpg",
    )
    recorder.stop(
        success=True,
        failure_reason="",
        intervention=False,
        end_wall_ns=30,
        end_monotonic_ns=40,
    )

    assert front_path == "camera_front/000000.jpg"
    assert dump_path == "camera_dump/000000.jpg"
    assert (episode / front_path).read_bytes() == b"front-jpeg"
    assert (episode / dump_path).read_bytes() == b"dump-jpeg"
    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "excavator_demo_raw.v2"
    assert tuple(metadata["cameras"]) == ("front", "dump")
    assert metadata["collection_protocol"] == {
        "task_variant": "dig_transport_dump",
        "soil_reset_block_id": "block_03",
        "dig_point_id": "dig_02",
    }
    assert metadata["target_source_provenance"] == {
        "repository": "airylidar",
        "path": "mission/config/excavation_demo.json",
        "sha256": "a" * 64,
        "commit": "b" * 40,
        "dirty": False,
    }
    assert "0,110,camera_dump/000000.jpg" in (
        episode / "camera_dump_timestamps.csv"
    ).read_text(encoding="utf-8")


def test_episode_recorder_rejects_dual_camera_without_trial_protocol(tmp_path):
    recorder = EpisodeRecorder(tmp_path)

    with pytest.raises(ValueError, match="dual-camera Episode requires"):
        recorder.start(
            EpisodeStart(
                task="ExecuteDig",
                operator_id="zhaoshuai",
                dig_target_m=(0.8, 0.0, -0.2),
                material_id="dry_soil_01",
                provenance={},
                camera_front={"device_id": "/dev/v4l/by-path/front"},
                camera_dump={"device_id": "/dev/v4l/by-path/dump"},
            ),
            start_wall_ns=10,
            start_monotonic_ns=20,
        )

    assert not tuple(tmp_path.glob("episode_*"))


def test_episode_recorder_rejects_formal_protocol_without_target_source(tmp_path):
    recorder = EpisodeRecorder(tmp_path)

    with pytest.raises(ValueError, match="target_source_provenance"):
        recorder.start(
            EpisodeStart(
                task="ExecuteDig",
                operator_id="zhaoshuai",
                dig_target_m=(1.0, 0.0, 0.0),
                material_id="soil",
                provenance={},
                camera_front={"device_id": "/dev/front"},
                task_variant="dig_only",
                soil_reset_block_id="block_01",
                dig_point_id="dig_01",
            ),
            start_wall_ns=10,
            start_monotonic_ns=20,
        )

    assert not tuple(tmp_path.glob("episode_*"))


def test_episode_recorder_allows_explicit_dual_camera_diagnostic_without_protocol(
    tmp_path,
):
    recorder = EpisodeRecorder(tmp_path)

    episode = recorder.start(
        EpisodeStart(
            task="zero_command_soak",
            operator_id="zhaoshuai",
            dig_target_m=(0.8, 0.0, -0.2),
            material_id="diagnostic",
            provenance={},
            camera_front={"device_id": "/dev/v4l/by-path/front"},
            camera_dump={"device_id": "/dev/v4l/by-path/dump"},
            recording_purpose="diagnostic",
        ),
        start_wall_ns=10,
        start_monotonic_ns=20,
    )
    recorder.stop(
        success=False,
        failure_reason="zero_command_soak_complete",
        intervention=False,
        end_wall_ns=30,
        end_monotonic_ns=40,
        aborted=True,
    )

    metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert metadata["recording_purpose"] == "diagnostic"
    assert "collection_protocol" not in metadata


def test_episode_recorder_rejects_unknown_recording_purpose(tmp_path):
    recorder = EpisodeRecorder(tmp_path)

    with pytest.raises(ValueError, match="recording_purpose"):
        recorder.start(
            EpisodeStart(
                task="ExecuteDig",
                operator_id="zhaoshuai",
                dig_target_m=(0.8, 0.0, -0.2),
                material_id="dry_soil_01",
                provenance={},
                camera_front={"device_id": "/dev/video0"},
                recording_purpose="training-ish",
            ),
            start_wall_ns=10,
            start_monotonic_ns=20,
        )

    assert not tuple(tmp_path.glob("episode_*"))


def test_episode_recorder_rejects_unconfigured_camera_role(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.start(
        EpisodeStart(
            task="ExecuteDig",
            operator_id="operator_01",
            dig_target_m=(0.8, 0.0, -0.2),
            material_id="soil",
            provenance={},
            camera_front={"device_id": "/dev/video0"},
        ),
        start_wall_ns=1,
        start_monotonic_ns=2,
    )

    with pytest.raises(ValueError, match="not configured"):
        recorder.record_camera(
            camera_id="dump",
            encoded_image=b"jpeg",
            capture_monotonic_ns=3,
            extension="jpg",
        )
    recorder.stop(
        success=False,
        failure_reason="fixture",
        intervention=False,
        end_wall_ns=4,
        end_monotonic_ns=5,
    )
