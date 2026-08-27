import csv
import json

import pytest

from excavator_il.raw_episode import EpisodeValidationError, validate_episode

def test_validate_episode_accepts_synchronized_rgb_demonstration(rgb_episode_factory):
    episode = rgb_episode_factory()
    report = validate_episode(episode)

    assert report.episode_id == "episode_0001"
    assert report.step_count == 3
    assert report.camera_frame_count == 3
    assert report.image_shape == (24, 32, 3)
    assert report.max_camera_age_ms == 5.0
    assert report.max_action_age_ms == 10.0


def test_validate_episode_accepts_optional_dump_camera(rgb_episode_factory):
    report = validate_episode(rgb_episode_factory(dual_camera=True))

    assert report.cameras["dump"].frame_count == 3


def test_validate_episode_rejects_invalid_optional_collection_labels(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(dual_camera=True)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["collection_labels"] = {
        "collection_zone_id": "zone_07",
        "dig_repeat_index": 4,
        "operator_note": "invalid fixture",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="collection_labels"):
        validate_episode(episode)


def test_validate_episode_accepts_explicit_dual_camera_diagnostic_without_protocol(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(dual_camera=True)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["recording_purpose"] = "diagnostic"
    metadata.pop("collection_protocol")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_episode(episode)

    assert tuple(report.cameras) == ("front", "dump")


def test_validate_episode_rejects_camera_frame_older_than_sync_limit(rgb_episode_factory):
    episode = rgb_episode_factory()
    timestamps_path = episode / "camera_front_timestamps.csv"
    with timestamps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["camera_stamp_monotonic_ns"] = str(int(row["camera_stamp_monotonic_ns"]) - 500_000_000)
    with timestamps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="front camera frame age"):
        validate_episode(episode, max_camera_age_ms=120.0)


def test_validate_episode_rejects_expert_action_outside_normalized_range(rgb_episode_factory):
    episode = rgb_episode_factory()
    steps_path = episode / "steps.csv"
    with steps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["action_bucket"] = "1.01"
    with steps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="action_bucket"):
        validate_episode(episode)


def test_validate_episode_rejects_missing_camera_provenance(rgb_episode_factory):
    episode = rgb_episode_factory()
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("camera_front", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="camera_front"):
        validate_episode(episode)


def test_validate_episode_rejects_non_manual_training_rows(rgb_episode_factory):
    episode = rgb_episode_factory()
    steps_path = episode / "steps.csv"
    with steps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["control_mode"] = "safe_zero"
    with steps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="control_mode"):
        validate_episode(episode)


def test_validate_episode_rejects_future_expert_action(rgb_episode_factory):
    episode = rgb_episode_factory()
    steps_path = episode / "steps.csv"
    with steps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["action_stamp_monotonic_ns"] = str(
        int(rows[0]["state_receive_monotonic_ns"]) + 1
    )
    with steps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="future expert action"):
        validate_episode(episode)


def test_validate_episode_rejects_joystick_timeout_in_quality_report(
    rgb_episode_factory,
):
    episode = rgb_episode_factory()
    (episode / "quality_report.json").write_text(
        json.dumps({"joystick_timeout_count": 1}),
        encoding="utf-8",
    )

    with pytest.raises(EpisodeValidationError, match="joystick timeout"):
        validate_episode(episode)


def test_validate_episode_requires_manifest_for_legacy_state_sequence_gap(
    rgb_episode_factory,
):
    episode = rgb_episode_factory()
    steps_path = episode / "steps.csv"
    with steps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["state_seq"] = str(int(rows[0]["state_seq"]) + 2)
    rows[2]["state_seq"] = str(int(rows[1]["state_seq"]) + 1)
    with steps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="rerun build-steps"):
        validate_episode(episode)


def test_validate_episode_accepts_recovered_timeout_with_segment_boundaries(
    rgb_episode_factory,
):
    episode = rgb_episode_factory()
    (episode / "quality_report.json").write_text(
        json.dumps(
            {
                "joystick_timeout_count": 1,
                "training_segment_count": 2,
                "excluded_training_step_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (episode / "training_segments.json").write_text(
        json.dumps(
            {
                "schema_version": "excavator_training_segments.v1",
                "parent_episode_id": "episode_0001",
                "strategy": "lerobot_episode_boundaries",
                "recovery_joystick_sample_count": 10,
                "fault_events": [
                    {
                        "event_type": "joystick_timeout",
                        "event_stamp_monotonic_ns": 1_040_000_000,
                        "recovery_stamp_monotonic_ns": 1_050_000_000,
                        "recovered": True,
                    }
                ],
                "unresolved_safety_event_count": 0,
                "excluded_training_step_count": 0,
                "segments": [
                    {
                        "segment_id": "episode_0001_segment_0000",
                        "start_frame_index": 0,
                        "end_frame_index_exclusive": 1,
                        "step_count": 1,
                        "start_state_receive_monotonic_ns": 1_000_000_000,
                        "end_state_receive_monotonic_ns": 1_000_000_000,
                    },
                    {
                        "segment_id": "episode_0001_segment_0001",
                        "start_frame_index": 1,
                        "end_frame_index_exclusive": 3,
                        "step_count": 2,
                        "start_state_receive_monotonic_ns": 1_100_000_000,
                        "end_state_receive_monotonic_ns": 1_200_000_000,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_episode(episode)

    assert report.training_segment_count == 2


def test_validate_episode_rejects_tampered_segment_timestamp_provenance(
    rgb_episode_factory,
):
    episode = rgb_episode_factory()
    (episode / "quality_report.json").write_text(
        json.dumps({"joystick_timeout_count": 0, "training_segment_count": 1}),
        encoding="utf-8",
    )
    (episode / "training_segments.json").write_text(
        json.dumps(
            {
                "schema_version": "excavator_training_segments.v1",
                "parent_episode_id": "episode_0001",
                "strategy": "lerobot_episode_boundaries",
                "recovery_joystick_sample_count": 10,
                "fault_events": [],
                "unresolved_safety_event_count": 0,
                "excluded_training_step_count": 0,
                "segments": [
                    {
                        "segment_id": "episode_0001_segment_0000",
                        "start_frame_index": 0,
                        "end_frame_index_exclusive": 3,
                        "step_count": 3,
                        "start_state_receive_monotonic_ns": 999,
                        "end_state_receive_monotonic_ns": 1_200_000_000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EpisodeValidationError, match="timestamp provenance"):
        validate_episode(episode)


def test_validate_episode_rejects_segment_crossing_recovered_timeout(
    rgb_episode_factory,
):
    episode = rgb_episode_factory()
    (episode / "quality_report.json").write_text(
        json.dumps({"joystick_timeout_count": 1, "training_segment_count": 1}),
        encoding="utf-8",
    )
    (episode / "training_segments.json").write_text(
        json.dumps(
            {
                "schema_version": "excavator_training_segments.v1",
                "parent_episode_id": "episode_0001",
                "strategy": "lerobot_episode_boundaries",
                "recovery_joystick_sample_count": 10,
                "fault_events": [
                    {
                        "event_type": "joystick_timeout",
                        "event_stamp_monotonic_ns": 1_040_000_000,
                        "recovery_stamp_monotonic_ns": 1_050_000_000,
                        "recovered": True,
                    }
                ],
                "unresolved_safety_event_count": 0,
                "excluded_training_step_count": 0,
                "segments": [
                    {
                        "segment_id": "episode_0001_segment_0000",
                        "start_frame_index": 0,
                        "end_frame_index_exclusive": 3,
                        "step_count": 3,
                        "start_state_receive_monotonic_ns": 1_000_000_000,
                        "end_state_receive_monotonic_ns": 1_200_000_000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EpisodeValidationError, match="crosses a safety event"):
        validate_episode(episode)


def test_validate_episode_rejects_pending_operator_review(rgb_episode_factory):
    episode = rgb_episode_factory()
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "pending_review"
    metadata["success"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="pending_review"):
        validate_episode(episode)


def test_validate_episode_v2_reports_each_rgb_stream_and_alignment(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(dual_camera=True)

    report = validate_episode(episode)

    assert tuple(report.cameras) == ("front", "dump")
    assert report.cameras["front"].frame_count == 3
    assert report.cameras["dump"].frame_count == 3
    assert report.cameras["front"].estimated_rate_hz == 10.0
    assert report.cameras["dump"].image_shape == (24, 32, 3)
    assert report.cameras["front"].max_age_ms == 5.0
    assert report.cameras["dump"].max_age_ms == 8.0
    assert report.max_intercamera_skew_ms == 3.0
    # Front aliases remain stable for existing ACT v1 callers.
    assert report.camera_frame_count == 3
    assert report.image_shape == (24, 32, 3)


def test_validate_episode_v2_fails_closed_when_configured_dump_stream_is_missing(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(dual_camera=True)
    (episode / "camera_dump_timestamps.csv").unlink()

    with pytest.raises(EpisodeValidationError, match="camera_dump_timestamps.csv"):
        validate_episode(episode)


def test_validate_episode_v2_rejects_stale_dump_frames(rgb_episode_factory):
    episode = rgb_episode_factory(dual_camera=True)
    timestamps_path = episode / "camera_dump_timestamps.csv"
    with timestamps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["camera_stamp_monotonic_ns"] = str(
            int(row["camera_stamp_monotonic_ns"]) - 500_000_000
        )
    with timestamps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="dump camera frame age"):
        validate_episode(episode)


def test_validate_episode_v2_rejects_low_rate_dump_even_when_frames_are_fresh(
    rgb_episode_factory,
):
    episode = rgb_episode_factory(step_count=8, dual_camera=True)
    timestamps_path = episode / "camera_dump_timestamps.csv"
    with timestamps_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[::2]
    for index, row in enumerate(rows):
        row["camera_frame_index"] = str(index)
    with timestamps_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(EpisodeValidationError, match="dump camera rate .*nominal"):
        validate_episode(episode)
