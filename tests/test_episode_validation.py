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

    with pytest.raises(EpisodeValidationError, match="camera frame age"):
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
