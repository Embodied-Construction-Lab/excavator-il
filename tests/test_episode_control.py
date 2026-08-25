from excavator_il.collector.config import (
    CameraConfig,
    EpisodeDefaults,
)
from excavator_il.collector.control import EpisodeController
from excavator_il.collector.recorder import EpisodeRecorder
import json
import pytest


def test_episode_controller_start_stop_and_abort_are_explicit(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    ticks = iter((100, 200, 300, 400, 500, 600, 700, 800))
    controller = EpisodeController(
        recorder=recorder,
        defaults=EpisodeDefaults(
            dig_target_m=(0.8, 0.1, -0.2),
            material_id="soil",
            provenance={},
        ),
        camera=CameraConfig("/dev/video0", 640, 480, 30, 95),
        monotonic_ns=lambda: next(ticks),
        wall_ns=lambda: next(ticks),
    )

    started = controller.handle(
        {"command": "start", "task": "ExecuteDig", "operator_id": "operator_01"}
    )
    status = controller.handle({"command": "status"})
    stopped = controller.handle({"command": "stop", "success": True})

    assert started["ok"] is True
    assert status == {"ok": True, "active": True, "episode_id": "episode_0001"}
    assert stopped["status"] == "complete"

    controller.handle(
        {"command": "start", "task": "ExecuteDig", "operator_id": "operator_01"}
    )
    aborted = controller.handle(
        {"command": "abort", "reason": "emergency_stop"}
    )
    assert aborted["status"] == "aborted"


def test_episode_controller_seals_then_classifies_without_recording_tail(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    ticks = iter((100, 200, 300, 400))
    controller = EpisodeController(
        recorder=recorder,
        defaults=EpisodeDefaults(
            dig_target_m=(0.8, 0.0, -0.2),
            material_id="soil",
            provenance={},
        ),
        camera=CameraConfig("/dev/video0", 640, 480, 30, 95),
        monotonic_ns=lambda: next(ticks),
        wall_ns=lambda: next(ticks),
    )
    started = controller.handle(
        {"command": "start", "task": "ExecuteDig", "operator_id": "operator_01"}
    )

    sealed = controller.handle({"command": "seal"})
    finalized = controller.handle(
        {
            "command": "finalize",
            "path": sealed["path"],
            "result": "failure",
            "failure_reason": "diagnostic_task_failed",
        }
    )

    assert sealed == {
        "ok": True,
        "active": False,
        "episode_id": started["episode_id"],
        "path": started["path"],
        "status": "pending_review",
    }
    assert finalized["status"] == "failed"


def test_episode_controller_dual_camera_requires_and_persists_trial_labels(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    controller = EpisodeController(
        recorder=recorder,
        defaults=EpisodeDefaults(
            dig_target_m=(0.8, 0.0, -0.2),
            material_id="soil",
            provenance={},
        ),
        cameras={
            "front": CameraConfig("/dev/front", 640, 480, 30, 95),
            "dump": CameraConfig("/dev/dump", 640, 480, 30, 95),
        },
        monotonic_ns=lambda: 100,
        wall_ns=lambda: 200,
    )

    with pytest.raises(ValueError, match="must be provided together"):
        controller.handle(
            {
                "command": "start",
                "task": "ExecuteDig",
                "operator_id": "zhaoshuai",
                "task_variant": "dig_only",
            }
        )

    started = controller.handle(
        {
            "command": "start",
            "task": "ExecuteDig",
            "operator_id": "zhaoshuai",
            "task_variant": "dig_only",
            "soil_reset_block_id": "block_01",
            "dig_point_id": "dig_02",
            "target_source_provenance": {
                "repository": "airylidar",
                "path": "mission/config/excavation_demo.json",
                "sha256": "a" * 64,
                "commit": "b" * 40,
                "dirty": False,
            },
        }
    )
    controller.handle({"command": "stop", "success": True})

    metadata = json.loads(
        (tmp_path / started["episode_id"] / "episode.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["collection_protocol"] == {
        "task_variant": "dig_only",
        "soil_reset_block_id": "block_01",
        "dig_point_id": "dig_02",
    }
    assert metadata["target_source_provenance"]["commit"] == "b" * 40


def test_episode_controller_rejects_dirty_target_source_before_creating_episode(
    tmp_path,
):
    controller = EpisodeController(
        recorder=EpisodeRecorder(tmp_path),
        defaults=EpisodeDefaults((1.0, 0.0, 0.0), "soil", {}),
        cameras={
            "front": CameraConfig("/dev/front", 640, 480, 30, 95),
            "dump": CameraConfig("/dev/dump", 640, 480, 30, 95),
        },
    )

    with pytest.raises(ValueError, match="dirty must be exactly false"):
        controller.handle(
            {
                "command": "start",
                "task": "ExecuteDig",
                "operator_id": "zhaoshuai",
                "task_variant": "dig_only",
                "soil_reset_block_id": "block_01",
                "dig_point_id": "dig_01",
                "target_source_provenance": {
                    "repository": "airylidar",
                    "path": "mission/config/excavation_demo.json",
                    "sha256": "a" * 64,
                    "commit": "b" * 40,
                    "dirty": True,
                },
            }
        )

    assert not tuple(tmp_path.glob("episode_*"))


def test_episode_controller_starts_explicit_dual_camera_diagnostic_without_labels(
    tmp_path,
):
    recorder = EpisodeRecorder(tmp_path)
    controller = EpisodeController(
        recorder=recorder,
        defaults=EpisodeDefaults((0.8, 0.0, -0.2), "soil", {}),
        cameras={
            "front": CameraConfig("/dev/front", 640, 480, 30, 95),
            "dump": CameraConfig("/dev/dump", 640, 480, 30, 95),
        },
        monotonic_ns=lambda: 100,
        wall_ns=lambda: 200,
    )

    started = controller.handle(
        {
            "command": "start",
            "task": "zero_command_soak",
            "operator_id": "zhaoshuai",
            "recording_purpose": "diagnostic",
        }
    )
    controller.handle({"command": "abort", "reason": "diagnostic_complete"})

    metadata = json.loads(
        (tmp_path / started["episode_id"] / "episode.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["recording_purpose"] == "diagnostic"
    assert "collection_protocol" not in metadata


def test_episode_controller_rejects_unknown_recording_purpose(tmp_path):
    controller = EpisodeController(
        recorder=EpisodeRecorder(tmp_path),
        defaults=EpisodeDefaults((0.8, 0.0, -0.2), "soil", {}),
        camera=CameraConfig("/dev/front", 640, 480, 30, 95),
    )

    with pytest.raises(ValueError, match="recording_purpose"):
        controller.handle(
            {
                "command": "start",
                "task": "ExecuteDig",
                "operator_id": "zhaoshuai",
                "recording_purpose": "unknown",
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task_variant", "other", "task_variant"),
        ("soil_reset_block_id", "Soil Block", "normalized lowercase"),
        ("dig_point_id", "DIG 01", "normalized lowercase"),
    ],
)
def test_episode_controller_rejects_invalid_trial_labels(
    tmp_path, field, value, message
):
    recorder = EpisodeRecorder(tmp_path)
    controller = EpisodeController(
        recorder=recorder,
        defaults=EpisodeDefaults((0.8, 0.0, -0.2), "soil", {}),
        cameras={
            "front": CameraConfig("/dev/front", 640, 480, 30, 95),
            "dump": CameraConfig("/dev/dump", 640, 480, 30, 95),
        },
    )
    request = {
        "command": "start",
        "task": "ExecuteDig",
        "operator_id": "zhaoshuai",
        "task_variant": "dig_only",
        "soil_reset_block_id": "block_01",
        "dig_point_id": "dig_01",
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        controller.handle(request)
