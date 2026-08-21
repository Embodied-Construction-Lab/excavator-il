from excavator_il.collector.config import (
    CameraConfig,
    EpisodeDefaults,
)
from excavator_il.collector.control import EpisodeController
from excavator_il.collector.recorder import EpisodeRecorder


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
