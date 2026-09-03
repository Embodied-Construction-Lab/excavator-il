import threading
import time

from excavator_il.hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from excavator_il.resident_fixed_cycle_system import ResidentFixedCycleSupervisor
from excavator_il.resident_fixed_cycle_visualization import (
    ResidentFixedCycleRemoteStatus,
)


def _status(stage: str, *, terminal: bool = False) -> ResidentFixedCycleRemoteStatus:
    return ResidentFixedCycleRemoteStatus(
        run_id="cancel-race",
        mission_id="fixed_target_hybrid",
        active_behavior_id="" if terminal else "onnx_rl_tracking",
        stage=stage,
        requested_cycles=1,
        completed_cycles=0,
        current_dig_point_id="dig_01",
        dig_group_id="all",
        terminal=terminal,
        outcome="CANCELLED" if terminal else "",
        reason_code="",
    )


class _SlowCancelOperations:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release_terminal_disarmed: bool | None = None

    def start(self, **_kwargs):
        self.started.set()
        return _status("go_current_dig")

    def status(self):
        return _status("go_current_dig")

    def cancel(self):
        time.sleep(0.1)
        return _status("CANCELLED", terminal=True)

    def release(self, *, terminal_disarmed):
        self.release_terminal_disarmed = terminal_disarmed


def test_cancel_acknowledgement_wins_over_background_stop_poll() -> None:
    operations = _SlowCancelOperations()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01",),
        poll_interval_s=0.02,
    )
    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
    )
    assert operations.started.wait(1.0)

    supervisor.stop()
    supervisor.close()

    assert supervisor.snapshot().stage == "cancelled"
    assert operations.release_terminal_disarmed is True
