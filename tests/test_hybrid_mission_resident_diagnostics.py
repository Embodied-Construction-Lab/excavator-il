import pytest

from excavator_il.hybrid_mission_resident import (
    ResidentControlStatus,
    ResidentHybridMissionOperations,
    ResidentPolicyBinding,
)


def _status(
    *,
    phase: str,
    generation: int,
    active: ResidentPolicyBinding | None,
    target: ResidentPolicyBinding | None,
    act_complete: bool,
) -> ResidentControlStatus:
    return ResidentControlStatus(
        phase=phase,
        control_generation=generation,
        active=active,
        target=target,
        last_handoff_latency_ms=None,
        rl_is_active=False,
        act_is_active=active
        == ResidentPolicyBinding("act_dig", "manual_action"),
        act_worker_ready=True,
        act_segment_generation=11,
        act_segment_max_steps=130,
        act_segment_completed_steps=130 if act_complete else 129,
        act_segment_complete=act_complete,
        mission_lease_active=True,
        is_operational=True,
    )


def test_revoked_act_to_rl_error_contains_the_atomic_control_status() -> None:
    act = ResidentPolicyBinding("act_dig", "manual_action")

    class Control:
        def ensure_ready(self):
            return _status(
                phase="active",
                generation=10,
                active=ResidentPolicyBinding(
                    "rl_follow", "velocity_reference"
                ),
                target=None,
                act_complete=False,
            )

        def activate_act(self, max_steps):
            assert max_steps == 130
            return _status(
                phase="active",
                generation=11,
                active=act,
                target=None,
                act_complete=False,
            )

        def status(self):
            return _status(
                phase="terminal_zero_pending",
                generation=13,
                active=act,
                target=None,
                act_complete=True,
            )

        def terminal_disarm(self):
            return _status(
                phase="idle",
                generation=14,
                active=None,
                target=None,
                act_complete=True,
            )

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        act_run_timeout_s=1.0,
        poll_interval_s=0.01,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError) as raised:
        operations.run_act_dig(130)

    message = str(raised.value)
    assert "ACT-to-RL handoff was revoked" in message
    assert "phase=terminal_zero_pending" in message
    assert "generation=13" in message
    assert "active=act_dig/manual_action" in message
    assert "target=none" in message
    assert "act_segment=11:130/130:complete" in message
