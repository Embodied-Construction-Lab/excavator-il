import json
import subprocess
from dataclasses import replace

import pytest

from excavator_il.hybrid_mission_resident import (
    ExistingRlBehaviorAdapter,
    PreparedDumpActivation,
    ResidentControlStatus,
    ResidentHybridMissionOperations,
    ResidentPolicyBinding,
    SshResidentControlAdapter,
)


def _status(
    *,
    control_generation=1,
    rl_is_active=False,
    act_is_active=False,
    act_segment_generation=None,
    act_segment_max_steps=None,
    act_segment_completed_steps=0,
    act_segment_complete=False,
    mission_lease_active=True,
):
    active = None
    if rl_is_active:
        active = ResidentPolicyBinding("rl_follow", "velocity_reference")
    elif act_is_active:
        active = ResidentPolicyBinding("act_dig", "manual_action")
    return ResidentControlStatus(
        phase="active" if rl_is_active or act_is_active else "idle",
        control_generation=control_generation,
        active=active,
        target=None,
        last_handoff_latency_ms=None,
        rl_is_active=rl_is_active,
        act_is_active=act_is_active,
        act_worker_ready=True,
        act_segment_generation=act_segment_generation,
        act_segment_max_steps=act_segment_max_steps,
        act_segment_completed_steps=act_segment_completed_steps,
        act_segment_complete=act_segment_complete,
        mission_lease_active=mission_lease_active,
        is_operational=True,
    )


def _wire_response(command="status", **status_overrides):
    status = {
        "phase": "active",
        "control_generation": 9,
        "active": {"source": "rl_follow", "mode": "velocity_reference"},
        "target": None,
        "last_handoff_latency_ms": 37.25,
        "rl_is_active": True,
        "act_is_active": False,
        "act_worker_ready": True,
        "act_segment_generation": 8,
        "act_segment_max_steps": 130,
        "act_segment_completed_steps": 130,
        "act_segment_complete": True,
        "mission_lease_active": True,
        "is_operational": True,
    }
    status.update(status_overrides)
    return {
        "schema_version": "resident_motion_control.v1",
        "ok": True,
        "command": command,
        "status": status,
        "error": None,
    }


class _ReadyRlControl:
    def __init__(self, events):
        self._events = events

    def ensure_ready(self):
        self._events.append(("ensure_ready",))
        return _status()

    def activate_rl(self):
        self._events.append(("activate_rl",))
        return _status(rl_is_active=True)


class _RecordingBehavior:
    def __init__(self, events):
        self._events = events

    def run_rl_to_dig(self, target_id):
        self._events.append(("run_rl_to_dig", target_id))

    def run_rl_to_dump_and_dump(self):
        self._events.append(("run_rl_to_dump_and_dump",))

    def run_rl_return_to_dig(self, target_id):
        self._events.append(("run_rl_return_to_dig", target_id))


def test_resident_poll_interval_rejects_an_ssh_flooding_rate():
    with pytest.raises(ValueError, match="poll_interval_s"):
        ResidentHybridMissionOperations(
            control=object(),
            behavior=object(),
            act_run_timeout_s=90,
            poll_interval_s=0.009,
        )


def test_rl_to_dig_activates_resident_rl_then_runs_existing_behavior():
    events = []
    operations = ResidentHybridMissionOperations(
        control=_ReadyRlControl(events),
        behavior=_RecordingBehavior(events),
        act_run_timeout_s=90,
    )

    operations.run_rl_to_dig("dig_03")

    assert events == [
        ("ensure_ready",),
        ("activate_rl",),
        ("run_rl_to_dig", "dig_03"),
    ]


def test_rl_behavior_waits_for_the_resident_handoff_to_become_active():
    events = []

    class Control:
        def ensure_ready(self):
            events.append(("ensure_ready",))
            return _status()

        def activate_rl(self):
            events.append(("activate_rl",))
            return _status()

        def status(self):
            events.append(("status",))
            return _status(rl_is_active=True)

        def activate_act(self, max_steps):
            raise AssertionError(max_steps)

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=_RecordingBehavior(events),
        act_run_timeout_s=90,
        poll_interval_s=0.1,
        sleep=lambda seconds: events.append(("sleep", seconds)),
    )

    operations.run_rl_to_dig("dig_02")

    assert events == [
        ("ensure_ready",),
        ("activate_rl",),
        ("sleep", 0.1),
        ("status",),
        ("run_rl_to_dig", "dig_02"),
    ]


def test_rl_dump_segment_activates_rl_and_uses_the_existing_behavior_seam():
    events = []
    operations = ResidentHybridMissionOperations(
        control=_ReadyRlControl(events),
        behavior=_RecordingBehavior(events),
        act_run_timeout_s=90,
    )

    operations.run_rl_to_dump_and_dump()

    assert events == [
        ("ensure_ready",),
        ("activate_rl",),
        ("run_rl_to_dump_and_dump",),
    ]


def test_rl_return_segment_activates_rl_and_uses_the_existing_behavior_seam():
    events = []
    operations = ResidentHybridMissionOperations(
        control=_ReadyRlControl(events),
        behavior=_RecordingBehavior(events),
        act_run_timeout_s=90,
    )

    operations.run_rl_return_to_dig("dig_04")

    assert events == [
        ("ensure_ready",),
        ("activate_rl",),
        ("run_rl_return_to_dig", "dig_04"),
    ]


def test_act_dig_waits_for_same_segment_completion_and_automatic_rl_handoff():
    events = []
    statuses = iter(
        (
            replace(
                _status(
                    control_generation=8,
                    act_segment_generation=7,
                    act_segment_max_steps=130,
                    act_segment_completed_steps=130,
                    act_segment_complete=True,
                ),
                phase="terminal_zero_pending",
                active=ResidentPolicyBinding("act_dig", "manual_action"),
                target=ResidentPolicyBinding("rl_follow", "velocity_reference"),
            ),
            _status(
                rl_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=130,
                act_segment_completed_steps=130,
                act_segment_complete=True,
            ),
        )
    )

    class Control:
        def ensure_ready(self):
            events.append(("ensure_ready",))
            return _status()

        def activate_act(self, max_steps):
            events.append(("activate_act", max_steps))
            return _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=max_steps,
            )

        def status(self):
            events.append(("status",))
            return next(statuses)

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    class Behavior:
        pass

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=Behavior(),
        act_run_timeout_s=90,
        poll_interval_s=0.1,
        sleep=lambda seconds: events.append(("sleep", seconds)),
    )

    operations.run_act_dig(130)

    assert events == [
        ("ensure_ready",),
        ("activate_act", 130),
        ("sleep", 0.1),
        ("status",),
        ("sleep", 0.1),
        ("status",),
    ]


def test_act_warms_dump_planner_then_triggers_fresh_plan_near_completion():
    events = []
    statuses = iter(
        (
            _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=130,
                act_segment_completed_steps=109,
            ),
            _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=130,
                act_segment_completed_steps=110,
            ),
            _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=130,
                act_segment_completed_steps=124,
            ),
            _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=130,
                act_segment_completed_steps=125,
            ),
            _status(
                control_generation=8,
                rl_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=130,
                act_segment_completed_steps=130,
                act_segment_complete=True,
            ),
        )
    )

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            events.append(("activate_act", max_steps))
            return _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=max_steps,
            )

        def status(self):
            events.append("status")
            return next(statuses)

        def terminal_disarm(self):
            events.append("terminal_disarm")
            return _status()

    class Prepared:
        def start_prepare(self):
            events.append("start_prepare")

        def trigger_prepare(self):
            events.append("trigger_prepare")

        def trigger_refresh(self):
            events.append("trigger_refresh")

        def activate_prepared(self):
            events.append("activate_prepared")
            return PreparedDumpActivation.ACTIVATED

        def cancel(self):
            events.append("cancel_prepare")

    class Behavior:
        def run_dump_action(self):
            events.append("run_dump_action")

        def run_rl_to_dump_and_dump(self):
            events.append("legacy_dump")

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=Behavior(),
        prepared_dump=Prepared(),
        prepared_dump_lead_steps=20,
        prepared_dump_refresh_lead_steps=5,
        act_run_timeout_s=90,
        poll_interval_s=0.1,
        sleep=lambda _seconds: events.append("sleep"),
    )

    operations.run_act_dig(130)
    operations.run_rl_to_dump_and_dump()

    assert events.count("start_prepare") == 1
    assert events.count("trigger_prepare") == 1
    assert events.count("trigger_refresh") == 1
    assert events.index("start_prepare") < events.index("status")
    assert events.index("trigger_prepare") > events.index("status")
    assert events.index("trigger_refresh") > events.index("trigger_prepare")
    assert events[-2:] == ["activate_prepared", "run_dump_action"]
    assert "legacy_dump" not in events


def test_prepared_dump_falls_back_only_after_an_explicit_fallback_safe_result():
    events = []

    class Control:
        def ensure_ready(self):
            events.append("ensure_ready")
            return _status()

        def activate_act(self, max_steps):
            events.append(("activate_act", max_steps))
            return _status(
                control_generation=7,
                rl_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=max_steps,
                act_segment_completed_steps=max_steps,
                act_segment_complete=True,
            )

        def activate_rl(self):
            events.append("activate_rl")
            return _status(rl_is_active=True)

        def terminal_disarm(self):
            events.append("terminal_disarm")
            return _status()

    class Prepared:
        def start_prepare(self):
            events.append("start_prepare")

        def trigger_prepare(self):
            events.append("trigger_prepare")

        def activate_prepared(self):
            events.append("activate_prepared")
            return PreparedDumpActivation.FALLBACK_SAFE

        def cancel(self):
            events.append("cancel_prepare")

    class Behavior:
        def run_rl_to_dump_and_dump(self):
            events.append("legacy_dump")

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=Behavior(),
        prepared_dump=Prepared(),
        prepared_dump_lead_steps=20,
        act_run_timeout_s=90,
    )

    operations.run_act_dig(130)
    events.clear()
    operations.run_rl_to_dump_and_dump()

    assert events == [
        "activate_prepared",
        "ensure_ready",
        "activate_rl",
        "legacy_dump",
    ]


def test_prepared_dump_failure_cancels_and_disarms_without_legacy_replan():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=7,
                rl_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=max_steps,
                act_segment_completed_steps=max_steps,
                act_segment_complete=True,
            )

        def terminal_disarm(self):
            events.append("terminal_disarm")
            return _status()

    class Prepared:
        def start_prepare(self):
            events.append("start_prepare")

        def trigger_prepare(self):
            events.append("trigger_prepare")

        def activate_prepared(self):
            events.append("activate_prepared")
            raise RuntimeError("prepared Follow failed")

        def cancel(self):
            events.append("cancel_prepare")

    class Behavior:
        def run_rl_to_dump_and_dump(self):
            events.append("legacy_dump")

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=Behavior(),
        prepared_dump=Prepared(),
        prepared_dump_lead_steps=20,
        act_run_timeout_s=90,
    )
    operations.run_act_dig(130)
    events.clear()

    with pytest.raises(RuntimeError, match="prepared Follow failed"):
        operations.run_rl_to_dump_and_dump()

    assert events == [
        "activate_prepared",
        "terminal_disarm",
        "cancel_prepare",
    ]
    assert "legacy_dump" not in events


def test_safe_stop_terminally_disarms_the_resident_owner():
    events = []

    class Control:
        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=object(), act_run_timeout_s=90
    )

    operations.safe_stop()

    assert events == [("terminal_disarm",)]


def test_safe_stop_is_idempotent_after_cancelled_rl_already_disarmed():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_rl(self):
            return _status(rl_is_active=True)

        def terminal_disarm(self):
            events.append("terminal_disarm")
            if events.count("terminal_disarm") > 1:
                raise RuntimeError("resident motion control request failed")
            return _status()

    class Behavior:
        def run_rl_to_dig(self, _target_id):
            raise KeyboardInterrupt

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=Behavior(), act_run_timeout_s=90
    )

    with pytest.raises(KeyboardInterrupt):
        operations.run_rl_to_dig("dig_01")
    operations.safe_stop()

    assert events == ["terminal_disarm"]


def test_safe_stop_disarms_before_a_potentially_blocking_prepared_cleanup():
    events = []

    class Control:
        def terminal_disarm(self):
            events.append("terminal_disarm")
            return _status()

    class Prepared:
        def cancel(self):
            assert events == ["terminal_disarm"]
            events.append("cancel_prepare")

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        prepared_dump=Prepared(),
        prepared_dump_lead_steps=20,
        act_run_timeout_s=90,
    )

    operations.safe_stop()

    assert events == ["terminal_disarm", "cancel_prepare"]


def test_prewarm_only_ensures_resident_services_are_ready():
    events = []

    class Control:
        def ensure_ready(self):
            events.append(("ensure_ready",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=object(), act_run_timeout_s=90
    )

    operations.prewarm_next_act(130)

    assert events == [("ensure_ready",)]


def test_act_dig_rejects_a_different_segment_generation_and_disarms():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=7,
                act_is_active=True,
                act_segment_generation=7,
                act_segment_max_steps=max_steps,
            )

        def status(self):
            return _status(
                rl_is_active=True,
                act_segment_generation=8,
                act_segment_max_steps=130,
                act_segment_completed_steps=130,
                act_segment_complete=True,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        act_run_timeout_s=90,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="generation changed"):
        operations.run_act_dig(130)

    assert events == [("terminal_disarm",)]


def test_act_dig_timeout_disarms_instead_of_accepting_stale_progress():
    events = []
    now = 0.0

    def monotonic():
        return now

    def sleep(seconds):
        nonlocal now
        events.append(("sleep", seconds))
        now += seconds

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=11,
                act_is_active=True,
                act_segment_generation=11,
                act_segment_max_steps=max_steps,
            )

        def status(self):
            events.append(("status",))
            return _status(
                act_is_active=True,
                act_segment_generation=11,
                act_segment_max_steps=130,
                act_segment_completed_steps=12,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        act_run_timeout_s=0.2,
        poll_interval_s=0.1,
        monotonic=monotonic,
        sleep=sleep,
    )

    with pytest.raises(TimeoutError, match="ACT segment timed out"):
        operations.run_act_dig(130)

    assert events[-1] == ("terminal_disarm",)


def test_act_authority_loss_fails_immediately_instead_of_waiting_for_timeout():
    events = []
    now = [0.0]

    def sleep(seconds):
        events.append(("sleep", seconds))
        now[0] += seconds

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=11,
                act_is_active=True,
                act_segment_generation=11,
                act_segment_max_steps=max_steps,
                act_segment_completed_steps=1,
            )

        def status(self):
            events.append(("status",))
            return _status(
                control_generation=12,
                act_is_active=False,
                act_segment_generation=11,
                act_segment_max_steps=130,
                act_segment_completed_steps=1,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return replace(_status(), is_operational=False)

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        act_run_timeout_s=0.2,
        poll_interval_s=0.1,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    with pytest.raises(RuntimeError, match="ACT authority was revoked"):
        operations.run_act_dig(130)

    assert events == [("sleep", 0.1), ("status",), ("terminal_disarm",)]


def test_completed_act_fails_when_sensor_safety_cancels_the_rl_handoff():
    events = []
    now = [0.0]

    def sleep(seconds):
        events.append(("sleep", seconds))
        now[0] += seconds

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=11,
                act_is_active=True,
                act_segment_generation=11,
                act_segment_max_steps=max_steps,
                act_segment_completed_steps=129,
            )

        def status(self):
            events.append(("status",))
            return replace(
                _status(
                    control_generation=13,
                    act_segment_generation=11,
                    act_segment_max_steps=130,
                    act_segment_completed_steps=130,
                    act_segment_complete=True,
                ),
                phase="terminal_zero_pending",
                active=ResidentPolicyBinding("act_dig", "manual_action"),
                target=None,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return replace(_status(), is_operational=False)

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        act_run_timeout_s=0.2,
        poll_interval_s=0.1,
        monotonic=lambda: now[0],
        sleep=sleep,
    )

    with pytest.raises(RuntimeError, match="ACT-to-RL handoff was revoked"):
        operations.run_act_dig(130)

    assert events == [("sleep", 0.1), ("status",), ("terminal_disarm",)]


def test_act_run_timeout_starts_after_resident_services_are_ready():
    now = 0.0

    def monotonic():
        return now

    def advance(seconds):
        nonlocal now
        now += seconds

    class Control:
        def ensure_ready(self):
            advance(10.0)
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=31,
                act_is_active=True,
                act_segment_generation=31,
                act_segment_max_steps=max_steps,
            )

        def status(self):
            return _status(
                control_generation=32,
                rl_is_active=True,
                act_segment_generation=31,
                act_segment_max_steps=130,
                act_segment_completed_steps=130,
                act_segment_complete=True,
            )

        def terminal_disarm(self):
            raise AssertionError("ready time must not consume the ACT run budget")

    operations = ResidentHybridMissionOperations(
        control=Control(),
        behavior=object(),
        act_run_timeout_s=0.2,
        poll_interval_s=0.1,
        monotonic=monotonic,
        sleep=advance,
    )

    operations.run_act_dig(130)


def test_act_activation_generation_must_match_the_control_generation():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=12,
                act_is_active=True,
                act_segment_generation=11,
                act_segment_max_steps=max_steps,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=object(), act_run_timeout_s=90
    )

    with pytest.raises(RuntimeError, match="activation generation"):
        operations.run_act_dig(130)

    assert events == [("terminal_disarm",)]


def test_act_completion_requires_the_full_requested_step_budget():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return _status(
                control_generation=21,
                rl_is_active=True,
                act_segment_generation=21,
                act_segment_max_steps=max_steps,
                act_segment_completed_steps=max_steps - 1,
                act_segment_complete=True,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=object(), act_run_timeout_s=90
    )

    with pytest.raises(RuntimeError, match="completed step count"):
        operations.run_act_dig(130)

    assert events == [("terminal_disarm",)]


def test_act_completion_requires_an_unambiguous_active_rl_binding():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_act(self, max_steps):
            return replace(
                _status(
                    control_generation=22,
                    rl_is_active=True,
                    act_segment_generation=22,
                    act_segment_max_steps=max_steps,
                    act_segment_completed_steps=max_steps,
                    act_segment_complete=True,
                ),
                active=None,
            )

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=object(), act_run_timeout_s=90
    )

    with pytest.raises(RuntimeError, match="active RL binding"):
        operations.run_act_dig(130)

    assert events == [("terminal_disarm",)]


def test_rl_behavior_rejects_an_incoherent_active_status():
    events = []

    class Control:
        def ensure_ready(self):
            return _status()

        def activate_rl(self):
            return replace(_status(rl_is_active=True), active=None)

        def terminal_disarm(self):
            events.append(("terminal_disarm",))
            return _status()

    class Behavior:
        def run_rl_to_dig(self, target_id):
            events.append(("run_rl_to_dig", target_id))

    operations = ResidentHybridMissionOperations(
        control=Control(), behavior=Behavior(), act_run_timeout_s=90
    )

    with pytest.raises(RuntimeError, match="active RL binding"):
        operations.run_rl_to_dig("dig_01")

    assert events == [("terminal_disarm",)]


def test_existing_rl_behavior_adapter_reuses_only_follow_and_fixed_actions():
    calls = []

    class ExistingOperations:
        def run_rl_follow(self, phase, *, target_id=None):
            calls.append(("run_rl_follow", phase, target_id))

        def run_rl_fixed_action(self, behavior, *, behavior_port):
            calls.append(("run_rl_fixed_action", behavior, behavior_port))

    behavior = ExistingRlBehaviorAdapter(
        ExistingOperations(), behavior_port=18083
    )

    behavior.run_rl_to_dig("dig_02")
    behavior.run_rl_to_dump_and_dump()
    behavior.run_rl_return_to_dig("dig_03")

    assert calls == [
        ("run_rl_follow", "dig", "dig_02"),
        ("run_rl_follow", "dump", None),
        ("run_rl_fixed_action", "ExecuteDump", 18083),
        ("run_rl_follow", "dig", "dig_03"),
    ]


def test_ssh_control_status_uses_one_short_cli_and_parses_the_exact_contract():
    calls = []
    response = _wire_response()

    def run_command(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(response) + "\n", stderr=""
        )

    control = SshResidentControlAdapter(
        ssh_host="jetson16@192.168.50.2",
        orin_repo="/srv/excavator-orin-runtime",
        socket_path="/run/excavator/resident-control.sock",
        ensure_services_ready=lambda: None,
        run_command=run_command,
    )

    status = control.status()

    assert status.active == ResidentPolicyBinding(
        source="rl_follow", mode="velocity_reference"
    )
    assert status.act_segment_generation == 8
    argv, kwargs = calls[0]
    assert argv[:5] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
    ]
    assert argv[5] == "jetson16@192.168.50.2"
    assert argv[6] == (
        "cd /srv/excavator-orin-runtime && "
        "python3 -m edge_runtime.resident_control "
        "--socket /run/excavator/resident-control.sock status"
    )
    assert kwargs == {"capture_output": True, "text": True, "timeout": 30}


def test_ssh_control_exposes_ready_activate_and_disarm_as_short_requests():
    events = []

    def run_command(argv, **_kwargs):
        remote_command = argv[-1]
        events.append(("remote", remote_command))
        if remote_command.endswith(" activate_rl"):
            command = "activate_rl"
            response = _wire_response(command)
        elif remote_command.endswith(" activate_act --max-steps 130"):
            command = "activate_act"
            response = _wire_response(
                command,
                control_generation=10,
                active={"source": "act_dig", "mode": "manual_action"},
                rl_is_active=False,
                act_is_active=True,
                act_segment_generation=10,
                act_segment_completed_steps=0,
                act_segment_complete=False,
            )
        elif remote_command.endswith(" renew_lease"):
            command = "renew_lease"
            response = _wire_response(command, mission_lease_active=True)
        elif remote_command.endswith(" terminal_disarm"):
            command = "terminal_disarm"
            response = _wire_response(
                command,
                phase="idle",
                active=None,
                rl_is_active=False,
                mission_lease_active=False,
                is_operational=False,
            )
        else:
            command = "status"
            response = _wire_response(command)
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(response) + "\n", stderr=""
        )

    control = SshResidentControlAdapter(
        ssh_host="jetson16@192.168.50.2",
        orin_repo="/srv/excavator-orin-runtime",
        socket_path="/run/excavator/resident-control.sock",
        ensure_services_ready=lambda: events.append(("ensure_services_ready",)),
        run_command=run_command,
    )

    assert control.ensure_ready().act_worker_ready is True
    assert control.activate_rl().rl_is_active is True
    assert control.activate_act(130).act_segment_generation == 10
    assert control.renew_lease().mission_lease_active is True
    assert control.terminal_disarm().is_operational is False

    assert events[0] == ("ensure_services_ready",)
    assert [event[1].rsplit(" ", 1)[-1] for event in events[1:]] == [
        "status",
        "activate_rl",
        "130",
        "renew_lease",
        "terminal_disarm",
    ]


def test_ssh_lease_renewal_has_a_shorter_timeout_than_other_control_requests():
    calls = []

    def run_command(argv, **kwargs):
        calls.append((argv[-1], kwargs["timeout"]))
        command = "renew_lease" if argv[-1].endswith(" renew_lease") else "status"
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                _wire_response(command, mission_lease_active=True)
            )
            + "\n",
            stderr="",
        )

    control = SshResidentControlAdapter(
        ssh_host="jetson16@192.168.50.2",
        orin_repo="/srv/excavator-orin-runtime",
        socket_path="/run/excavator/resident-control.sock",
        ensure_services_ready=lambda: None,
        run_command=run_command,
        command_timeout_s=30,
        lease_command_timeout_s=1,
    )

    control.status()
    control.renew_lease()

    assert [timeout for _command, timeout in calls] == [30, 1]


def _json_with_status(**changes):
    response = _wire_response()
    response["status"] = {**response["status"], **changes}
    return json.dumps(response)


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({**_wire_response(), "extra": 1}),
        json.dumps(
            {
                **_wire_response(),
                "status": {**_wire_response()["status"], "extra": 1},
            }
        ),
        _json_with_status(control_generation=True),
        _json_with_status(
            active={"source": " rl_follow", "mode": "velocity_reference"}
        ),
        _json_with_status(act_segment_max_steps=2001),
        _json_with_status(mission_lease_active=1),
        json.dumps(_wire_response()).replace("37.25", "NaN", 1),
        json.dumps(_wire_response()).replace(
            '"ok": true', '"ok": true, "ok": true', 1
        ),
        json.dumps(_wire_response()) + " " * 4097,
    ],
)
def test_ssh_control_rejects_ambiguous_or_invalid_status_json(payload):
    def run_command(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv, 0, stdout=payload + "\n", stderr=""
        )

    control = SshResidentControlAdapter(
        ssh_host="jetson16@192.168.50.2",
        orin_repo="/srv/excavator-orin-runtime",
        socket_path="/run/excavator/resident-control.sock",
        ensure_services_ready=lambda: None,
        run_command=run_command,
    )

    with pytest.raises(RuntimeError, match="invalid response"):
        control.status()


def test_ssh_control_quotes_remote_paths_and_never_uses_a_password_helper():
    calls = []

    def run_command(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(_wire_response()) + "\n", stderr=""
        )

    control = SshResidentControlAdapter(
        ssh_host="jetson16@192.168.50.2",
        orin_repo="/srv/runtime; touch /tmp/repo-pwn",
        socket_path="/run/control; touch /tmp/socket-pwn.sock",
        ensure_services_ready=lambda: None,
        run_command=run_command,
    )

    control.status()

    argv = calls[0]
    assert argv[-1] == (
        "cd '/srv/runtime; touch /tmp/repo-pwn' && "
        "python3 -m edge_runtime.resident_control --socket "
        "'/run/control; touch /tmp/socket-pwn.sock' status"
    )
    assert "BatchMode=yes" in argv
    assert all("sshpass" not in value.lower() for value in argv)
    assert all("password" not in value.lower() for value in argv)


@pytest.mark.parametrize(
    "socket_path",
    ["relative.sock", "/" + "x" * 108, "/run/control\x00.sock"],
)
def test_ssh_control_requires_a_safe_absolute_unix_socket_path(socket_path):
    with pytest.raises(ValueError, match="socket_path"):
        SshResidentControlAdapter(
            ssh_host="jetson16@192.168.50.2",
            orin_repo="/srv/excavator-orin-runtime",
            socket_path=socket_path,
            ensure_services_ready=lambda: None,
        )


@pytest.mark.parametrize(
    "ssh_host",
    ["-oProxyCommand=touch /tmp/pwn", "jetson16@host; touch /tmp/pwn", "host"],
)
def test_ssh_control_rejects_unsafe_ssh_destinations(ssh_host):
    with pytest.raises(ValueError, match="ssh_host"):
        SshResidentControlAdapter(
            ssh_host=ssh_host,
            orin_repo="/srv/excavator-orin-runtime",
            socket_path="/run/excavator/resident-control.sock",
            ensure_services_ready=lambda: None,
        )
