import json
from pathlib import PurePosixPath

import pytest

from excavator_il.hybrid_mission import (
    REQUIRED_HYBRID_MOTION_AUTHORIZATION,
    HybridMissionConfig,
    HybridMissionSegment,
    ResidentMissionConfig,
    execute_hybrid_segment,
    remaining_hybrid_segments,
)


class _Operations:
    def __init__(self):
        self.calls = []

    def run_rl_to_dig(self, target_id):
        self.calls.append(("run_rl_to_dig", target_id))

    def run_act_dig(self, max_steps):
        self.calls.append(("run_act_dig", max_steps))

    def run_rl_to_dump_and_dump(self):
        self.calls.append(("run_rl_to_dump_and_dump",))

    def run_rl_return_to_dig(self, target_id):
        self.calls.append(("run_rl_return_to_dig", target_id))

    def safe_stop(self):
        self.calls.append(("safe_stop",))


def test_hybrid_segments_form_the_requested_closed_loop_in_order():
    assert remaining_hybrid_segments(HybridMissionSegment.RL_TO_DIG) == (
        HybridMissionSegment.RL_TO_DIG,
        HybridMissionSegment.ACT_DIG,
        HybridMissionSegment.RL_TO_DUMP_AND_DUMP,
        HybridMissionSegment.RL_RETURN_TO_DIG,
    )


def test_hybrid_segment_adapts_each_stage_to_one_deep_operation():
    operations = _Operations()

    execute_hybrid_segment(
        operations,
        segment=HybridMissionSegment.RL_TO_DIG,
        dig_target_id="dig_02",
        act_max_steps=130,
        motion_authorization=None,
    )
    execute_hybrid_segment(
        operations,
        segment=HybridMissionSegment.ACT_DIG,
        dig_target_id="dig_02",
        act_max_steps=130,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
    )
    execute_hybrid_segment(
        operations,
        segment=HybridMissionSegment.RL_TO_DUMP_AND_DUMP,
        dig_target_id="dig_02",
        act_max_steps=130,
        motion_authorization=None,
    )
    execute_hybrid_segment(
        operations,
        segment=HybridMissionSegment.RL_RETURN_TO_DIG,
        dig_target_id="dig_02",
        act_max_steps=130,
        motion_authorization=None,
    )

    assert operations.calls == [
        ("run_rl_to_dig", "dig_02"),
        ("run_act_dig", 130),
        ("run_rl_to_dump_and_dump",),
        ("run_rl_return_to_dig", "dig_02"),
    ]


def test_act_segment_requires_exact_explicit_motion_authorization():
    operations = _Operations()

    with pytest.raises(ValueError, match="authorization"):
        execute_hybrid_segment(
            operations,
            segment=HybridMissionSegment.ACT_DIG,
            dig_target_id="dig_01",
            act_max_steps=130,
            motion_authorization="yes",
        )

    assert operations.calls == []


def test_hybrid_config_resolves_paths_and_validates_step_budget(tmp_path):
    guided = tmp_path / "guided.json"
    guided.write_text("{}", encoding="utf-8")
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_hybrid_mission_config.v1",
                "guided_config": "guided.json",
                "act": {
                    "max_steps": 130,
                    "ready_timeout_s": 60,
                    "run_timeout_s": 90,
                    "remote_script": "scripts/run_act_motion.sh",
                },
                "rl": {"behavior_port": 18083},
            }
        ),
        encoding="utf-8",
    )

    config = HybridMissionConfig.load(path)

    assert config.guided_config == guided
    assert config.act_max_steps == 130
    assert config.act_remote_script == "scripts/run_act_motion.sh"
    assert config.rl_behavior_port == 18083
    assert config.runtime_backend == "legacy"
    assert config.resident is None


def test_hybrid_v2_config_selects_one_resident_runtime_for_the_whole_mission(
    tmp_path,
):
    guided = tmp_path / "guided.json"
    guided.write_text("{}", encoding="utf-8")
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_hybrid_mission_config.v2",
                "runtime_backend": "resident",
                "guided_config": "guided.json",
                "act": {
                    "max_steps": 130,
                    "ready_timeout_s": 60,
                    "run_timeout_s": 90,
                    "remote_script": "scripts/run_act_motion.sh",
                },
                "rl": {"behavior_port": 18083},
                "resident": {
                    "owner_script": "scripts/run_resident_mission_runtime.sh",
                    "act_worker_script": "scripts/run_act_resident.sh",
                    "runtime_root": "/home/jetson16/.local/run/excavator-resident",
                    "ready_timeout_s": 120,
                    "handoff_timeout_s": 3,
                    "poll_interval_ms": 50,
                    "prepared_dump_lead_steps": 20,
                    "prepared_ready_grace_ms": 300,
                    "prepared_start_tolerance_m": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )

    config = HybridMissionConfig.load(path)

    assert config.runtime_backend == "resident"
    assert config.resident == ResidentMissionConfig(
        owner_script="scripts/run_resident_mission_runtime.sh",
        act_worker_script="scripts/run_act_resident.sh",
        runtime_root=PurePosixPath(
            "/home/jetson16/.local/run/excavator-resident"
        ),
        ready_timeout_s=120,
        handoff_timeout_s=3,
        poll_interval_ms=50,
        prepared_dump_lead_steps=20,
        prepared_ready_grace_ms=300,
        prepared_start_tolerance_m=0.15,
    )


def test_hybrid_v2_config_rejects_prepared_dump_lead_at_the_act_budget(tmp_path):
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_hybrid_mission_config.v2",
                "runtime_backend": "resident",
                "guided_config": "guided.json",
                "act": {
                    "max_steps": 20,
                    "ready_timeout_s": 1,
                    "run_timeout_s": 3,
                    "remote_script": "scripts/run_act_motion.sh",
                },
                "rl": {"behavior_port": 18083},
                "resident": {
                    "owner_script": "scripts/run_resident_mission_runtime.sh",
                    "act_worker_script": "scripts/run_act_resident.sh",
                    "runtime_root": "/run/excavator-resident",
                    "ready_timeout_s": 120,
                    "handoff_timeout_s": 3,
                    "poll_interval_ms": 50,
                    "prepared_dump_lead_steps": 20,
                    "prepared_ready_grace_ms": 300,
                    "prepared_start_tolerance_m": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="prepared_dump_lead_steps must be less than act.max_steps"
    ):
        HybridMissionConfig.load(path)


def test_hybrid_v2_config_rejects_resident_backend_without_resident_contract(
    tmp_path,
):
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_hybrid_mission_config.v2",
                "runtime_backend": "resident",
                "guided_config": "guided.json",
                "act": {
                    "max_steps": 130,
                    "ready_timeout_s": 60,
                    "run_timeout_s": 90,
                    "remote_script": "scripts/run_act_motion.sh",
                },
                "rl": {"behavior_port": 18083},
                "resident": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resident"):
        HybridMissionConfig.load(path)


@pytest.mark.parametrize("steps", [0, -1, True, 2001])
def test_hybrid_config_rejects_unsafe_step_budget(tmp_path, steps):
    path = tmp_path / "hybrid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_hybrid_mission_config.v1",
                "guided_config": "guided.json",
                "act": {
                    "max_steps": steps,
                    "ready_timeout_s": 60,
                    "run_timeout_s": 90,
                    "remote_script": "scripts/run_act_motion.sh",
                },
                "rl": {"behavior_port": 18083},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="act.max_steps"):
        HybridMissionConfig.load(path)
