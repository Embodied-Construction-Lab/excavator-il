#!/usr/bin/env python3
"""Statically verify the three-phase ACT engineering-reference assets.

This command only reads versioned configuration and local model files.  It does
not start Docker, cameras, serial devices, or a Mission runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
SRC = REPOSITORY / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from excavator_il.act_deployment import verify_deployment_manifest
from excavator_il.act_phase_conditioning import PHASE_FEATURE_NAMES
from excavator_il.act_runtime_config import load_act_runtime_config
from excavator_il.lerobot_conversion import STATE_FIELDS
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


MODEL_SHA256 = "8faf364022695312714ccbac3299635e8fc8ee30b1935b0f3122b1c1fdedd637"
MISSION_ID = "engineering_act_transport_three_phase_reference"
BEHAVIOR_ID = "act_dig_transport_dump_three_phase"
RUNTIME_CONFIG = "act_runtime.icra2027_transport_dump_three_phase.orin.json"
RESIDENT_CONFIG = (
    "resident_fixed_cycle.act_dig_transport_dump_three_phase_reference."
    "commissioning.pc.json"
)
EVIDENCE_MANIFEST = (
    "act_deployment.icra2027_transport_dump_three_phase_step140000.json"
)
ORIN_IL_ROOT = PurePosixPath("/home/jetson16/workspace_excavator/excavator-il")
ORIN_RUNTIME_ROOT = PurePosixPath(
    "/home/jetson16/workspace_excavator/excavator-orin-runtime"
)


def _local_path(
    host_path: PurePosixPath,
    *,
    host_root: PurePosixPath,
    local_root: Path,
) -> Path:
    try:
        relative = host_path.relative_to(host_root)
    except ValueError as exc:
        raise ValueError(f"asset path is outside its repository: {host_path}") from exc
    return local_root.joinpath(*relative.parts)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_assets(
    *, repository_root: Path, machine_profile_path: Path
) -> dict[str, Any]:
    repository = repository_root.expanduser().resolve()
    machine_profile = machine_profile_path.expanduser().resolve()
    config_root = repository / "config"
    runtime = load_act_runtime_config(config_root / RUNTIME_CONFIG)
    resident = ResidentFixedCyclePcConfig.load(config_root / RESIDENT_CONFIG)

    if runtime.act_behavior_id != BEHAVIOR_ID:
        raise ValueError("runtime does not bind the three-phase ACT behavior")
    if runtime.checkpoint_model_sha256 != MODEL_SHA256:
        raise ValueError("runtime does not bind the selected three-phase checkpoint")
    if runtime.camera_roles != ("front", "dump"):
        raise ValueError("three-phase ACT must use front and dump RGB")
    if runtime.policy_state_fields != STATE_FIELDS + PHASE_FEATURE_NAMES:
        raise ValueError("three-phase ACT state contract is not the exact 14D contract")
    if runtime.phase_schedule is None:
        raise ValueError("three-phase ACT phase schedule is missing")
    if resident.expected_mission_id != MISSION_ID:
        raise ValueError("resident config does not bind the engineering Mission")
    if resident.expected_act_behavior_id != BEHAVIOR_ID:
        raise ValueError("resident config ACT behavior does not match runtime")
    if resident.expected_act_model_sha256 != MODEL_SHA256:
        raise ValueError("resident config ACT model does not match runtime")
    if resident.act_checkpoint_host_path is None:
        raise ValueError("resident config checkpoint path is missing")
    if resident.act_deployment_host_path is None:
        raise ValueError("resident config deployment path is missing")

    checkpoint = _local_path(
        resident.act_checkpoint_host_path,
        host_root=ORIN_IL_ROOT,
        local_root=repository,
    )
    deployment = _local_path(
        resident.act_deployment_host_path,
        host_root=ORIN_IL_ROOT,
        local_root=repository,
    )
    manifest_path = deployment / "deployment_manifest.json"
    manifest = verify_deployment_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint,
        machine_profile_path=machine_profile,
    )
    if tuple(manifest["contract"]["state_fields"]) != runtime.policy_state_fields:
        raise ValueError("runtime and deployment state contracts disagree")
    if manifest_path.read_bytes() != (config_root / EVIDENCE_MANIFEST).read_bytes():
        raise ValueError("deployed manifest does not match versioned evidence")

    orin_repository = repository.parent / "excavator-orin-runtime"
    plan_path = _local_path(
        resident.fixed_cycle_plan,
        host_root=ORIN_RUNTIME_ROOT,
        local_root=orin_repository,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    mission = plan.get("mission")
    if not isinstance(mission, dict):
        raise ValueError("resident plan Mission is invalid")
    if (
        mission.get("mission_id") != MISSION_ID
        or plan.get("mission_sha256") != resident.expected_mission_sha256
        or mission.get("act_policy_bindings") != {BEHAVIOR_ID: MODEL_SHA256}
    ):
        raise ValueError("resident plan identity does not match PC configuration")
    cycle = mission.get("cycle_behaviors")
    if not isinstance(cycle, list) or not cycle:
        raise ValueError("resident plan cycle behaviors are invalid")
    if cycle[0].get("behavior_id") != BEHAVIOR_ID:
        raise ValueError("resident plan does not schedule the three-phase behavior")
    if cycle[0].get("max_steps") != resident.act_max_steps:
        raise ValueError("resident plan and PC ACT step budgets disagree")

    return {
        "schema_version": "excavator_phase_conditioned_act_preflight.v1",
        "passed": True,
        "motion_commands_emitted": 0,
        "mission_id": MISSION_ID,
        "behavior_id": BEHAVIOR_ID,
        "checkpoint_model_sha256": MODEL_SHA256,
        "deployment_manifest_sha256": _sha256(manifest_path),
        "camera_roles": list(runtime.camera_roles),
        "state_dim": len(runtime.policy_state_fields),
        "phase_boundaries": [
            runtime.phase_schedule.dig_to_transport_step,
            runtime.phase_schedule.transport_to_dump_step,
        ],
        "act_max_steps": resident.act_max_steps,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--machine-profile",
        type=Path,
        default=REPOSITORY.parent / "shared/machine_profile.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = inspect_assets(
            repository_root=args.repository_root,
            machine_profile_path=args.machine_profile,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": "excavator_phase_conditioned_act_preflight.v1",
            "passed": False,
            "motion_commands_emitted": 0,
            "failure_reasons": [str(exc)],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
