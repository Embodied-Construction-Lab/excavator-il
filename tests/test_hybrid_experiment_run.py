import json
from pathlib import Path

import pytest

from excavator_il.hybrid_experiment_run import (
    HybridEvidenceArtifact,
    HybridEvidenceIncompleteError,
    HybridExperimentRunConfig,
    HybridExperimentRunFactory,
    HybridMissionRunRequest,
)


PROJECT_ROOT = Path(__file__).parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ACTIVE_CONFIG = PROJECT_ROOT / "config/hybrid_evidence.pc.json"
TEST_MISSION_SHA256 = "a" * 64


def _repository_paths():
    return {
        "excavator_il": PROJECT_ROOT,
        "excavator_orin_runtime": WORKSPACE_ROOT / "excavator-orin-runtime",
        "airy_lidar": WORKSPACE_ROOT / "AiryLidar",
        "rl_excavator": WORKSPACE_ROOT / "RLExcavator",
    }


def _scope_kwargs():
    return {
        "source_path": ACTIVE_CONFIG,
        "evidence_root": WORKSPACE_ROOT / "EvaluationReport/experiment_runs",
        "machine_profile_path": WORKSPACE_ROOT / "shared/machine_profile.json",
        "repository_paths": _repository_paths(),
        "mission_id": "fixed_target_hybrid",
        "mission_sha256": TEST_MISSION_SHA256,
    }


class _RecordingExperimentRun:
    def __init__(self, run_id="hybrid_run_001"):
        self.run_id = run_id
        self.events = []
        self.artifacts = []
        self.finalizations = []

    def append_event(self, event_type, payload=None):
        self.events.append((event_type, dict(payload or {})))

    def register_artifact(self, artifact_id, source_path, *, role, metadata=None):
        artifact = (artifact_id, str(source_path), role, dict(metadata or {}))
        self.artifacts.append(artifact)

    def finalize(self, status, *, metrics=None, summary=None):
        self.finalizations.append((status, dict(metrics or {}), summary))


class _HybridConfig:
    guided_config = PROJECT_ROOT / "config/guided.json"

    def __init__(self, runtime_backend="resident"):
        self.runtime_backend = runtime_backend


class _GuidedConfig:
    operator_id = "operator_07"
    material_id = "soil_default"


def _policy_artifacts():
    return (
        HybridEvidenceArtifact(
            artifact_id="act_deployment_manifest",
            source_path=PROJECT_ROOT / "models/act/deployment_manifest.json",
            role="act_deployment_manifest",
        ),
        HybridEvidenceArtifact(
            artifact_id="act_policy_checkpoint",
            source_path=PROJECT_ROOT / "models/act/checkpoint",
            role="act_policy_checkpoint",
        ),
        HybridEvidenceArtifact(
            artifact_id="rl_onnx_model",
            source_path=WORKSPACE_ROOT
            / "RLExcavator/Assets/AIModels/follow.onnx",
            role="rl_onnx_model",
        ),
    )


def _fixed_action_artifacts():
    return (
        HybridEvidenceArtifact(
            artifact_id="fixed_action_profile",
            source_path=WORKSPACE_ROOT
            / "excavator-orin-runtime/config/fixed_actions.json",
            role="fixed_action_profile",
        ),
    )


def _fixed_action_preflight_config(
    tmp_path,
    *,
    profile_text,
    expected_profile_id="fixed_dig_v1",
):
    workspace = tmp_path / "RL_prj"
    repository_paths = {
        "excavator_il": workspace / "excavator-il",
        "excavator_orin_runtime": workspace / "excavator-orin-runtime",
        "airy_lidar": workspace / "AiryLidar",
        "rl_excavator": workspace / "RLExcavator",
    }
    for repository in repository_paths.values():
        repository.mkdir(parents=True)
    source_path = repository_paths["excavator_il"] / "config/hybrid-evidence.json"
    profile_path = (
        repository_paths["excavator_orin_runtime"] / "config/fixed-actions.json"
    )
    machine_profile_path = workspace / "shared/machine.json"
    evidence_parent = workspace / "EvaluationReport"
    for directory in (
        source_path.parent,
        profile_path.parent,
        machine_profile_path.parent,
        evidence_parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}", encoding="utf-8")
    profile_path.write_text(profile_text, encoding="utf-8")
    machine_profile_path.write_text("{}", encoding="utf-8")
    return HybridExperimentRunConfig(
        evidence_root=evidence_parent / "experiment_runs",
        machine_profile_path=machine_profile_path,
        repository_paths=repository_paths,
        config_paths={},
        policy_ids={
            "dig_policy": f"fixed_action:{expected_profile_id}",
            "trajectory_controller": "cartesian_p:deterministic_v1",
        },
        host_topology={},
        mission_id="fixed_dig_hybrid",
        mission_sha256=TEST_MISSION_SHA256,
        artifacts=(
            HybridEvidenceArtifact(
                artifact_id="fixed_action_profile",
                source_path=profile_path,
                role="fixed_action_profile",
            ),
        ),
        source_path=source_path,
    )


def _policy_ids(
    *,
    dig_policy="lerobot_act:step200000",
    trajectory_controller="onnx_rl:scale-v3",
):
    return {
        "dig_policy": dig_policy,
        "trajectory_controller": trajectory_controller,
    }


def test_hybrid_experiment_run_config_loads_strict_json_and_resolves_paths(
    tmp_path,
):
    config_path = tmp_path / "config" / "hybrid-evidence.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
                {
                    "schema_version": "excavator_hybrid_evidence_config.v2",
                    "mission_id": "fixed_target_hybrid",
                    "mission_sha256": TEST_MISSION_SHA256,
                    "evaluation_scope": "training_internal",
                    "evidence_root": "../evidence",
                "machine_profile_path": "../shared/machine.json",
                "repository_paths": {
                    "excavator_il": "../repos/excavator-il",
                    "excavator_orin_runtime": "../repos/orin",
                    "airy_lidar": "../repos/AiryLidar",
                    "rl_excavator": "../repos/RLExcavator",
                },
                "config_paths": {"act_runtime": "act-runtime.json"},
                "policy_ids": {
                    "dig_policy": "lerobot_act:step200000",
                    "trajectory_controller": "onnx_rl:scale-v3",
                },
                "host_topology": {
                    "pc": {"host": "192.168.50.1", "role": "planning_ui"},
                    "orin": {
                        "host": "192.168.50.2",
                        "role": "resident_runtime",
                    },
                },
                "evidence_requirements": {
                    "act_deployment_manifest": {"required": True, "min_count": 1},
                    "act_policy_checkpoint": {"required": True, "min_count": 1},
                    "rl_onnx_model": {"required": True, "min_count": 1},
                },
                "task_context": {
                    "task_variant": "dig_transport_dump",
                    "soil_reset_block_id": None,
                    "operator_id": None,
                    "material_id": None,
                },
                "artifacts": [
                    {
                        "artifact_id": "act_deployment_manifest",
                        "source_path": "act-deployment.json",
                        "role": "act_deployment_manifest",
                        "metadata": {"policy": "dig_policy"},
                    },
                    {
                        "artifact_id": "act_policy_checkpoint",
                        "source_path": "../models/act",
                        "role": "act_policy_checkpoint",
                        "metadata": {},
                    },
                    {
                        "artifact_id": "rl_onnx_model",
                        "source_path": "../models/rl.onnx",
                        "role": "rl_onnx_model",
                        "metadata": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = HybridExperimentRunConfig.load(config_path)

    assert config.evidence_root == (tmp_path / "evidence").resolve()
    assert config.mission_id == "fixed_target_hybrid"
    assert config.machine_profile_path == (tmp_path / "shared/machine.json").resolve()
    assert config.repository_paths["excavator_il"] == (
        tmp_path / "repos/excavator-il"
    ).resolve()
    assert config.config_paths["act_runtime"] == (
        tmp_path / "config/act-runtime.json"
    ).resolve()
    assert config.artifacts[0].source_path == (
        tmp_path / "config/act-deployment.json"
    ).resolve()


def test_fixed_action_dig_policy_requires_fixed_action_profile_binding():
    config = HybridExperimentRunConfig(
        evidence_root=Path("/evidence"),
        machine_profile_path=Path("/profiles/machine_profile.json"),
        repository_paths={},
        config_paths={},
        policy_ids={
            "dig_policy": "fixed_action:dig_lift_dump_v1",
            "trajectory_controller": "cartesian_p:deterministic_v1",
        },
        host_topology={},
        mission_id="fixed_dig_hybrid",
        mission_sha256=TEST_MISSION_SHA256,
        artifacts=_fixed_action_artifacts(),
    )

    assert config.evidence_requirements["fixed_action_profile"].required is True
    assert config.evidence_requirements["fixed_action_profile"].min_count == 1


def test_fixed_action_policy_identity_must_match_bound_profile(tmp_path):
    config = _fixed_action_preflight_config(
        tmp_path,
        profile_text=json.dumps({"profile_id": "fixed_dig_v2"}),
    )

    with pytest.raises(
        ValueError,
        match="fixed_action policy identity does not match.*fixed_dig_v1.*fixed_dig_v2",
    ):
        HybridExperimentRunFactory(config).preflight()


def test_fixed_action_profile_identity_rejects_invalid_json(tmp_path):
    config = _fixed_action_preflight_config(
        tmp_path,
        profile_text="not-json",
    )

    with pytest.raises(ValueError, match="cannot load fixed_action_profile"):
        HybridExperimentRunFactory(config).preflight()


def test_fixed_action_profile_identity_requires_profile_id(tmp_path):
    config = _fixed_action_preflight_config(
        tmp_path,
        profile_text="{}",
    )

    with pytest.raises(ValueError, match="fixed_action_profile.profile_id"):
        HybridExperimentRunFactory(config).preflight()


def test_fixed_action_identity_mismatch_blocks_mission_without_preflight(tmp_path):
    config = _fixed_action_preflight_config(
        tmp_path,
        profile_text=json.dumps({"profile_id": "fixed_dig_v2"}),
    )
    created = []
    factory = HybridExperimentRunFactory(
        config,
        run_creator=lambda _root, **_kwargs: created.append("created"),
        hybrid_config_loader=lambda _path: _HybridConfig(),
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda _path: True,
    )

    with pytest.raises(ValueError, match="fixed_action policy identity does not match"):
        factory(
            HybridMissionRunRequest(
                config_path=Path("/configs/hybrid.json"),
                dig_target_id="dig_01",
                automatic=True,
                requested_cycles=1,
            )
        )

    assert created == []


def test_hybrid_experiment_run_config_loads_fixed_action_policy_requirements(
    tmp_path,
):
    config_path = tmp_path / "config" / "hybrid-evidence.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_hybrid_evidence_config.v2",
                "mission_id": "fixed_dig_hybrid",
                "mission_sha256": TEST_MISSION_SHA256,
                "evaluation_scope": "training_internal",
                "evidence_root": "../evidence",
                "machine_profile_path": "../shared/machine.json",
                "repository_paths": {
                    "excavator_il": "../repos/excavator-il",
                    "excavator_orin_runtime": "../repos/orin",
                    "airy_lidar": "../repos/AiryLidar",
                    "rl_excavator": "../repos/RLExcavator",
                },
                "config_paths": {"fixed_actions": "fixed-actions.json"},
                "policy_ids": {
                    "dig_policy": "fixed_action:dig_lift_dump_v1",
                    "trajectory_controller": "cartesian_p:deterministic_v1",
                },
                "host_topology": {
                    "pc": {"host": "192.168.50.1", "role": "planning_ui"},
                    "orin": {
                        "host": "192.168.50.2",
                        "role": "resident_runtime",
                    },
                },
                "evidence_requirements": {
                    "fixed_action_profile": {"required": True, "min_count": 1}
                },
                "task_context": {
                    "task_variant": "dig_transport_dump",
                    "soil_reset_block_id": None,
                    "operator_id": None,
                    "material_id": None,
                },
                "artifacts": [
                    {
                        "artifact_id": "fixed_action_profile",
                        "source_path": "fixed-actions.json",
                        "role": "fixed_action_profile",
                        "metadata": {"policy": "dig_policy"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = HybridExperimentRunConfig.load(config_path)

    assert config.policy_ids["dig_policy"] == "fixed_action:dig_lift_dump_v1"
    assert tuple(config.evidence_requirements) == ("fixed_action_profile",)


def test_fixed_action_with_onnx_rl_requires_fixed_action_and_rl_bindings():
    config = HybridExperimentRunConfig(
        evidence_root=Path("/evidence"),
        machine_profile_path=Path("/profiles/machine_profile.json"),
        repository_paths={},
        config_paths={},
        policy_ids=_policy_ids(
            dig_policy="fixed_action:dig_lift_dump_v1",
            trajectory_controller="onnx_rl:scale-v3",
        ),
        host_topology={},
        mission_id="fixed_dig_hybrid",
        mission_sha256=TEST_MISSION_SHA256,
        artifacts=(
            *_fixed_action_artifacts(),
            HybridEvidenceArtifact(
                artifact_id="rl_onnx_model",
                source_path=Path("/models/rl/follow.onnx"),
                role="rl_onnx_model",
            ),
        ),
    )

    assert set(config.evidence_requirements) == {
        "fixed_action_profile",
        "rl_onnx_model",
    }


def test_fixed_action_dig_policy_forbids_act_policy_artifacts():
    with pytest.raises(
        ValueError,
        match=(
            "unsupported policy artifacts: act_deployment_manifest, "
            "act_policy_checkpoint"
        ),
    ):
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={},
            config_paths={},
            policy_ids=_policy_ids(
                dig_policy="fixed_action:dig_lift_dump_v1",
                trajectory_controller="cartesian_p:deterministic_v1",
            ),
            host_topology={},
            mission_id="fixed_dig_hybrid",
            mission_sha256=TEST_MISSION_SHA256,
            artifacts=(
                *_fixed_action_artifacts(),
                HybridEvidenceArtifact(
                    artifact_id="act_deployment_manifest",
                    source_path=Path("/models/act/deployment_manifest.json"),
                    role="act_deployment_manifest",
                ),
                HybridEvidenceArtifact(
                    artifact_id="act_policy_checkpoint",
                    source_path=Path("/models/act/checkpoint"),
                    role="act_policy_checkpoint",
                ),
            ),
        )


def test_hybrid_evidence_config_requires_act_and_rl_model_artifact_bindings():
    with pytest.raises(ValueError, match="act_deployment_manifest"):
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={},
            config_paths={},
            policy_ids=_policy_ids(),
            host_topology={},
            mission_id="fixed_target_hybrid",
            mission_sha256=TEST_MISSION_SHA256,
        )


def test_experiment_run_factory_uses_explicit_evidence_paths_and_config_metadata():
    recorder = _RecordingExperimentRun()
    created = []

    def create(root, **kwargs):
        created.append((root, kwargs))
        return recorder

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={"act_runtime": PROJECT_ROOT / "config/act.json"},
            policy_ids={
                "dig_policy": "lerobot_act:checkpoint-200k",
                "trajectory_controller": "onnx_rl:7496592",
            },
            host_topology={"pc": "planner", "orin": "resident_runtime"},
            artifacts=_policy_artifacts(),
        ),
        run_creator=create,
        hybrid_config_loader=lambda _path: _HybridConfig(),
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda _path: True,
    )

    run = factory(
        HybridMissionRunRequest(
            config_path=PROJECT_ROOT / "config/hybrid.json",
            dig_target_id="dig_03",
            automatic=True,
            requested_cycles=3,
        )
    )

    assert run.run_id == "hybrid_run_001"
    assert created[0][0] == WORKSPACE_ROOT / "EvaluationReport/experiment_runs"
    kwargs = created[0][1]
    assert kwargs["run_kind"] == "hybrid_live"
    assert kwargs["task_context"].task_variant == "dig_transport_dump"
    assert kwargs["task_context"].dig_point_id == "dig_03"
    assert kwargs["task_context"].operator_id == "operator_07"
    assert kwargs["task_context"].material_id == "soil_default"
    assert kwargs["policy_ids"] == {
        "dig_policy": "lerobot_act:checkpoint-200k",
        "trajectory_controller": "onnx_rl:7496592",
    }
    assert kwargs["config_paths"] == {
        "act_runtime": PROJECT_ROOT / "config/act.json",
        "hybrid_evidence": ACTIVE_CONFIG,
        "guided_episode": PROJECT_ROOT / "config/guided.json",
        "hybrid_mission": PROJECT_ROOT / "config/hybrid.json",
    }
    assert all(
        kwargs["evidence_requirements"][role].required
        for role in (
            "act_deployment_manifest",
            "act_policy_checkpoint",
            "rl_onnx_model",
        )
    )
    assert recorder.events == []

    run.append_event("mission_started", {"run_id": run.run_id})

    assert recorder.events == [
        (
            "runtime_selected",
            {
                "run_id": "hybrid_run_001",
                "runtime_backend": "resident",
                "automatic": True,
                "requested_cycles": 3,
                "mission_id": "fixed_target_hybrid",
                "mission_sha256": TEST_MISSION_SHA256,
            },
        ),
        ("mission_started", {"run_id": "hybrid_run_001"}),
    ]
    assert {role for _artifact_id, _path, role, _metadata in recorder.artifacts} == {
        "act_deployment_manifest",
        "act_policy_checkpoint",
        "rl_onnx_model",
    }


def test_experiment_run_factory_supports_v3a_runtime_config_label():
    recorder = _RecordingExperimentRun(run_id="v3a_run_001")
    created = []
    runtime_config = type(
        "FixedConfig",
        (),
        {
            "guided_config": PROJECT_ROOT / "config/guided.json",
            "expected_mission_id": "fixed_target_hybrid",
            "expected_mission_sha256": TEST_MISSION_SHA256,
        },
    )()
    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={"act_runtime": PROJECT_ROOT / "config/act.json"},
            policy_ids={
                "dig_policy": "lerobot_act:checkpoint-200k",
                "trajectory_controller": "onnx_rl:7496592",
            },
            host_topology={"pc": "display", "orin": "resident_runtime"},
            artifacts=_policy_artifacts(),
        ),
        run_creator=lambda root, **kwargs: created.append((root, kwargs)) or recorder,
        hybrid_config_loader=lambda _path: runtime_config,
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda _path: True,
        runtime_config_label="resident_fixed_cycle",
        runtime_backend="resident_fixed_cycle",
    )

    run = factory(
        HybridMissionRunRequest(
            config_path=PROJECT_ROOT / "config/resident-fixed.json",
            dig_target_id="dig_01",
            automatic=True,
            requested_cycles=3,
        )
    )
    run.append_event("mission_started", {"run_id": run.run_id})

    assert created[0][1]["config_paths"]["resident_fixed_cycle"] == Path(
        PROJECT_ROOT / "config/resident-fixed.json"
    )
    assert "hybrid_mission" not in created[0][1]["config_paths"]
    assert recorder.events[0][1]["runtime_backend"] == "resident_fixed_cycle"


def test_v3a_runtime_mission_id_must_match_evidence_before_run_creation():
    created = []
    runtime_config = type(
        "FixedConfig",
        (),
        {
            "guided_config": PROJECT_ROOT / "config/guided.json",
            "expected_mission_id": "classical_tracking_hybrid",
            "expected_mission_sha256": TEST_MISSION_SHA256,
        },
    )()
    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={},
            policy_ids=_policy_ids(),
            host_topology={},
            artifacts=_policy_artifacts(),
        ),
        run_creator=lambda _root, **_kwargs: created.append("created"),
        hybrid_config_loader=lambda _path: runtime_config,
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda _path: True,
        runtime_config_label="resident_fixed_cycle",
        runtime_backend="resident_fixed_cycle",
    )

    with pytest.raises(ValueError, match="runtime mission_id does not match evidence"):
        factory(
            HybridMissionRunRequest(
                config_path=PROJECT_ROOT / "config/resident-fixed.json",
                dig_target_id="dig_01",
                automatic=True,
                requested_cycles=1,
            )
        )

    assert created == []


def test_experiment_run_factory_registers_only_available_configured_logs():
    recorder = _RecordingExperimentRun()

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={},
            policy_ids=_policy_ids(),
            host_topology={},
            artifacts=(
                *_policy_artifacts(),
                HybridEvidenceArtifact(
                    artifact_id="mission_log",
                    source_path=PROJECT_ROOT / "logs/mission.log",
                    role="mission_log",
                ),
                HybridEvidenceArtifact(
                    artifact_id="runtime_log",
                    source_path=PROJECT_ROOT / "logs/runtime.log",
                    role="runtime_log",
                ),
            ),
        ),
        run_creator=lambda _root, **_kwargs: recorder,
        hybrid_config_loader=lambda _path: _HybridConfig("legacy"),
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda path: path.name != "runtime.log",
    )
    run = factory(
        HybridMissionRunRequest(
            config_path=PROJECT_ROOT / "config/hybrid.json",
            dig_target_id="dig_01",
            automatic=False,
            requested_cycles=1,
        )
    )

    run.finalize(
        "success",
        metrics={"requested_cycles": 1, "completed_cycles": 1},
        summary="hybrid Mission completed",
    )

    assert (
        "mission_log",
        str(PROJECT_ROOT / "logs/mission.log"),
        "mission_log",
        {},
    ) in recorder.artifacts
    assert {role for _artifact_id, _path, role, _metadata in recorder.artifacts} >= {
        "act_deployment_manifest",
        "act_policy_checkpoint",
        "rl_onnx_model",
    }
    assert recorder.events[-1] == (
        "artifact_unavailable",
        {
            "run_id": "hybrid_run_001",
            "artifact_id": "runtime_log",
            "role": "runtime_log",
            "path": str(PROJECT_ROOT / "logs/runtime.log"),
        },
    )
    assert recorder.finalizations == [
        (
                "success",
                {
                    "requested_cycles": 1,
                    "completed_cycles": 1,
                    "evaluation_scope": "training_internal",
                    "mission_id": "fixed_target_hybrid",
                    "mission_sha256": TEST_MISSION_SHA256,
                },
            "hybrid Mission completed",
        )
    ]


def test_registered_policy_snapshot_remains_authoritative_if_source_disappears():
    recorder = _RecordingExperimentRun()
    rl_checks = 0

    def path_is_available(path):
        nonlocal rl_checks
        if path.name != "follow.onnx":
            return True
        rl_checks += 1
        return rl_checks == 1

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={},
            policy_ids=_policy_ids(),
            host_topology={},
            artifacts=_policy_artifacts(),
        ),
        run_creator=lambda _root, **_kwargs: recorder,
        hybrid_config_loader=lambda _path: _HybridConfig(),
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=path_is_available,
    )
    run = factory(
        HybridMissionRunRequest(
            config_path=PROJECT_ROOT / "config/hybrid.json",
            dig_target_id="dig_01",
            automatic=True,
            requested_cycles=1,
        )
    )

    run.finalize(
        "success",
        metrics={"requested_cycles": 1, "completed_cycles": 1},
        summary="hybrid Mission completed",
    )

    assert recorder.finalizations == [
        (
            "success",
            {
                "requested_cycles": 1,
                "completed_cycles": 1,
                "evaluation_scope": "training_internal",
                "mission_id": "fixed_target_hybrid",
                "mission_sha256": TEST_MISSION_SHA256,
            },
            "hybrid Mission completed",
        )
    ]
    assert all(event[0] != "artifact_unavailable" for event in recorder.events)


def test_factory_rejects_unavailable_required_policy_artifact_before_run_creation():
    created = []

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={},
            policy_ids=_policy_ids(),
            host_topology={},
            artifacts=_policy_artifacts(),
        ),
        run_creator=lambda _root, **_kwargs: created.append("created"),
        hybrid_config_loader=lambda _path: _HybridConfig(),
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda path: path.name != "follow.onnx",
    )

    with pytest.raises(HybridEvidenceIncompleteError, match="rl_onnx_model"):
        factory(
            HybridMissionRunRequest(
                config_path=PROJECT_ROOT / "config/hybrid.json",
                dig_target_id="dig_01",
                automatic=True,
                requested_cycles=1,
            )
        )

    assert created == []


def test_factory_rejects_unavailable_fixed_action_profile_before_run_creation():
    created = []

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            **_scope_kwargs(),
            config_paths={},
            policy_ids=_policy_ids(
                dig_policy="fixed_action:dig_lift_dump_v1",
                trajectory_controller="cartesian_p:deterministic_v1",
            ),
            host_topology={},
            artifacts=_fixed_action_artifacts(),
        ),
        run_creator=lambda _root, **_kwargs: created.append("created"),
        hybrid_config_loader=lambda _path: _HybridConfig(),
        guided_config_loader=lambda _path: _GuidedConfig(),
        path_is_available=lambda path: path.name != "fixed_actions.json",
    )

    with pytest.raises(HybridEvidenceIncompleteError, match="fixed_action_profile"):
        factory(
            HybridMissionRunRequest(
                config_path=PROJECT_ROOT / "config/hybrid.json",
                dig_target_id="dig_01",
                automatic=True,
                requested_cycles=1,
            )
        )

    assert created == []


def test_factory_preflight_validates_all_static_evidence_paths_before_mission(
    tmp_path,
):
    workspace = tmp_path / "RL_prj"
    repositories = {
        "excavator_il": workspace / "excavator-il",
        "excavator_orin_runtime": workspace / "excavator-orin-runtime",
        "airy_lidar": workspace / "AiryLidar",
        "rl_excavator": workspace / "RLExcavator",
    }
    profile = workspace / "shared/machine.json"
    source_path = repositories["excavator_il"] / "config/hybrid-evidence.json"
    runtime_config = repositories["excavator_il"] / "config/runtime.json"
    act_manifest = repositories["excavator_il"] / "models/act.json"
    act_checkpoint = repositories["excavator_il"] / "models/checkpoint"
    rl_model = repositories["rl_excavator"] / "Assets/AIModels/rl.onnx"
    for directory in (
        profile.parent,
        *repositories.values(),
        source_path.parent,
        runtime_config.parent,
        act_checkpoint,
        rl_model.parent,
        workspace / "EvaluationReport",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for file_path in (profile, source_path, runtime_config, act_manifest):
        file_path.write_text("{}", encoding="utf-8")

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=workspace / "EvaluationReport/experiment_runs",
            machine_profile_path=profile,
            repository_paths=repositories,
            config_paths={"act_runtime": runtime_config},
            policy_ids={
                "dig_policy": "lerobot_act:step200000",
                "trajectory_controller": "onnx_rl:scale-v3",
            },
            host_topology={"pc": {}, "orin": {}},
            mission_id="fixed_target_hybrid",
            mission_sha256=TEST_MISSION_SHA256,
            artifacts=(
                HybridEvidenceArtifact(
                    "act_deployment_manifest",
                    act_manifest,
                    "act_deployment_manifest",
                ),
                HybridEvidenceArtifact(
                    "act_policy_checkpoint",
                    act_checkpoint,
                    "act_policy_checkpoint",
                ),
                HybridEvidenceArtifact(
                    "rl_onnx_model",
                    rl_model,
                    "rl_onnx_model",
                ),
            ),
            source_path=source_path,
        )
    )

    with pytest.raises(ValueError, match="rl_onnx_model.*does not exist"):
        factory.preflight()

    rl_model.write_bytes(b"onnx")

    factory.preflight()
