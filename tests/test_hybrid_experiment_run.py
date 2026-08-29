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
    guided_config = Path("/configs/guided.json")

    def __init__(self, runtime_backend="resident"):
        self.runtime_backend = runtime_backend


class _GuidedConfig:
    operator_id = "operator_07"
    material_id = "soil_default"


def _policy_artifacts():
    return (
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
        HybridEvidenceArtifact(
            artifact_id="rl_onnx_model",
            source_path=Path("/models/rl/follow.onnx"),
            role="rl_onnx_model",
        ),
    )


def test_hybrid_experiment_run_config_loads_strict_json_and_resolves_paths(
    tmp_path,
):
    config_path = tmp_path / "config" / "hybrid-evidence.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
                {
                    "schema_version": "excavator_hybrid_evidence_config.v1",
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


def test_hybrid_evidence_config_requires_act_and_rl_model_artifact_bindings():
    with pytest.raises(ValueError, match="act_deployment_manifest"):
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={},
            config_paths={},
            policy_ids={},
            host_topology={},
        )


def test_experiment_run_factory_uses_explicit_evidence_paths_and_config_metadata():
    recorder = _RecordingExperimentRun()
    created = []

    def create(root, **kwargs):
        created.append((root, kwargs))
        return recorder

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={"excavator_il": Path("/repos/excavator-il")},
            config_paths={"act_runtime": Path("/configs/act.json")},
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
            config_path=Path("/configs/hybrid.json"),
            dig_target_id="dig_03",
            automatic=True,
            requested_cycles=3,
        )
    )

    assert run.run_id == "hybrid_run_001"
    assert created[0][0] == Path("/evidence")
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
        "act_runtime": Path("/configs/act.json"),
        "guided_episode": Path("/configs/guided.json"),
        "hybrid_mission": Path("/configs/hybrid.json"),
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
        {"guided_config": Path("/configs/guided.json")},
    )()
    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={"excavator_il": Path("/repos/excavator-il")},
            config_paths={"act_runtime": Path("/configs/act.json")},
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
            config_path=Path("/configs/resident-fixed.json"),
            dig_target_id="dig_01",
            automatic=True,
            requested_cycles=3,
        )
    )
    run.append_event("mission_started", {"run_id": run.run_id})

    assert created[0][1]["config_paths"]["resident_fixed_cycle"] == Path(
        "/configs/resident-fixed.json"
    )
    assert "hybrid_mission" not in created[0][1]["config_paths"]
    assert recorder.events[0][1]["runtime_backend"] == "resident_fixed_cycle"


def test_experiment_run_factory_registers_only_available_configured_logs():
    recorder = _RecordingExperimentRun()

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={},
            config_paths={},
            policy_ids={},
            host_topology={},
            artifacts=(
                *_policy_artifacts(),
                HybridEvidenceArtifact(
                    artifact_id="mission_log",
                    source_path=Path("/logs/mission.log"),
                    role="mission_log",
                ),
                HybridEvidenceArtifact(
                    artifact_id="runtime_log",
                    source_path=Path("/logs/runtime.log"),
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
            config_path=Path("/configs/hybrid.json"),
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
        "/logs/mission.log",
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
            "path": "/logs/runtime.log",
        },
    )
    assert recorder.finalizations == [
        (
                "success",
                {
                    "requested_cycles": 1,
                    "completed_cycles": 1,
                    "evaluation_scope": "training_internal",
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
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={},
            config_paths={},
            policy_ids={},
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
            config_path=Path("/configs/hybrid.json"),
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
            },
            "hybrid Mission completed",
        )
    ]
    assert all(event[0] != "artifact_unavailable" for event in recorder.events)


def test_factory_rejects_unavailable_required_policy_artifact_before_run_creation():
    created = []

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/profiles/machine_profile.json"),
            repository_paths={},
            config_paths={},
            policy_ids={},
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
                config_path=Path("/configs/hybrid.json"),
                dig_target_id="dig_01",
                automatic=True,
                requested_cycles=1,
            )
        )

    assert created == []


def test_factory_preflight_validates_all_static_evidence_paths_before_mission(
    tmp_path,
):
    profile = tmp_path / "shared/machine.json"
    repository = tmp_path / "repo"
    runtime_config = tmp_path / "config/runtime.json"
    act_manifest = tmp_path / "models/act.json"
    act_checkpoint = tmp_path / "models/checkpoint"
    rl_model = tmp_path / "models/rl.onnx"
    for directory in (
        profile.parent,
        repository,
        runtime_config.parent,
        act_checkpoint,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for file_path in (profile, runtime_config, act_manifest):
        file_path.write_text("{}", encoding="utf-8")

    factory = HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=tmp_path / "evidence/runs",
            machine_profile_path=profile,
            repository_paths={"excavator_il": repository},
            config_paths={"act_runtime": runtime_config},
            policy_ids={
                "dig_policy": "lerobot_act:step200000",
                "trajectory_controller": "onnx_rl:scale-v3",
            },
            host_topology={"pc": {}, "orin": {}},
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
        )
    )

    with pytest.raises(ValueError, match="rl_onnx_model.*does not exist"):
        factory.preflight()

    rl_model.write_bytes(b"onnx")
    (tmp_path / "evidence").mkdir()

    factory.preflight()
