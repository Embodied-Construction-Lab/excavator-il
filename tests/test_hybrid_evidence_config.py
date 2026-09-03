import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.hybrid_experiment_run import (
    HybridEvidenceArtifact,
    HybridExperimentRunConfig,
    HybridExperimentRunFactory,
    HybridMissionRunRequest,
)


PROJECT_ROOT = Path(__file__).parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ACTIVE_CONFIG = PROJECT_ROOT / "config/hybrid_evidence.pc.json"
FIXED_TARGET_MISSION_SHA256 = (
    "3a5c7edd6a228863e3d5eefe3228173848756a46e9ce441da53cc2b0c164d786"
)
ACTIVE_EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "config/hybrid_evidence.fixed_target_selection.commissioning.pc.json"
)
ACTIVE_EXPERIMENT_PROFILE = (
    PROJECT_ROOT
    / "config/experiments/icra2027/fixed_target_selection.json"
)
ACTIVE_CLASSICAL_EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "config/hybrid_evidence.classical_tracking.commissioning.pc.json"
)
ACTIVE_FIXED_DIG_EXPERIMENT_CONFIG = (
    PROJECT_ROOT / "config/hybrid_evidence.fixed_dig.commissioning.pc.json"
)


class _RecordingRun:
    run_id = "hybrid_run_scope_001"

    def __init__(self):
        self.finalizations = []
        self.events = []

    def append_event(self, event_type, payload=None):
        self.events.append((event_type, dict(payload or {})))
        return None

    def register_artifact(
        self, _artifact_id, _source_path, *, role, metadata=None
    ):
        return {"role": role, "metadata": dict(metadata or {})}

    def finalize(self, status, *, metrics=None, summary=None):
        self.finalizations.append((status, dict(metrics or {}), summary))


def _artifacts():
    return (
        HybridEvidenceArtifact(
            "act_manifest",
            PROJECT_ROOT / "models/act/deployment_manifest.json",
            "act_deployment_manifest",
        ),
        HybridEvidenceArtifact(
            "act_checkpoint",
            PROJECT_ROOT / "models/act/checkpoint",
            "act_policy_checkpoint",
        ),
        HybridEvidenceArtifact(
            "rl_model",
            WORKSPACE_ROOT / "RLExcavator/Assets/AIModels/rl.onnx",
            "rl_onnx_model",
        ),
    )


def _repository_paths():
    return {
        "excavator_il": PROJECT_ROOT,
        "excavator_orin_runtime": WORKSPACE_ROOT / "excavator-orin-runtime",
        "airy_lidar": WORKSPACE_ROOT / "AiryLidar",
        "rl_excavator": WORKSPACE_ROOT / "RLExcavator",
    }


def _factory(
    recorder,
    *,
    scope="training_internal",
    path_is_available=lambda _p: True,
):
    return HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=WORKSPACE_ROOT / "EvaluationReport/experiment_runs",
            machine_profile_path=WORKSPACE_ROOT / "shared/machine_profile.json",
            repository_paths=_repository_paths(),
            config_paths={},
            policy_ids={
                "dig_policy": "lerobot_act:step200000",
                "trajectory_controller": "onnx_rl:scale-v3",
            },
            host_topology={"pc": {}, "orin": {}},
            mission_id="fixed_target_hybrid",
            mission_sha256=FIXED_TARGET_MISSION_SHA256,
            evaluation_scope=scope,
            artifacts=_artifacts(),
            source_path=ACTIVE_CONFIG,
        ),
        run_creator=lambda _root, **_kwargs: recorder,
        hybrid_config_loader=lambda _path: SimpleNamespace(
            guided_config=Path("/guided.json"), runtime_backend="resident"
        ),
        guided_config_loader=lambda _path: SimpleNamespace(
            operator_id="operator_01", material_id="soil_default"
        ),
        path_is_available=path_is_available,
    )


def _request():
    return HybridMissionRunRequest(
        config_path=Path("/hybrid.json"),
        dig_target_id="dig_01",
        automatic=True,
        requested_cycles=1,
    )


@pytest.mark.parametrize("status", ["success", "failure"])
def test_hybrid_run_wrapper_always_injects_configured_evaluation_scope(status):
    recorder = _RecordingRun()
    run = _factory(recorder, scope="held_out_experiment")(_request())

    run.finalize(status, metrics={"completed_cycles": 1})

    assert recorder.finalizations == [
        (
            status,
            {
                "completed_cycles": 1,
                "evaluation_scope": "held_out_experiment",
                "mission_id": "fixed_target_hybrid",
                "mission_sha256": FIXED_TARGET_MISSION_SHA256,
            },
            None,
        )
    ]


def test_registered_policy_snapshot_preserves_scope_if_source_disappears():
    recorder = _RecordingRun()
    rl_checks = 0

    def availability(path):
        nonlocal rl_checks
        if path.name != "rl.onnx":
            return True
        rl_checks += 1
        return rl_checks == 1

    run = _factory(recorder, path_is_available=availability)(_request())

    run.finalize("success", metrics={"completed_cycles": 1})

    assert recorder.finalizations[0][0] == "success"
    assert recorder.finalizations[0][1]["evaluation_scope"] == "training_internal"


def test_strict_loader_rejects_unknown_fields_and_invalid_scope(tmp_path):
    raw = json.loads(ACTIVE_CONFIG.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    path = tmp_path / "hybrid-evidence.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain exactly"):
        HybridExperimentRunConfig.load(path)

    raw.pop("unexpected")
    raw["evaluation_scope"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation_scope must be non-empty text"):
        HybridExperimentRunConfig.load(path)


def test_commissioned_hybrid_evidence_config_passes_static_preflight():
    config = HybridExperimentRunConfig.load(ACTIVE_CONFIG)

    HybridExperimentRunFactory(config).preflight()

    assert config.evaluation_scope == "training_internal"
    assert set(config.repository_paths) == {
        "excavator_il",
        "excavator_orin_runtime",
        "airy_lidar",
        "rl_excavator",
    }


def test_loaded_evidence_config_cannot_live_outside_declared_excavator_repo(
    tmp_path,
):
    active = HybridExperimentRunConfig.load(ACTIVE_EXPERIMENT_CONFIG)
    payload = json.loads(ACTIVE_EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    payload["evidence_root"] = str(tmp_path / "evidence")
    payload["machine_profile_path"] = str(active.machine_profile_path)
    payload["repository_paths"] = {
        label: str(path) for label, path in active.repository_paths.items()
    }
    payload["config_paths"] = {
        label: str(path) for label, path in active.config_paths.items()
    }
    payload["artifacts"] = [
        {
            "artifact_id": artifact.artifact_id,
            "source_path": str(artifact.source_path),
            "role": artifact.role,
            "metadata": dict(artifact.metadata),
        }
        for artifact in active.artifacts
    ]
    outside = tmp_path / "hybrid-evidence.json"
    outside.write_text(json.dumps(payload), encoding="utf-8")

    config = HybridExperimentRunConfig.load(outside)

    with pytest.raises(ValueError, match="hybrid evidence config.*excavator_il"):
        HybridExperimentRunFactory(config).preflight()


def test_evidence_config_path_cannot_escape_declared_repository_roots(tmp_path):
    config = HybridExperimentRunConfig.load(ACTIVE_EXPERIMENT_CONFIG)
    outside = tmp_path / "borrowed-config.json"
    outside.write_text("{}", encoding="utf-8")
    forged = replace(
        config,
        config_paths={**config.config_paths, "borrowed": outside},
    )

    with pytest.raises(ValueError, match="config_paths.borrowed.*repository root"):
        HybridExperimentRunFactory(forged).preflight()


def test_evidence_config_path_traversal_cannot_escape_workspace(tmp_path):
    config = HybridExperimentRunConfig.load(ACTIVE_EXPERIMENT_CONFIG)
    traversal = PROJECT_ROOT / "config/../../../../tmp/forged-config.json"
    forged = replace(
        config,
        config_paths={**config.config_paths, "traversal": traversal},
    )

    with pytest.raises(ValueError, match="config_paths.traversal.*repository root"):
        HybridExperimentRunFactory(forged).preflight()


def test_hybrid_evidence_loader_rejects_a_symlinked_config(tmp_path):
    redirect = tmp_path / "hybrid-evidence.json"
    redirect.symlink_to(ACTIVE_EXPERIMENT_CONFIG)

    with pytest.raises(ValueError, match="symlink"):
        HybridExperimentRunConfig.load(redirect)


def test_factory_rejects_programmatic_config_without_source_provenance():
    config = HybridExperimentRunConfig(
        evidence_root=PROJECT_ROOT.parent / "EvaluationReport/experiment_runs",
        machine_profile_path=PROJECT_ROOT.parent / "shared/machine_profile.json",
        repository_paths={},
        config_paths={},
        policy_ids={
            "dig_policy": "lerobot_act:test",
            "trajectory_controller": "onnx_rl:test",
        },
        host_topology={},
        mission_id="fixed_target_hybrid",
        mission_sha256=FIXED_TARGET_MISSION_SHA256,
        artifacts=_artifacts(),
    )

    with pytest.raises(ValueError, match="source_path.*required"):
        HybridExperimentRunFactory(config)


def test_commissioning_profile_is_snapshotted_and_emitted_in_run_identity():
    recorder = _RecordingRun()
    captured = {}
    config = HybridExperimentRunConfig.load(ACTIVE_EXPERIMENT_CONFIG)
    factory = HybridExperimentRunFactory(
        config,
        run_creator=lambda _root, **kwargs: (captured.update(kwargs) or recorder),
        hybrid_config_loader=lambda _path: SimpleNamespace(
            guided_config=Path("/guided.json"),
            runtime_backend="resident_fixed_cycle",
            expected_mission_id="fixed_target_hybrid",
            expected_mission_sha256=FIXED_TARGET_MISSION_SHA256,
        ),
        guided_config_loader=lambda _path: SimpleNamespace(
            operator_id="operator_01", material_id="soil_default"
        ),
        path_is_available=lambda _path: True,
        runtime_config_label="resident_fixed_cycle",
        runtime_backend="resident_fixed_cycle",
    )

    run = factory(
        HybridMissionRunRequest(
            config_path=PROJECT_ROOT / "config/resident_fixed_cycle.pc.json",
            dig_target_id="dig_near_01",
            automatic=True,
            requested_cycles=1,
        )
    )
    run.finalize("success", metrics={"completed_cycles": 1})

    assert captured["config_paths"]["experiment_profile"] == (
        ACTIVE_EXPERIMENT_PROFILE
    )
    assert captured["config_paths"]["hybrid_evidence"] == (
        ACTIVE_EXPERIMENT_CONFIG
    )
    assert recorder.events[0] == (
        "runtime_selected",
        {
            "run_id": recorder.run_id,
            "runtime_backend": "resident_fixed_cycle",
            "automatic": True,
            "requested_cycles": 1,
            "experiment_profile_id": "fixed_target_selection",
            "mission_id": "fixed_target_hybrid",
            "mission_sha256": FIXED_TARGET_MISSION_SHA256,
        },
    )
    assert recorder.finalizations[0][1]["experiment_profile_id"] == (
        "fixed_target_selection"
    )
    assert recorder.finalizations[0][1]["mission_id"] == "fixed_target_hybrid"
    assert recorder.finalizations[0][1]["mission_sha256"] == (
        FIXED_TARGET_MISSION_SHA256
    )


def test_experiment_profile_rejects_a_different_runtime_config_before_launch():
    recorder = _RecordingRun()
    config = HybridExperimentRunConfig.load(ACTIVE_CLASSICAL_EXPERIMENT_CONFIG)
    hybrid_loader_called = False

    def load_hybrid(_path):
        nonlocal hybrid_loader_called
        hybrid_loader_called = True
        raise AssertionError("runtime config must be rejected before loading")

    factory = HybridExperimentRunFactory(
        config,
        run_creator=lambda _root, **_kwargs: recorder,
        hybrid_config_loader=load_hybrid,
        path_is_available=lambda _path: True,
        runtime_config_label="resident_fixed_cycle",
        runtime_backend="resident_fixed_cycle",
    )

    with pytest.raises(ValueError, match="experiment_profile.*runtime config"):
        factory(
            HybridMissionRunRequest(
                config_path=PROJECT_ROOT / "config/resident_fixed_cycle.pc.json",
                dig_target_id="dig_near_01",
                automatic=True,
                requested_cycles=1,
            )
        )

    assert hybrid_loader_called is False


def test_commissioning_profile_cannot_be_attached_to_held_out_run():
    config = HybridExperimentRunConfig.load(ACTIVE_EXPERIMENT_CONFIG)

    with pytest.raises(ValueError, match="training_internal"):
        HybridExperimentRunFactory(
            replace(config, evaluation_scope="held_out_experiment"),
        )


def test_classical_tracking_evidence_preflight_has_no_fake_rl_artifact():
    config = HybridExperimentRunConfig.load(ACTIVE_CLASSICAL_EXPERIMENT_CONFIG)

    HybridExperimentRunFactory(config).preflight()

    assert config.policy_ids["trajectory_controller"] == (
        "cartesian_p:deterministic_v1"
    )
    assert "rl_onnx_model" not in config.evidence_requirements
    assert {artifact.role for artifact in config.artifacts} == {
        "act_deployment_manifest",
        "act_policy_checkpoint",
    }


def test_fixed_dig_evidence_preflight_binds_exact_fixed_action_profile():
    config = HybridExperimentRunConfig.load(ACTIVE_FIXED_DIG_EXPERIMENT_CONFIG)

    HybridExperimentRunFactory(config).preflight()

    assert config.policy_ids["dig_policy"] == (
        "fixed_action:icra2027_fixed_dig_fixed_dump_commissioning_v1"
    )


def test_cartesian_p_evidence_requires_no_rl_model_artifact():
    base = _factory(_RecordingRun())._config
    config = replace(
        base,
        policy_ids={
            **base.policy_ids,
            "trajectory_controller": "cartesian_p:deterministic_v1",
        },
        evidence_requirements={
            "act_deployment_manifest": {
                "required": True,
                "min_count": 1,
            },
            "act_policy_checkpoint": {
                "required": True,
                "min_count": 1,
            },
        },
        artifacts=tuple(
            artifact for artifact in base.artifacts if artifact.role != "rl_onnx_model"
        ),
    )

    assert config.policy_ids["trajectory_controller"] == (
        "cartesian_p:deterministic_v1"
    )
    assert {artifact.role for artifact in config.artifacts} == {
        "act_deployment_manifest",
        "act_policy_checkpoint",
    }


def test_cartesian_p_evidence_forbids_an_optional_rl_model_artifact():
    base = _factory(_RecordingRun())._config

    with pytest.raises(ValueError, match="roles do not match.*rl_onnx_model"):
        replace(
            base,
            policy_ids={
                **base.policy_ids,
                "trajectory_controller": "cartesian_p:deterministic_v1",
            },
            evidence_requirements={
                **base.evidence_requirements,
                "rl_onnx_model": {"required": False, "min_count": 0},
            },
        )


@pytest.mark.parametrize(
    ("trajectory_controller", "required", "minimum", "message"),
    [
        ("onnx_rl:scale-v3", False, 0, "must be required"),
        ("cartesian_p:deterministic_v1", True, 1, "roles do not match"),
        ("unknown_controller:v1", False, 0, "trajectory_controller"),
    ],
)
def test_trajectory_controller_identity_must_match_rl_artifact_requirement(
    trajectory_controller,
    required,
    minimum,
    message,
):
    base = _factory(_RecordingRun())._config

    with pytest.raises(ValueError, match=message):
        replace(
            base,
            policy_ids={
                **base.policy_ids,
                "trajectory_controller": trajectory_controller,
            },
            evidence_requirements={
                **base.evidence_requirements,
                "rl_onnx_model": {
                    "required": required,
                    "min_count": minimum,
                },
            },
        )
