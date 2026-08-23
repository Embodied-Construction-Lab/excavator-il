import json
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
ACTIVE_CONFIG = PROJECT_ROOT / "config/hybrid_evidence.pc.json"


class _RecordingRun:
    run_id = "hybrid_run_scope_001"

    def __init__(self):
        self.finalizations = []

    def append_event(self, _event_type, _payload=None):
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
            "act_manifest", Path("/models/act.json"), "act_deployment_manifest"
        ),
        HybridEvidenceArtifact(
            "act_checkpoint", Path("/models/act"), "act_policy_checkpoint"
        ),
        HybridEvidenceArtifact(
            "rl_model", Path("/models/rl.onnx"), "rl_onnx_model"
        ),
    )


def _factory(
    recorder,
    *,
    scope="training_internal",
    path_is_available=lambda _p: True,
):
    return HybridExperimentRunFactory(
        HybridExperimentRunConfig(
            evidence_root=Path("/evidence"),
            machine_profile_path=Path("/machine.json"),
            repository_paths={},
            config_paths={},
            policy_ids={
                "dig_policy": "lerobot_act:step200000",
                "trajectory_controller": "onnx_rl:scale-v3",
            },
            host_topology={"pc": {}, "orin": {}},
            evaluation_scope=scope,
            artifacts=_artifacts(),
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
