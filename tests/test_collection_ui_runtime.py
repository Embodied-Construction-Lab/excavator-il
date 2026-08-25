from types import SimpleNamespace

import pytest

from excavator_il.collection_ui_app import CollectionUiMetadata
from excavator_il.collection_ui_config import CollectionUiConfig
from excavator_il import collection_ui_runtime
from excavator_il.collection_ui_runtime import (
    OrinCampaignInspector,
    build_collection_ui_runtime,
    metadata_from_guided_config,
    run_collection_ui,
)
from excavator_il.guided_episode import GuidedEpisodeConfig
from excavator_il.hybrid_mission import HybridMissionConfig
from excavator_il.hybrid_experiment_run import HybridExperimentRunConfig
from excavator_il.resident_fixed_cycle_system import ResidentFixedCyclePcConfig


def test_collection_ui_metadata_comes_from_authoritative_guided_config():
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)

    assert metadata_from_guided_config(guided) == CollectionUiMetadata(
        operator_id="zhaoshuai",
        task="ExecuteDig",
        dig_target_m=(0.8, 0.0, -0.2),
        orin_host="192.168.50.2",
        rl_dig_targets=(),
    )


def test_orin_campaign_inspector_reads_remote_collection_config_without_writes():
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "orin_repo", "/opt/excavator-il")
    object.__setattr__(guided, "orin_executable", "/opt/env/bin/excavator-il")
    object.__setattr__(guided, "orin_collection_config", "config/collection.orin.json")
    calls = []

    class Host:
        def run(self, command, *, accepted_returncodes=(0,)):
            calls.append((command, accepted_returncodes))
            return (
                '{"schema_version":"excavator_collection_campaign.v1",'
                '"summary":{"planned":200,"completed":0,'
                '"ignored_diagnostics":0,"complete_and_valid":false},'
                '"next_expected_slot":{"slot_id":"slot_001",'
                '"task_variant":"dig_only","soil_reset_block_id":"block_01",'
                '"dig_point_id":"dig_01"}}'
            )

    report = OrinCampaignInspector(guided, remote_host=Host())()

    assert report["next_expected_slot"]["slot_id"] == "slot_001"
    assert calls[0][1] == (0, 2)
    assert "cd /opt/excavator-il" in calls[0][0]
    assert "/opt/env/bin/python" in calls[0][0]
    assert "scripts/inspect_collection_campaign.py" in calls[0][0]
    assert "--collection-config config/collection.orin.json --next" in calls[0][0]


def test_collection_ui_runtime_composes_config_supervisor_and_app(
    monkeypatch, tmp_path
):
    ui_config = CollectionUiConfig(
        guided_config=tmp_path / "guided.json",
        host="127.0.0.1",
        port=8088,
        camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
        visualization_url="",
    )
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    supervisor = object()
    app = object()

    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(
        collection_ui_runtime,
        "GuidedCollectionSupervisor",
        lambda **_kwargs: supervisor,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "create_collection_ui_app",
        lambda **kwargs: app if kwargs["supervisor"] is supervisor else None,
    )

    runtime = build_collection_ui_runtime(tmp_path / "ui.json")

    assert runtime.config is ui_config
    assert runtime.app is app


def test_collection_ui_runtime_composes_optional_hybrid_supervisor(
    monkeypatch, tmp_path
):
    guided_path = tmp_path / "guided.json"
    hybrid_path = tmp_path / "hybrid.json"
    ui_config = CollectionUiConfig(
        guided_config=guided_path,
        hybrid_mission_config=hybrid_path,
        host="127.0.0.1",
        port=8088,
        camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
        visualization_url="",
    )
    hybrid_config = HybridMissionConfig(
        guided_config=guided_path,
        act_max_steps=130,
        act_ready_timeout_s=60,
        act_run_timeout_s=90,
        act_remote_script="scripts/run_act_motion.sh",
        rl_behavior_port=18083,
    )
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    collection = object()
    hybrid = object()
    captured = {}
    hybrid_kwargs = {}

    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(HybridMissionConfig, "load", lambda _path: hybrid_config)
    monkeypatch.setattr(
        collection_ui_runtime,
        "load_rl_dig_targets",
        lambda _config: (
            ("dig_01", (1.0, 0.2, 0.0)),
            ("dig_02", (1.0, 0.0, 0.0)),
            ("dig_03", (1.0, -0.2, 0.0)),
        ),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "GuidedCollectionSupervisor",
        lambda **_kwargs: collection,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "HybridMissionSupervisor",
        lambda **kwargs: hybrid_kwargs.update(kwargs) or hybrid,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "create_collection_ui_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    build_collection_ui_runtime(tmp_path / "ui.json")

    assert captured["hybrid_supervisor"] is hybrid
    assert captured["metadata"].hybrid_act_max_steps == 130
    assert hybrid_kwargs["dig_target_ids"] == ("dig_01", "dig_02", "dig_03")


def test_collection_ui_runtime_selects_v3a_local_cycle_without_pc_operator(
    monkeypatch, tmp_path
):
    guided_path = tmp_path / "guided.json"
    fixed_path = tmp_path / "fixed-cycle.json"
    evidence_path = tmp_path / "hybrid-evidence.json"
    ui_config = CollectionUiConfig(
        guided_config=guided_path,
        resident_fixed_cycle_config=fixed_path,
        hybrid_evidence_config=evidence_path,
        host="127.0.0.1",
        port=8088,
        camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
        visualization_url="",
    )
    fixed_config = object.__new__(ResidentFixedCyclePcConfig)
    object.__setattr__(fixed_config, "guided_config", guided_path)
    object.__setattr__(fixed_config, "status_poll_s", 0.1)
    object.__setattr__(fixed_config, "act_max_steps", 130)
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    operations = object()
    local_supervisor = object()
    captured = {}
    operation_kwargs = {}
    supervisor_kwargs = {}
    evidence_config = object()

    class EvidenceFactory:
        def __init__(self, config, **kwargs):
            assert config is evidence_config
            self.kwargs = kwargs
            self.preflighted = False

        def preflight(self):
            self.preflighted = True

    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(
        ResidentFixedCyclePcConfig,
        "load",
        classmethod(lambda _cls, path: fixed_config if path == fixed_path else None),
    )
    monkeypatch.setattr(
        HybridExperimentRunConfig,
        "load",
        classmethod(lambda _cls, path: evidence_config if path == evidence_path else None),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "HybridExperimentRunFactory",
        EvidenceFactory,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "load_rl_dig_targets",
        lambda _config: (
            ("dig_01", (1.0, 0.26, 0.0)),
            ("dig_02", (1.0, 0.0, 0.0)),
            ("dig_03", (1.0, -0.26, 0.0)),
        ),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "SshResidentFixedCycleOperations",
        lambda *_args, **kwargs: operation_kwargs.update(kwargs) or operations,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "ResidentFixedCycleSupervisor",
        lambda **kwargs: supervisor_kwargs.update(kwargs) or local_supervisor,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "create_collection_ui_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    build_collection_ui_runtime(tmp_path / "ui.json")

    assert captured["hybrid_supervisor"] is local_supervisor
    assert captured["operator_supervisor"] is None
    assert captured["metadata"].hybrid_runtime_backend == "resident_fixed_cycle"
    assert captured["metadata"].hybrid_act_max_steps == 130
    assert callable(operation_kwargs["output"])
    factory = supervisor_kwargs["evidence_run_factory"]
    assert factory.preflighted is True
    assert factory.kwargs["runtime_config_label"] == "resident_fixed_cycle"
    assert factory.kwargs["runtime_backend"] == "resident_fixed_cycle"
    assert supervisor_kwargs["config_path"] == fixed_path


def test_runtime_preflights_and_injects_hybrid_evidence_before_supervisors(
    monkeypatch,
    tmp_path,
):
    guided_path = tmp_path / "guided.json"
    hybrid_path = tmp_path / "hybrid.json"
    evidence_path = tmp_path / "hybrid-evidence.json"
    ui_config = CollectionUiConfig(
        guided_config=guided_path,
        hybrid_mission_config=hybrid_path,
        hybrid_evidence_config=evidence_path,
        host="127.0.0.1",
        port=8088,
        camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
        visualization_url="",
    )
    hybrid_config = HybridMissionConfig(
        guided_config=guided_path,
        act_max_steps=130,
        act_ready_timeout_s=60,
        act_run_timeout_s=90,
        act_remote_script="scripts/run_act_motion.sh",
        rl_behavior_port=18083,
    )
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    evidence_config = object.__new__(HybridExperimentRunConfig)
    order = []
    hybrid_kwargs = {}

    class Factory:
        def __init__(self, config):
            assert config is evidence_config
            order.append("factory")

        def preflight(self):
            order.append("preflight")

    factory_type = Factory
    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(HybridMissionConfig, "load", lambda _path: hybrid_config)
    monkeypatch.setattr(
        HybridExperimentRunConfig,
        "load",
        classmethod(
            lambda _cls, path: evidence_config if path == evidence_path else None
        ),
    )
    monkeypatch.setattr(
        collection_ui_runtime, "HybridExperimentRunFactory", factory_type
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "GuidedCollectionSupervisor",
        lambda **_kwargs: order.append("collection") or object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "HybridMissionSupervisor",
        lambda **kwargs: hybrid_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "AiryOperatorSupervisor",
        lambda **_kwargs: order.append("operator") or object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "create_collection_ui_app",
        lambda **_kwargs: object(),
    )

    build_collection_ui_runtime(tmp_path / "ui.json")

    assert order[:2] == ["factory", "preflight"]
    assert order.index("preflight") < order.index("collection")
    assert order.index("preflight") < order.index("operator")
    assert isinstance(hybrid_kwargs["evidence_run_factory"], Factory)


def test_collection_ui_runtime_fails_before_supervisor_when_evidence_path_is_missing(
    monkeypatch, tmp_path
):
    guided_path = tmp_path / "guided.json"
    hybrid_path = tmp_path / "hybrid.json"
    evidence_path = tmp_path / "hybrid-evidence.json"
    ui_config = CollectionUiConfig(
        guided_config=guided_path,
        hybrid_mission_config=hybrid_path,
        hybrid_evidence_config=evidence_path,
        host="127.0.0.1",
        port=8088,
        camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
        visualization_url="",
    )
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    hybrid_config = SimpleNamespace(guided_config=guided_path)
    evidence_config = object.__new__(HybridExperimentRunConfig)

    class BrokenFactory:
        def __init__(self, _config):
            pass

        def preflight(self):
            raise ValueError("artifact rl_onnx_model does not exist")

    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(HybridMissionConfig, "load", lambda _path: hybrid_config)
    monkeypatch.setattr(
        HybridExperimentRunConfig,
        "load",
        classmethod(lambda _cls, _path: evidence_config),
    )
    monkeypatch.setattr(
        collection_ui_runtime, "HybridExperimentRunFactory", BrokenFactory
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "GuidedCollectionSupervisor",
        lambda **_kwargs: pytest.fail("supervisor constructed before preflight"),
    )

    with pytest.raises(ValueError, match="rl_onnx_model does not exist"):
        build_collection_ui_runtime(tmp_path / "ui.json")


def test_run_collection_ui_uses_loopback_config_without_browser(monkeypatch):
    app = object()
    runtime = SimpleNamespace(
        config=SimpleNamespace(host="127.0.0.1", port=8088), app=app
    )
    calls = []
    monkeypatch.setattr(
        collection_ui_runtime,
        "build_collection_ui_runtime",
        lambda _path: runtime,
    )
    monkeypatch.setattr(
        "uvicorn.run",
        lambda passed_app, **kwargs: calls.append((passed_app, kwargs)),
    )

    run_collection_ui("ui.json", open_browser=False)

    assert calls == [
        (
            app,
            {
                "host": "127.0.0.1",
                "port": 8088,
                "log_level": "info",
                "access_log": False,
            },
        )
    ]
