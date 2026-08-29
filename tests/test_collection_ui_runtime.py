from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.collection_ui_app import CollectionUiMetadata
from excavator_il.collection_ui_config import CollectionUiConfig
from excavator_il import collection_ui_runtime
from excavator_il.collection_ui_runtime import (
    build_collection_ui_runtime,
    metadata_from_guided_config,
    run_collection_ui,
)
from excavator_il.dig_point_catalog import DigPointCatalog
from excavator_il.guided_episode import GuidedEpisodeConfig
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
    captured = {}

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
        lambda **kwargs: captured.update(kwargs) or app,
    )
    runtime = build_collection_ui_runtime(tmp_path / "ui.json")

    assert runtime.config is ui_config
    assert runtime.app is app
    assert captured["supervisor"] is supervisor
    assert "campaign_inspector" not in captured


def test_collection_ui_runtime_rejects_legacy_v2_hybrid_supervisor(
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
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)

    with pytest.raises(ValueError, match="legacy V2 hybrid Mission"):
        build_collection_ui_runtime(tmp_path / "ui.json")


def test_collection_ui_runtime_rejects_fixed_cycle_without_dig_point_catalog(
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
    object.__setattr__(fixed_config, "dig_point_catalog", None)
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    object.__setattr__(guided, "log_dir", tmp_path / "logs")
    operations = object()
    local_supervisor = object()
    captured = {}
    operation_kwargs = {}
    supervisor_kwargs = {}
    operator_kwargs = {}
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
        "SshResidentFixedCycleOperations",
        lambda *_args, **kwargs: operation_kwargs.update(kwargs) or operations,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "ResidentFixedCycleSupervisor",
        lambda **kwargs: supervisor_kwargs.update(kwargs) or local_supervisor,
    )
    operator = object()
    monkeypatch.setattr(
        collection_ui_runtime,
        "AiryOperatorSupervisor",
        lambda **kwargs: operator_kwargs.update(kwargs) or operator,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "create_collection_ui_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    with pytest.raises(ValueError, match="dig_point_catalog"):
        build_collection_ui_runtime(tmp_path / "ui.json")


def test_grouped_fixed_cycle_uses_catalog_for_ui_and_supervisor(
    monkeypatch, tmp_path
):
    guided_path = tmp_path / "guided.json"
    fixed_path = tmp_path / "fixed.json"
    catalog_relative = Path("mission/config/dig-points.json")
    ui_config = CollectionUiConfig(
        guided_config=guided_path,
        resident_fixed_cycle_config=fixed_path,
        host="127.0.0.1",
        port=8088,
        camera_preview_url="http://192.168.50.2:18092/camera/front.mjpg",
        visualization_url="",
    )
    fixed_config = object.__new__(ResidentFixedCyclePcConfig)
    object.__setattr__(fixed_config, "guided_config", guided_path)
    object.__setattr__(fixed_config, "status_poll_s", 0.1)
    object.__setattr__(fixed_config, "act_max_steps", 130)
    object.__setattr__(fixed_config, "dig_point_catalog", catalog_relative)
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (1.0, 0.0, 0.0))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    object.__setattr__(guided, "rl_airy_repo", tmp_path / "AiryLidar")
    object.__setattr__(guided, "log_dir", tmp_path / "logs")
    points = {
        "dig_near_01": (1.0, 0.4, 0.0),
        "dig_near_02": (1.0, 0.15, 0.0),
        "dig_near_03": (1.0, -0.1, 0.0),
        "dig_near_04": (1.0, -0.35, 0.0),
        "dig_far_01": (1.3, 0.4, 0.0),
        "dig_far_02": (1.3, 0.15, 0.0),
        "dig_far_03": (1.3, -0.1, 0.0),
        "dig_far_04": (1.3, -0.35, 0.0),
    }
    catalog = DigPointCatalog(
        points=points,
        groups={
            "all": tuple(points),
            "near": tuple(points)[:4],
            "far": tuple(points)[4:],
        },
        default_group_id="all",
    )
    captured = {}
    supervisor_kwargs = {}

    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(
        ResidentFixedCyclePcConfig,
        "load",
        classmethod(lambda _cls, _path: fixed_config),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "load_dig_point_catalog",
        lambda path: catalog
        if path == tmp_path / "AiryLidar" / catalog_relative
        else None,
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "GuidedCollectionSupervisor",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "SshResidentFixedCycleOperations",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "ResidentFixedCycleSupervisor",
        lambda **kwargs: supervisor_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "AiryOperatorSupervisor",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "create_collection_ui_app",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    build_collection_ui_runtime(tmp_path / "ui.json")

    assert supervisor_kwargs["dig_target_ids"] == tuple(points)
    assert supervisor_kwargs["dig_groups"]["near"] == tuple(points)[:4]
    assert supervisor_kwargs["default_dig_group_id"] == "all"
    assert captured["metadata"].rl_dig_targets == tuple(points.items())
    assert captured["metadata"].hybrid_default_dig_group_id == "all"
    assert captured["metadata"].hybrid_dig_groups[1].group_id == "near"


def test_legacy_v2_hybrid_is_rejected_before_evidence_or_supervisors(
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
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")
    object.__setattr__(guided, "rl_demo_config", None)
    monkeypatch.setattr(
        collection_ui_runtime, "load_collection_ui_config", lambda _path: ui_config
    )
    monkeypatch.setattr(GuidedEpisodeConfig, "load", lambda _path: guided)
    monkeypatch.setattr(
        HybridExperimentRunConfig,
        "load",
        classmethod(lambda _cls, _path: pytest.fail("legacy evidence loaded")),
    )
    monkeypatch.setattr(
        collection_ui_runtime,
        "GuidedCollectionSupervisor",
        lambda **_kwargs: pytest.fail("legacy supervisor constructed"),
    )

    with pytest.raises(ValueError, match="legacy V2 hybrid Mission"):
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
