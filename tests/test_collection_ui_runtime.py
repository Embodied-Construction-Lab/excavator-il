from types import SimpleNamespace

from excavator_il.collection_ui_app import CollectionUiMetadata
from excavator_il.collection_ui_config import CollectionUiConfig
from excavator_il import collection_ui_runtime
from excavator_il.collection_ui_runtime import (
    build_collection_ui_runtime,
    metadata_from_guided_config,
    run_collection_ui,
)
from excavator_il.guided_episode import GuidedEpisodeConfig


def test_collection_ui_metadata_comes_from_authoritative_guided_config():
    guided = object.__new__(GuidedEpisodeConfig)
    object.__setattr__(guided, "operator_id", "zhaoshuai")
    object.__setattr__(guided, "task", "ExecuteDig")
    object.__setattr__(guided, "dig_target_m", (0.8, 0.0, -0.2))
    object.__setattr__(guided, "orin_ssh_host", "jetson16@192.168.50.2")

    assert metadata_from_guided_config(guided) == CollectionUiMetadata(
        operator_id="zhaoshuai",
        task="ExecuteDig",
        dig_target_m=(0.8, 0.0, -0.2),
        orin_host="192.168.50.2",
    )


def test_collection_ui_runtime_composes_config_supervisor_and_app(monkeypatch, tmp_path):
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
