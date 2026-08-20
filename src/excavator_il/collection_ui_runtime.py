"""Composition root for the local guided-collection Web UI."""

from __future__ import annotations

import threading
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path

from .collection_ui_app import CollectionUiMetadata, create_collection_ui_app
from .collection_ui_config import CollectionUiConfig, load_collection_ui_config
from .collection_ui_session import GuidedCollectionSupervisor
from .guided_episode import GuidedEpisodeConfig, load_rl_dig_targets
from .hybrid_mission import HybridMissionConfig
from .hybrid_mission_session import HybridMissionSupervisor


@dataclass(frozen=True)
class CollectionUiRuntime:
    config: CollectionUiConfig
    app: object


def metadata_from_guided_config(
    config: GuidedEpisodeConfig,
) -> CollectionUiMetadata:
    _user, orin_host = config.orin_ssh_host.split("@", maxsplit=1)
    return CollectionUiMetadata(
        operator_id=config.operator_id,
        task=config.task,
        dig_target_m=config.dig_target_m,
        orin_host=orin_host,
        rl_dig_targets=load_rl_dig_targets(config),
    )


def build_collection_ui_runtime(
    config_path: str | Path,
) -> CollectionUiRuntime:
    ui_config = load_collection_ui_config(config_path)
    guided_config = GuidedEpisodeConfig.load(ui_config.guided_config)
    supervisor = GuidedCollectionSupervisor(config_path=ui_config.guided_config)
    hybrid_supervisor = None
    metadata = metadata_from_guided_config(guided_config)
    if ui_config.hybrid_mission_config is not None:
        hybrid_config = HybridMissionConfig.load(ui_config.hybrid_mission_config)
        if hybrid_config.guided_config != ui_config.guided_config:
            raise ValueError(
                "collection UI and hybrid Mission must reference the same guided config"
            )
        hybrid_supervisor = HybridMissionSupervisor(
            config_path=ui_config.hybrid_mission_config
        )
        metadata = replace(
            metadata,
            hybrid_act_max_steps=hybrid_config.act_max_steps,
        )
    app = create_collection_ui_app(
        config=ui_config,
        metadata=metadata,
        supervisor=supervisor,
        hybrid_supervisor=hybrid_supervisor,
    )
    return CollectionUiRuntime(config=ui_config, app=app)


def run_collection_ui(
    config_path: str | Path,
    *,
    open_browser: bool = True,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "collection UI dependencies are missing; install excavator-il[ui]"
        ) from exc

    runtime = build_collection_ui_runtime(config_path)
    url = f"http://{runtime.config.host}:{runtime.config.port}/"
    print(f"本地示教采集 UI: {url}", flush=True)
    if open_browser:
        opener = threading.Timer(0.8, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    uvicorn.run(
        runtime.app,
        host=runtime.config.host,
        port=runtime.config.port,
        log_level="info",
        access_log=False,
    )
