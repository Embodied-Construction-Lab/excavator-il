"""Composition root for the local guided-collection Web UI."""

from __future__ import annotations

import threading
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .airy_operator import AiryOperatorSupervisor
from .collection_ui_app import (
    CollectionUiMetadata,
    HybridDigGroupMetadata,
    create_collection_ui_app,
)
from .collection_ui_config import CollectionUiConfig, load_collection_ui_config
from .collection_ui_session import GuidedCollectionSupervisor
from .guided_episode import GuidedEpisodeConfig
from .dig_point_catalog import load_dig_point_catalog
from .hybrid_experiment_run import (
    HybridExperimentRunConfig,
    HybridExperimentRunFactory,
)
from .machine_state_telemetry import MachineStateTelemetryService
from .resident_fixed_cycle_system import (
    ResidentFixedCyclePcConfig,
    ResidentFixedCycleSupervisor,
    SshResidentFixedCycleOperations,
)
from .resident_fixed_cycle_visualization import v3a_trajectory_path


@dataclass(frozen=True)
class CollectionUiRuntime:
    config: CollectionUiConfig
    app: object


def metadata_from_guided_config(
    config: GuidedEpisodeConfig,
    *,
    rl_dig_targets: tuple[tuple[str, tuple[float, float, float]], ...] = (),
) -> CollectionUiMetadata:
    _user, orin_host = config.orin_ssh_host.split("@", maxsplit=1)
    return CollectionUiMetadata(
        operator_id=config.operator_id,
        task=config.task,
        dig_target_m=config.dig_target_m,
        orin_host=orin_host,
        rl_dig_targets=rl_dig_targets,
    )


def build_collection_ui_runtime(
    config_path: str | Path,
) -> CollectionUiRuntime:
    ui_config = load_collection_ui_config(config_path)
    guided_config = GuidedEpisodeConfig.load(ui_config.guided_config)
    metadata = metadata_from_guided_config(guided_config)
    evidence_run_factory = None
    evidence_config = None
    if ui_config.hybrid_mission_config is not None:
        raise ValueError(
            "legacy V2 hybrid Mission is no longer supported by the collection UI; "
            "configure resident_fixed_cycle_config with the authoritative "
            "dig-point catalog"
        )
    if (
        ui_config.hybrid_evidence_config is not None
        and ui_config.hybrid_mission_config is None
        and ui_config.resident_fixed_cycle_config is None
    ):
        raise ValueError(
            "hybrid_evidence_config requires a hybrid runtime config"
        )
    if ui_config.hybrid_evidence_config is not None:
        evidence_config = HybridExperimentRunConfig.load(
            ui_config.hybrid_evidence_config
        )
    supervisor = GuidedCollectionSupervisor(config_path=ui_config.guided_config)
    telemetry_source = MachineStateTelemetryService(guided_config=guided_config)
    hybrid_supervisor = None
    operator_supervisor = None
    if ui_config.resident_fixed_cycle_config is not None:
        fixed_config = ResidentFixedCyclePcConfig.load(
            ui_config.resident_fixed_cycle_config
        )
        if fixed_config.guided_config != ui_config.guided_config:
            raise ValueError(
                "collection UI and V3-A fixed cycle must reference the same guided config"
            )
        if evidence_config is not None:
            evidence_run_factory = HybridExperimentRunFactory(
                evidence_config,
                hybrid_config_loader=ResidentFixedCyclePcConfig.load,
                runtime_config_label="resident_fixed_cycle",
                runtime_backend="resident_fixed_cycle",
            )
            evidence_run_factory.preflight()
        def forward_fixed_cycle_log(message: str) -> None:
            if hybrid_supervisor is not None:
                hybrid_supervisor.append_external_log(message)

        operations = SshResidentFixedCycleOperations(
            fixed_config,
            guided_config=guided_config,
            output=forward_fixed_cycle_log,
        )
        if fixed_config.dig_point_catalog is None:
            raise ValueError(
                "resident fixed cycle requires dig_point_catalog; "
                "the legacy three-point fallback has been removed"
            )
        catalog = load_dig_point_catalog(
            Path(guided_config.rl_airy_repo) / fixed_config.dig_point_catalog
        )
        fixed_target_ids = tuple(catalog.points)
        fixed_groups = dict(catalog.groups)
        fixed_default_group = catalog.default_group_id
        group_metadata = tuple(
            HybridDigGroupMetadata(
                group_id=group_id,
                label=_dig_group_label(group_id, len(point_ids)),
                point_ids=point_ids,
            )
            for group_id, point_ids in catalog.groups.items()
        )
        hybrid_supervisor = ResidentFixedCycleSupervisor(
            operations=operations,
            dig_target_ids=fixed_target_ids,
            dig_groups=fixed_groups,
            default_dig_group_id=fixed_default_group,
            poll_interval_s=fixed_config.status_poll_s,
            config_path=ui_config.resident_fixed_cycle_config,
            evidence_run_factory=evidence_run_factory,
        )
        operator_supervisor = AiryOperatorSupervisor(
            guided_config=guided_config,
            behavior_port=None,
            profile="live_shadow",
            trajectory_path=v3a_trajectory_path(guided_config.log_dir),
            external_state_bridge=True,
        )
        metadata = replace(
            metadata,
            rl_dig_targets=tuple(catalog.points.items()),
            hybrid_act_max_steps=fixed_config.act_max_steps,
            hybrid_runtime_backend="resident_fixed_cycle",
            hybrid_dig_groups=group_metadata,
            hybrid_default_dig_group_id=fixed_default_group,
        )
    app = create_collection_ui_app(
        config=ui_config,
        metadata=metadata,
        supervisor=supervisor,
        hybrid_supervisor=hybrid_supervisor,
        operator_supervisor=operator_supervisor,
        telemetry_source=telemetry_source,
    )
    return CollectionUiRuntime(config=ui_config, app=app)


def _dig_group_label(group_id: str, point_count: int) -> str:
    names = {"all": "全部", "near": "近端", "far": "远端"}
    return f"{names.get(group_id, group_id)} {point_count} 点"


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
