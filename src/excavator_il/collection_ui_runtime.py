"""Composition root for the local guided-collection Web UI."""

from __future__ import annotations

import json
import shlex
import threading
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .airy_operator import AiryOperatorSupervisor
from .collection_ui_app import CollectionUiMetadata, create_collection_ui_app
from .collection_ui_config import CollectionUiConfig, load_collection_ui_config
from .collection_ui_session import GuidedCollectionSupervisor
from .guided_episode import GuidedEpisodeConfig, load_rl_dig_targets
from .hybrid_experiment_run import (
    HybridExperimentRunConfig,
    HybridExperimentRunFactory,
)
from .hybrid_mission import HybridMissionConfig
from .hybrid_mission_session import HybridMissionSupervisor
from .remote_runtime import SshRuntimeHost
from .resident_fixed_cycle_system import (
    ResidentFixedCyclePcConfig,
    ResidentFixedCycleSupervisor,
    SshResidentFixedCycleOperations,
)


@dataclass(frozen=True)
class CollectionUiRuntime:
    config: CollectionUiConfig
    app: object


class OrinCampaignInspector:
    """Read-only adapter for the campaign state beside the Orin raw Episodes."""

    def __init__(
        self,
        config: GuidedEpisodeConfig,
        *,
        remote_host: SshRuntimeHost | None = None,
    ) -> None:
        self._config = config
        self._remote_host = remote_host or SshRuntimeHost(config.orin_ssh_host)

    def __call__(self) -> Mapping[str, Any]:
        repo = PurePosixPath(self._config.orin_repo)
        executable = PurePosixPath(self._config.orin_executable)
        python = executable.parent / "python"
        argv = [
            "env",
            "PYTHONPATH=src",
            str(python),
            "scripts/inspect_collection_campaign.py",
            "--collection-config",
            str(self._config.orin_collection_config),
            "--next",
        ]
        command = f"cd {shlex.quote(str(repo))} && {shlex.join(argv)}"
        output = self._remote_host.run(command, accepted_returncodes=(0, 2))
        try:
            report = json.loads(output)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"remote campaign inspector returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(report, dict):
            raise RuntimeError("remote campaign inspector must return an object")
        return report


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
    metadata = metadata_from_guided_config(guided_config)
    hybrid_config = None
    evidence_run_factory = None
    evidence_config = None
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
    if ui_config.hybrid_mission_config is not None:
        hybrid_config = HybridMissionConfig.load(ui_config.hybrid_mission_config)
        if hybrid_config.guided_config != ui_config.guided_config:
            raise ValueError(
                "collection UI and hybrid Mission must reference the same guided config"
            )
        if evidence_config is not None:
            evidence_run_factory = HybridExperimentRunFactory(evidence_config)
            evidence_run_factory.preflight()

    supervisor = GuidedCollectionSupervisor(config_path=ui_config.guided_config)
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
        hybrid_supervisor = ResidentFixedCycleSupervisor(
            operations=operations,
            dig_target_ids=tuple(target_id for target_id, _ in metadata.rl_dig_targets),
            poll_interval_s=fixed_config.status_poll_s,
            config_path=ui_config.resident_fixed_cycle_config,
            evidence_run_factory=evidence_run_factory,
        )
        metadata = replace(
            metadata,
            hybrid_act_max_steps=fixed_config.act_max_steps,
            hybrid_runtime_backend="resident_fixed_cycle",
        )
    if hybrid_config is not None:
        hybrid_supervisor = HybridMissionSupervisor(
            config_path=ui_config.hybrid_mission_config,
            dig_target_ids=tuple(target_id for target_id, _ in metadata.rl_dig_targets),
            evidence_run_factory=evidence_run_factory,
        )
        operator_supervisor = AiryOperatorSupervisor(
            guided_config=guided_config,
            behavior_port=hybrid_config.rl_behavior_port,
        )
        metadata = replace(
            metadata,
            hybrid_act_max_steps=hybrid_config.act_max_steps,
            hybrid_runtime_backend="resident_v2",
        )
    app = create_collection_ui_app(
        config=ui_config,
        metadata=metadata,
        supervisor=supervisor,
        hybrid_supervisor=hybrid_supervisor,
        operator_supervisor=operator_supervisor,
        campaign_inspector=OrinCampaignInspector(guided_config),
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
