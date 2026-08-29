import json
import io
import signal
import subprocess
import tomllib
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.collection_campaign import inspect_collection_campaign
from excavator_il.guided_episode import (
    GuidedEpisodeConfig,
    GuidedEpisodeStage,
    PositioningMode,
    SystemGuidedEpisodeOperations,
    _read_positioning_choice,
    load_rl_dig_targets,
    main as guided_main,
    run_guided_episode,
    run_standalone_teleop,
)
from excavator_il.remote_runtime import LineProcess


class _FakeOperations:
    def __init__(self):
        self.events = []
        self.episode_index = 0
        self.episode_targets = []
        self.episode_purposes = []
        self.target_source_requests = []

    def preflight(self):
        self.events.append("preflight")

    def start_collector(self):
        self.events.append("start_collector")

    def start_rl_runtime(self):
        self.events.append("start_rl_runtime")

    def run_rl_preposition(self, target_id=None):
        self.events.append("run_rl_preposition")
        return (1.0, 0.0, 0.0)

    def capture_target_source_provenance(self, point_id, expected_target_m):
        self.events.append("capture_target_source_provenance")
        self.target_source_requests.append((point_id, expected_target_m))
        return {
            "repository": "airylidar",
            "path": "mission/config/excavation_demo.json",
            "sha256": "a" * 64,
            "commit": "b" * 40,
            "dirty": False,
        }

    def stop_rl_runtime_and_wait_for_serial(self):
        self.events.append("stop_rl_runtime_and_wait_for_serial")

    def start_teleop(self):
        self.events.append("start_teleop")

    def wait_for_ack(self, timeout_s):
        self.events.append(("wait_for_ack", timeout_s))

    def wait_for_deadman_pressed(self):
        self.events.append("wait_for_deadman_pressed")

    def wait_for_deadman_released(self):
        self.events.append("wait_for_deadman_released")

    def start_episode(
        self,
        dig_target_m=None,
        *,
        recording_purpose="demonstration",
        **protocol,
    ):
        self.events.append("start_episode")
        self.episode_targets.append(dig_target_m)
        self.episode_purposes.append(recording_purpose)
        target_source = protocol.pop("target_source_provenance", None)
        if target_source is not None:
            self.events.append(("target_source_provenance", target_source))
        labels = {
            name: protocol.pop(name)
            for name in (
                "collection_zone_id",
                "dig_repeat_index",
                "operator_note",
            )
            if name in protocol
        }
        if labels:
            self.events.append(("collection_labels", labels))
        if protocol:
            self.events.append(("collection_protocol", protocol))
        self.episode_index += 1
        return f"/data/raw/episode_{self.episode_index:04d}"

    def seal_episode(self):
        self.events.append("seal_episode")
        return f"/data/raw/episode_{self.episode_index:04d}"

    def finalize_episode(self, episode_path, result, reason=""):
        self.events.append(("finalize_episode", episode_path, result, reason))
        return episode_path

    def abort_episode(self, reason):
        self.events.append(("abort_episode", reason))
        return f"/data/raw/episode_{self.episode_index:04d}"

    def discard_episode(self, episode_path):
        self.events.append(("discard_episode", episode_path))
        self.episode_index -= 1

    def stop_teleop(self):
        self.events.append("stop_teleop")

    def stop_collector(self):
        self.events.append("stop_collector")

    def build_and_validate(self, episode_path):
        self.events.append(("build_and_validate", episode_path))


@pytest.mark.parametrize(
    "mode",
    (PositioningMode.RL, PositioningMode.MANUAL, PositioningMode.DIRECT),
)
def test_formal_collection_validates_and_propagates_live_target_source_before_preflight(
    tmp_path, mode
):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.0, 0.0]}
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = _FakeOperations()
    answers = iter(("c", "s") if mode is PositioningMode.MANUAL else ("s",))

    run_guided_episode(
        config,
        operations,
        positioning_mode=mode,
        input_fn=lambda _prompt: next(answers),
        output=lambda _message: None,
        task_variant="dig_only",
        soil_reset_block_id="block_01",
        dig_point_id="dig_01",
        collection_zone_id="zone_04",
        dig_repeat_index=2,
        operator_note="近排左侧第二次",
    )

    assert operations.events.index("capture_target_source_provenance") < (
        operations.events.index("preflight")
    )
    assert operations.target_source_requests == [
        ("dig_01", (1.0, 0.0, 0.0)),
        ("dig_01", (1.0, 0.0, 0.0)),
    ]
    capture_indices = [
        index
        for index, event in enumerate(operations.events)
        if event == "capture_target_source_provenance"
    ]
    gate_indices = [
        index
        for index, event in enumerate(operations.events)
        if isinstance(event, tuple)
        and event[0] == "require_expected_campaign_slot"
    ]
    assert gate_indices == []
    assert capture_indices[-1] < operations.events.index("start_episode")
    assert operations.episode_purposes == ["demonstration"]
    labels_event = next(
        event
        for event in operations.events
        if isinstance(event, tuple) and event[0] == "collection_labels"
    )
    assert labels_event[1] == {
        "collection_zone_id": "zone_04",
        "dig_repeat_index": 2,
        "operator_note": "近排左侧第二次",
    }
    provenance_event = next(
        event
        for event in operations.events
        if isinstance(event, tuple) and event[0] == "target_source_provenance"
    )
    assert provenance_event[1]["dirty"] is False


def test_formal_collection_rejects_target_source_drift_before_episode_creation(
    tmp_path,
):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.0, 0.0]}
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)

    class DriftingSourceOperations(_FakeOperations):
        def capture_target_source_provenance(self, point_id, expected_target_m):
            value = super().capture_target_source_provenance(
                point_id, expected_target_m
            )
            if len(self.target_source_requests) == 2:
                return {**value, "commit": "c" * 40}
            return value

    operations = DriftingSourceOperations()

    with pytest.raises(ValueError, match="changed before Episode creation"):
        run_guided_episode(
            config,
            operations,
            positioning_mode=PositioningMode.DIRECT,
            input_fn=lambda _prompt: "s",
            output=lambda _message: None,
            task_variant="dig_only",
            soil_reset_block_id="block_01",
            dig_point_id="dig_01",
        )

    assert "start_episode" not in operations.events


def test_single_episode_collection_does_not_consult_campaign_slot(
    tmp_path,
):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.0, 0.0]}
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)

    class OfflineCampaignOperations(_FakeOperations):
        def require_expected_campaign_slot(self, **protocol):
            self.events.append(("require_expected_campaign_slot", protocol))
            raise AssertionError("single Episode collection must not inspect campaign")

    operations = OfflineCampaignOperations()

    run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.DIRECT,
        input_fn=lambda _prompt: "s",
        output=lambda _message: None,
        task_variant="dig_only",
        soil_reset_block_id="block_07",
        dig_point_id="dig_01",
    )

    assert not any(
        isinstance(event, tuple)
        and event[0] == "require_expected_campaign_slot"
        for event in operations.events
    )
    assert "start_episode" in operations.events


def test_guided_episode_passes_collection_protocol_to_every_recording_attempt(tmp_path):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_03", "position_m": [1.0, -0.2, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = _FakeOperations()
    answers = iter(("r", "s"))

    completed = run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.DIRECT,
        input_fn=lambda _prompt: next(answers),
        task_variant="dig_transport_dump",
        soil_reset_block_id="block_06",
        dig_point_id="dig_03",
    )

    assert completed == "/data/raw/episode_0001"
    assert operations.episode_purposes == ["demonstration", "demonstration"]
    protocol_events = [
        event
        for event in operations.events
        if isinstance(event, tuple) and event[0] == "collection_protocol"
    ]
    assert protocol_events == [
        (
            "collection_protocol",
            {
                "task_variant": "dig_transport_dump",
                "soil_reset_block_id": "block_06",
                "dig_point_id": "dig_03",
            },
        ),
        (
            "collection_protocol",
            {
                "task_variant": "dig_transport_dump",
                "soil_reset_block_id": "block_06",
                "dig_point_id": "dig_03",
            },
        ),
    ]
    gate_indices = [
        index
        for index, event in enumerate(operations.events)
        if isinstance(event, tuple)
        and event[0] == "require_expected_campaign_slot"
    ]
    start_indices = [
        index
        for index, event in enumerate(operations.events)
        if event == "start_episode"
    ]
    assert gate_indices == []
    assert len(start_indices) == 2


def test_direct_collection_persists_selected_dig_point_coordinates(tmp_path):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "demo_id": "field_demo_001",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.2, 0.0]},
                    {"point_id": "dig_02", "position_m": [1.0, 0.0, 0.0]},
                    {"point_id": "dig_03", "position_m": [1.0, -0.2, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = _FakeOperations()

    run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.DIRECT,
        input_fn=lambda _prompt: "s",
        output=lambda _message: None,
        task_variant="dig_only",
        soil_reset_block_id="block_01",
        dig_point_id="dig_03",
    )

    assert operations.episode_targets == [(1.0, -0.2, 0.0)]


def test_manual_collection_persists_selected_dig_point_coordinates(tmp_path):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_02", "position_m": [1.1, 0.0, 0.05]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = _FakeOperations()
    answers = iter(("c", "s"))

    run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.MANUAL,
        input_fn=lambda _prompt: next(answers),
        output=lambda _message: None,
        task_variant="dig_transport_dump",
        soil_reset_block_id="block_02",
        dig_point_id="dig_02",
    )

    assert operations.episode_targets == [(1.1, 0.0, 0.05)]


def test_protocol_free_cli_collection_keeps_configured_legacy_target(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()

    run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.DIRECT,
        input_fn=lambda _prompt: "s",
        output=lambda _message: None,
    )

    assert operations.episode_targets == [config.dig_target_m]
    assert operations.episode_purposes == ["demonstration"]
    assert not any(
        isinstance(event, tuple)
        and event[0] == "require_expected_campaign_slot"
        for event in operations.events
    )


def test_non_formal_guided_terminal_episode_is_diagnostic_and_ignored_by_campaign(
    tmp_path,
    monkeypatch,
):
    raw_root = tmp_path / "raw"
    episode_path = raw_root / "episode_0001"

    class DiagnosticEpisodeOperations(_FakeOperations):
        @property
        def log_paths(self):
            return (
                tmp_path / "collector.log",
                tmp_path / "teleop.log",
                tmp_path / "validation.log",
            )

        def start_episode(
            self,
            dig_target_m=None,
            *,
            recording_purpose="demonstration",
            **protocol,
        ):
            episode_path.mkdir(parents=True)
            metadata = {
                "schema_version": "excavator_demo_raw.v2",
                "episode_id": episode_path.name,
                "recording_purpose": recording_purpose,
                "status": "recording",
                "success": None,
            }
            if protocol:
                metadata = {**metadata, "collection_protocol": protocol}
            (episode_path / "episode.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            return str(episode_path)

        def seal_episode(self):
            metadata_path = episode_path / "episode.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_path.write_text(
                json.dumps({**metadata, "status": "pending_review"}),
                encoding="utf-8",
            )
            return str(episode_path)

        def finalize_episode(self, completed_path, result, reason=""):
            assert completed_path == str(episode_path)
            metadata_path = episode_path / "episode.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_path.write_text(
                json.dumps(
                    {
                        **metadata,
                        "status": "complete" if result == "success" else "failed",
                        "success": result == "success",
                    }
                ),
                encoding="utf-8",
            )
            return completed_path

    config = _guided_config(tmp_path)
    operations = DiagnosticEpisodeOperations()
    original_run = run_guided_episode

    def run_from_terminal(config_arg, operations_arg, **kwargs):
        return original_run(
            config_arg,
            operations_arg,
            input_fn=lambda _prompt: "s",
            output=lambda _message: None,
            **kwargs,
        )

    monkeypatch.setattr(
        "excavator_il.guided_episode.GuidedEpisodeConfig.load",
        lambda _path: config,
    )
    monkeypatch.setattr(
        "excavator_il.guided_episode.SystemGuidedEpisodeOperations",
        lambda _config: operations,
    )
    monkeypatch.setattr(
        "excavator_il.guided_episode._read_positioning_choice",
        lambda _input, _output: PositioningMode.DIRECT,
    )
    monkeypatch.setattr(
        "excavator_il.guided_episode.run_guided_episode",
        run_from_terminal,
    )

    assert guided_main(["--config", str(tmp_path / "guided.json")]) == 0

    metadata = json.loads(
        (episode_path / "episode.json").read_text(encoding="utf-8")
    )
    report = inspect_collection_campaign(raw_root)
    assert metadata["recording_purpose"] == "diagnostic"
    assert "collection_protocol" not in metadata
    assert report["summary"]["ignored_diagnostics"] == 1
    assert report["summary"]["malformed"] == 0
    assert report["next_expected_slot"]["slot_id"] == "slot_001"


def test_standalone_teleop_never_creates_episode_and_cleans_up_on_interrupt(tmp_path):
    operations = _FakeOperations()
    stages = []

    def wait_until_stopped():
        operations.events.append("operator_wait")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_standalone_teleop(
            _guided_config(tmp_path),
            operations,
            wait_fn=wait_until_stopped,
            stage_callback=stages.append,
        )

    assert operations.events == [
        "preflight",
        "start_collector",
        "start_teleop",
        ("wait_for_ack", 8),
        "operator_wait",
        "stop_teleop",
        "stop_collector",
    ]
    assert stages == [
        GuidedEpisodeStage.PREFLIGHT,
        GuidedEpisodeStage.COLLECTOR_STARTING,
        GuidedEpisodeStage.TELEOPERATION,
    ]


def test_guided_episode_config_resolves_pc_paths_and_validates_contract(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "guided.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_guided_episode_config.v3",
                "teleop_config": "teleop.pc.json",
                "orin": {
                    "ssh_host": "operator@192.0.2.10",
                    "repo": "/srv/excavator-il",
                    "executable": "/opt/excavator/bin/excavator-il",
                    "collection_config": "config/collection.orin.json",
                },
                "rl_preposition": {
                    "airy_repo": "../../AiryLidar",
                    "ros_setup": "/opt/ros/jazzy/setup.zsh",
                    "workspace_setup": "ros2_ws/install/setup.zsh",
                    "mission_config": "mission/config/excavation_cycle.json",
                    "phase": "dig",
                    "timeout_s": 90,
                    "serial_port": "/dev/ttyTHS1",
                    "serial_release_timeout_s": 8,
                    "orin_repo": "/srv/excavator-orin-runtime",
                    "orin_python": "/opt/excavator-orin/bin/python",
                    "edge_config": "deploy/edge_runtime.remote.json",
                    "pc_host": "192.0.2.20",
                    "ready_timeout_s": 15,
                },
                "episode": {
                    "task": "DiagnosticBoomJog",
                    "operator_id": "operator_01",
                    "dig_target_m": [0.8, 0.0, -0.2],
                    "material_id": "soil_default",
                },
                "runtime": {
                    "collector_ready_timeout_s": 8,
                    "ack_timeout_s": 8,
                    "teleop_print_every": 1,
                    "zero_soak_duration_s": 30,
                    "log_dir": "../logs",
                },
            }
        ),
        encoding="utf-8",
    )

    config = GuidedEpisodeConfig.load(config_path)

    assert config.teleop_config == config_dir / "teleop.pc.json"
    assert config.log_dir == tmp_path / "logs"
    assert config.orin_ssh_host == "operator@192.0.2.10"
    assert config.dig_target_m == (0.8, 0.0, -0.2)
    assert config.teleop_print_every == 1
    assert config.zero_soak_duration_s == 30
    assert config.rl_airy_repo == tmp_path.parent / "AiryLidar"
    assert config.rl_mission_config == tmp_path.parent / "AiryLidar/mission/config/excavation_cycle.json"
    assert config.rl_phase == "dig"
    assert str(config.rl_serial_port) == "/dev/ttyTHS1"
    assert config.orin_experiment_run_config is None

    configured_root = json.loads(config_path.read_text(encoding="utf-8"))
    configured_root["orin"]["experiment_run_config"] = (
        "config/collection_evidence.orin.json"
    )
    config_path.write_text(json.dumps(configured_root), encoding="utf-8")
    configured = GuidedEpisodeConfig.load(config_path)
    assert str(configured.orin_experiment_run_config) == (
        "config/collection_evidence.orin.json"
    )


def test_guided_episode_config_rejects_unsafe_or_inconsistent_values(tmp_path):
    config_path = tmp_path / "guided.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_guided_episode_config.v3",
                "teleop_config": "teleop.pc.json",
                "orin": {
                    "ssh_host": "operator@host; reboot",
                    "repo": "/srv/excavator-il",
                    "executable": "/opt/excavator/bin/excavator-il",
                    "collection_config": "config/collection.orin.json",
                },
                "rl_preposition": {
                    "airy_repo": "../AiryLidar",
                    "ros_setup": "/opt/ros/jazzy/setup.zsh",
                    "workspace_setup": "ros2_ws/install/setup.zsh",
                    "mission_config": "mission/config/excavation_cycle.json",
                    "phase": "dig",
                    "timeout_s": 90,
                    "serial_port": "/dev/ttyTHS1",
                    "serial_release_timeout_s": 8,
                    "orin_repo": "/srv/excavator-orin-runtime",
                    "orin_python": "/opt/excavator-orin/bin/python",
                    "edge_config": "deploy/edge_runtime.remote.json",
                    "pc_host": "192.0.2.20",
                    "ready_timeout_s": 15,
                },
                "episode": {
                    "task": "DiagnosticBoomJog",
                    "operator_id": "operator_01",
                    "dig_target_m": [0.8, 0.0, -0.2],
                    "material_id": "soil_default",
                },
                "runtime": {
                    "collector_ready_timeout_s": 8,
                    "ack_timeout_s": 8,
                    "teleop_print_every": 1,
                    "log_dir": "../logs",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        GuidedEpisodeConfig.load(config_path)


def _guided_config(tmp_path):
    return GuidedEpisodeConfig(
        teleop_config=tmp_path / "teleop.json",
        orin_ssh_host="operator@192.0.2.10",
        orin_repo="/srv/excavator-il",
        orin_executable="/opt/excavator/bin/excavator-il",
        orin_collection_config="config/collection.orin.json",
        task="ExecuteDig",
        operator_id="operator_01",
        dig_target_m=(0.8, 0.0, -0.2),
        material_id="soil_default",
        collector_ready_timeout_s=8,
        ack_timeout_s=8,
        teleop_print_every=1,
        log_dir=tmp_path / "logs",
        failure_reason="diagnostic_task_failed",
        rl_airy_repo=tmp_path / "AiryLidar",
        rl_ros_setup=Path("/opt/ros/jazzy/setup.zsh"),
        rl_workspace_setup=tmp_path / "AiryLidar/ros2_ws/install/setup.zsh",
        rl_mission_config=tmp_path / "AiryLidar/mission/config/excavation_cycle.json",
        rl_phase="dig",
        rl_timeout_s=90,
        rl_serial_port="/dev/ttyTHS1",
        rl_serial_release_timeout_s=8,
        rl_orin_repo="/srv/excavator-orin-runtime",
        rl_orin_python="/opt/excavator-orin/bin/python",
        rl_edge_config="deploy/edge_runtime.remote.json",
        rl_pc_host="192.0.2.20",
        rl_ready_timeout_s=15,
    )


def _prepare_clean_airy_repo(tmp_path):
    repo = tmp_path / "AiryLidar"
    demo_path = repo / "mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.2, 0.0]},
                    {"point_id": "dig_02", "position_m": [1.0, 0.0, 0.0]},
                    {"point_id": "dig_03", "position_m": [1.0, -0.2, 0.0]},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for command in (
        ("git", "init", "-b", "main"),
        ("git", "config", "user.name", "Fixture"),
        ("git", "config", "user.email", "fixture@example.invalid"),
        ("git", "add", "mission/config/excavation_demo.json"),
        ("git", "commit", "-m", "fixture"),
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    return repo, demo_path


def test_system_operations_captures_actual_clean_airy_target_source(tmp_path):
    repo, demo_path = _prepare_clean_airy_repo(tmp_path)
    config = _guided_config(tmp_path)
    object.__setattr__(config, "rl_airy_repo", repo)
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = SystemGuidedEpisodeOperations(config, output=lambda _message: None)

    provenance = operations.capture_target_source_provenance(
        "dig_02", (1.0, 0.0, 0.0)
    )

    assert dict(provenance) == {
        "repository": "airylidar",
        "path": "mission/config/excavation_demo.json",
        "sha256": hashlib.sha256(demo_path.read_bytes()).hexdigest(),
        "commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dirty": False,
    }


def test_system_operations_allows_unrelated_dirty_airy_files(tmp_path):
    repo, demo_path = _prepare_clean_airy_repo(tmp_path)
    (repo / "untracked.txt").write_text("drift", encoding="utf-8")
    config = _guided_config(tmp_path)
    object.__setattr__(config, "rl_airy_repo", repo)
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = SystemGuidedEpisodeOperations(config, output=lambda _message: None)

    provenance = operations.capture_target_source_provenance(
        "dig_02", (1.0, 0.0, 0.0)
    )

    assert provenance["path"] == "mission/config/excavation_demo.json"
    assert provenance["dirty"] is False


def test_system_operations_rejects_modified_airy_target_config(tmp_path):
    repo, demo_path = _prepare_clean_airy_repo(tmp_path)
    payload = json.loads(demo_path.read_text(encoding="utf-8"))
    payload["note"] = "uncommitted target change"
    demo_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _guided_config(tmp_path)
    object.__setattr__(config, "rl_airy_repo", repo)
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = SystemGuidedEpisodeOperations(config, output=lambda _message: None)

    with pytest.raises(ValueError, match="target config to match HEAD"):
        operations.capture_target_source_provenance(
            "dig_02", (1.0, 0.0, 0.0)
        )


def test_system_operations_rejects_missing_airy_source(tmp_path):
    repo, demo_path = _prepare_clean_airy_repo(tmp_path)
    demo_path.unlink()
    config = _guided_config(tmp_path)
    object.__setattr__(config, "rl_airy_repo", repo)
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = SystemGuidedEpisodeOperations(config, output=lambda _message: None)

    with pytest.raises(ValueError, match="must be a file inside"):
        operations.capture_target_source_provenance(
            "dig_02", (1.0, 0.0, 0.0)
        )


def test_preflight_reclaims_only_known_stale_serial_owner(tmp_path):
    config = _guided_config(tmp_path)
    config.teleop_config.write_text("{}", encoding="utf-8")
    calls = []
    messages = []

    class FakeRemoteHost:
        def run(self, command, *, accepted_returncodes=(0,)):
            calls.append(("run", command, accepted_returncodes))
            return ""

        def reclaim_serial_owner(self, **kwargs):
            calls.append(("reclaim", kwargs))
            return "reclaimed"

    operations = SystemGuidedEpisodeOperations(config, output=messages.append)
    operations._remote_host = FakeRemoteHost()

    operations.preflight()

    reclaim = next(call for call in calls if call[0] == "reclaim")[1]
    assert reclaim["serial_path"] == "/dev/ttyTHS1"
    assert (
        "/opt/excavator/bin/excavator-il",
        "collect",
        "--config",
        "config/collection.orin.json",
    ) in reclaim["known_argv_suffixes"]
    assert (
        "-u",
        "orin_state_sender.py",
        "--serial-port",
        "/dev/ttyTHS1",
        "--control-enabled",
        "--pc-host",
        "192.0.2.20",
        "--edge-config",
        "deploy/edge_runtime.remote.json",
        "--edge-motion-authorization",
        "ALLOW_EDGE_MACHINE_MOTION",
        "--print-every",
        "100",
    ) in reclaim["known_argv_suffixes"]
    assert messages == ["检测到并释放了上一次遗留的 Orin 串口 Runtime。"]


def test_operator_preview_owns_only_camera_during_rl_stage(tmp_path):
    config = _guided_config(tmp_path)
    calls = []

    class _RemoteHost:
        def argv(self, command):
            calls.append(("argv", command))
            return ["ssh", command]

        def reclaim_serial_owner(self, **kwargs):
            calls.append(("reclaim", kwargs))
            return "idle"

        def stop_owned_process(self, **kwargs):
            calls.append(("stop", kwargs))

    class _PreviewProcess:
        def wait_for(self, predicate, _timeout_s, *, after_index=-1):
            del after_index
            lines = ("GUIDED_PREVIEW_PID=4321", "camera preview ready: test")
            line = next(value for value in lines if predicate(value))
            return lines.index(line), line

        def wait(self, timeout_s=5.0):
            assert timeout_s == 2.0

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda _message: None,
        line_process_factory=lambda _argv, **_kwargs: _PreviewProcess(),
    )
    operations._remote_host = _RemoteHost()

    operations.start_operator_preview()
    operations.stop_operator_preview_and_wait_for_camera()

    reclaim = next(value for kind, value in calls if kind == "reclaim")
    assert reclaim["serial_path"] == "/dev/video0"
    assert reclaim["known_argv_suffixes"] == (
        (
            "/opt/excavator/bin/excavator-il",
            "camera-preview",
            "--config",
            "config/collection.orin.json",
        ),
    )
    stop = next(value for kind, value in calls if kind == "stop")
    assert stop["pid"] == 4321
    assert stop["serial_path"] == "/dev/video0"
    assert stop["identity_ere"] == r"[c]amera-preview"


def test_guided_episode_loads_selectable_demo_dig_targets(tmp_path):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "demo_id": "field_demo_001",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.2, 0.0]},
                    {"point_id": "dig_02", "position_m": [1.0, 0.0, 0.0]},
                    {"point_id": "dig_03", "position_m": [1.0, -0.2, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)

    assert load_rl_dig_targets(config) == (
        ("dig_01", (1.0, 0.2, 0.0)),
        ("dig_02", (1.0, 0.0, 0.0)),
        ("dig_03", (1.0, -0.2, 0.0)),
    )


def test_guided_episode_stands_by_before_deadman_and_seals_immediately_on_release(
    tmp_path,
):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()
    prompts = []

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: prompts.append(prompt) or "成功",
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert len(prompts) == 1
    assert operations.events == [
        "preflight",
        "start_collector",
        "start_episode",
        "start_teleop",
        ("wait_for_ack", 8),
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "success",
            "",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_guided_episode_accepts_bracketed_paste_outcome(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()
    answers = iter(("\x1b[200~s\x1b[201~", "s"))
    prompts = []
    messages = []

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        output=messages.append,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert len(prompts) == 1
    assert not any("无法识别结果" in message for message in messages)


def test_guided_episode_optional_preposition_happens_before_episode_is_created(
    tmp_path,
):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()
    answers = iter(("完成", "成功"))

    episode_path = run_guided_episode(
        config,
        operations,
        preposition=True,
        input_fn=lambda prompt: next(answers),
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert operations.events == [
        "preflight",
        "start_collector",
        "start_teleop",
        ("wait_for_ack", 8),
        "wait_for_deadman_released",
        "stop_teleop",
        "start_episode",
        "start_teleop",
        ("wait_for_ack", 8),
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "success",
            "",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_positioning_choice_supports_rl_manual_and_direct_modes():
    messages = []
    answers = iter(("unknown", "l", "y"))

    assert _read_positioning_choice(lambda prompt: "", messages.append) is PositioningMode.DIRECT
    assert _read_positioning_choice(lambda prompt: next(answers), messages.append) is PositioningMode.RL
    assert _read_positioning_choice(lambda prompt: next(answers), messages.append) is PositioningMode.MANUAL
    assert messages == ["无法识别选择，请输入：RL定位/l、人工预定位/y 或直接采集/n。"]


def test_rl_positioning_finishes_and_releases_serial_before_collector_starts(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()

    episode_path = run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.RL,
        input_fn=lambda prompt: "成功",
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert operations.events[:5] == [
        "preflight",
        "start_rl_runtime",
        "run_rl_preposition",
        "stop_rl_runtime_and_wait_for_serial",
        "start_collector",
    ]
    assert operations.events.index("stop_rl_runtime_and_wait_for_serial") < operations.events.index("start_episode")
    assert operations.episode_targets == [(1.0, 0.0, 0.0)]


def test_selected_rl_target_is_persisted_as_episode_target(tmp_path):
    class SelectedTargetOperations(_FakeOperations):
        def run_rl_preposition(self, target_id=None):
            self.events.append(("run_rl_preposition", target_id))
            return (1.0, -0.2, 0.0)

    operations = SelectedTargetOperations()

    run_guided_episode(
        _guided_config(tmp_path),
        operations,
        positioning_mode=PositioningMode.RL,
        rl_target_id="dig_03",
        input_fn=lambda _prompt: "成功",
        output=lambda _message: None,
    )

    assert ("run_rl_preposition", "dig_03") in operations.events
    assert operations.episode_targets == [(1.0, -0.2, 0.0)]


def test_campaign_rl_collection_uses_protocol_dig_point_for_follow_and_provenance(
    tmp_path,
):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_02", "position_m": [1.05, 0.0, 0.02]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)

    class SelectedTargetOperations(_FakeOperations):
        def run_rl_preposition(self, target_id=None):
            self.events.append(("run_rl_preposition", target_id))
            return (1.05, 0.0, 0.02)

    operations = SelectedTargetOperations()

    run_guided_episode(
        config,
        operations,
        positioning_mode=PositioningMode.RL,
        input_fn=lambda _prompt: "s",
        output=lambda _message: None,
        task_variant="dig_only",
        soil_reset_block_id="block_03",
        dig_point_id="dig_02",
    )

    assert ("run_rl_preposition", "dig_02") in operations.events
    assert operations.episode_targets == [(1.05, 0.0, 0.02)]


def test_collection_rejects_unknown_protocol_dig_point_before_preflight(tmp_path):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.2, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = _FakeOperations()

    with pytest.raises(ValueError, match="not configured.*dig_missing"):
        run_guided_episode(
            config,
            operations,
            positioning_mode=PositioningMode.DIRECT,
            input_fn=lambda _prompt: "s",
            output=lambda _message: None,
            task_variant="dig_only",
            soil_reset_block_id="block_01",
            dig_point_id="dig_missing",
        )

    assert operations.events == []


def test_collection_rejects_mismatched_rl_and_protocol_point_ids_before_preflight(
    tmp_path,
):
    config = _guided_config(tmp_path)
    demo_path = tmp_path / "AiryLidar/mission/config/excavation_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "dig_points": [
                    {"point_id": "dig_01", "position_m": [1.0, 0.2, 0.0]},
                    {"point_id": "dig_02", "position_m": [1.0, 0.0, 0.0]},
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    operations = _FakeOperations()

    with pytest.raises(ValueError, match="must match.*dig_point_id"):
        run_guided_episode(
            config,
            operations,
            positioning_mode=PositioningMode.RL,
            rl_target_id="dig_01",
            input_fn=lambda _prompt: "s",
            output=lambda _message: None,
            task_variant="dig_only",
            soil_reset_block_id="block_01",
            dig_point_id="dig_02",
        )

    assert operations.events == []


def test_guided_episode_reports_operator_relevant_stages_through_one_callback(tmp_path):
    stages = []
    answers = iter(("c", "s"))

    run_guided_episode(
        _guided_config(tmp_path),
        _FakeOperations(),
        positioning_mode=PositioningMode.MANUAL,
        input_fn=lambda _prompt: next(answers),
        output=lambda _message: None,
        stage_callback=stages.append,
    )

    assert stages == [
        GuidedEpisodeStage.PREFLIGHT,
        GuidedEpisodeStage.COLLECTOR_STARTING,
        GuidedEpisodeStage.MANUAL_POSITIONING,
        GuidedEpisodeStage.RECORDER_STANDBY,
        GuidedEpisodeStage.RECORDING,
        GuidedEpisodeStage.REVIEW,
        GuidedEpisodeStage.FINALIZING,
        GuidedEpisodeStage.VALIDATING,
        GuidedEpisodeStage.COMPLETED,
    ]


def test_rl_positioning_failure_stops_runtime_without_starting_collector(tmp_path):
    class FailedRlOperations(_FakeOperations):
        def run_rl_preposition(self):
            self.events.append("run_rl_preposition")
            raise RuntimeError("Follow failed")

    operations = FailedRlOperations()

    with pytest.raises(RuntimeError, match="Follow failed"):
        run_guided_episode(
            _guided_config(tmp_path),
            operations,
            positioning_mode=PositioningMode.RL,
            input_fn=lambda prompt: "成功",
            output=lambda message: None,
        )

    assert operations.events == [
        "preflight",
        "start_rl_runtime",
        "run_rl_preposition",
        "stop_rl_runtime_and_wait_for_serial",
    ]


def test_system_rl_positioning_uses_mission_target_and_live_plan_follow(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    config.rl_airy_repo.mkdir()
    assert config.rl_ros_setup.is_file()
    config.rl_workspace_setup.parent.mkdir(parents=True)
    config.rl_workspace_setup.touch()
    config.rl_mission_config.parent.mkdir(parents=True)
    config.rl_mission_config.write_text(
        json.dumps(
            {
                "schema_version": "excavation_mission.v1",
                "targets": {"dig": {"position_m": [1.1, -0.2, 0.05]}},
            }
        ),
        encoding="utf-8",
    )
    process_calls = []

    class FakeLineProcess:
        returncode = 0

        def __init__(self, argv, **kwargs):
            process_calls.append((argv, kwargs))

        def wait(self, timeout_s=5.0):
            assert timeout_s == 90

        def stop(self, signum, *, timeout_s=5.0):
            raise AssertionError("successful RL positioning must not be stopped")

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda message: None,
        line_process_factory=FakeLineProcess,
    )

    target = operations.run_rl_preposition()

    assert target == (1.1, -0.2, 0.05)
    command = process_calls[0][0]
    assert command[:2] == ["/bin/zsh", "-lc"]
    assert "mission.runtime_ros.run_plan_follow_live" in command[2]
    assert str(config.rl_mission_config) in command[2]


def test_system_rl_positioning_uses_selected_demo_dig_target(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    config.rl_airy_repo.mkdir()
    config.rl_workspace_setup.parent.mkdir(parents=True)
    config.rl_workspace_setup.touch()
    config.rl_mission_config.parent.mkdir(parents=True)
    config.rl_mission_config.touch()
    demo_path = config.rl_airy_repo / "mission/config/excavation_demo.json"
    demo_path.write_text(
        json.dumps(
            {
                "schema_version": "excavation_demo.v1",
                "demo_id": "field_demo_001",
                "dig_points": [
                    {"point_id": "dig_03", "position_m": [1.0, -0.2, 0.0]}
                ],
            }
        ),
        encoding="utf-8",
    )
    object.__setattr__(config, "rl_demo_config", demo_path)
    process_calls = []

    class FakeLineProcess:
        returncode = 0

        def __init__(self, argv, **kwargs):
            process_calls.append((argv, kwargs))

        def wait(self, timeout_s=5.0):
            assert timeout_s == 90

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda _message: None,
        line_process_factory=FakeLineProcess,
    )

    target = operations.run_rl_preposition("dig_03")

    assert target == (1.0, -0.2, 0.0)
    command = process_calls[0][0][2]
    assert f"--demo {demo_path}" in command
    assert "--dig-point dig_03" in command


def test_system_rl_follow_supports_dump_phase_without_demo_target(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    config.rl_airy_repo.mkdir()
    config.rl_workspace_setup.parent.mkdir(parents=True)
    config.rl_workspace_setup.touch()
    config.rl_mission_config.parent.mkdir(parents=True)
    config.rl_mission_config.write_text(
        json.dumps(
            {
                "schema_version": "excavation_mission.v1",
                "targets": {
                    "dig": {"position_m": [1.0, 0.0, 0.0]},
                    "dump": {"position_m": [0.0, -1.0, 0.0]},
                },
            }
        ),
        encoding="utf-8",
    )
    process_calls = []

    class FakeLineProcess:
        returncode = 0

        def __init__(self, argv, **kwargs):
            process_calls.append((argv, kwargs))

        def wait(self, timeout_s=5.0):
            assert timeout_s == 90

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda _message: None,
        line_process_factory=FakeLineProcess,
    )

    target = operations.run_rl_follow("dump")

    assert target == (0.0, -1.0, 0.0)
    command = process_calls[0][0][2]
    assert "run_plan_follow_live dump" in command
    assert "--dig-point" not in command


def test_system_runs_existing_orin_fixed_dump_client(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    config.rl_airy_repo.mkdir()
    config.rl_workspace_setup.parent.mkdir(parents=True)
    config.rl_workspace_setup.touch()
    process_calls = []

    class FakeLineProcess:
        returncode = 0

        def __init__(self, argv, **kwargs):
            process_calls.append((argv, kwargs))

        def wait(self, timeout_s=5.0):
            assert timeout_s == 90

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda _message: None,
        line_process_factory=FakeLineProcess,
    )

    operations.run_rl_fixed_action("ExecuteDump", behavior_port=18083)

    command = process_calls[0][0][2]
    assert "runtime_bridge.apps.run_orin_fixed_action ExecuteDump" in command
    assert "--host 192.0.2.10" in command
    assert "--port 18083" in command


def test_system_starts_owned_rl_runtime_and_waits_for_ready(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    process_calls = []
    preflight_commands = []

    class FakeRuntimeProcess:
        def __init__(self, argv, **kwargs):
            process_calls.append((argv, kwargs))

        def wait_for(self, predicate, timeout_s, *, after_index=-1):
            candidates = (
                "GUIDED_RL_PID=4242",
                "REMOTE EDGE CONTROL ARMED IDLE",
                "sent seq=0 stm32_t=100 sensor_valid=True",
            )
            line = next(candidate for candidate in candidates if predicate(candidate))
            return candidates.index(line), line

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda message: None,
        line_process_factory=FakeRuntimeProcess,
    )

    def fake_run_ssh(command):
        preflight_commands.append(command)
        if "serial owner is not reclaimable" in command:
            return "idle\n"
        return "ready\n"

    monkeypatch.setattr(operations, "_run_ssh", fake_run_ssh)

    operations.start_rl_runtime()

    assert operations._rl_runtime_pid == 4242
    rendered = " ".join(process_calls[0][0])
    assert "orin_state_sender.py" in rendered
    assert "--edge-motion-authorization ALLOW_EDGE_MACHINE_MOTION" in rendered
    assert "--pc-host 192.0.2.20" in rendered
    assert len(preflight_commands) == 2
    assert "serial owner is not reclaimable" in preflight_commands[0]
    assert "allowed_client_host" in preflight_commands[1]
    assert "192.0.2.20" in preflight_commands[1]
    assert "fuser" in preflight_commands[1]


def test_system_prewarms_rl_without_serial_then_releases_gate_after_serial_check(
    tmp_path, monkeypatch
):
    config = _guided_config(tmp_path)
    process_calls = []
    remote_commands = []

    class FakeRuntimeProcess:
        def __init__(self, argv, **kwargs):
            process_calls.append((argv, kwargs))

        def wait_for(self, predicate, timeout_s, *, after_index=-1):
            del timeout_s, after_index
            candidates = (
                "GUIDED_RL_PID=4242",
                "RL prewarm ready: waiting for hardware start gate",
                "REMOTE EDGE CONTROL ARMED IDLE",
                "sent seq=0 stm32_t=100 sensor_valid=True",
            )
            line = next(candidate for candidate in candidates if predicate(candidate))
            return candidates.index(line), line

    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda _message: None,
        line_process_factory=FakeRuntimeProcess,
    )

    def fake_run_ssh(command):
        remote_commands.append(command)
        if "fuser -s" in command:
            return "released\n"
        return "ready\n"

    monkeypatch.setattr(operations, "_run_ssh", fake_run_ssh)
    gate = "/tmp/excavator-rl-control/hybrid_test.start"

    operations.prewarm_rl_runtime(gate)

    rendered = " ".join(process_calls[0][0])
    assert "--hardware-start-gate" in rendered
    assert gate in rendered
    assert not any("touch --" in command for command in remote_commands)

    operations.start_rl_runtime()

    assert operations._rl_runtime_pid == 4242
    serial_check_index = next(
        index
        for index, command in enumerate(remote_commands)
        if "fuser -s" in command
    )
    gate_release_index = next(
        index
        for index, command in enumerate(remote_commands)
        if "touch --" in command
    )
    assert serial_check_index < gate_release_index


def test_system_rl_release_targets_one_runtime_and_waits_for_serial(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    remote_commands = []
    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)
    waits = []

    class RunningRlRuntime:
        def wait(self, timeout_s=5.0):
            waits.append(timeout_s)

    operations._rl_runtime = RunningRlRuntime()
    operations._rl_runtime_pid = 4242
    monkeypatch.setattr(
        operations,
        "_run_ssh",
        lambda command: remote_commands.append(command) or "released\n",
    )

    operations.stop_rl_runtime_and_wait_for_serial()

    assert len(remote_commands) == 1
    assert "kill -TERM" in remote_commands[0]
    assert "[o]rin_state_sender" in remote_commands[0]
    assert "fuser" in remote_commands[0]
    assert "/dev/ttyTHS1" in remote_commands[0]
    assert "pid=4242" in remote_commands[0]
    assert waits == [2.0]


def test_guided_episode_interrupt_aborts_before_stopping_motion_io(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()

    def interrupt_during_result(unused_prompt):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_guided_episode(
            config,
            operations,
            input_fn=interrupt_during_result,
            output=lambda message: None,
        )

    assert operations.events[-4:] == [
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "aborted",
            "guided_episode_interrupted",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_guided_episode_interrupt_while_standing_by_does_not_build_empty_attempt(
    tmp_path,
):
    config = _guided_config(tmp_path)

    class InterruptWhileStandingBy(_FakeOperations):
        def wait_for_deadman_pressed(self):
            self.events.append("wait_for_deadman_pressed")
            raise KeyboardInterrupt

    operations = InterruptWhileStandingBy()

    with pytest.raises(KeyboardInterrupt):
        run_guided_episode(
            config,
            operations,
            input_fn=lambda prompt: "成功",
            output=lambda message: None,
        )

    assert ("abort_episode", "guided_episode_interrupted") in operations.events
    assert not any(
        isinstance(event, tuple) and event[0] == "build_and_validate"
        for event in operations.events
    )


def test_guided_episode_retake_deletes_attempt_and_reuses_episode_id(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()
    outcomes = iter(("重录", "失败"))

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: next(outcomes),
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert operations.events == [
        "preflight",
        "start_collector",
        "start_episode",
        "start_teleop",
        ("wait_for_ack", 8),
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        ("discard_episode", "/data/raw/episode_0001"),
        "start_episode",
        "wait_for_deadman_pressed",
        "wait_for_deadman_released",
        "seal_episode",
        (
            "finalize_episode",
            "/data/raw/episode_0001",
            "failure",
            "diagnostic_task_failed",
        ),
        "stop_teleop",
        "stop_collector",
        ("build_and_validate", "/data/raw/episode_0001"),
    ]


def test_guided_episode_can_retain_a_failed_attempt_and_validate_it(tmp_path):
    config = _guided_config(tmp_path)
    operations = _FakeOperations()

    episode_path = run_guided_episode(
        config,
        operations,
        input_fn=lambda prompt: "f",
        output=lambda message: None,
    )

    assert episode_path == "/data/raw/episode_0001"
    assert (
        "finalize_episode",
        "/data/raw/episode_0001",
        "failure",
        "diagnostic_task_failed",
    ) in operations.events
    assert operations.events[-1] == (
        "build_and_validate",
        "/data/raw/episode_0001",
    )


def test_guided_episode_validates_saved_episode_after_cleanup_error(tmp_path):
    class CleanupFailureOperations(_FakeOperations):
        def stop_collector(self):
            self.events.append("stop_collector")
            raise RuntimeError("collector SSH cleanup timed out")

    config = _guided_config(tmp_path)
    operations = CleanupFailureOperations()

    with pytest.raises(RuntimeError, match="Collector cleanup failed"):
        run_guided_episode(
            config,
            operations,
            input_fn=lambda prompt: "s",
            output=lambda message: None,
        )

    assert operations.events[-1] == (
        "build_and_validate",
        "/data/raw/episode_0001",
    )


def test_system_operations_manage_exact_collector_and_episode_commands(
    tmp_path, monkeypatch
):
    (tmp_path / "teleop.json").write_text("{}", encoding="utf-8")
    config = GuidedEpisodeConfig(
        teleop_config=tmp_path / "teleop.json",
        orin_ssh_host="operator@192.0.2.10",
        orin_repo="/srv/excavator-il",
        orin_executable="/opt/excavator/bin/excavator-il",
        orin_collection_config="config/collection.orin.json",
        task="DiagnosticBoomJog",
        operator_id="operator_01",
        dig_target_m=(0.8, 0.0, -0.2),
        material_id="soil_default",
        collector_ready_timeout_s=8,
        ack_timeout_s=8,
        teleop_print_every=1,
        log_dir=tmp_path / "logs",
        rl_airy_repo=tmp_path / "AiryLidar",
        rl_ros_setup=Path("/opt/ros/jazzy/setup.zsh"),
        rl_workspace_setup=tmp_path / "AiryLidar/ros2_ws/install/setup.zsh",
        rl_mission_config=tmp_path / "AiryLidar/mission/config/excavation_cycle.json",
        rl_phase="dig",
        rl_timeout_s=90,
        rl_serial_port="/dev/ttyTHS1",
        rl_serial_release_timeout_s=8,
        rl_orin_repo="/srv/excavator-orin-runtime",
        rl_orin_python="/opt/excavator-orin/bin/python",
        rl_edge_config="deploy/edge_runtime.remote.json",
        rl_pc_host="192.0.2.20",
        rl_ready_timeout_s=15,
    )
    popen_calls = []
    signal_calls = []
    run_calls = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            popen_calls.append(argv)
            if "collect" in " ".join(argv):
                output = "GUIDED_COLLECTOR_PID=4321\ncollector ready: serial=/dev/ttyTHS1\n"
            else:
                output = (
                    "teleop seq=20 ack=19 ack_lag=1 accepted_acks=20 "
                    "rejected_acks=0 deadman=False axes=(0,0,0,0,0,0)\n"
                    "teleop seq=21 ack=20 ack_lag=1 accepted_acks=21 "
                    "rejected_acks=0 deadman=True axes=(0,0,0,0,0,0)\n"
                    "teleop seq=22 ack=21 ack_lag=1 accepted_acks=22 "
                    "rejected_acks=0 deadman=False axes=(0,0,0,0,0,0)\n"
                )
            self.stdout = io.StringIO(output)
            self.returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            signal_calls.append(signum)
            self.returncode = 0

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    def fake_run(argv, **kwargs):
        run_calls.append(argv)
        rendered = " ".join(argv)
        if "serial owner is not reclaimable" in rendered:
            stdout = "idle\n"
        elif " episode " in rendered and " start" in rendered:
            stdout = json.dumps(
                {"ok": True, "active": True, "path": "/data/raw/episode_0001"}
            )
        elif " episode " in rendered and " seal" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "pending_review",
                    "path": "/data/raw/episode_0001",
                }
            )
        elif " episode " in rendered and " finalize" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "complete",
                    "path": "/data/raw/episode_0001",
                }
            )
        elif " episode " in rendered and " stop" in rendered:
            stdout = json.dumps(
                {"ok": True, "active": False, "path": "/data/raw/episode_0001"}
            )
        elif " episode " in rendered and " abort" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "aborted",
                    "path": "/data/raw/episode_0001",
                }
            )
        elif "quality_report.json" in rendered:
            stdout = json.dumps({"episode_id": "episode_0001", "valid": True})
        else:
            stdout = "{}"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("excavator_il.guided_episode.subprocess.run", fake_run)
    operations = SystemGuidedEpisodeOperations(
        config,
        output=lambda message: None,
        timestamp="20260811_200000",
        line_process_factory=lambda argv, **kwargs: LineProcess(
            argv,
            popen_command=FakePopen,
            **kwargs,
        ),
    )

    operations.preflight()
    operations.start_collector()
    assert operations.start_episode() == "/data/raw/episode_0001"
    operations.start_teleop()
    operations.wait_for_ack(8)
    operations.wait_for_deadman_pressed()
    operations.wait_for_deadman_released()
    assert operations.seal_episode() == "/data/raw/episode_0001"
    assert (
        operations.finalize_episode("/data/raw/episode_0001", "success")
        == "/data/raw/episode_0001"
    )
    operations.stop_teleop()
    operations.stop_collector()
    operations.build_and_validate("/data/raw/episode_0001")

    assert len(popen_calls) == 2
    teleop_argv = next(call for call in popen_calls if "teleop" in call)
    assert teleop_argv[1:3] == ["-u", "-m"]
    assert any("kill -TERM -- -4321" in " ".join(call) for call in run_calls)
    assert any("build-steps" in " ".join(call) for call in run_calls)
    assert any("validate" in " ".join(call) for call in run_calls)
    assert signal_calls


def test_inspect_zero_soak_preserves_failed_quality_report(tmp_path, monkeypatch):
    (tmp_path / "teleop.json").write_text("{}", encoding="utf-8")
    config = _guided_config(tmp_path)
    report = {
        "episode_id": "episode_0001",
        "passed": False,
        "failure_reasons": ["new_sensor_state rate is outside its allowed range"],
    }

    def fake_run(argv, **kwargs):
        return SimpleNamespace(
            returncode=3,
            stdout=json.dumps(report),
            stderr="",
        )

    monkeypatch.setattr("excavator_il.guided_episode.subprocess.run", fake_run)
    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)

    assert operations.inspect_zero_soak("/data/raw/episode_0001") == report


def test_build_and_validate_records_collection_evidence_after_quality_check(tmp_path):
    config = replace(
        _guided_config(tmp_path),
        orin_experiment_run_config="config/collection_evidence.orin.json",
    )
    operations = SystemGuidedEpisodeOperations(config, output=lambda _message: None)
    events = []

    def remote_cli(argv, *, accepted_returncodes=(0,)):
        events.append(("remote_cli", tuple(argv), accepted_returncodes))
        return {"command": argv[0]}

    def run_ssh(command, *, accepted_returncodes=(0,)):
        events.append(("quality", command, accepted_returncodes))
        return json.dumps({"episode_id": "episode_0001", "passed": True})

    operations._remote_cli = remote_cli
    operations._run_ssh = run_ssh

    operations.build_and_validate("/data/raw/episode_0001")

    assert [event[0] for event in events] == [
        "remote_cli",
        "remote_cli",
        "quality",
        "remote_cli",
    ]
    assert events[-1][1] == (
        "record-collection-run",
        "/data/raw/episode_0001",
        "--config",
        "config/collection_evidence.orin.json",
    )


def test_collection_evidence_failure_fails_validation_without_deleting_raw_episode(
    tmp_path,
):
    config = replace(
        _guided_config(tmp_path),
        orin_experiment_run_config="config/collection_evidence.orin.json",
    )
    operations = SystemGuidedEpisodeOperations(config, output=lambda _message: None)
    remote_calls = []

    def remote_cli(argv, *, accepted_returncodes=(0,)):
        remote_calls.append(tuple(argv))
        if argv[0] == "record-collection-run":
            raise RuntimeError("evidence write failed")
        return {"command": argv[0]}

    operations._remote_cli = remote_cli
    operations._run_ssh = lambda *_args, **_kwargs: json.dumps(
        {"episode_id": "episode_0001", "passed": True}
    )

    with pytest.raises(RuntimeError, match="evidence write failed"):
        operations.build_and_validate("/data/raw/episode_0001")

    assert [call[0] for call in remote_calls] == [
        "build-steps",
        "validate",
        "record-collection-run",
    ]
    assert all("rm" not in call for call in remote_calls)


def test_system_operations_only_discard_retake_episode_started_by_this_run(
    tmp_path, monkeypatch
):
    (tmp_path / "teleop.json").write_text("{}", encoding="utf-8")
    config = _guided_config(tmp_path)
    run_calls = []

    def fake_run(argv, **kwargs):
        run_calls.append(argv)
        rendered = " ".join(argv)
        if " episode " in rendered and " start" in rendered:
            stdout = json.dumps(
                {"ok": True, "active": True, "path": "/data/raw/episode_0007"}
            )
        elif " episode " in rendered and " seal" in rendered:
            stdout = json.dumps(
                {
                    "ok": True,
                    "active": False,
                    "status": "pending_review",
                    "path": "/data/raw/episode_0007",
                }
            )
        else:
            stdout = ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("excavator_il.guided_episode.subprocess.run", fake_run)
    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)

    with pytest.raises(RuntimeError, match="unapproved"):
        operations.discard_episode("/data/raw/episode_0006")

    assert operations.start_episode() == "/data/raw/episode_0007"
    assert operations.seal_episode() == "/data/raw/episode_0007"
    operations.discard_episode("/data/raw/episode_0007")

    delete_command = " ".join(run_calls[-1])
    assert "rm -rf -- /data/raw/episode_0007" in delete_command
    assert "/data/raw/episode_0006" not in delete_command

    with pytest.raises(RuntimeError, match="unapproved"):
        operations.discard_episode("/data/raw/episode_0007")


def test_system_start_episode_sends_target_source_as_one_immutable_json_argument(
    tmp_path,
):
    operations = SystemGuidedEpisodeOperations(
        _guided_config(tmp_path), output=lambda _message: None
    )
    calls = []

    def remote_cli(command, *, accepted_returncodes=(0,)):
        calls.append((tuple(command), accepted_returncodes))
        return {
            "ok": True,
            "active": True,
            "path": "/data/raw/episode_0001",
        }

    operations._remote_cli = remote_cli
    provenance = {
        "repository": "airylidar",
        "path": "mission/config/excavation_demo.json",
        "sha256": "a" * 64,
        "commit": "b" * 40,
        "dirty": False,
    }

    operations.start_episode(
        (1.0, 0.0, 0.0),
        task_variant="dig_only",
        soil_reset_block_id="block_01",
        dig_point_id="dig_02",
        collection_zone_id="zone_05",
        dig_repeat_index=2,
        operator_note="中排偏硬",
        target_source_provenance=provenance,
    )

    command = calls[0][0]
    wire_index = command.index("--target-source-provenance-json")
    assert json.loads(command[wire_index + 1]) == provenance
    assert command[command.index("--collection-zone-id") + 1] == "zone_05"
    assert command[command.index("--dig-repeat-index") + 1] == "2"
    assert command[command.index("--operator-note") + 1] == "中排偏硬"


def test_stop_collector_does_not_kill_remote_pid_after_collector_exited(
    tmp_path, monkeypatch
):
    config = _guided_config(tmp_path)
    remote_commands = []

    class ExitedCollector:
        running = False

        def wait(self, timeout_s=5.0):
            return None

    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)
    operations._collector = ExitedCollector()
    operations._collector_pid = 15303
    monkeypatch.setattr(
        operations,
        "_run_ssh",
        lambda command: remote_commands.append(command) or "",
    )

    operations.stop_collector()

    assert remote_commands == []
    assert operations._collector is None
    assert operations._collector_pid is None


def test_stop_collector_allows_bounded_remote_shutdown_time(tmp_path, monkeypatch):
    config = _guided_config(tmp_path)
    wait_timeouts = []
    remote_commands = []

    class RunningCollector:
        running = True

        def wait(self, timeout_s=5.0):
            wait_timeouts.append(timeout_s)

    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)
    operations._collector = RunningCollector()
    operations._collector_pid = 14313
    monkeypatch.setattr(
        operations,
        "_run_ssh",
        lambda command: remote_commands.append(command) or "",
    )

    operations.stop_collector()

    assert remote_commands == ["kill -TERM -- -14313"]
    assert wait_timeouts == [2.0]


def test_stop_collector_closes_stale_ssh_after_remote_process_exited(
    tmp_path, monkeypatch
):
    config = _guided_config(tmp_path)
    remote_commands = []
    local_stop_calls = []

    class StaleSshCollector:
        running = True

        def wait(self, timeout_s=5.0):
            raise subprocess.TimeoutExpired("ssh collector", timeout_s)

        def stop(self, signum, *, timeout_s=5.0):
            local_stop_calls.append((signum, timeout_s))

    def fake_run_ssh(command):
        remote_commands.append(command)
        return "exited\n" if "for attempt in" in command else ""

    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)
    operations._collector = StaleSshCollector()
    operations._collector_pid = 15867
    monkeypatch.setattr(operations, "_run_ssh", fake_run_ssh)

    operations.stop_collector()

    assert remote_commands == [
        "kill -TERM -- -15867",
        (
            "for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 "
            "18 19 20; do if ! kill -0 -- -15867 2>/dev/null; then "
            "echo exited; exit 0; fi; sleep 0.25; done; echo running"
        ),
    ]
    assert local_stop_calls == [(signal.SIGKILL, 2.0)]
    assert operations._collector is None
    assert operations._collector_pid is None


def test_stop_collector_reports_remote_process_still_running_after_timeout(
    tmp_path, monkeypatch
):
    config = _guided_config(tmp_path)
    local_stop_calls = []

    class HungCollector:
        running = True

        def wait(self, timeout_s=5.0):
            raise subprocess.TimeoutExpired("ssh collector", timeout_s)

        def stop(self, signum, *, timeout_s=5.0):
            local_stop_calls.append((signum, timeout_s))

    operations = SystemGuidedEpisodeOperations(config, output=lambda message: None)
    operations._collector = HungCollector()
    operations._collector_pid = 15867
    monkeypatch.setattr(
        operations,
        "_run_ssh",
        lambda command: "running\n" if "for attempt in" in command else "",
    )

    with pytest.raises(RuntimeError, match="still running"):
        operations.stop_collector()

    assert local_stop_calls == [(signal.SIGKILL, 2.0)]
    assert operations._collector is None
    assert operations._collector_pid is None


def test_line_process_stop_escalates_to_kill_when_ssh_ignores_term(
    tmp_path, monkeypatch
):
    calls = []

    class TermIgnoringPopen:
        def __init__(self, argv, **kwargs):
            self.stdout = io.StringIO("")
            self.returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            calls.append(("send_signal", signum))

        def terminate(self):
            calls.append(("terminate", signal.SIGTERM))

        def kill(self):
            calls.append(("kill", signal.SIGKILL))
            self.returncode = -signal.SIGKILL

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("ssh collector", timeout)
            return self.returncode

    process = LineProcess(
        ["ssh", "collector"],
        log_path=tmp_path / "collector.log",
        prefix="collector",
        output=lambda message: None,
        popen_command=TermIgnoringPopen,
    )

    process.stop(signal.SIGTERM, timeout_s=0.01)

    assert calls == [
        ("send_signal", signal.SIGTERM),
        ("terminate", signal.SIGTERM),
        ("kill", signal.SIGKILL),
    ]


def test_line_process_stop_signals_owned_process_group_after_leader_exits(
    tmp_path, monkeypatch
):
    calls = []

    class ExitedLeaderPopen:
        pid = 43210

        def __init__(self, _argv, **_kwargs):
            self.stdout = io.StringIO("")

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(
        "excavator_il.remote_runtime.os.killpg",
        lambda pgid, signum: calls.append((pgid, signum)),
    )
    process = LineProcess(
        ["ros2", "launch"],
        log_path=tmp_path / "operator.log",
        prefix="airy-operator",
        output=lambda _message: None,
        popen_command=ExitedLeaderPopen,
    )

    process.stop(signal.SIGINT, timeout_s=0.01)

    assert calls == [(43210, signal.SIGINT)]


def test_collector_extra_includes_episode_image_validation_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    collector_dependencies = pyproject["project"]["optional-dependencies"]["collector"]

    assert any(dependency.startswith("Pillow>=") for dependency in collector_dependencies)
