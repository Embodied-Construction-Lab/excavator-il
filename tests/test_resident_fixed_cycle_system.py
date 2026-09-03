import json
import hashlib
import threading
import time
from pathlib import Path

import pytest

import excavator_il.resident_fixed_cycle_system as resident_module
from excavator_il._resident_fixed_cycle_support import parse_owner_readiness
from excavator_il.hybrid_mission import REQUIRED_HYBRID_MOTION_AUTHORIZATION
from excavator_il.guided_episode import GuidedEpisodeConfig
from excavator_il.resident_fixed_cycle_system import (
    ResidentFixedCyclePcConfig,
    ResidentFixedCycleProcesses,
    ResidentFixedCycleRemoteStatus,
    ResidentFixedCycleSupervisor,
    SshResidentFixedCycleOperations,
)


def _write_config(tmp_path: Path, **overrides: object) -> Path:
    document = {
        "schema_version": "excavator_resident_fixed_cycle_pc.v6",
        "guided_config": "guided.json",
        "fixed_cycle_plan": "/home/jetson16/workspace_excavator/"
        "excavator-orin-runtime/deploy/v3a/fixed_cycle.field.json",
        "runtime_root": "/home/jetson16/.local/run/excavator-resident-v3a",
        "owner_script": "scripts/run_resident_mission_runtime.sh",
        "act_worker_script": "scripts/run_act_resident.sh",
        "control_socket": "/home/jetson16/.local/run/"
        "excavator-resident-v3a/fixed-cycle.sock",
        "ready_timeout_s": 120,
        "status_poll_ms": 100,
        "act_max_steps": 130,
        "commissioning_authorization": "",
        "expected_mission_id": "fixed_target_hybrid",
        "expected_mission_sha256": "3a5c7edd6a228863e3d5eefe3228173848756a46e9ce441da53cc2b0c164d786",
        "expected_act_worker_required": True,
        "expected_act_behavior_id": "act_dig_lift",
        "expected_act_model_sha256": "742a07ad6175af60ab0f14e4cdf409b790b35aad5a33f1e26d3d378952b7a475",
        "act_runtime_config": "/home/jetson16/workspace_excavator/"
        "excavator-il/config/act_runtime.orin.json",
        "act_checkpoint_host_path": "/home/jetson16/workspace_excavator/"
        "excavator-il/models/icra2027_dig_only_front_swing_zero_seed2027_"
        "step400000/checkpoint",
        "act_deployment_host_path": "/home/jetson16/workspace_excavator/"
        "excavator-il/models/icra2027_dig_only_front_swing_zero_seed2027_"
        "step400000/deployment",
        "dig_point_catalog": "mission/config/excavation_dig_point_catalog.v1.json",
        "edge_runtime_config": "deploy/edge_runtime.resident.remote.json",
        "trajectory_controller_backend": "onnx_rl",
        "trajectory_controller_commissioning_authorization": "",
    }
    document.update(overrides)
    path = tmp_path / "resident-fixed-cycle.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_act_reference_config(tmp_path: Path, **overrides: object) -> Path:
    document = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    document.update(
        {
            "expected_mission_id": "engineering_act_transport_reference",
            "expected_mission_sha256": "3631027f71a30faf92000a7c03f0d7905ca8f00f60daa0b9afeb3482da4cc83c",
            "expected_act_worker_required": True,
            "expected_act_behavior_id": "act_dig_transport_dump",
            "expected_act_model_sha256": "54a3ba90e6c2186787b8b7eb1b9e5211e2bcf81e41551e866283ace41ed04f4a",
            "act_runtime_config": "/home/jetson16/workspace_excavator/"
            "excavator-il/config/act_runtime.transport_dump_reference.orin.json",
            "act_checkpoint_host_path": "/home/jetson16/workspace_excavator/"
            "excavator-il/models/act-dig-transport-dump-reference",
            "act_deployment_host_path": "/home/jetson16/workspace_excavator/"
            "excavator-il/models/act-dig-transport-dump-reference/deployment",
            "act_max_steps": 260,
        }
    )
    document.update(overrides)
    path = tmp_path / "resident-act-reference.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_grouped_cycle_config(tmp_path: Path, **overrides: object) -> Path:
    document = json.loads(_write_config(tmp_path).read_text(encoding="utf-8"))
    document.update(
        {
            "expected_mission_id": "fixed_target_hybrid",
            "expected_act_worker_required": True,
            "dig_point_catalog": (
                "mission/config/excavation_dig_point_catalog.v1.json"
            ),
        }
    )
    document.update(overrides)
    path = tmp_path / "resident-grouped-cycle.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_classical_tracking_config(
    tmp_path: Path, **overrides: object
) -> Path:
    document = json.loads(
        _write_grouped_cycle_config(tmp_path).read_text(encoding="utf-8")
    )
    document.update(
        {
            "expected_mission_id": "classical_tracking_hybrid",
            "expected_mission_sha256": "629ecaa1dcff9b17c8b5497d7fae7dc8e0223b0eb1c1c5d8aec22465eea1e1a7",
            "edge_runtime_config": (
                "deploy/edge_runtime.resident.cartesian_p.commissioning.json"
            ),
            "trajectory_controller_backend": "cartesian_p",
            "trajectory_controller_commissioning_authorization": (
                "ALLOW_CARTESIAN_P_MACHINE_MOTION"
            ),
        }
    )
    document.update(overrides)
    path = tmp_path / "resident-classical-tracking.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _write_fixed_dig_config(tmp_path: Path, **overrides: object) -> Path:
    document = json.loads(
        _write_grouped_cycle_config(tmp_path).read_text(encoding="utf-8")
    )
    document.update(
        {
            "expected_mission_id": "fixed_dig_hybrid",
            "expected_mission_sha256": "a52462867a4c81e28e623d02c736b3f6e91cec62cbb71fa0505ad6035ad80101",
            "expected_act_worker_required": False,
            "expected_act_behavior_id": None,
            "expected_act_model_sha256": None,
            "act_runtime_config": None,
            "act_checkpoint_host_path": None,
            "act_deployment_host_path": None,
            "edge_runtime_config": (
                "deploy/edge_runtime.resident.fixed_dig.commissioning.json"
            ),
            "trajectory_controller_backend": "onnx_rl",
            "trajectory_controller_commissioning_authorization": "",
        }
    )
    document.update(overrides)
    path = tmp_path / "resident-fixed-dig.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_grouped_cycle_config_references_one_airy_point_catalog(tmp_path: Path) -> None:
    config = ResidentFixedCyclePcConfig.load(_write_grouped_cycle_config(tmp_path))

    assert config.expected_mission_id == "fixed_target_hybrid"
    assert config.expected_act_worker_required is True
    assert str(config.dig_point_catalog) == (
        "mission/config/excavation_dig_point_catalog.v1.json"
    )


def test_classical_tracking_config_selects_one_explicit_commissioned_backend(
    tmp_path: Path,
) -> None:
    config = ResidentFixedCyclePcConfig.load(
        _write_classical_tracking_config(tmp_path)
    )

    assert config.trajectory_controller_backend == "cartesian_p"
    assert str(config.edge_runtime_config) == (
        "deploy/edge_runtime.resident.cartesian_p.commissioning.json"
    )
    assert config.trajectory_controller_commissioning_authorization == (
        "ALLOW_CARTESIAN_P_MACHINE_MOTION"
    )

    with pytest.raises(ValueError, match="commissioning authorization"):
        ResidentFixedCyclePcConfig.load(
            _write_classical_tracking_config(
                tmp_path,
                trajectory_controller_commissioning_authorization="",
            )
        )


def test_fixed_dig_config_reuses_edge_runtime_without_parallel_act_assets(
    tmp_path: Path,
) -> None:
    config = ResidentFixedCyclePcConfig.load(_write_fixed_dig_config(tmp_path))

    assert config.expected_mission_id == "fixed_dig_hybrid"
    assert config.expected_act_worker_required is False
    assert config.act_runtime_config is None
    assert config.act_checkpoint_host_path is None
    assert config.act_deployment_host_path is None
    assert config.trajectory_controller_backend == "onnx_rl"
    assert str(config.edge_runtime_config) == (
        "deploy/edge_runtime.resident.fixed_dig.commissioning.json"
    )

    with pytest.raises(ValueError, match="ACT assets require"):
        ResidentFixedCyclePcConfig.load(
            _write_fixed_dig_config(
                tmp_path,
                act_runtime_config="/opt/act/runtime.json",
                act_checkpoint_host_path="/opt/act/checkpoint",
                act_deployment_host_path="/opt/act/deployment",
            )
        )


def test_required_act_worker_requires_explicit_asset_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required ACT worker assets"):
        ResidentFixedCyclePcConfig.load(
            _write_config(
                tmp_path,
                act_runtime_config=None,
                act_checkpoint_host_path=None,
                act_deployment_host_path=None,
            )
        )


def test_fixed_cycle_pc_config_is_strict_and_resolves_guided_config(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))

    assert config.guided_config == (tmp_path / "guided.json").resolve()
    assert config.fixed_cycle_plan.is_absolute()
    assert config.status_poll_s == pytest.approx(0.1)
    assert config.act_max_steps == 130
    assert config.commissioning_authorization == ""

    invalid = _write_config(tmp_path, unexpected=True)
    with pytest.raises(ValueError, match="fields"):
        ResidentFixedCyclePcConfig.load(invalid)

    commissioned = ResidentFixedCyclePcConfig.load(
        _write_config(
            tmp_path,
            commissioning_authorization="ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING",
        )
    )
    assert commissioned.commissioning_authorization.startswith("ALLOW_V3A")

    with pytest.raises(ValueError, match="commissioning_authorization"):
        ResidentFixedCyclePcConfig.load(
            _write_config(tmp_path, commissioning_authorization="wrong")
        )


def test_act_reference_pc_config_requires_explicit_parallel_act_assets(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_act_reference_config(tmp_path))

    assert config.expected_mission_id == "engineering_act_transport_reference"
    assert config.expected_act_worker_required is True
    assert config.act_max_steps == 260
    assert str(config.act_runtime_config).endswith(
        "act_runtime.transport_dump_reference.orin.json"
    )
    assert str(config.act_checkpoint_host_path).endswith(
        "models/act-dig-transport-dump-reference"
    )
    assert str(config.act_deployment_host_path).endswith(
        "models/act-dig-transport-dump-reference/deployment"
    )

    with pytest.raises(ValueError, match="act_runtime_config"):
        ResidentFixedCyclePcConfig.load(
            _write_act_reference_config(tmp_path, act_runtime_config="relative.json")
        )


def test_fixed_dig_processes_start_only_owner_without_act_worker(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_fixed_dig_config(tmp_path))
    lines = iter(
        [
            "RESIDENT_OWNER_PID=4321",
            _owner_ready_line(config),
            "RESIDENT_HARDWARE_READY sensor_valid=True",
        ]
    )
    commands = []

    class Owner:
        returncode = None

        def wait_for(self, predicate, timeout_s):
            assert timeout_s == config.ready_timeout_s
            while True:
                line = next(lines)
                if predicate(line):
                    return (time.monotonic(), line)

        def wait(self, *, timeout_s):
            assert timeout_s == 10.0
            return 0

    class Host:
        def argv(self, command):
            commands.append(command)
            return ["ssh", command]

        def stop_owned_process(self, **_kwargs):
            return None

    calls = 0

    def factory(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise AssertionError("fixed_dig must not launch ACT worker")
        return Owner()

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
    )

    processes.start()
    processes.require_running()
    processes.stop()

    assert calls == 1
    assert len(commands) == 1
    expected_edge_config = (
        "--edge-config "
        "deploy/edge_runtime.resident.fixed_dig.commissioning.json"
    )
    assert expected_edge_config in commands[0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "wrong"}, "schema"),
        ({"control_socket": "/tmp/outside.sock"}, "runtime_root"),
        ({"fixed_cycle_plan": "relative.json"}, "absolute"),
        ({"owner_script": "../owner.sh"}, "normalized relative"),
        ({"ready_timeout_s": True}, "numeric"),
        ({"status_poll_ms": 1}, "allowed range"),
        ({"act_max_steps": 0}, "allowed range"),
    ],
)
def test_fixed_cycle_pc_config_rejects_invalid_boundaries(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        ResidentFixedCyclePcConfig.load(_write_config(tmp_path, **overrides))


def test_fixed_cycle_pc_config_wraps_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load"):
        ResidentFixedCyclePcConfig.load(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"unexpected": 1}, "fields"),
        ({"stage": "bad stage"}, "stage"),
        ({"requested_cycles": 0, "completed_cycles": 1}, "cannot exceed"),
        ({"terminal": 1}, "boolean"),
        ({"stage": "COMPLETED", "terminal": False}, "disagree"),
        ({"run_id": "bad id"}, "identifier"),
    ],
)
def test_remote_status_rejects_invalid_wire_values(change, message):
    value = _status("go_current_dig")
    value.update(change)
    with pytest.raises(ValueError, match=message):
        ResidentFixedCycleRemoteStatus.from_mapping(value)


def test_ssh_operations_start_once_and_use_only_local_cycle_control(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    calls = []
    trajectory_updates = []

    class Processes:
        def start(self):
            calls.append(("processes", "start"))

        def stop(self, *, terminal_disarmed=False):
            calls.append(("processes", "stop", terminal_disarmed))

        def require_running(self):
            calls.append(("processes", "running"))

    responses = {
        "start": _status("go_current_dig", behavior_id="onnx_rl_tracking"),
        "heartbeat": _status(
            "dig", behavior_id="act_dig_lift", active_trajectory=_trajectory()
        ),
        "cancel": _status("CANCELLED", terminal=True, outcome="CANCELLED"),
    }

    class Host:
        def run(self, command, *, accepted_returncodes=(0,)):
            calls.append(("ssh", command, accepted_returncodes))
            selected = next(name for name in responses if command.endswith(name))
            return json.dumps({"schema_version": "resident_fixed_cycle_control.v4",
                               "ok": True, "command": selected,
                               "status": responses[selected], "error": None})

    class TrajectoryFile:
        def update(self, trajectory):
            trajectory_updates.append(trajectory)

    operations = SshResidentFixedCycleOperations(
        config,
        guided_config=_guided(),
        processes=Processes(),
        remote_host=Host(),
        trajectory_file=TrajectoryFile(),
    )
    assert trajectory_updates == [None]

    start = operations.start(
        run_id="run-001", requested_cycles=3, first_dig_point_id="dig_02"
    )
    status = operations.status()
    cancelled = operations.cancel()
    operations.release(terminal_disarmed=True)

    assert start.stage == "go_current_dig"
    assert start.mission_id == "fixed_target_hybrid"
    assert status.active_behavior_id == "act_dig_lift"
    assert cancelled.terminal is True
    ssh_commands = [entry[1] for entry in calls if entry[0] == "ssh"]
    assert any("resident_fixed_cycle_control" in command for command in ssh_commands)
    assert any("--run-id run-001 --cycles 3 --first-dig-point-id dig_02 "
               "--dig-group-id all start" in command
               for command in ssh_commands)
    assert not any("18083" in command or "Plan" in command for command in ssh_commands)
    assert any(update is not None for update in trajectory_updates)
    assert trajectory_updates[-1] is None


def test_ssh_operations_accept_real_guided_posix_path_fields(tmp_path):
    """Exercise the production config types at the final shlex boundary."""

    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    repo_root = Path(__file__).parents[1]
    guided = GuidedEpisodeConfig.load(repo_root / "config/guided_episode.pc.json")
    commands = []

    class Processes:
        def start(self):
            return None

    class Host:
        def run(self, command, *, accepted_returncodes=(0,)):
            commands.append(command)
            return json.dumps(
                {
                    "schema_version": "resident_fixed_cycle_control.v4",
                    "ok": True,
                    "command": "start",
                    "status": _status("go_current_dig"),
                    "error": None,
                }
            )

    operations = SshResidentFixedCycleOperations(
        config,
        guided_config=guided,
        processes=Processes(),
        remote_host=Host(),
    )

    status = operations.start(
        run_id="run-real-config",
        requested_cycles=1,
        first_dig_point_id="dig_01",
    )

    assert status.stage == "go_current_dig"
    assert commands and str(guided.rl_orin_python) in commands[0]


def test_commissioning_owner_command_forwards_flat_exact_authorization(tmp_path):
    config = ResidentFixedCyclePcConfig.load(
        _write_grouped_cycle_config(
            tmp_path,
            commissioning_authorization="ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING",
        )
    )
    commands = []

    class Host:
        def argv(self, command):
            commands.append(command)
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=123",
                _owner_ready_line(config),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    processes._start_owner()

    expected_catalog = (
        Path(__file__).resolve().parents[2]
        / "AiryLidar/mission/config/excavation_dig_point_catalog.v1.json"
    )
    expected_digest = hashlib.sha256(
        expected_catalog.read_bytes()
    ).hexdigest()
    assert "--expected-dig-catalog-sha256" in commands[0]
    assert expected_digest in commands[0]

    assert len(commands) == 1
    assert (
        "--commissioning-authorization "
        "ALLOW_V3A_FIXED_TRAJECTORY_COMMISSIONING"
    ) in commands[0]


def test_classical_tracking_owner_forwards_backend_and_independent_authorization(
    tmp_path: Path,
) -> None:
    config = ResidentFixedCyclePcConfig.load(
        _write_classical_tracking_config(tmp_path)
    )
    commands: list[str] = []

    class Host:
        def argv(self, command):
            commands.append(command)
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=123",
                _owner_ready_line(config),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    processes._start_owner()

    assert len(commands) == 1
    command = commands[0]
    assert (
        "--edge-config "
        "deploy/edge_runtime.resident.cartesian_p.commissioning.json"
    ) in command
    assert (
        "--trajectory-controller-commissioning-authorization "
        "ALLOW_CARTESIAN_P_MACHINE_MOTION"
    ) in command


def test_owner_readiness_rejects_mismatched_fixed_cycle_socket(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))

    class Host:
        def argv(self, command):
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=123",
                "RESIDENT_FIXED_CYCLE_READY "
                "control_socket=/home/jetson16/.local/run/"
                "excavator-resident/fixed-cycle.sock "
                    "act_socket=/home/jetson16/.local/run/"
                    "excavator-resident-v3a/act.sock "
                    "trajectory_controller_backend=onnx_rl "
                    "mission_id=fixed_target_hybrid "
                    f"mission_sha256={config.expected_mission_sha256} "
                    "act_worker_required=true "
                    "act_worker_behavior_id=act_dig_lift "
                    "act_worker_model_sha256="
                    f"{config.expected_act_model_sha256}",
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    with pytest.raises(RuntimeError, match="control socket"):
        processes._start_owner()


def test_owner_readiness_accepts_real_structured_log_prefix(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))

    class Host:
        def argv(self, command):
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            lines = (
                "RESIDENT_OWNER_PID=123",
                "2026-09-01 15:20:09,708 INFO orin_state_sender: "
                + _owner_ready_line(config),
                "2026-09-01 15:20:09,715 INFO orin_state_sender: "
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            )
            for line in lines:
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    processes._start_owner()

    assert processes._owner_act_worker_required is True


def test_owner_readiness_rejects_arbitrary_text_before_protocol(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))

    with pytest.raises(RuntimeError, match="readiness prefix"):
        parse_owner_readiness("not-a-log-prefix " + _owner_ready_line(config))


@pytest.mark.parametrize(
    ("ready_overrides", "message"),
    [
        ({"mission_id": "different_mission"}, "mission_id"),
        ({"mission_sha256": "f" * 64}, "mission_sha256"),
        ({"act_worker_required": False}, "act_worker_required"),
        ({"act_worker_behavior_id": "act_dig_transport_dump"}, "behavior"),
        ({"act_worker_model_sha256": "f" * 64}, "model"),
    ],
)
def test_owner_readiness_must_match_expected_mission_and_worker_contract(
    tmp_path, ready_overrides, message
):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))

    class Host:
        def argv(self, command):
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            values = {
                "mission_id": config.expected_mission_id,
                "mission_sha256": config.expected_mission_sha256,
                "act_worker_required": config.expected_act_worker_required,
                "act_worker_behavior_id": config.expected_act_behavior_id,
                "act_worker_model_sha256": config.expected_act_model_sha256,
            }
            values.update(ready_overrides)
            if values["act_worker_required"] is False:
                values["act_worker_behavior_id"] = "none"
                values["act_worker_model_sha256"] = "none"
            lines = (
                "RESIDENT_OWNER_PID=123",
                _owner_ready_line(
                    config,
                    mission_id=values["mission_id"],
                    mission_sha256=values["mission_sha256"],
                    act_worker_required=values["act_worker_required"],
                    act_worker_behavior_id=values["act_worker_behavior_id"],
                    act_worker_model_sha256=values["act_worker_model_sha256"],
                ),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            )
            for line in lines:
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    with pytest.raises(RuntimeError, match=message):
        processes._start_owner()


@pytest.mark.parametrize(
    "line",
    [
        "RESIDENT_FIXED_CYCLE_READY control_socket=/run/owner.sock",
        (
            "RESIDENT_FIXED_CYCLE_READY control_socket=/run/owner.sock "
            "act_socket=/run/act.sock trajectory_controller_backend=onnx_rl "
            "mission_id=mission/unsafe act_worker_required=true"
        ),
        (
            "RESIDENT_FIXED_CYCLE_READY control_socket=/run/owner.sock "
            "act_socket=/run/act.sock trajectory_controller_backend=onnx_rl "
            "mission_id=safe_mission act_worker_required=True"
        ),
    ],
)
def test_owner_readiness_contract_rejects_missing_or_unsafe_fields(line):
    with pytest.raises(RuntimeError, match="readiness"):
        parse_owner_readiness(line)


def test_process_start_uses_owner_worker_requirement_not_mission_name(tmp_path):
    config = ResidentFixedCyclePcConfig.load(
        _write_config(
            tmp_path,
            expected_mission_id="custom_declarative_mission",
            expected_act_worker_required=True,
        )
    )
    spawned = []

    class Host:
        def argv(self, command):
            return ["ssh", command]

        def stop_owned_process(self, **_kwargs):
            return None

    class Process:
        returncode = None

        def __init__(self, lines):
            self.lines = lines

        def wait_for(self, predicate, _timeout):
            for line in self.lines:
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

        def wait(self, *, timeout_s):
            assert timeout_s == 10.0

    def factory(*_args, **_kwargs):
        process = Process(
            (
                "RESIDENT_OWNER_PID=321",
                _owner_ready_line(config),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            )
            if not spawned
            else ("RESIDENT_ACT_PID=654", "ACT resident worker ready: connected")
        )
        spawned.append(process)
        return process

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
    )

    processes.start()

    assert len(spawned) == 2


def test_process_start_skips_act_worker_when_owner_contract_says_not_required(
    tmp_path,
):
    config = ResidentFixedCyclePcConfig.load(_write_fixed_dig_config(tmp_path))
    spawned = []

    class Host:
        def argv(self, command):
            return ["ssh", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=321",
                _owner_ready_line(config),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    def factory(*_args, **_kwargs):
        process = Process()
        spawned.append(process)
        return process

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
    )

    processes.start()

    assert len(spawned) == 1
    processes.require_running()


def test_remote_process_lifecycle_starts_once_and_releases_owned_devices(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    spawned = []
    stopped = []

    class Host:
        def argv(self, command):
            return ["ssh", "orin", command]

        def stop_owned_process(self, **kwargs):
            stopped.append(kwargs)

    class Process:
        def __init__(self, lines):
            self.lines = lines
            self.returncode = None

        def wait_for(self, predicate, _timeout):
            for line in self.lines:
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

        def wait(self, *, timeout_s):
            assert timeout_s == 10.0
            self.returncode = 0

    def factory(argv, **kwargs):
        index = len(spawned)
        lines = (
            [
                "RESIDENT_OWNER_PID=321",
                _owner_ready_line(config),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ]
            if index == 0
            else ["RESIDENT_ACT_PID=654", "ACT resident worker ready: connected"]
        )
        spawned.append((argv, kwargs))
        return Process(lines)

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
        timestamp="20260825_120000",
    )

    processes.start()
    processes.start()
    processes.stop()

    assert len(spawned) == 2
    assert spawned[0][1]["prefix"] == "v3a-owner"
    assert spawned[1][1]["prefix"] == "v3a-act"
    assert len(stopped) == 2
    assert stopped[0]["serial_path"] == Path("/dev/ttyTHS1")
    assert str(stopped[1]["serial_path"]) == "/dev/video0"


def test_act_reference_act_worker_receives_explicit_parallel_assets(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_act_reference_config(tmp_path))
    commands = []

    class Host:
        def argv(self, command):
            commands.append(command)
            return ["ssh", "orin", command]

    class Process:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_ACT_PID=654",
                "ACT resident worker ready: connected",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError("readiness predicate did not match")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=lambda *_args, **_kwargs: Process(),
    )

    processes._start_act_worker()

    command = commands[0]
    assert "ACT_RUNTIME_CONFIG_PATH=" in command
    assert "ACT_CHECKPOINT_HOST_PATH=" in command
    assert "ACT_DEPLOYMENT_HOST_PATH=" in command


def test_operations_reject_owner_plan_with_wrong_mission_id(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_act_reference_config(tmp_path))

    class Processes:
        def start(self):
            return None

    class Host:
        def run(self, _command, *, accepted_returncodes=(0,)):
            return json.dumps(
                {
                    "schema_version": "resident_fixed_cycle_control.v4",
                    "ok": True,
                    "command": "start",
                    "status": _status(
                        "go_current_dig", mission_id="fixed_target_hybrid"
                    ),
                    "error": None,
                }
            )

    operations = SshResidentFixedCycleOperations(
        config,
        guided_config=_guided(),
        processes=Processes(),
        remote_host=Host(),
    )

    with pytest.raises(RuntimeError, match="mission_id"):
        operations.start(
            run_id="run-profile-mismatch",
            requested_cycles=1,
            first_dig_point_id="dig_01",
        )
def test_process_start_failure_cleans_owner_and_reports_cleanup_error(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    messages = []

    class Host:
        def argv(self, command):
            return [command]

        def stop_owned_process(self, **_kwargs):
            raise OSError("cleanup failed")

    class Owner:
        returncode = None

        def wait_for(self, predicate, _timeout):
            for line in (
                "RESIDENT_OWNER_PID=321",
                _owner_ready_line(config),
                "RESIDENT_HARDWARE_READY sensor_valid=True",
            ):
                if predicate(line):
                    return 0, line
            raise AssertionError

    calls = 0

    def factory(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Owner()
        raise RuntimeError("ACT failed to launch")

    processes = ResidentFixedCycleProcesses(
        config,
        guided_config=_guided(),
        remote_host=Host(),
        line_process_factory=factory,
        output=messages.append,
    )

    with pytest.raises(RuntimeError, match="ACT failed"):
        processes.start()

    assert any("startup cleanup failed" in message for message in messages)


def test_process_require_running_and_wait_fail_closed(tmp_path):
    config = ResidentFixedCyclePcConfig.load(_write_config(tmp_path))
    processes = ResidentFixedCycleProcesses(config, guided_config=_guided())
    with pytest.raises(RuntimeError, match="not started"):
        processes.require_running()

    class Failed:
        returncode = 3

        def wait(self, *, timeout_s):
            assert timeout_s == 10.0

    with pytest.raises(RuntimeError, match="return code 3"):
        resident_module._wait_process(Failed(), allow_nonzero=False)
    resident_module._wait_process(Failed(), allow_nonzero=True)


def test_supervisor_maps_complete_local_cycle_to_existing_ui_snapshot():
    operations = _ScriptedOperations(
        [
            _remote_status(
                "go_current_dig", "onnx_rl_tracking", completed=0, group="near"
            ),
            _remote_status("dig", "act_dig_lift", completed=0, group="near"),
            _remote_status(
                "carry_to_dump",
                "onnx_rl_tracking",
                completed=0,
                group="near",
                target_id="dump",
            ),
            _remote_status("dump", "fixed_dump", completed=0, group="near"),
            _remote_status(
                "return_to_dig", "onnx_rl_tracking", completed=2, group="near"
            ),
            _remote_status(
                "COMPLETED", "",
                completed=2,
                terminal=True,
                outcome="SUCCEEDED",
                group="near",
            ),
        ]
    )
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("near_01", "near_02", "far_01", "far_02"),
        dig_groups={
            "all": ("near_01", "near_02", "far_01", "far_02"),
            "near": ("near_01", "near_02"),
            "far": ("far_01", "far_02"),
        },
        default_dig_group_id="all",
        poll_interval_s=0.02,
    )

    supervisor.start(
        "near_02",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=2,
        dig_group_id="near",
    )
    deadline = time.monotonic() + 1.0
    while supervisor.snapshot().stage != "completed" and time.monotonic() < deadline:
        time.sleep(0.005)

    snapshot = supervisor.snapshot()
    assert snapshot.stage == "completed"
    assert snapshot.run_completed_cycles == 2
    assert snapshot.requested_cycles == 2
    assert snapshot.dig_target_id == "dig_02"
    assert snapshot.can_stop is False
    assert snapshot.dig_group_id == "near"
    assert operations.started == [(snapshot.run_id, 2, "near_02", "near")]
    assert operations.released == [True]


def test_supervisor_records_v3a_status_and_finalizes_experiment_evidence():
    operations = _ScriptedOperations(
        [
            _remote_status("go_current_dig", "onnx_rl_tracking", completed=0),
            _remote_status("dig", "act_dig_lift", completed=0),
            _remote_status(
                "COMPLETED", "", completed=1, terminal=True, outcome="SUCCEEDED"
            ),
        ]
    )

    class Evidence:
        run_id = "v3a_evidence_001"

        def __init__(self):
            self.events = []
            self.finalizations = []

        def append_event(self, event_type, payload):
            self.events.append((event_type, dict(payload)))

        def finalize(self, status, *, metrics, summary):
            self.finalizations.append((status, dict(metrics), summary))

    evidence = Evidence()
    requests = []
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
        config_path=Path("/configs/resident-fixed.json"),
        evidence_run_factory=lambda request: requests.append(request) or evidence,
    )

    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=1,
    )
    deadline = time.monotonic() + 1.0
    while supervisor.snapshot().stage != "completed" and time.monotonic() < deadline:
        time.sleep(0.005)

    assert supervisor.snapshot().run_id == "v3a_evidence_001"
    assert requests[0].config_path == Path("/configs/resident-fixed.json")
    assert any(event == "resident_fixed_cycle_status" for event, _ in evidence.events)
    status_events = [
        payload
        for event, payload in evidence.events
        if event == "resident_fixed_cycle_status"
    ]
    assert all(
        payload["mission_id"] == "fixed_target_hybrid"
        for payload in status_events
    )
    assert evidence.finalizations[0][0] == "success"
    assert evidence.finalizations[0][1]["completed_cycles"] == 1


def test_initial_v3a_evidence_failure_prevents_remote_motion_start():
    operations = _ScriptedOperations([])

    class Evidence:
        run_id = "v3a_evidence_failed"

        def __init__(self):
            self.finalizations = []

        def append_event(self, _event_type, _payload):
            raise OSError("evidence disk unavailable")

        def finalize(self, status, *, metrics, summary):
            self.finalizations.append((status, dict(metrics), summary))

    evidence = Evidence()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
        config_path=Path("/configs/resident-fixed.json"),
        evidence_run_factory=lambda _request: evidence,
    )

    with pytest.raises(RuntimeError, match="initial V3-A evidence"):
        supervisor.start(
            "dig_01",
            automatic=True,
            motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
            cycle_count=1,
        )

    assert operations.started == []
    assert evidence.finalizations[0][0] == "failure"


def test_supervisor_rejects_segmented_v3_and_cancel_is_immediate():
    operations = _BlockingOperations()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )
    with pytest.raises(ValueError, match="automatic"):
        supervisor.start(
            "dig_01", automatic=False, motion_authorization=None, cycle_count=1
        )

    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=1,
    )
    assert operations.entered.wait(1.0)
    supervisor.stop()
    operations.cancelled.wait(1.0)
    deadline = time.monotonic() + 1.0
    while supervisor.snapshot().stage != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert supervisor.snapshot().stage == "cancelled"

    with pytest.raises(RuntimeError, match="not support segmented"):
        supervisor.advance(motion_authorization=None)


def test_cancel_transport_failure_stops_heartbeats_and_forces_owner_release():
    operations = _CancelFailureOperations()
    supervisor = ResidentFixedCycleSupervisor(
        operations=operations,
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )
    supervisor.start(
        "dig_01",
        automatic=True,
        motion_authorization=REQUIRED_HYBRID_MOTION_AUTHORIZATION,
        cycle_count=1,
    )
    assert operations.entered.wait(1.0)

    supervisor.stop()
    assert operations.released.wait(1.0)

    snapshot = supervisor.snapshot()
    assert snapshot.stage == "failed"
    assert "cancel was not acknowledged" in snapshot.error
    assert operations.status_calls == 0
    assert operations.release_terminal_disarmed is False


def test_supervisor_maps_local_dump_and_return_phases_to_existing_ui_contract():
    supervisor = ResidentFixedCycleSupervisor(
        operations=_BlockingOperations(),
        dig_target_ids=("dig_01", "dig_02", "dig_03"),
        poll_interval_s=0.02,
    )

    supervisor._apply_status(
        _remote_status(
            "carry_to_dump", "onnx_rl_tracking", completed=0, target_id="dump"
        )
    )
    assert supervisor.snapshot().stage == "running_rl_to_dump_and_dump"

    supervisor._apply_status(
        _remote_status("return_to_dig", "onnx_rl_tracking", completed=1)
    )
    assert supervisor.snapshot().stage == "running_rl_return_to_dig"

    supervisor._apply_status(
        _remote_status("return_to_dig", "onnx_rl_tracking", completed=2)
    )
    assert supervisor.snapshot().stage == "running_rl_return_to_dig"
    assert supervisor.snapshot().run_completed_cycles == 2

    supervisor.append_external_log("[resident-owner] hardware ready")
    assert supervisor.snapshot().logs[-1] == "[resident-owner] hardware ready"

    stage_before_clear = supervisor.snapshot().stage
    supervisor.clear_logs()
    assert supervisor.snapshot().logs == ()
    assert supervisor.snapshot().stage == stage_before_clear


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "invalid JSON"),
        (json.dumps({"schema_version": "wrong"}), "fields"),
        (
            json.dumps(
                {
                    "schema_version": "resident_fixed_cycle_control.v4",
                    "ok": False,
                    "command": "status",
                    "status": None,
                    "error": {"code": "failed", "message": "failed"},
                }
            ),
            "rejected",
        ),
    ],
)
def test_control_response_parser_rejects_untrusted_payload(payload, message):
    with pytest.raises(RuntimeError, match=message):
        resident_module._parse_control_response(payload, "status")


def test_readiness_pid_parser_rejects_malformed_line():
    with pytest.raises(RuntimeError, match="invalid RESIDENT_OWNER_PID"):
        resident_module._remote_pid("RESIDENT_OWNER_PID=0", "RESIDENT_OWNER_PID")


def _guided():
    class Guided:
        orin_ssh_host = "jetson16@192.168.50.2"
        rl_orin_repo = Path("/home/jetson16/workspace_excavator/excavator-orin-runtime")
        orin_repo = Path("/home/jetson16/workspace_excavator/excavator-il")
        rl_orin_python = "/opt/orin/bin/python"
        rl_pc_host = "192.168.50.1"
        rl_serial_port = Path("/dev/ttyTHS1")
        rl_serial_release_timeout_s = 8.0
        log_dir = Path("/tmp")
        rl_airy_repo = Path(__file__).resolve().parents[2] / "AiryLidar"

    return Guided()


def _owner_ready_line(
    config,
    *,
    mission_id=None,
    mission_sha256=None,
    act_worker_required=None,
    act_worker_behavior_id=None,
    act_worker_model_sha256=None,
):
    mission_id = mission_id or config.expected_mission_id
    mission_sha256 = mission_sha256 or config.expected_mission_sha256
    if act_worker_required is None:
        act_worker_required = config.expected_act_worker_required
    if act_worker_behavior_id is None:
        act_worker_behavior_id = config.expected_act_behavior_id or "none"
    if act_worker_model_sha256 is None:
        act_worker_model_sha256 = config.expected_act_model_sha256 or "none"
    return (
        "RESIDENT_FIXED_CYCLE_READY "
        f"control_socket={config.control_socket} "
        f"act_socket={config.runtime_root / 'act.sock'} "
        "trajectory_controller_backend="
        f"{config.trajectory_controller_backend} "
        f"mission_id={mission_id} "
        f"mission_sha256={mission_sha256} "
        "act_worker_required="
        f"{str(act_worker_required).lower()} "
        f"act_worker_behavior_id={act_worker_behavior_id} "
        f"act_worker_model_sha256={act_worker_model_sha256}"
    )


def _status(
    stage,
    *,
    behavior_id="onnx_rl_tracking",
    mission_id="fixed_target_hybrid",
    terminal=False,
    outcome="",
    active_trajectory=None,
):
    return {
        "run_id": "run-001",
        "mission_id": mission_id,
        "active_behavior_id": "" if terminal else behavior_id,
        "stage": stage,
        "requested_cycles": 3,
        "completed_cycles": 0,
        "current_dig_point_id": "dig_02",
        "dig_group_id": "all",
        "terminal": terminal,
        "outcome": outcome,
        "reason_code": "",
        "active_trajectory": active_trajectory,
    }


def _trajectory():
    return {
        "frame_id": "machine_root_ros",
        "target_id": "dig_02",
        "waypoints": [[0.2, 0.0, 0.1], [0.3, 0.1, 0.1], [0.4, 0.2, 0.1]],
        "current_waypoint_index": 1,
        "waypoint_tolerance_m": 0.4,
    }


def _remote_status(
    stage,
    behavior_id,
    *,
    completed,
    terminal=False,
    outcome="",
    group="all",
    target_id=None,
):
    return ResidentFixedCycleRemoteStatus(
        run_id="run-local",
        mission_id="fixed_target_hybrid",
        active_behavior_id=behavior_id,
        stage=stage,
        requested_cycles=2,
        completed_cycles=completed,
        current_dig_point_id="dig_02",
        dig_group_id=group,
        terminal=terminal,
        outcome=outcome,
        reason_code="",
        active_trajectory=(
            None
            if target_id is None
            else ResidentFixedCycleRemoteStatus.from_mapping(
                _status(
                    stage,
                    behavior_id=behavior_id,
                    active_trajectory={**_trajectory(), "target_id": target_id},
                )
            ).active_trajectory
        ),
    )


class _ScriptedOperations:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.started = []
        self.released = []

    def start(
        self,
        *,
        run_id,
        requested_cycles,
        first_dig_point_id,
        dig_group_id="all",
    ):
        self.started.append(
            (run_id, requested_cycles, first_dig_point_id, dig_group_id)
        )
        return self.statuses.pop(0)

    def status(self):
        return self.statuses.pop(0)

    def cancel(self):
        raise AssertionError("cancel is not expected")

    def release(self, *, terminal_disarmed):
        self.released.append(terminal_disarmed)


class _BlockingOperations:
    def __init__(self):
        self.entered = threading.Event()
        self.cancelled = threading.Event()

    def start(self, **_kwargs):
        self.entered.set()
        return _remote_status("go_current_dig", "onnx_rl_tracking", completed=0)

    def status(self):
        if self.cancelled.is_set():
            return _remote_status(
                "CANCELLED", "", completed=0, terminal=True, outcome="CANCELLED"
            )
        return _remote_status(
            "go_current_dig", "onnx_rl_tracking", completed=0
        )

    def cancel(self):
        self.cancelled.set()
        return _remote_status(
            "CANCELLED", "", completed=0, terminal=True, outcome="CANCELLED"
        )

    def release(self, *, terminal_disarmed):
        assert terminal_disarmed is True


class _CancelFailureOperations:
    def __init__(self):
        self.entered = threading.Event()
        self.released = threading.Event()
        self.status_calls = 0
        self.release_terminal_disarmed = None

    def start(self, **_kwargs):
        self.entered.set()
        return _remote_status("go_current_dig", "onnx_rl_tracking", completed=0)

    def status(self):
        self.status_calls += 1
        return _remote_status("go_current_dig", "onnx_rl_tracking", completed=0)

    def cancel(self):
        raise OSError("control socket unavailable")

    def release(self, *, terminal_disarmed):
        self.release_terminal_disarmed = terminal_disarmed
        self.released.set()
