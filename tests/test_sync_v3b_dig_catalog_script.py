import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
SCRIPT = REPOSITORY / "scripts/sync_v3b_dig_catalog.py"
CATALOG = (
    WORKSPACE
    / "AiryLidar/mission/config/excavation_dig_point_catalog.v1.json"
)


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_v3b_dig_catalog", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_reports_the_authoritative_catalog_without_remote_access(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "remote-command-was-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for command in ("ssh", "rsync"):
        executable = fake_bin / command
        executable.write_text(
            f"#!/bin/sh\ntouch '{marker}'\nexit 97\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        cwd=REPOSITORY,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    source_bytes = CATALOG.read_bytes()
    source = json.loads(source_bytes)
    assert report == {
        "catalog_path": str(CATALOG),
        "default_group": source["default_dig_group"],
        "groups": source["dig_groups"],
        "point_count": len(source["dig_points"]),
        "points": source["dig_points"],
        "source_catalog_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "status": "dry_run",
    }
    assert not marker.exists()


def test_sync_refuses_to_build_or_copy_while_orin_resources_are_active(
    tmp_path: Path,
) -> None:
    rsync_marker = tmp_path / "rsync-was-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        "echo '901 python3 orin_state_sender.py --serial-port /dev/ttyTHS1'\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    rsync = fake_bin / "rsync"
    rsync.write_text(
        f"#!/bin/sh\ntouch '{rsync_marker}'\nexit 0\n",
        encoding="utf-8",
    )
    rsync.chmod(0o755)

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPOSITORY,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Orin hardware resources are active" in result.stderr
    assert not rsync_marker.exists()


def test_idle_check_does_not_match_its_own_remote_shell_command(
    tmp_path: Path,
) -> None:
    rsync_marker = tmp_path / "rsync-was-called"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *\"pgrep -af 'python.*\"*)\n"
        "    echo \"777 zsh -c $*\"\n"
        "    ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    rsync = fake_bin / "rsync"
    rsync.write_text(
        f"#!/bin/sh\ntouch '{rsync_marker}'\nexit 0\n",
        encoding="utf-8",
    )
    rsync.chmod(0o755)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-idle"],
        cwd=REPOSITORY,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "orin_idle"
    assert not rsync_marker.exists()


def test_builder_regenerates_every_declarative_mission_from_one_catalog(
    tmp_path: Path,
) -> None:
    sync = _load_sync_module()
    report = sync._catalog_report()

    generated = sync._build_deployments(tmp_path)
    artifact_hashes = sync._verify_deployments(generated, report)

    expected_missions = {
        "fixed_target_hybrid",
        "classical_tracking_hybrid",
        "fixed_dig_hybrid",
        "engineering_act_transport_reference",
    }
    assert set(generated) == expected_missions
    assert set(artifact_hashes) == expected_missions
    for mission_id, directory in generated.items():
        plan = json.loads(
            (directory / "fixed_cycle.candidate.json").read_text(
                encoding="utf-8"
            )
        )
        target_catalog = json.loads(
            (directory / "target_catalog.candidate.json").read_text(
                encoding="utf-8"
            )
        )
        assert plan["schema_version"] == "resident_fixed_cycle_plan.v5"
        assert plan["mission"]["mission_id"] == mission_id
        assert target_catalog["waypoint_tolerance_m"] == 0.25
        assert target_catalog["intermediate_waypoint_tolerance_m"] == 0.40


def test_remote_verifier_covers_every_declarative_mission(monkeypatch) -> None:
    sync = _load_sync_module()
    report = sync._catalog_report()
    mission_ids = {
        mission_id for mission_id, _definition, _destination in sync.DEPLOYMENTS
    }
    commands = []

    def capture(command, *, cwd=None):
        commands.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(sync, "_run", capture)
    sync._verify_remote(
        "orin-under-test",
        report,
        {mission_id: "a" * 64 for mission_id in mission_ids},
    )

    assert len(commands) == 1
    command, cwd = commands[0]
    assert cwd is None
    assert command[:2] == ["ssh", "orin-under-test"]
    remote_argv = shlex.split(command[2])
    assert remote_argv[:2] == ["python3", "-c"]
    compile(remote_argv[2], "<remote-deployment-verifier>", "exec")
    expected = json.loads(remote_argv[4])
    assert set(expected["deployments"]) == mission_ids
    assert set(expected["artifact_hashes"]) == mission_ids
