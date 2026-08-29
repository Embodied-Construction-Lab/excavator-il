import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
SCRIPT = REPOSITORY / "scripts/sync_v3b_dig_catalog.py"
CATALOG = (
    WORKSPACE
    / "AiryLidar/mission/config/excavation_dig_point_catalog.v1.json"
)


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
