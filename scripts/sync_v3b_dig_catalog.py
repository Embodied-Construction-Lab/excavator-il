#!/usr/bin/env python3
"""Validate, build, and synchronize the V3-B dig-point catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
CATALOG_PATH = (
    WORKSPACE
    / "AiryLidar/mission/config/excavation_dig_point_catalog.v1.json"
)
DEFAULT_ORIN_HOST = "jetson16@192.168.50.2"
LOCAL_ORIN_REPOSITORY = WORKSPACE / "excavator-orin-runtime"
REMOTE_ORIN_REPOSITORY = (
    "/home/jetson16/workspace_excavator/excavator-orin-runtime"
)
MISSION_PATH = WORKSPACE / "AiryLidar/mission/config/excavation_cycle.json"
BUILDER = LOCAL_ORIN_REPOSITORY / "scripts/build_v3a_fixed_cycle_candidate.py"
DEPLOYMENTS = (
    (
        "fixed_target_hybrid",
        Path("deploy/missions/fixed_target_hybrid.json"),
        Path("deploy/v3b/catalog/candidate"),
    ),
    (
        "classical_tracking_hybrid",
        Path("deploy/missions/classical_tracking_hybrid.json"),
        Path("deploy/v3b/classical-tracking/catalog/candidate"),
    ),
    (
        "fixed_dig_hybrid",
        Path("deploy/missions/fixed_dig_hybrid.json"),
        Path("deploy/v3b/fixed-dig/catalog/candidate"),
    ),
    (
        "engineering_act_transport_reference",
        Path("deploy/missions/engineering_act_transport_reference.json"),
        Path("deploy/v3b/act-dig-transport-dump-reference/catalog/candidate"),
    ),
)


class CatalogSyncError(RuntimeError):
    """A safe, user-facing catalog synchronization failure."""

sys.path.insert(0, str(REPOSITORY / "src"))

from excavator_il.dig_point_catalog import load_dig_point_catalog


def _catalog_report() -> dict[str, object]:
    source_bytes = CATALOG_PATH.read_bytes()
    source = json.loads(source_bytes)
    load_dig_point_catalog(CATALOG_PATH)
    return {
        "catalog_path": str(CATALOG_PATH),
        "default_group": source["default_dig_group"],
        "groups": source["dig_groups"],
        "point_count": len(source["dig_points"]),
        "points": source["dig_points"],
        "source_catalog_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _require_idle_orin(orin_host: str) -> None:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            orin_host,
            (
                "fuser -v /dev/ttyTHS1 /dev/video0 /dev/video2 2>&1 || true; "
                "pgrep -af '[p]ython.*([o]rin_state_sender|"
                "[r]esident_act_runtime)' "
                "|| true"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CatalogSyncError("could not inspect Orin hardware resources")
    owners = result.stdout.strip()
    if owners:
        raise CatalogSyncError(
            f"Orin hardware resources are active; stop them first:\n{owners}"
        )


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise CatalogSyncError(
            f"command failed ({result.returncode}): {command[0]}: {detail}"
        )
    return result


def _build_deployments(staging_root: Path) -> dict[str, Path]:
    generated: dict[str, Path] = {}
    for mission_id, mission_definition, relative_destination in DEPLOYMENTS:
        output_dir = staging_root / mission_id
        deployed_root = f"{REMOTE_ORIN_REPOSITORY}/{relative_destination}"
        _run(
            [
                sys.executable,
                str(BUILDER),
                "--mission-config",
                str(MISSION_PATH),
                "--mission-definition",
                str(LOCAL_ORIN_REPOSITORY / mission_definition),
                "--dig-point-catalog",
                str(CATALOG_PATH),
                "--output-dir",
                str(output_dir),
                "--deployed-root",
                deployed_root,
                "--intermediate-waypoint-tolerance-m",
                "0.40",
            ],
            cwd=LOCAL_ORIN_REPOSITORY,
        )
        generated[mission_id] = output_dir
    return generated


def _verify_deployments(
    generated: dict[str, Path], report: dict[str, object]
) -> dict[str, str]:
    artifact_hashes: dict[str, str] = {}
    for mission_id, _definition, _destination in DEPLOYMENTS:
        directory = generated[mission_id]
        files = sorted(path.name for path in directory.iterdir())
        expected_files = [
            "fixed_cycle.candidate.json",
            "target_catalog.candidate.json",
        ]
        if files != expected_files:
            raise CatalogSyncError(
                f"unexpected {mission_id} deployment files: {files}"
            )
        plan = json.loads((directory / expected_files[0]).read_text())
        catalog_path = directory / expected_files[1]
        catalog = json.loads(catalog_path.read_text())
        artifact_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        if plan.get("source_catalog_sha256") != report["source_catalog_sha256"]:
            raise CatalogSyncError(f"{mission_id} source catalog hash mismatch")
        if plan.get("dig_sequence") != list(report["points"]):
            raise CatalogSyncError(f"{mission_id} point order mismatch")
        if plan.get("dig_groups") != report["groups"]:
            raise CatalogSyncError(f"{mission_id} group mismatch")
        if catalog.get("dig_points") != report["points"]:
            raise CatalogSyncError(f"{mission_id} point coordinates mismatch")
        if plan.get("target_catalog", {}).get("sha256") != artifact_sha:
            raise CatalogSyncError(f"{mission_id} artifact hash mismatch")
        if plan.get("mission", {}).get("mission_id") != mission_id:
            raise CatalogSyncError(f"{mission_id} Mission identity mismatch")
        mission_sha256 = plan.get("mission_sha256")
        if catalog.get("mission_id") != mission_id:
            raise CatalogSyncError(f"{mission_id} catalog Mission identity mismatch")
        if catalog.get("mission_sha256") != mission_sha256:
            raise CatalogSyncError(f"{mission_id} catalog Mission digest mismatch")
        artifact_hashes[mission_id] = artifact_sha
    return artifact_hashes


def _remote_backup(orin_host: str) -> str:
    backup = _run(["ssh", orin_host, "mktemp -d /tmp/v3b-catalog-sync.XXXXXX"])
    backup_path = backup.stdout.strip()
    if not backup_path.startswith("/tmp/v3b-catalog-sync."):
        raise CatalogSyncError("Orin returned an invalid backup path")
    commands = ["set -e"]
    for mission_id, _definition, relative_destination in DEPLOYMENTS:
        source = f"{REMOTE_ORIN_REPOSITORY}/{relative_destination}"
        commands.append(
            f"if [ -d '{source}' ]; then cp -a '{source}' "
            f"'{backup_path}/{mission_id}'; fi"
        )
        commands.append(f"mkdir -p '{source}'")
    _run(["ssh", orin_host, "; ".join(commands)])
    return backup_path


def _upload_deployments(orin_host: str, generated: dict[str, Path]) -> None:
    for mission_id, _definition, relative_destination in DEPLOYMENTS:
        remote = f"{REMOTE_ORIN_REPOSITORY}/{relative_destination}/"
        _run(
            [
                "rsync",
                "-a",
                "--delete",
                f"{generated[mission_id]}/",
                f"{orin_host}:{remote}",
            ]
        )


def _verify_remote(
    orin_host: str,
    report: dict[str, object],
    artifact_hashes: dict[str, str],
) -> None:
    expected = json.dumps(
        {
            "source_sha": report["source_catalog_sha256"],
            "points": report["points"],
            "groups": report["groups"],
            "artifact_hashes": artifact_hashes,
            "deployments": {
                mission_id: str(relative_destination)
                for mission_id, _definition, relative_destination in DEPLOYMENTS
            },
        },
        separators=(",", ":"),
    )
    verifier = (
        "import hashlib,json,sys; from pathlib import Path\n"
        "root=Path(sys.argv[1]); expected=json.loads(sys.argv[2])\n"
        "for mission_id,relative in expected[\"deployments\"].items():\n"
        " directory=root/relative\n"
        " p=json.loads((directory/\"fixed_cycle.candidate.json\").read_text()); "
        "cpath=directory/\"target_catalog.candidate.json\"; "
        "c=json.loads(cpath.read_text()); "
        "sha=hashlib.sha256(cpath.read_bytes()).hexdigest(); "
        "assert p[\"source_catalog_sha256\"]==expected[\"source_sha\"]; "
        "assert p[\"dig_sequence\"]==list(expected[\"points\"]); "
        "assert p[\"dig_groups\"]==expected[\"groups\"]; "
        "assert c[\"dig_points\"]==expected[\"points\"]; "
        "assert p[\"mission\"][\"mission_id\"]==mission_id; "
        "assert c[\"mission_id\"]==mission_id; "
        "assert c[\"mission_sha256\"]==p[\"mission_sha256\"]; "
        "assert sha==expected[\"artifact_hashes\"][mission_id]\n"
    )
    remote_command = " ".join(
        shlex.quote(value)
        for value in (
            "python3",
            "-c",
            verifier,
            REMOTE_ORIN_REPOSITORY,
            expected,
        )
    )
    _run(["ssh", orin_host, remote_command])


def _publish_local(generated: dict[str, Path]) -> str:
    backup_root = Path(tempfile.mkdtemp(prefix="v3b-catalog-local-backup-"))
    for mission_id, _definition, relative_destination in DEPLOYMENTS:
        destination = LOCAL_ORIN_REPOSITORY / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.move(str(destination), str(backup_root / mission_id))
        shutil.move(str(generated[mission_id]), str(destination))
    return str(backup_root)


def _synchronize(orin_host: str, report: dict[str, object]) -> dict[str, object]:
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".catalog-sync-",
            dir=LOCAL_ORIN_REPOSITORY / "deploy/v3b",
        )
    )
    try:
        generated = _build_deployments(staging_root)
        artifact_hashes = _verify_deployments(generated, report)
        remote_backup = _remote_backup(orin_host)
        _upload_deployments(orin_host, generated)
        _verify_remote(orin_host, report, artifact_hashes)
        local_backup = _publish_local(generated)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return {
        **report,
        "artifact_sha256": artifact_hashes,
        "local_backup": local_backup,
        "remote_backup": remote_backup,
        "status": "synchronized",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report the catalog without writing or connecting",
    )
    mode.add_argument(
        "--check-idle",
        action="store_true",
        help="only verify that Orin motion resources are idle",
    )
    parser.add_argument("--orin-host", default=DEFAULT_ORIN_HOST)
    args = parser.parse_args()
    report = _catalog_report()
    if args.dry_run:
        print(
            json.dumps(
                {**report, "status": "dry_run"},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        _require_idle_orin(args.orin_host)
        if args.check_idle:
            print(
                json.dumps(
                    {"orin_host": args.orin_host, "status": "orin_idle"},
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        result = _synchronize(args.orin_host, report)
    except CatalogSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
