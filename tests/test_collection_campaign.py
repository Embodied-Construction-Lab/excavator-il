import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from excavator_il.collection_campaign import (
    build_default_campaign,
    inspect_collection_campaign,
)


def _write_episode(
    root: Path,
    episode_id: str,
    *,
    task_variant: str = "dig_only",
    soil_reset_block_id: str = "block_01",
    dig_point_id: str = "dig_01",
    status: str = "complete",
    success: bool = True,
    recording_purpose: str = "demonstration",
) -> Path:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode.json").write_text(
        json.dumps(
            {
                "schema_version": "excavator_demo_raw.v2",
                "episode_id": episode_id,
                "status": status,
                "success": success,
                "recording_purpose": recording_purpose,
                "collection_protocol": {
                    "task_variant": task_variant,
                    "soil_reset_block_id": soil_reset_block_id,
                    "dig_point_id": dig_point_id,
                },
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "quality_report.json").write_text(
        json.dumps({"episode_id": episode_id, "passed": success}),
        encoding="utf-8",
    )
    return episode_dir


def test_diagnostic_episode_is_ignored_without_polluting_campaign_counts(
    tmp_path: Path,
) -> None:
    episode = _write_episode(
        tmp_path,
        "episode_0001",
        recording_purpose="diagnostic",
        status="aborted",
        success=False,
    )
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("collection_protocol")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = inspect_collection_campaign(tmp_path)

    assert report["summary"]["ignored_diagnostics"] == 1
    assert report["summary"]["malformed"] == 0
    assert report["summary"]["failed"] == 0
    assert report["ignored_diagnostics"] == [
        {
            "episode_id": "episode_0001",
            "episode_path": str(episode.resolve()),
            "recording_purpose": "diagnostic",
            "status": "aborted",
            "success": False,
        }
    ]


def test_default_campaign_is_the_deterministic_200_episode_protocol() -> None:
    campaign = build_default_campaign()

    assert set(campaign) == {
        "schema_version",
        "planned_episode_count",
        "slots",
    }
    assert campaign["schema_version"] == "excavator_collection_campaign.v1"
    assert campaign["planned_episode_count"] == 200
    assert len(campaign["slots"]) == 200
    assert [slot["slot_id"] for slot in campaign["slots"]] == [
        f"slot_{number:03d}" for number in range(1, 201)
    ]

    for block_index in range(1, 21):
        slots = campaign["slots"][(block_index - 1) * 10 : block_index * 10]
        expected_variant = (
            "dig_only" if block_index % 2 else "dig_transport_dump"
        )
        assert {slot["soil_reset_block_id"] for slot in slots} == {
            f"block_{block_index:02d}"
        }
        assert {slot["task_variant"] for slot in slots} == {expected_variant}
        assert [slot["dig_point_id"] for slot in slots] == [
            "dig_01",
            "dig_02",
            "dig_03",
            "dig_01",
            "dig_02",
            "dig_03",
            "dig_01",
            "dig_02",
            "dig_03",
            "dig_01",
        ]

    variants = [slot["task_variant"] for slot in campaign["slots"]]
    assert variants.count("dig_only") == 100
    assert variants.count("dig_transport_dump") == 100


def test_inspection_only_completes_a_slot_for_a_successful_complete_episode(
    tmp_path: Path,
) -> None:
    _write_episode(tmp_path, "episode_0007")

    report = inspect_collection_campaign(tmp_path)

    assert set(report) == {
        "schema_version",
        "raw_root",
        "planned_episode_count",
        "slots",
        "summary",
        "completed",
        "missing",
        "failed",
        "duplicate",
        "unplanned",
        "malformed",
        "ignored_diagnostics",
        "next_expected_slot",
    }
    assert report["schema_version"] == "excavator_collection_campaign.v1"
    assert report["summary"] == {
        "planned": 200,
        "completed": 1,
        "missing": 199,
        "failed": 0,
        "duplicate": 0,
        "unplanned": 0,
        "malformed": 0,
        "ignored_diagnostics": 0,
        "complete_and_valid": False,
    }
    assert report["completed"] == [
        {
            "slot_id": "slot_001",
            "episode_id": "episode_0007",
            "episode_path": str((tmp_path / "episode_0007").resolve()),
        }
    ]
    assert report["missing"][0] == {
        "slot_id": "slot_002",
        "task_variant": "dig_only",
        "soil_reset_block_id": "block_01",
        "dig_point_id": "dig_02",
    }
    assert report["next_expected_slot"] == report["missing"][0]
    assert report["failed"] == []
    assert report["duplicate"] == []
    assert report["unplanned"] == []
    assert report["malformed"] == []
    assert report["ignored_diagnostics"] == []


def test_failed_episode_is_reported_and_does_not_complete_its_slot(
    tmp_path: Path,
) -> None:
    _write_episode(
        tmp_path,
        "episode_0001",
        status="failed",
        success=False,
    )

    report = inspect_collection_campaign(tmp_path)

    assert report["summary"]["completed"] == 0
    assert report["summary"]["missing"] == 200
    assert report["summary"]["failed"] == 1
    assert report["failed"] == [
        {
            "episode_id": "episode_0001",
            "episode_path": str((tmp_path / "episode_0001").resolve()),
            "task_variant": "dig_only",
            "soil_reset_block_id": "block_01",
            "dig_point_id": "dig_01",
            "status": "failed",
            "success": False,
        }
    ]
    assert report["next_expected_slot"]["slot_id"] == "slot_001"


def test_successful_metadata_without_passing_quality_does_not_complete_slot(
    tmp_path: Path,
) -> None:
    episode = _write_episode(tmp_path, "episode_0001")
    (episode / "quality_report.json").write_text(
        json.dumps({"episode_id": episode.name, "passed": False}),
        encoding="utf-8",
    )

    report = inspect_collection_campaign(tmp_path)

    assert report["summary"]["completed"] == 0
    assert report["summary"]["malformed"] == 1
    assert "quality_report.json passed must be true" in report["malformed"][0][
        "reason"
    ]


def test_successes_beyond_matching_capacity_are_duplicates_and_unknown_slots_are_unplanned(
    tmp_path: Path,
) -> None:
    for episode_number in range(1, 6):
        _write_episode(tmp_path, f"episode_{episode_number:04d}")
    _write_episode(
        tmp_path,
        "episode_0006",
        task_variant="dig_transport_dump",
        soil_reset_block_id="block_01",
    )

    report = inspect_collection_campaign(tmp_path)

    assert [item["slot_id"] for item in report["completed"]] == [
        "slot_001",
        "slot_004",
        "slot_007",
        "slot_010",
    ]
    assert report["summary"]["duplicate"] == 1
    assert report["duplicate"][0]["episode_id"] == "episode_0005"
    assert report["summary"]["unplanned"] == 1
    assert report["unplanned"][0]["episode_id"] == "episode_0006"
    assert report["summary"]["complete_and_valid"] is False


def test_inspection_reports_malformed_v1_invalid_json_and_non_exact_protocol(
    tmp_path: Path,
) -> None:
    legacy = _write_episode(tmp_path, "episode_0001")
    legacy_metadata = json.loads((legacy / "episode.json").read_text(encoding="utf-8"))
    legacy_metadata["schema_version"] = "excavator_demo_raw.v1"
    (legacy / "episode.json").write_text(json.dumps(legacy_metadata), encoding="utf-8")

    invalid = tmp_path / "episode_0002"
    invalid.mkdir()
    (invalid / "episode.json").write_text("{broken", encoding="utf-8")

    extra_key = _write_episode(tmp_path, "episode_0003")
    extra_metadata = json.loads(
        (extra_key / "episode.json").read_text(encoding="utf-8")
    )
    extra_metadata["collection_protocol"]["operator_note"] = "unexpected"
    (extra_key / "episode.json").write_text(
        json.dumps(extra_metadata), encoding="utf-8"
    )

    report = inspect_collection_campaign(tmp_path)

    assert report["summary"]["malformed"] == 3
    assert [item["episode_id"] for item in report["malformed"]] == [
        "episode_0001",
        "episode_0002",
        "episode_0003",
    ]
    assert "excavator_demo_raw.v2" in report["malformed"][0]["reason"]
    assert "invalid episode.json" in report["malformed"][1]["reason"]
    assert "exactly" in report["malformed"][2]["reason"]


@pytest.mark.parametrize("target_kind", ["directory", "file", "broken"])
def test_episode_symlinks_are_malformed_and_never_advance_the_campaign(
    tmp_path: Path,
    target_kind: str,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    episode_link = raw_root / "episode_0001"
    if target_kind == "directory":
        target = _write_episode(tmp_path / "targets", "episode_0001")
        episode_link.symlink_to(target, target_is_directory=True)
    elif target_kind == "file":
        target = tmp_path / "episode_payload.json"
        target.write_text("{}", encoding="utf-8")
        episode_link.symlink_to(target)
    else:
        episode_link.symlink_to(tmp_path / "missing_episode")

    report = inspect_collection_campaign(raw_root)

    assert report["summary"]["completed"] == 0
    assert report["summary"]["malformed"] == 1
    assert report["completed"] == []
    assert report["next_expected_slot"]["slot_id"] == "slot_001"
    assert report["malformed"] == [
        {
            "episode_id": "episode_0001",
            "episode_path": str(episode_link.absolute()),
            "reason": "Episode entry must be a real directory, not a symbolic link",
        }
    ]

    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "inspect_collection_campaign.py"),
            str(raw_root),
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["malformed"] == report["malformed"]


@pytest.mark.parametrize("authority_name", ["episode.json", "quality_report.json"])
@pytest.mark.parametrize("replacement_kind", ["symlink", "fifo"])
def test_authoritative_episode_json_must_be_a_real_regular_file(
    tmp_path: Path,
    authority_name: str,
    replacement_kind: str,
) -> None:
    raw_root = tmp_path / "raw"
    episode = _write_episode(raw_root, "episode_0001")
    authority_path = episode / authority_name
    if replacement_kind == "symlink":
        target = tmp_path / f"{authority_name}.target"
        authority_path.replace(target)
        authority_path.symlink_to(target)
    else:
        authority_path.unlink()
        os.mkfifo(authority_path)

    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "inspect_collection_campaign.py"),
            str(raw_root),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=2,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    report = json.loads(completed.stdout)
    assert report["summary"]["completed"] == 0
    assert report["summary"]["malformed"] == 1
    assert report["next_expected_slot"]["slot_id"] == "slot_001"
    assert report["malformed"][0]["reason"] == (
        f"{authority_name} must be a regular non-symbolic-link file"
    )


@pytest.mark.parametrize("entry_kind", ["file", "fifo"])
def test_non_directory_episode_entries_are_malformed(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    episode_entry = raw_root / "episode_0001"
    if entry_kind == "file":
        episode_entry.write_text("not an Episode directory", encoding="utf-8")
    else:
        os.mkfifo(episode_entry)

    report = inspect_collection_campaign(raw_root)

    assert report["summary"]["completed"] == 0
    assert report["summary"]["malformed"] == 1
    assert report["next_expected_slot"]["slot_id"] == "slot_001"
    assert report["malformed"] == [
        {
            "episode_id": "episode_0001",
            "episode_path": str(episode_entry.resolve()),
            "reason": "Episode entry must be a real directory",
        }
    ]

    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "inspect_collection_campaign.py"),
            str(raw_root),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=2,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["malformed"] == report["malformed"]


@pytest.mark.parametrize("replaced_entry", ["episode_directory", "episode.json"])
def test_concurrent_symlink_replacement_cannot_forge_a_completed_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_entry: str,
) -> None:
    raw_root = tmp_path / "raw"
    episode = _write_episode(raw_root, "episode_0001")
    original_lstat = Path.lstat
    replaced = False

    if replaced_entry == "episode_directory":
        watched_path = episode
        parked_path = tmp_path / "parked_episode"
        forged_target = _write_episode(tmp_path / "forged", "episode_0001")
    else:
        watched_path = episode / "episode.json"
        parked_path = tmp_path / "parked_episode.json"
        forged_target = tmp_path / "forged_episode.json"
        forged_target.write_bytes(watched_path.read_bytes())

    def replace_after_lstat(path: Path):
        nonlocal replaced
        result = original_lstat(path)
        if path == watched_path and not replaced:
            path.rename(parked_path)
            path.symlink_to(
                forged_target,
                target_is_directory=replaced_entry == "episode_directory",
            )
            replaced = True
        return result

    monkeypatch.setattr(Path, "lstat", replace_after_lstat)

    report = inspect_collection_campaign(raw_root)

    assert replaced is True
    assert report["summary"]["completed"] == 0
    assert report["summary"]["malformed"] == 1
    assert report["next_expected_slot"]["slot_id"] == "slot_001"


def test_exact_campaign_is_complete_valid_and_deterministic(tmp_path: Path) -> None:
    campaign = build_default_campaign()
    for episode_number, slot in enumerate(campaign["slots"], start=1):
        _write_episode(
            tmp_path,
            f"episode_{episode_number:04d}",
            task_variant=slot["task_variant"],
            soil_reset_block_id=slot["soil_reset_block_id"],
            dig_point_id=slot["dig_point_id"],
        )

    first = inspect_collection_campaign(tmp_path)
    second = inspect_collection_campaign(tmp_path)

    assert first == second
    assert first["summary"] == {
        "planned": 200,
        "completed": 200,
        "missing": 0,
        "failed": 0,
        "duplicate": 0,
        "unplanned": 0,
        "malformed": 0,
        "ignored_diagnostics": 0,
        "complete_and_valid": True,
    }
    assert first["missing"] == []
    assert first["next_expected_slot"] is None


def test_retained_failed_attempt_does_not_invalidate_200_successful_slots(
    tmp_path: Path,
) -> None:
    campaign = build_default_campaign()
    for episode_number, slot in enumerate(campaign["slots"], start=1):
        _write_episode(
            tmp_path,
            f"episode_{episode_number:04d}",
            task_variant=slot["task_variant"],
            soil_reset_block_id=slot["soil_reset_block_id"],
            dig_point_id=slot["dig_point_id"],
        )
    _write_episode(
        tmp_path,
        "episode_0201",
        status="failed",
        success=False,
    )

    report = inspect_collection_campaign(tmp_path)

    assert report["summary"]["completed"] == 200
    assert report["summary"]["failed"] == 1
    assert report["summary"]["complete_and_valid"] is True
    assert report["next_expected_slot"] is None


def test_cli_next_is_concise_deterministic_json_and_incomplete_exits_two(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "scripts" / "inspect_collection_campaign.py"
    source_root = Path(__file__).parents[1] / "src"

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "--next"],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(source_root)},
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "schema_version": "excavator_collection_campaign.v1",
        "raw_root": str(tmp_path.resolve()),
        "complete_and_valid": False,
        "summary": {
            "planned": 200,
            "completed": 0,
            "ignored_diagnostics": 0,
            "complete_and_valid": False,
        },
        "next_expected_slot": {
            "slot_id": "slot_001",
            "task_variant": "dig_only",
            "soil_reset_block_id": "block_01",
            "dig_point_id": "dig_01",
        },
    }


def test_cli_can_resolve_authoritative_raw_root_from_collection_config(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[1]
    raw = json.loads(
        (repository / "config" / "collection.orin.json").read_text(
            encoding="utf-8"
        )
    )
    raw_root = tmp_path / "raw"
    raw["data_root"] = str(raw_root)
    config_path = tmp_path / "collection.orin.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "inspect_collection_campaign.py"),
            "--collection-config",
            str(config_path),
            "--next",
        ],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["raw_root"] == str(raw_root.resolve())
    assert payload["summary"]["planned"] == 200
    assert payload["next_expected_slot"]["slot_id"] == "slot_001"


def test_cli_exits_zero_only_for_a_complete_valid_campaign(tmp_path: Path) -> None:
    for episode_number, slot in enumerate(
        build_default_campaign()["slots"], start=1
    ):
        _write_episode(
            tmp_path,
            f"episode_{episode_number:04d}",
            task_variant=slot["task_variant"],
            soil_reset_block_id=slot["soil_reset_block_id"],
            dig_point_id=slot["dig_point_id"],
        )
    script = Path(__file__).parents[1] / "scripts" / "inspect_collection_campaign.py"
    source_root = Path(__file__).parents[1] / "src"

    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(source_root)},
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["summary"]["complete_and_valid"] is True
