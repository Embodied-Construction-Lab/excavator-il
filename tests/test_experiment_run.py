from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import excavator_il._experiment_artifact_store as artifact_store_module
import excavator_il._experiment_run_support as experiment_support_module
import excavator_il.experiment_run as experiment_run_module
from excavator_il.experiment_run import (
    EXPERIMENT_RUN_SCHEMA_VERSION,
    EvidenceRequirement,
    ExperimentRun,
    ExperimentRunFinalizedError,
    ExperimentRunValidationError,
    TaskContext,
    capture_repository_state,
    fingerprint_path,
    load_experiment_run,
)


def _git_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Experiment Test",
            "-c",
            "user.email=experiment@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        check=True,
    )
    return path


def _create_run(tmp_path: Path, *, run_id: str = "run_test_001") -> ExperimentRun:
    repository = _git_repository(tmp_path / "repo")
    config = tmp_path / "mission.json"
    config.write_text('{"target":"dig_01","rate_hz":10}\n', encoding="utf-8")
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_text('{"action_order":["boom","stick","bucket","swing"]}\n')
    return ExperimentRun.create(
        tmp_path / "evidence",
        run_id=run_id,
        run_kind="evaluation",
        task_context=TaskContext(
            task_variant="dig_transport_dump",
            soil_reset_block_id="soil_block_03",
            dig_point_id="dig_01",
            operator_id="zhaoshuai",
            material_id="dry_sand_batch_a",
        ),
        policy_ids={
            "digging_policy": "act:swing-zero-200k",
            "trajectory_controller": "rl:follow-v1",
        },
        host_topology={
            "pc": {"host": "192.168.50.1", "role": "planner"},
            "orin": {"host": "192.168.50.2", "role": "motion_owner"},
        },
        repository_paths={"excavator_il": repository},
        config_paths={"hybrid_mission": config},
        machine_profile_path=machine_profile,
        evidence_requirements={
            "runtime_log": EvidenceRequirement(required=True, min_count=1),
            "video": EvidenceRequirement(required=False, min_count=0),
        },
    )


def test_create_captures_reproducible_start_manifest_and_config_content(tmp_path):
    run = _create_run(tmp_path)

    assert run.run_dir == tmp_path / "evidence" / "active" / "run_test_001"
    assert run.state == "active"
    start_path = run.run_dir / "start.json"
    start_bytes = start_path.read_bytes()
    start = json.loads(start_bytes)
    assert start["schema_version"] == EXPERIMENT_RUN_SCHEMA_VERSION
    assert start["run_id"] == "run_test_001"
    assert start["run_kind"] == "evaluation"
    assert start["task_context"] == {
        "task_variant": "dig_transport_dump",
        "soil_reset_block_id": "soil_block_03",
        "dig_point_id": "dig_01",
        "operator_id": "zhaoshuai",
        "material_id": "dry_sand_batch_a",
    }
    assert start["repositories"]["excavator_il"]["dirty"] is False
    assert len(start["repositories"]["excavator_il"]["commit"]) == 40
    assert start["evidence_requirements"]["runtime_log"] == {
        "required": True,
        "min_count": 1,
    }

    snapshot_relpath = start["config_snapshots"]["hybrid_mission"]["snapshot_path"]
    snapshot_path = run.run_dir / snapshot_relpath
    assert snapshot_path.read_text(encoding="utf-8") == (
        '{"target":"dig_01","rate_hz":10}\n'
    )
    assert start["config_snapshots"]["hybrid_mission"]["sha256"] == hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    profile_snapshot = run.run_dir / start["machine_profile"]["snapshot_path"]
    assert start["machine_profile"]["sha256"] == hashlib.sha256(
        profile_snapshot.read_bytes()
    ).hexdigest()

    run.append_event("phase_started", {"phase": "rl_to_dig", "cycle_index": 0})
    assert start_path.read_bytes() == start_bytes


def test_create_is_unique_and_rejects_traversal_and_symlinked_inputs(tmp_path):
    existing = _create_run(tmp_path)
    profile = tmp_path / "another-profile.json"
    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ExperimentRun.create(
            existing.root,
            run_id="run_test_001",
            run_kind="diagnostic",
            task_context=TaskContext("dig", None, None, "operator", None),
            policy_ids={},
            host_topology={},
            repository_paths={},
            config_paths={},
            machine_profile_path=profile,
        )

    with pytest.raises(ExperimentRunValidationError, match="run_id"):
        ExperimentRun.create(
            tmp_path / "other",
            run_id="../escape",
            run_kind="diagnostic",
            task_context=TaskContext("dig", None, None, "operator", None),
            policy_ids={},
            host_topology={},
            repository_paths={},
            config_paths={},
            machine_profile_path=tmp_path / "missing",
        )

    target = tmp_path / "profile-target.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "profile-link.json"
    symlink.symlink_to(target)
    with pytest.raises(ExperimentRunValidationError, match="symlink"):
        ExperimentRun.create(
            tmp_path / "other",
            run_kind="diagnostic",
            task_context=TaskContext("dig", None, None, "operator", None),
            policy_ids={},
            host_topology={},
            repository_paths={},
            config_paths={},
            machine_profile_path=symlink,
        )


def test_append_events_is_ordered_timestamped_and_rejects_nonfinite_payload(tmp_path):
    run = _create_run(tmp_path)
    first = run.append_event(
        "phase_started", {"phase": "act_dig", "cycle_index": 0, "score": 0.5}
    )
    second = run.append_event("phase_completed", {"phase": "act_dig"})

    assert first["schema_version"] == "experiment_run_event.v1"
    assert [first["sequence"], second["sequence"]] == [0, 1]
    assert first["monotonic_ns"] <= second["monotonic_ns"]
    assert first["wall_time_utc"].endswith("Z")
    persisted = [
        json.loads(line)
        for line in (run.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert persisted == [first, second]

    with pytest.raises(ExperimentRunValidationError, match="finite"):
        run.append_event("invalid_metric", {"value": float("nan")})
    assert len((run.run_dir / "events.jsonl").read_text().splitlines()) == 2


def test_artifacts_snapshot_files_and_directories_inside_the_run(tmp_path):
    run = _create_run(tmp_path)
    artifact_file = tmp_path / "runtime.jsonl"
    artifact_file.write_text('{"passed":true}\n', encoding="utf-8")
    file_record = run.register_artifact(
        "runtime_000", artifact_file, role="runtime_log", metadata={"mode": "motion"}
    )
    assert file_record["schema_version"] == "experiment_run_artifact.v2"
    assert file_record["object_type"] == "file"
    assert file_record["sha256"] == hashlib.sha256(artifact_file.read_bytes()).hexdigest()
    assert file_record["source_path"] == str(artifact_file.resolve())
    assert file_record["snapshot_method"] in {"copy", "reflink"}
    file_snapshot = run.run_dir / file_record["snapshot_path"]
    assert file_snapshot.read_bytes() == artifact_file.read_bytes()
    assert file_snapshot != artifact_file.resolve()

    tree = tmp_path / "episode_0001"
    (tree / "nested").mkdir(parents=True)
    (tree / "z.txt").write_text("z", encoding="utf-8")
    (tree / "nested" / "a.txt").write_text("a", encoding="utf-8")
    first_fingerprint = fingerprint_path(tree)
    directory_record = run.register_artifact(
        "episode_0001", tree, role="raw_episode", metadata={"accepted": True}
    )
    assert directory_record["object_type"] == "directory"
    assert directory_record["sha256"] == first_fingerprint.sha256
    assert directory_record["file_count"] == 2
    assert directory_record["size_bytes"] == 2
    assert directory_record["source_path"] == str(tree.resolve())
    assert directory_record["snapshot_method"] in {"copy", "reflink", "mixed"}
    directory_snapshot = run.run_dir / directory_record["snapshot_path"]
    assert (directory_snapshot / "z.txt").read_text(encoding="utf-8") == "z"
    assert (directory_snapshot / "nested" / "a.txt").read_text(
        encoding="utf-8"
    ) == "a"
    assert fingerprint_path(tree).sha256 == first_fingerprint.sha256

    with pytest.raises(ExperimentRunValidationError, match="already registered"):
        run.register_artifact("episode_0001", tree, role="raw_episode")

    (tree / "unsafe-link").symlink_to(tree / "z.txt")
    with pytest.raises(ExperimentRunValidationError, match="symlink"):
        fingerprint_path(tree)


def test_artifact_snapshot_falls_back_to_honest_byte_copy(tmp_path, monkeypatch):
    run = _create_run(tmp_path)
    artifact_file = tmp_path / "runtime.jsonl"
    content = ("0123456789abcdef" * 100_000).encode()
    artifact_file.write_bytes(content)

    def unsupported_reflink(destination_descriptor, operation, source_descriptor):
        raise OSError(errno.EOPNOTSUPP, "reflink unsupported")

    monkeypatch.setattr(artifact_store_module.fcntl, "ioctl", unsupported_reflink)

    record = run.register_artifact("runtime", artifact_file, role="runtime_log")

    assert record["snapshot_method"] == "copy"
    assert (run.run_dir / record["snapshot_path"]).read_bytes() == content


def test_artifact_source_is_always_opened_nonblocking_before_fstat(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    artifact_file = tmp_path / "runtime.jsonl"
    artifact_file.write_text("{}\n", encoding="utf-8")
    original_open = artifact_store_module.os.open
    source_open_flags = []

    def inspect_open(path, flags, *args):
        if Path(path) == artifact_file:
            source_open_flags.append(flags)
        return original_open(path, flags, *args)

    monkeypatch.setattr(artifact_store_module.os, "open", inspect_open)

    run.register_artifact("runtime", artifact_file, role="runtime_log")

    assert source_open_flags
    assert all(flags & os.O_NONBLOCK for flags in source_open_flags)


def test_register_artifact_recovers_orphan_after_snapshot_fsync_failure(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    artifact_file = tmp_path / "runtime.jsonl"
    artifact_file.write_text("{}\n", encoding="utf-8")
    snapshot_root = run.run_dir / "artifact_snapshots"
    original_fsync_directory = artifact_store_module._fsync_directory
    failed = False

    def fail_after_snapshot_publish(path):
        nonlocal failed
        if Path(path) == snapshot_root and not failed:
            failed = True
            raise OSError("injected snapshot directory fsync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(
        artifact_store_module,
        "_fsync_directory",
        fail_after_snapshot_publish,
    )
    with pytest.raises(OSError, match="snapshot directory fsync failure"):
        run.register_artifact("runtime", artifact_file, role="runtime_log")

    assert list(snapshot_root.iterdir())
    monkeypatch.setattr(
        artifact_store_module,
        "_fsync_directory",
        original_fsync_directory,
    )

    record = run.register_artifact("runtime", artifact_file, role="runtime_log")
    assert record["artifact_id"] == "runtime"
    assert len(list(snapshot_root.iterdir())) == 1


def test_register_artifact_treats_atomic_jsonl_replace_as_commit_point(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    artifact_file = tmp_path / "runtime.jsonl"
    artifact_file.write_text("{}\n", encoding="utf-8")
    original_fsync_directory = experiment_support_module._fsync_directory
    failed = False

    def fail_after_replace(path):
        nonlocal failed
        if Path(path) == run.run_dir and not failed:
            failed = True
            raise OSError("injected post-replace fsync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(
        experiment_support_module,
        "_fsync_directory",
        fail_after_replace,
    )

    record = run.register_artifact("runtime", artifact_file, role="runtime_log")

    assert record["artifact_id"] == "runtime"
    assert len(run.snapshot().artifacts) == 1
    run.snapshot().verify_artifacts()


def test_file_source_change_during_snapshot_fails_without_registering_artifact(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    artifact_file = tmp_path / "runtime.jsonl"
    artifact_file.write_text("before\n", encoding="utf-8")
    original_copy = artifact_store_module._copy_regular_file

    def copy_then_change_source(source, destination):
        method = original_copy(source, destination)
        artifact_file.write_text("after\n", encoding="utf-8")
        return method

    monkeypatch.setattr(
        artifact_store_module,
        "_copy_regular_file",
        copy_then_change_source,
    )

    with pytest.raises(ExperimentRunValidationError, match="changed while snapshotting"):
        run.register_artifact("runtime", artifact_file, role="runtime_log")

    assert (run.run_dir / "artifacts.jsonl").read_text(encoding="utf-8") == ""
    assert list((run.run_dir / "artifact_snapshots").iterdir()) == []


def test_directory_source_change_during_snapshot_fails_without_partial_snapshot(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    artifact_directory = tmp_path / "runtime"
    artifact_directory.mkdir()
    (artifact_directory / "events.jsonl").write_text("before\n", encoding="utf-8")
    original_copy = artifact_store_module._copy_directory

    def copy_then_change_source(source, destination):
        methods = original_copy(source, destination)
        (artifact_directory / "late.jsonl").write_text("late\n", encoding="utf-8")
        return methods

    monkeypatch.setattr(
        artifact_store_module,
        "_copy_directory",
        copy_then_change_source,
    )

    with pytest.raises(ExperimentRunValidationError, match="changed while snapshotting"):
        run.register_artifact("runtime", artifact_directory, role="runtime_log")

    assert (run.run_dir / "artifacts.jsonl").read_text(encoding="utf-8") == ""
    assert list((run.run_dir / "artifact_snapshots").iterdir()) == []


def test_finalize_enforces_required_evidence_and_atomically_publishes_run(tmp_path):
    run = _create_run(tmp_path)
    with pytest.raises(ExperimentRunValidationError, match="runtime_log"):
        run.finalize("success", metrics={"task_success": 1.0})

    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    run.append_event("mission_completed", {"cycle_count": 1})
    snapshot = run.finalize(
        "success",
        metrics={"task_success": 1.0, "phase_duration_s": {"act": 12.3}},
        summary="One excavation cycle completed.",
    )

    expected_dir = tmp_path / "evidence" / "runs" / "run_test_001"
    assert run.run_dir == expected_dir
    assert snapshot.run_dir == expected_dir
    assert snapshot.state == "success"
    assert not (tmp_path / "evidence" / "active" / "run_test_001").exists()
    assert (expected_dir / "manifest.json").is_file()
    index = json.loads((expected_dir / "index.json").read_text())
    assert index["schema_version"] == "experiment_run_index.v1"
    assert index["status"] == "success"
    assert index["manifest_sha256"] == hashlib.sha256(
        (expected_dir / "manifest.json").read_bytes()
    ).hexdigest()

    with pytest.raises(ExperimentRunFinalizedError):
        run.append_event("too_late", {})
    with pytest.raises(ExperimentRunFinalizedError):
        run.register_artifact("too_late", runtime_log, role="runtime_log")
    with pytest.raises(ExperimentRunFinalizedError):
        run.finalize("failure", metrics={})


def test_finalize_uses_file_snapshot_when_external_source_changes(tmp_path):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    original_content = "{}\n"
    runtime_log.write_text(original_content, encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")

    record = run.snapshot().artifacts[0]
    runtime_log.write_text('{"changed":true}\n', encoding="utf-8")

    snapshot = run.finalize("success", metrics={"task_success": 1})
    assert snapshot.state == "success"
    assert (snapshot.run_dir / record["snapshot_path"]).read_text(
        encoding="utf-8"
    ) == original_content
    snapshot.verify_artifacts()
    with pytest.raises(ExperimentRunFinalizedError):
        run.finalize("success", metrics={"task_success": 1})


def test_finalize_publishes_stable_snapshot_when_source_is_replaced_at_publication(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    active_dir = run.run_dir
    final_dir = run.root / "runs" / run.run_id
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text('{"replaced":true}\n', encoding="utf-8")
    original_replace = experiment_run_module.os.replace

    def replace_source_at_publication(source, destination):
        if Path(source) == active_dir and Path(destination) == final_dir:
            original_replace(replacement, runtime_log)
        return original_replace(source, destination)

    monkeypatch.setattr(
        experiment_run_module.os,
        "replace",
        replace_source_at_publication,
    )

    snapshot = run.finalize("success", metrics={"task_success": 1})

    assert runtime_log.read_text(encoding="utf-8") == '{"replaced":true}\n'
    snapshot.verify_artifacts()


@pytest.mark.parametrize("failed_name", ["manifest.json", "index.json"])
def test_finalize_metadata_write_failure_leaves_active_run_clean_and_retryable(
    tmp_path,
    monkeypatch,
    failed_name,
):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    original_write = experiment_run_module._atomic_write_json

    def fail_selected_write(path, value, *, read_only):
        if Path(path).name == failed_name:
            raise OSError(f"injected {failed_name} write failure")
        return original_write(path, value, read_only=read_only)

    monkeypatch.setattr(
        experiment_run_module,
        "_atomic_write_json",
        fail_selected_write,
    )
    with pytest.raises(OSError, match="injected"):
        run.finalize("success", metrics={"task_success": 1})

    active_dir = run.root / "active" / run.run_id
    assert run.state == "active"
    assert not (active_dir / "manifest.json").exists()
    assert not (active_dir / "index.json").exists()

    monkeypatch.setattr(
        experiment_run_module,
        "_atomic_write_json",
        original_write,
    )
    snapshot = run.finalize("success", metrics={"task_success": 1})
    assert snapshot.state == "success"
    snapshot.verify_artifacts()


def test_finalize_index_publish_failure_rolls_back_manifest_and_is_retryable(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    active_dir = run.run_dir
    original_replace = experiment_run_module.os.replace

    def fail_index_publication(source, destination):
        if Path(destination) == active_dir / "index.json":
            raise OSError("injected index publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(
        experiment_run_module.os,
        "replace",
        fail_index_publication,
    )
    with pytest.raises(OSError, match="index publication failure"):
        run.finalize("success", metrics={"task_success": 1})

    assert run.state == "active"
    assert not (active_dir / "manifest.json").exists()
    assert not (active_dir / "index.json").exists()

    monkeypatch.setattr(experiment_run_module.os, "replace", original_replace)
    snapshot = run.finalize("success", metrics={"task_success": 1})
    assert snapshot.state == "success"
    snapshot.verify_artifacts()


@pytest.mark.parametrize("reload_run", [False, True])
def test_finalize_recovers_stale_active_metadata_after_process_crash(
    tmp_path,
    reload_run,
):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    active_dir = run.run_dir
    (active_dir / "manifest.json").write_text("stale\n", encoding="utf-8")
    (active_dir / "index.json").write_text("stale\n", encoding="utf-8")
    stale_staging = active_dir / ".finalizing-interrupted"
    stale_staging.mkdir()
    (stale_staging / "manifest.json").write_text("stale\n", encoding="utf-8")
    (active_dir / "events.jsonl").chmod(0o444)
    (active_dir / "artifacts.jsonl").chmod(0o444)
    (active_dir / "artifact_snapshots").chmod(0o555)

    recovered = ExperimentRun.load(run.root, run.run_id) if reload_run else run
    if reload_run:
        assert recovered.state == "active"
        assert not (active_dir / "manifest.json").exists()
        assert not (active_dir / "index.json").exists()
        assert not stale_staging.exists()
    snapshot = recovered.finalize("success", metrics={"task_success": 1})
    assert snapshot.state == "success"
    assert not (snapshot.run_dir / ".finalizing-interrupted").exists()
    snapshot.verify_artifacts()


def test_finalize_returns_committed_snapshot_when_parent_fsync_fails(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    final_dir = run.root / "runs" / run.run_id
    original_fsync_directory = experiment_run_module._fsync_directory

    def fail_after_commit(path):
        if final_dir.is_dir():
            raise OSError("injected post-commit parent fsync failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(
        experiment_run_module,
        "_fsync_directory",
        fail_after_commit,
    )

    snapshot = run.finalize("success", metrics={"task_success": 1})

    assert snapshot.state == "success"
    assert snapshot.run_dir == final_dir
    snapshot.verify_artifacts()


def test_final_directory_publish_failure_restores_clean_retryable_active_run(
    tmp_path,
    monkeypatch,
):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    active_dir = run.run_dir
    final_dir = run.root / "runs" / run.run_id
    original_replace = experiment_run_module.os.replace

    def fail_run_publication(source, destination):
        if Path(source) == active_dir and Path(destination) == final_dir:
            raise OSError("injected final directory publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(
        experiment_run_module.os,
        "replace",
        fail_run_publication,
    )
    with pytest.raises(OSError, match="publication failure"):
        run.finalize("success", metrics={"task_success": 1})

    assert run.state == "active"
    assert not (active_dir / "manifest.json").exists()
    assert not (active_dir / "index.json").exists()
    run.append_event("retry_ready", {})

    monkeypatch.setattr(experiment_run_module.os, "replace", original_replace)
    snapshot = run.finalize("success", metrics={"task_success": 1})
    assert snapshot.state == "success"
    snapshot.verify_artifacts()


def test_finalize_uses_directory_snapshot_when_external_source_changes(
    tmp_path,
):
    run = _create_run(tmp_path)
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    nested_log = runtime_directory / "events.jsonl"
    nested_log.write_text("{}\n", encoding="utf-8")
    record = run.register_artifact("runtime", runtime_directory, role="runtime_log")

    nested_log.write_text('{"changed":true}\n', encoding="utf-8")
    (runtime_directory / "late.jsonl").write_text("late\n", encoding="utf-8")

    snapshot = run.finalize("success", metrics={"task_success": 1})

    directory_snapshot = snapshot.run_dir / record["snapshot_path"]
    assert (directory_snapshot / "events.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert not (directory_snapshot / "late.jsonl").exists()
    snapshot.verify_artifacts()


def test_failure_can_finalize_without_required_artifact_but_metrics_must_be_finite(
    tmp_path,
):
    run = _create_run(tmp_path)
    with pytest.raises(ExperimentRunValidationError, match="finite"):
        run.finalize("failure", metrics={"loss": float("inf")})

    snapshot = run.finalize(
        "failure", metrics={"completed_cycles": 0}, summary="battery depleted"
    )
    assert snapshot.state == "failure"
    assert snapshot.final["summary"] == "battery depleted"


def test_loader_is_stable_frozen_and_detects_finalized_evidence_tampering(tmp_path):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    run.append_event("mission_completed", {"cycle_count": 1})
    finalized = run.finalize("success", metrics={"task_success": 1})

    loaded = load_experiment_run(tmp_path / "evidence", "run_test_001")
    loaded_by_path = load_experiment_run(finalized.run_dir)
    assert loaded == loaded_by_path
    assert loaded.start["task_context"]["task_variant"] == "dig_transport_dump"
    assert loaded.events[0]["event_type"] == "mission_completed"
    assert loaded.artifacts[0]["role"] == "runtime_log"
    with pytest.raises(TypeError):
        loaded.start["run_kind"] = "tampered"

    loaded.verify_artifacts()
    artifact_snapshot = loaded.run_dir / loaded.artifacts[0]["snapshot_path"]
    artifact_snapshot.chmod(0o644)
    artifact_snapshot.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ExperimentRunValidationError, match="fingerprint mismatch"):
        loaded.verify_artifacts()

    events_path = finalized.run_dir / "events.jsonl"
    events_path.chmod(0o644)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ExperimentRunValidationError, match="events.jsonl"):
        load_experiment_run(tmp_path / "evidence", "run_test_001")


def test_active_loader_rejects_unknown_start_fields(tmp_path):
    run = _create_run(tmp_path)
    start_path = run.run_dir / "start.json"
    start = json.loads(start_path.read_text())
    start["unexpected"] = True
    start_path.chmod(0o644)
    start_path.write_text(json.dumps(start), encoding="utf-8")
    with pytest.raises(ExperimentRunValidationError, match="start manifest fields"):
        load_experiment_run(tmp_path / "evidence", "run_test_001")


def test_active_loader_rejects_artifact_snapshot_path_traversal(tmp_path):
    run = _create_run(tmp_path)
    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text("{}\n", encoding="utf-8")
    run.register_artifact("runtime", runtime_log, role="runtime_log")
    artifacts_path = run.run_dir / "artifacts.jsonl"
    artifact = json.loads(artifacts_path.read_text(encoding="utf-8"))
    artifact["snapshot_path"] = "../outside"
    artifacts_path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentRunValidationError, match="snapshot_path"):
        load_experiment_run(tmp_path / "evidence", "run_test_001")


def test_capture_repository_state_reports_commit_and_dirty_status(tmp_path):
    repository = _git_repository(tmp_path / "repo")
    clean = capture_repository_state(repository)
    assert clean.dirty is False
    assert len(clean.commit) == 40
    assert clean.source_path == str(repository.resolve())

    (repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    dirty = capture_repository_state(repository)
    assert dirty.commit == clean.commit
    assert dirty.dirty is True


@pytest.mark.parametrize(
    "context",
    [
        {"task_variant": "dig", "operator_id": "operator"},
        {
            "task_variant": "dig",
            "soil_reset_block_id": None,
            "dig_point_id": None,
            "operator_id": "operator",
            "material_id": None,
            "extra": "not-allowed",
        },
    ],
)
def test_task_context_mapping_requires_exact_five_field_contract(tmp_path, context):
    profile = tmp_path / "machine.json"
    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ExperimentRunValidationError, match="task_context"):
        ExperimentRun.create(
            tmp_path / "runs",
            run_kind="diagnostic",
            task_context=context,
            policy_ids={},
            host_topology={},
            repository_paths={},
            config_paths={},
            machine_profile_path=profile,
        )


def test_generated_run_ids_are_unique_and_safe(tmp_path):
    profile = tmp_path / "machine.json"
    profile.write_text("{}", encoding="utf-8")
    kwargs = dict(
        run_kind="diagnostic",
        task_context=TaskContext("dig", None, None, "operator", None),
        policy_ids={},
        host_topology={},
        repository_paths={},
        config_paths={},
        machine_profile_path=profile,
    )
    first = ExperimentRun.create(tmp_path / "runs", **kwargs)
    second = ExperimentRun.create(tmp_path / "runs", **kwargs)
    assert first.run_id != second.run_id
    assert "/" not in first.run_id and ".." not in first.run_id


def test_run_kind_is_a_strict_research_lifecycle_dimension(tmp_path):
    profile = tmp_path / "machine.json"
    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ExperimentRunValidationError, match="run_kind must be one of"):
        ExperimentRun.create(
            tmp_path / "runs",
            run_kind="ad_hoc_unknown_kind",
            task_context=TaskContext("dig", None, None, "operator", None),
            policy_ids={},
            host_topology={},
            repository_paths={},
            config_paths={},
            machine_profile_path=profile,
        )


def test_management_cli_runs_create_event_finalize_and_show(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    repository = _git_repository(tmp_path / "repo")
    profile = tmp_path / "machine.json"
    profile.write_text("{}", encoding="utf-8")
    config = tmp_path / "diagnostic.json"
    config.write_text('{"duration_s":30}', encoding="utf-8")
    spec = tmp_path / "run-spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "experiment_run_create.v1",
                "run_kind": "diagnostic",
                "task_context": {
                    "task_variant": "zero_command_soak",
                    "soil_reset_block_id": None,
                    "dig_point_id": None,
                    "operator_id": "zhaoshuai",
                    "material_id": None,
                },
                "policy_ids": {},
                "host_topology": {"orin": {"role": "collector"}},
                "repository_paths": {"excavator_il": str(repository)},
                "config_paths": {"diagnostic": str(config)},
                "machine_profile_path": str(profile),
                "evidence_requirements": {},
            }
        ),
        encoding="utf-8",
    )
    script = project_root / "scripts" / "manage_experiment_run.py"
    evidence_root = tmp_path / "evidence"
    environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}

    def invoke(*arguments: str) -> dict:
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(completed.stdout)

    created = invoke(
        "create",
        "--root",
        str(evidence_root),
        "--spec",
        str(spec),
        "--run-id",
        "cli_run_001",
    )
    assert created["state"] == "active"
    event = invoke(
        "event",
        "--root",
        str(evidence_root),
        "--run-id",
        "cli_run_001",
        "--event-type",
        "diagnostic_completed",
        "--payload-json",
        '{"passed":true}',
    )
    assert event["sequence"] == 0
    invoke(
        "finalize",
        "--root",
        str(evidence_root),
        "--run-id",
        "cli_run_001",
        "--status",
        "failure",
        "--metrics-json",
        '{"completed_s":30}',
        "--summary",
        "battery depleted",
    )
    shown = invoke(
        "show", "--root", str(evidence_root), "--run-id", "cli_run_001"
    )
    assert shown["state"] == "failure"
    assert shown["start"]["config_snapshots"]["experiment_run_spec"]["sha256"]
