from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from excavator_il.collection_experiment_run import (
    CollectionExperimentRunRequest,
    load_collection_experiment_run_config,
    record_collection_experiment_run,
)
from excavator_il.experiment_run import ExperimentRun, fingerprint_path


_FIRMWARE_COMMIT = "a" * 40
_MACHINE_PROFILE_BYTES = (
    b'{"action_order":["boom","stick","bucket","swing"]}\n'
)
_MACHINE_PROFILE_SHA256 = hashlib.sha256(_MACHINE_PROFILE_BYTES).hexdigest()
_MISSION_CONFIG_SHA256 = "b" * 64


def _git_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tracked.txt").write_text("collection baseline\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Collection Test",
            "-c",
            "user.email=collection@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        check=True,
    )
    return path


def _episode(
    root: Path,
    *,
    status: str = "complete",
    success: bool = True,
    include_protocol: bool = True,
    include_quality: bool = True,
) -> Path:
    episode = root / "episode_0001"
    (episode / "camera_front").mkdir(parents=True)
    (episode / "camera_dump").mkdir()
    (episode / "camera_front" / "000000.jpg").write_bytes(b"front")
    (episode / "camera_dump" / "000000.jpg").write_bytes(b"dump")
    metadata = {
        "schema_version": "excavator_demo_raw.v2",
        "episode_id": episode.name,
        "task": "ExecuteDig",
        "operator_id": "zhaoshuai",
        "material_id": "dry_soil_batch_a",
        "recording_purpose": "demonstration",
        "status": status,
        "success": success,
        "failure_reason": "" if success else "operator_marked_failure",
        "dig_target_m": [1.0, 0.0, 0.0],
        "firmware_commit": _FIRMWARE_COMMIT,
        "machine_profile_hash": _MACHINE_PROFILE_SHA256,
        "target_source_provenance": {
            "repository": "airylidar",
            "path": "mission/config/excavation_demo.json",
            "sha256": _MISSION_CONFIG_SHA256,
            "commit": "e" * 40,
            "dirty": False,
        },
        "cameras": {
            "front": {"device_id": "front-camera"},
            "dump": {"device_id": "dump-camera"},
        },
    }
    if include_protocol:
        metadata["collection_protocol"] = {
            "task_variant": "dig_transport_dump",
            "soil_reset_block_id": "block_03",
            "dig_point_id": "dig_02",
        }
    (episode / "episode.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    if include_quality:
        (episode / "quality_report.json").write_text(
            json.dumps({"episode_id": episode.name, "passed": success}),
            encoding="utf-8",
        )
    return episode


def _request(tmp_path: Path, episode: Path) -> CollectionExperimentRunRequest:
    collection_config = tmp_path / "collection.orin.json"
    collection_config.write_text(
        '{"schema_version":"excavator_collection_config.v2"}\n',
        encoding="utf-8",
    )
    machine_profile = tmp_path / "machine_profile.json"
    machine_profile.write_bytes(_MACHINE_PROFILE_BYTES)
    campaign_provenance = tmp_path / "campaign_provenance.json"
    campaign_provenance.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_campaign_provenance.v1",
                "campaign_id": "icra2027-dual-rgb-campaign-v1",
                "frozen_baseline": {
                    "baseline_id": "icra2027-live-baseline-20260823",
                    "tag": "icra2027-live-baseline-20260823",
                    "repository_commits": {
                        "excavator_il": "c" * 40,
                        "excavator_orin_runtime": "d" * 40,
                        "airylidar": "e" * 40,
                        "f407": _FIRMWARE_COMMIT,
                    },
                },
                "airylidar_mission_targets": {
                    "repository": "airylidar",
                    "path": "mission/config/excavation_demo.json",
                    "sha256": _MISSION_CONFIG_SHA256,
                },
                "f407_firmware_commit": _FIRMWARE_COMMIT,
                "machine_profile_sha256": _MACHINE_PROFILE_SHA256,
                "dig_targets_m": {
                    "dig_01": [1.0, 0.26, 0.0],
                    "dig_02": [1.0, 0.0, 0.0],
                    "dig_03": [1.0, -0.26, 0.0],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    repository = _git_repository(tmp_path / "repository")
    return CollectionExperimentRunRequest(
        experiment_run_root=tmp_path / "experiment_runs",
        raw_episode_path=episode,
        collection_config_path=collection_config,
        machine_profile_path=machine_profile,
        campaign_provenance_path=campaign_provenance,
        repository_paths={"excavator_il": repository},
        policy_ids={"collection_policy": "human_demonstration:v1"},
        host_topology={
            "pc": {"host": "192.168.50.1", "role": "operator"},
            "orin": {"host": "192.168.50.2", "role": "collector"},
        },
        run_id="collection_episode_0001",
    )


def _tree_snapshot(path: Path) -> dict[str, tuple[bool, int, bytes | None]]:
    entries = [path, *sorted(path.rglob("*"))]
    return {
        item.relative_to(path).as_posix(): (
            item.is_dir(),
            item.stat().st_mtime_ns,
            None if item.is_dir() else item.read_bytes(),
        )
        for item in entries
    }


def _leave_active_run_after_create_crash(request, monkeypatch) -> None:
    original_register = ExperimentRun.register_artifact

    def crash_before_first_artifact(self, *args, **kwargs):
        raise RuntimeError("simulated crash after Experiment Run create")

    monkeypatch.setattr(
        ExperimentRun, "register_artifact", crash_before_first_artifact
    )
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            record_collection_experiment_run(request)
    finally:
        monkeypatch.setattr(
            ExperimentRun, "register_artifact", original_register
        )


def _leave_active_run_after_raw_artifact_crash(request, monkeypatch) -> None:
    original_register = ExperimentRun.register_artifact

    def crash_after_raw_episode(self, artifact_id, *args, **kwargs):
        record = original_register(self, artifact_id, *args, **kwargs)
        if artifact_id == "raw_episode":
            raise RuntimeError("simulated crash after raw Episode artifact")
        return record

    monkeypatch.setattr(
        ExperimentRun, "register_artifact", crash_after_raw_episode
    )
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            record_collection_experiment_run(request)
    finally:
        monkeypatch.setattr(
            ExperimentRun, "register_artifact", original_register
        )


def test_strict_evidence_config_resolves_paths_relative_to_its_file(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "collection_evidence.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_evidence_config.v2",
                "evidence_root": "../evidence",
                "collection_config": "collection.orin.json",
                "machine_profile": "../../shared/machine_profile.json",
                "campaign_provenance": "../../shared/campaign_provenance.json",
                "repository_paths": {"excavator_il": ".."},
                "policy_ids": {
                    "collection_policy": "human_expert_dual_stick.v1"
                },
                "host_topology": {
                    "pc": {"host": "192.168.50.1", "role": "operator"},
                    "orin": {"host": "192.168.50.2", "role": "collector"},
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_collection_experiment_run_config(config_path)

    assert config.evidence_root == tmp_path / "evidence"
    assert config.collection_config_path == config_dir / "collection.orin.json"
    assert config.machine_profile_path == tmp_path.parent / "shared/machine_profile.json"
    assert config.campaign_provenance_path == (
        tmp_path.parent / "shared/campaign_provenance.json"
    )
    assert config.repository_paths == {"excavator_il": tmp_path}
    assert config.policy_ids == {
        "collection_policy": "human_expert_dual_stick.v1"
    }


def test_evidence_config_rejects_unknown_fields(tmp_path):
    config_path = tmp_path / "collection_evidence.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "excavator_collection_evidence_config.v2",
                "evidence_root": "evidence",
                "collection_config": "collection.orin.json",
                "machine_profile": "machine_profile.json",
                "campaign_provenance": "campaign_provenance.json",
                "repository_paths": {"excavator_il": "."},
                "policy_ids": {"collection_policy": "human:v1"},
                "host_topology": {"orin": {"role": "collector"}},
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields are invalid"):
        load_collection_experiment_run_config(config_path)


def test_record_rejects_invalid_campaign_provenance_before_run_creation(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    provenance["schema_version"] = "unknown"
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="campaign provenance schema_version"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_campaign_provenance_requires_exact_frozen_baseline_contract(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    del provenance["frozen_baseline"]["tag"]
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="frozen_baseline fields"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_campaign_provenance_requires_nonempty_campaign_id(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    provenance["campaign_id"] = ""
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="campaign_id"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "f407", "repository must be airylidar"),
        ("path", "../targets.json", "repository-relative path"),
        ("sha256", "not-a-sha", "sha256"),
    ],
)
def test_campaign_provenance_rejects_invalid_mission_target_binding(
    tmp_path, field, value, message
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    provenance["airylidar_mission_targets"][field] = value
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=message):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_campaign_provenance_binds_f407_commit_to_frozen_baseline(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    provenance["f407_firmware_commit"] = "f" * 40
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="must match frozen_baseline"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_campaign_provenance_requires_machine_profile_sha256(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    provenance["machine_profile_sha256"] = "not-a-sha"
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="machine_profile_sha256"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda targets: targets.pop("dig_03"), "dig_targets_m fields"),
        (
            lambda targets: targets.__setitem__("dig_02", [1.0, 0.0]),
            "dig_targets_m.dig_02",
        ),
        (
            lambda targets: targets.__setitem__("dig_02", [1.0, float("nan"), 0.0]),
            "finite coordinates",
        ),
    ],
)
def test_campaign_provenance_requires_exact_finite_dig_targets(
    tmp_path, mutate, message
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    mutate(provenance["dig_targets_m"])
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match=message):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_campaign_provenance_must_match_the_machine_profile_file(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    request.machine_profile_path.write_text('{"changed":true}\n')

    with pytest.raises(ValueError, match="machine profile file SHA-256"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_formal_collection_evidence_rejects_dirty_repository(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    repository = request.repository_paths["excavator_il"]
    (repository / "tracked.txt").write_text("uncommitted campaign change\n")

    with pytest.raises(ValueError, match="repository must be clean"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda metadata: metadata.__setitem__("firmware_commit", "f" * 40),
            "firmware_commit does not match",
        ),
        (
            lambda metadata: metadata.__setitem__("machine_profile_hash", "f" * 64),
            "machine_profile_hash does not match",
        ),
        (
            lambda metadata: metadata["collection_protocol"].__setitem__(
                "dig_point_id", "dig_99"
            ),
            "dig_point_id is not defined",
        ),
        (
            lambda metadata: metadata.__setitem__("dig_target_m", [1.0, 0.1, 0.0]),
            "dig_target_m does not match",
        ),
        (
            lambda metadata: metadata.pop("firmware_commit"),
            "firmware_commit does not match",
        ),
    ],
)
def test_raw_episode_must_match_campaign_provenance(
    tmp_path, mutate, message
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text())
    mutate(metadata)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=message):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda source: source.__setitem__("commit", "f" * 40),
            "target source commit does not match",
        ),
        (
            lambda source: source.__setitem__("path", "mission/config/other.json"),
            "target source path does not match",
        ),
        (
            lambda source: source.__setitem__("sha256", "f" * 64),
            "target source SHA-256 does not match",
        ),
        (
            lambda source: source.__setitem__("dirty", True),
            "dirty must be exactly false",
        ),
    ],
)
def test_live_target_source_must_match_campaign_manifest(
    tmp_path, mutate, message
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata["target_source_provenance"])
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_formal_evidence_requires_live_target_source_provenance(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["target_source_provenance"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="target_source_provenance"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_record_dual_camera_demonstration_creates_hashed_success_evidence(tmp_path):
    episode = _episode(tmp_path / "raw")
    raw_fingerprint = fingerprint_path(episode)
    quality_bytes = (episode / "quality_report.json").read_bytes()
    request = _request(tmp_path, episode)
    provenance_bytes = request.campaign_provenance_path.read_bytes()

    snapshot = record_collection_experiment_run(request)

    assert snapshot.state == "success"
    assert snapshot.start["run_kind"] == "collection_episode"
    assert snapshot.start["task_context"] == {
        "task_variant": "dig_transport_dump",
        "soil_reset_block_id": "block_03",
        "dig_point_id": "dig_02",
        "operator_id": "zhaoshuai",
        "material_id": "dry_soil_batch_a",
    }
    artifacts = {artifact["role"]: artifact for artifact in snapshot.artifacts}
    assert set(artifacts) == {"raw_episode", "quality_report"}
    assert artifacts["raw_episode"]["sha256"] == raw_fingerprint.sha256
    assert artifacts["quality_report"]["sha256"] == hashlib.sha256(
        quality_bytes
    ).hexdigest()
    assert snapshot.start["evidence_requirements"] == {
        "quality_report": {"required": True, "min_count": 1},
        "raw_episode": {"required": True, "min_count": 1},
    }
    provenance_snapshot = snapshot.start["config_snapshots"][
        "campaign_provenance"
    ]
    assert provenance_snapshot["sha256"] == hashlib.sha256(
        provenance_bytes
    ).hexdigest()
    assert snapshot.final["metrics"]["evaluation_scope"] == "training_internal"
    snapshot.verify_artifacts()


@pytest.mark.parametrize(
    ("episode_status", "episode_success"),
    [
        ("failed", False),
        ("failed", True),
        ("aborted", False),
        ("complete", False),
    ],
)
def test_non_success_episode_is_retained_as_final_failure(
    tmp_path, episode_status, episode_success
):
    episode = _episode(
        tmp_path / "raw", status=episode_status, success=episode_success
    )

    snapshot = record_collection_experiment_run(_request(tmp_path, episode))

    assert snapshot.state == "failure"
    assert {artifact["role"] for artifact in snapshot.artifacts} == {
        "raw_episode",
        "quality_report",
    }
    assert snapshot.final["metrics"] == {
        "episode_id": "episode_0001",
        "episode_status": episode_status,
        "episode_success": episode_success,
        "evaluation_scope": "training_internal",
    }


@pytest.mark.parametrize(
    ("include_protocol", "include_quality", "message"),
    [
        (False, True, "collection_protocol"),
        (True, False, "quality_report.json"),
    ],
)
def test_required_protocol_and_quality_are_checked_before_run_creation(
    tmp_path, include_protocol, include_quality, message
):
    episode = _episode(
        tmp_path / "raw",
        include_protocol=include_protocol,
        include_quality=include_quality,
    )
    request = _request(tmp_path, episode)

    with pytest.raises(ValueError, match=message):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_diagnostic_recording_is_not_eligible_for_collection_evidence(tmp_path):
    episode = _episode(tmp_path / "raw")
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["recording_purpose"] = "diagnostic"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    request = _request(tmp_path, episode)

    with pytest.raises(ValueError, match="recording_purpose=demonstration"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_quality_report_must_identify_the_episode(tmp_path):
    episode = _episode(tmp_path / "raw")
    (episode / "quality_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    request = _request(tmp_path, episode)

    with pytest.raises(ValueError, match="quality_report.json episode_id"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_success_evidence_requires_a_passing_quality_report(tmp_path):
    episode = _episode(tmp_path / "raw")
    (episode / "quality_report.json").write_text(
        json.dumps({"episode_id": episode.name, "passed": False}),
        encoding="utf-8",
    )
    request = _request(tmp_path, episode)

    with pytest.raises(ValueError, match="quality_report.json passed must be true"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_adapter_never_writes_into_the_raw_episode_directory(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    before = _tree_snapshot(episode)

    record_collection_experiment_run(request)

    assert _tree_snapshot(episode) == before


def test_request_does_not_alias_nested_host_topology(tmp_path):
    episode = _episode(tmp_path / "raw")
    hosts = {"pc": {"host": "192.168.50.1", "roles": ["operator"]}}
    request = _request(tmp_path, episode)
    request = CollectionExperimentRunRequest(
        experiment_run_root=request.experiment_run_root,
        raw_episode_path=request.raw_episode_path,
        collection_config_path=request.collection_config_path,
        machine_profile_path=request.machine_profile_path,
        campaign_provenance_path=request.campaign_provenance_path,
        repository_paths=request.repository_paths,
        policy_ids=request.policy_ids,
        host_topology=hosts,
    )

    hosts["pc"]["host"] = "203.0.113.9"
    hosts["pc"]["roles"].append("mutated")

    assert request.host_topology["pc"]["host"] == "192.168.50.1"
    assert request.host_topology["pc"]["roles"] == ("operator",)


def test_record_is_idempotent_for_an_unchanged_finalized_episode(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = replace(_request(tmp_path, episode), run_id=None)

    first = record_collection_experiment_run(request)
    second = record_collection_experiment_run(request)

    assert second.run_id == "collection_episode_0001"
    assert second.run_dir == first.run_dir
    assert second.manifest == first.manifest
    assert [path.name for path in (request.experiment_run_root / "runs").iterdir()] == [
        "collection_episode_0001"
    ]


def test_record_resumes_unchanged_active_run_after_create_crash(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_create_crash(request, monkeypatch)

    active_run = (
        request.experiment_run_root / "active" / "collection_episode_0001"
    )
    assert active_run.is_dir()

    recovered = record_collection_experiment_run(request)

    assert recovered.state == "success"
    assert recovered.run_dir == (
        request.experiment_run_root / "runs" / "collection_episode_0001"
    )
    assert [artifact["artifact_id"] for artifact in recovered.artifacts] == [
        "raw_episode",
        "quality_report",
    ]


@pytest.mark.parametrize("drift_target", ["raw_episode", "quality_report"])
def test_record_rejects_source_drift_after_create_crash_before_artifact(
    tmp_path, monkeypatch, drift_target
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_create_crash(request, monkeypatch)
    if drift_target == "raw_episode":
        (episode / "camera_front" / "000000.jpg").write_bytes(
            b"changed-front"
        )
    else:
        quality_path = episode / "quality_report.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["review_note"] = "changed after Run create"
        quality_path.write_text(json.dumps(quality), encoding="utf-8")

    with pytest.raises(ValueError, match="source binding"):
        record_collection_experiment_run(request)

    active = ExperimentRun.load(
        request.experiment_run_root, "collection_episode_0001"
    ).snapshot()
    assert active.state == "active"
    assert active.artifacts == ()


def test_record_resumes_unchanged_active_run_after_partial_artifact_crash(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_raw_artifact_crash(request, monkeypatch)

    active = ExperimentRun.load(
        request.experiment_run_root, "collection_episode_0001"
    ).snapshot()
    assert [artifact["artifact_id"] for artifact in active.artifacts] == [
        "raw_episode"
    ]

    recovered = record_collection_experiment_run(request)

    assert recovered.state == "success"
    assert [artifact["artifact_id"] for artifact in recovered.artifacts] == [
        "raw_episode",
        "quality_report",
    ]
    assert [artifact["sequence"] for artifact in recovered.artifacts] == [0, 1]


def test_record_rejects_active_run_when_registered_artifact_drifted(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_raw_artifact_crash(request, monkeypatch)
    (episode / "camera_front" / "000000.jpg").write_bytes(b"changed-front")

    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        record_collection_experiment_run(request)

    assert (
        request.experiment_run_root / "active" / "collection_episode_0001"
    ).is_dir()


def test_record_rejects_active_run_when_collection_config_drifted(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_create_crash(request, monkeypatch)

    request.collection_config_path.write_text(
        '{"schema_version":"excavator_collection_config.v2","drift":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collection config snapshot"):
        record_collection_experiment_run(request)

    assert (
        request.experiment_run_root / "active" / "collection_episode_0001"
    ).is_dir()
    assert not (
        request.experiment_run_root / "runs" / "collection_episode_0001"
    ).exists()


def test_record_rejects_active_run_when_policy_ids_drifted(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_create_crash(request, monkeypatch)
    drifted_request = replace(
        request,
        policy_ids={"collection_policy": "human_demonstration:v2"},
    )

    with pytest.raises(ValueError, match="policy_ids"):
        record_collection_experiment_run(drifted_request)

    assert (
        request.experiment_run_root / "active" / "collection_episode_0001"
    ).is_dir()
    assert not (
        request.experiment_run_root / "runs" / "collection_episode_0001"
    ).exists()


def test_record_rejects_active_run_when_host_topology_drifted(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_create_crash(request, monkeypatch)
    drifted_request = replace(
        request,
        host_topology={
            "pc": {"host": "192.168.50.9", "role": "operator"},
            "orin": {"host": "192.168.50.2", "role": "collector"},
        },
    )

    with pytest.raises(ValueError, match="host_topology"):
        record_collection_experiment_run(drifted_request)

    assert (
        request.experiment_run_root / "active" / "collection_episode_0001"
    ).is_dir()
    assert not (
        request.experiment_run_root / "runs" / "collection_episode_0001"
    ).exists()


def test_record_rejects_active_run_when_repository_commit_drifted(
    tmp_path, monkeypatch
):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    _leave_active_run_after_create_crash(request, monkeypatch)
    repository = request.repository_paths["excavator_il"]
    (repository / "tracked.txt").write_text(
        "different clean collection baseline\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "-C", str(repository), "add", "tracked.txt"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Collection Test",
            "-c",
            "user.email=collection@example.invalid",
            "commit",
            "-q",
            "-m",
            "different baseline",
        ],
        check=True,
    )

    with pytest.raises(ValueError, match="repository state"):
        record_collection_experiment_run(request)

    assert (
        request.experiment_run_root / "active" / "collection_episode_0001"
    ).is_dir()
    assert not (
        request.experiment_run_root / "runs" / "collection_episode_0001"
    ).exists()


def test_collection_run_id_is_derived_from_episode_id(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = replace(_request(tmp_path, episode), run_id="custom_run")

    with pytest.raises(ValueError, match="collection_episode_0001"):
        record_collection_experiment_run(request)

    assert not request.experiment_run_root.exists()


def test_idempotent_record_rejects_raw_artifact_drift(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    record_collection_experiment_run(request)
    (episode / "camera_front" / "000000.jpg").write_bytes(b"changed-front")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        record_collection_experiment_run(request)


def test_idempotent_record_rejects_campaign_provenance_drift(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    record_collection_experiment_run(request)
    provenance = json.loads(request.campaign_provenance_path.read_text())
    provenance["airylidar_mission_targets"]["sha256"] = "f" * 64
    request.campaign_provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="target source SHA-256"):
        record_collection_experiment_run(request)


def test_idempotent_record_rejects_task_context_drift(tmp_path):
    episode = _episode(tmp_path / "raw")
    request = _request(tmp_path, episode)
    record_collection_experiment_run(request)
    metadata_path = episode / "episode.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["collection_protocol"]["dig_point_id"] = "dig_03"
    metadata["dig_target_m"] = [1.0, -0.26, 0.0]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="TaskContext"):
        record_collection_experiment_run(request)
