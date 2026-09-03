import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/preflight_act_dig_transport_dump_reference_assets.py"


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "excavator-il"
    config = repository / "config"
    checkpoint = (
        repository
        / "models/icra2027_transport_dump_dual_rgb_step115000/checkpoint"
    )
    deployment = checkpoint.parent / "deployment"
    config.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    deployment.mkdir()

    manifest = json.loads(
        (
            REPOSITORY
            / "config/act_deployment.icra2027_transport_dump_dual_rgb_step115000.json"
        ).read_text(encoding="utf-8")
    )
    file_hashes = {}
    for index, name in enumerate(manifest["checkpoint"]["files_sha256"], start=1):
        content = f"fixture-{index}-{name}".encode()
        (checkpoint / name).write_bytes(content)
        file_hashes[name] = hashlib.sha256(content).hexdigest()
    manifest["checkpoint"]["files_sha256"] = file_hashes

    machine_profile = tmp_path / "shared/machine_profile.json"
    machine_profile.parent.mkdir()
    machine_profile.write_text(
        json.dumps({"action_order": ["boom", "stick", "bucket", "swing"]}),
        encoding="utf-8",
    )
    manifest["machine_profile_sha256"] = hashlib.sha256(
        machine_profile.read_bytes()
    ).hexdigest()
    manifest_text = json.dumps(manifest)
    (deployment / "deployment_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    evidence_manifest = (
        config
        / "act_deployment.icra2027_transport_dump_dual_rgb_step115000.json"
    )
    evidence_manifest.write_text(manifest_text, encoding="utf-8")

    runtime = json.loads(
        (
            REPOSITORY
            / "config/act_runtime.icra2027_transport_dump_dual_rgb.orin.json"
        ).read_text(encoding="utf-8")
    )
    runtime["checkpoint_files_sha256"] = file_hashes
    runtime["checkpoint_model_sha256"] = file_hashes["model.safetensors"]
    (config / "act_runtime.icra2027_transport_dump_dual_rgb.orin.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )

    resident = json.loads(
        (
            REPOSITORY
            / "config/resident_fixed_cycle.act_dig_transport_dump_reference.commissioning.pc.json"
        ).read_text(encoding="utf-8")
    )
    (config / "resident_fixed_cycle.act_dig_transport_dump_reference.commissioning.pc.json").write_text(
        json.dumps(resident), encoding="utf-8"
    )
    return repository, machine_profile


def test_preflight_accepts_a_complete_pc_or_orin_asset_copy(tmp_path: Path):
    repository, machine_profile = _write_fixture(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(repository),
            "--machine-profile",
            str(machine_profile),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["camera_roles"] == ["front", "dump"]
    assert report["checkpoint_file_count"] == 7
    model = (
        repository
        / "models/icra2027_transport_dump_dual_rgb_step115000/"
        "checkpoint/model.safetensors"
    )
    assert report["checkpoint_model_sha256"] == hashlib.sha256(
        model.read_bytes()
    ).hexdigest()
    assert report["deployment_manifest_matches_evidence"] is True


def test_preflight_rejects_checkpoint_drift_before_hardware_start(tmp_path: Path):
    repository, machine_profile = _write_fixture(tmp_path)
    (
        repository
        / "models/icra2027_transport_dump_dual_rgb_step115000/"
        "checkpoint/model.safetensors"
    ).write_bytes(b"replaced-after-commissioning")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(repository),
            "--machine-profile",
            str(machine_profile),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["passed"] is False
    assert report["failure_reasons"] == [
        "ACT dig-transport-dump reference checkpoint SHA-256 mismatch: model.safetensors"
    ]


def test_preflight_rejects_an_unbound_model_directory(tmp_path: Path):
    repository, machine_profile = _write_fixture(tmp_path)
    source = repository / "models/icra2027_transport_dump_dual_rgb_step115000"
    replacement = repository / "models/unbound-act-reference-copy"
    shutil.copytree(source, replacement)
    resident_path = (
        repository
        / "config/resident_fixed_cycle.act_dig_transport_dump_reference.commissioning.pc.json"
    )
    resident = json.loads(resident_path.read_text(encoding="utf-8"))
    resident["act_checkpoint_host_path"] = (
        "/home/jetson16/workspace_excavator/excavator-il/"
        "models/unbound-act-reference-copy/checkpoint"
    )
    resident["act_deployment_host_path"] = (
        "/home/jetson16/workspace_excavator/excavator-il/"
        "models/unbound-act-reference-copy/deployment"
    )
    resident_path.write_text(json.dumps(resident), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(repository),
            "--machine-profile",
            str(machine_profile),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["failure_reasons"] == [
        "resident ACT checkpoint host path is not the commissioned asset"
    ]


def test_preflight_rejects_action_field_contract_drift(tmp_path: Path):
    repository, machine_profile = _write_fixture(tmp_path)
    deployed = (
        repository
        / "models/icra2027_transport_dump_dual_rgb_step115000/"
        "deployment/deployment_manifest.json"
    )
    evidence = (
        repository
        / "config/act_deployment.icra2027_transport_dump_dual_rgb_step115000.json"
    )
    manifest = json.loads(deployed.read_text(encoding="utf-8"))
    manifest["contract"]["action_fields"] = [
        "action_swing",
        "action_boom",
        "action_stick",
        "action_bucket",
    ]
    manifest_text = json.dumps(manifest)
    deployed.write_text(manifest_text, encoding="utf-8")
    evidence.write_text(manifest_text, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(repository),
            "--machine-profile",
            str(machine_profile),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["failure_reasons"] == [
        "ACT dig-transport-dump reference observation/action contract is invalid"
    ]


def test_preflight_rejects_incomplete_validation_evidence(tmp_path: Path):
    repository, machine_profile = _write_fixture(tmp_path)
    deployed = (
        repository
        / "models/icra2027_transport_dump_dual_rgb_step115000/"
        "deployment/deployment_manifest.json"
    )
    evidence = (
        repository
        / "config/act_deployment.icra2027_transport_dump_dual_rgb_step115000.json"
    )
    manifest = json.loads(deployed.read_text(encoding="utf-8"))
    del manifest["evaluation"]["saturated_value_count"]
    manifest_text = json.dumps(manifest)
    deployed.write_text(manifest_text, encoding="utf-8")
    evidence.write_text(manifest_text, encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repository-root",
            str(repository),
            "--machine-profile",
            str(machine_profile),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["failure_reasons"] == [
        "ACT dig-transport-dump reference validation evidence is unsafe"
    ]
