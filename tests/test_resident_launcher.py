import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_resident_act_launcher_has_valid_bash_syntax():
    script = ROOT / "scripts" / "run_act_resident.sh"

    subprocess.run(["bash", "-n", str(script)], check=True)


def test_resident_act_launcher_requires_exact_hybrid_authorization_and_direct_docker():
    script = (ROOT / "scripts" / "run_act_resident.sh").read_text(encoding="utf-8")

    assert '"--authorization"' in script
    assert '[[ "${authorization}" != "ALLOW_HYBRID_MACHINE_MOTION" ]]' in script
    assert "ALLOW_ACT_MACHINE_MOTION" not in script
    assert "docker_command=(docker)" in script
    assert "docker info" in script
    assert "direct docker access" in script
    assert "sudo docker" not in script
    assert "--runtime=nvidia --gpus all" in script


def test_resident_act_launcher_maps_camera_and_shared_runtime_dir_but_not_serial():
    script = (ROOT / "scripts" / "run_act_resident.sh").read_text(encoding="utf-8")

    assert "--device /dev/video0" in script
    assert "--device /dev/ttyTHS1" not in script
    assert "fuser /dev/ttyTHS1" not in script
    assert "/opt/excavator-resident" in script
    assert "--network=host" in script
    assert "--cap-drop=ALL" in script
    assert "--privileged" not in script


def test_resident_act_launcher_invokes_resident_module_with_socket_and_configs():
    script = (ROOT / "scripts" / "run_act_resident.sh").read_text(encoding="utf-8")

    assert "python3 -m excavator_il.resident_act_runtime" in script
    assert "--config /opt/act-runtime.json" in script
    assert "--socket-path /opt/excavator-resident/act.sock" in script
    assert (
        "--operator-observation-config /opt/collection-runtime.json"
        in script
    )
    assert "collection.orin.json" in script
    assert "/opt/collection-runtime.json" in script
    assert "act_runtime.orin.json" in script
    assert "/opt/act-runtime.json" in script
    assert "operator-observation-config" in script


def test_incremental_image_build_verifies_the_resident_runtime_entrypoint():
    dockerfile = (
        ROOT / "docker" / "act-inference.incremental.Dockerfile"
    ).read_text(encoding="utf-8")

    assert "import excavator_il.resident_act_runtime" in dockerfile
