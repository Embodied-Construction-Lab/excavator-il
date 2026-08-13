from pathlib import Path


def test_act_shadow_runs_as_operator_with_hardware_device_groups():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_act_shadow.sh"
    ).read_text(encoding="utf-8")

    assert '--user "${runtime_uid}:${runtime_gid}"' in script
    assert '--group-add "${serial_gid}"' in script
    assert '--group-add "${camera_gid}"' in script
    assert "-e PYTHONUNBUFFERED=1" in script
    assert "/dev/video0" in script
    assert "/dev/video1" not in script
    assert "chmod 777" not in script


def test_act_runtime_uses_the_verified_uvc_capture_node():
    config = (
        Path(__file__).resolve().parents[1] / "config" / "act_runtime.orin.json"
    ).read_text(encoding="utf-8")

    assert '"device": "/dev/video0"' in config
