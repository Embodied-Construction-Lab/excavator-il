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


def test_act_motion_requires_local_authorization_without_pc_runtime_input():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_act_motion.sh"
    )
    script = script_path.read_text(encoding="utf-8")

    assert script_path.stat().st_mode & 0o111
    assert "act_operator_hmac" not in script
    assert "authentication_key" not in script
    assert "--network=none" in script
    assert "PC teleop" in script
    assert "模型可能立即发送非零杆量" in script
    assert '[[ "${confirmation}" != "ALLOW_ACT_MACHINE_MOTION" ]]' in script
    assert "--motion-authorization ALLOW_ACT_MACHINE_MOTION" in script
    assert "pgrep -f" in script
    assert "fuser /dev/ttyTHS1 /dev/video0" in script
    assert "--cap-drop=ALL" in script
    assert "--privileged" not in script


def test_act_motion_supports_bounded_noninteractive_hybrid_segment():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_act_motion.sh"
    ).read_text(encoding="utf-8")

    assert '"--authorization"' in script
    assert '"--max-steps"' in script
    assert 'runtime_args+=(--max-steps "${max_steps}")' in script
    assert "sudo -n docker" not in script
    assert "非交互 ACT 启动需要 jetson16 直接访问 Docker" in script
    assert "docker group" in script
