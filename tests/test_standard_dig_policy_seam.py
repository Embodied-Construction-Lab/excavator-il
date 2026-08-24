import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.act_runtime_contract import REQUIRED_MOTION_AUTHORIZATION
from excavator_il.dig_policy import DigPolicyDescriptor, DigPolicyFactory


def test_standard_runtime_service_imports_when_lerobot_is_unavailable():
    source_root = Path(__file__).parents[1] / "src"
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class _BlockLeRobot(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "lerobot" or fullname.startswith("lerobot."):
                    raise ModuleNotFoundError(
                        f"blocked optional dependency for test: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, _BlockLeRobot())
        sys.path.insert(0, sys.argv[1])
        import excavator_il.act_runtime_service
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(source_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_standard_runtime_fails_closed_for_an_unregistered_policy_backend(
    monkeypatch,
    tmp_path,
):
    import excavator_il.act_policy_provider as act_policy_provider_module
    import excavator_il.act_runtime_service as runtime_service_module

    config = SimpleNamespace(
        dig_policy_backend="diffusion_policy",
        checkpoint_path=tmp_path,
        checkpoint_files_sha256={},
    )
    monkeypatch.setattr(
        runtime_service_module,
        "load_act_runtime_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "_load_lerobot_policy_api",
        lambda: pytest.fail("unregistered backend must not load LeRobot ACT"),
    )

    with pytest.raises(ValueError, match="unknown dig policy backend"):
        runtime_service_module.run_act_runtime("/unused/runtime.json")


@pytest.mark.parametrize(
    "motion_authorization,expected_mode",
    [
        (None, "shadow"),
        (REQUIRED_MOTION_AUTHORIZATION, "motion"),
    ],
)
def test_standard_runtime_uses_an_injected_policy_backend_without_act_loading(
    monkeypatch,
    tmp_path,
    caplog,
    motion_authorization,
    expected_mode,
):
    import excavator_il.act_policy_provider as act_policy_provider_module
    import excavator_il.act_runtime_service as runtime_service_module

    events = []

    class _AlternatePolicy:
        descriptor = DigPolicyDescriptor(
            backend_id="diffusion_policy",
            implementation="tests.AlternatePolicy",
        )

        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, 0.0)

        def warmup(self):
            events.append("warmup")
            return (0.1, -0.2, 0.3, 0.0)

        def reset(self):
            events.append("reset")

    config = SimpleNamespace(
        dig_policy_backend="diffusion_policy",
        log_root=tmp_path / "logs",
        serial=SimpleNamespace(port="/dev/test", baudrate=460800),
        camera=SimpleNamespace(
            device="/dev/video-test", width=640, height=480, nominal_fps=30
        ),
        max_inference_state_age_ms=100.0,
        state_silence_timeout_ms=250.0,
        max_camera_age_ms=120.0,
        max_inference_ms=100.0,
    )

    class _HardwareSerial:
        def close(self):
            events.append("serial_close")

    class _Camera:
        def __init__(self, _config):
            pass

        def close(self):
            events.append("camera_close")

    class _Service:
        def __init__(self, **kwargs):
            events.append(kwargs["processor"]._engine._session.descriptor.backend_id)
            events.append(kwargs["command_channel"].mode.value)

        def run(self):
            events.append("run")

        def request_stop(self):
            pass

    monkeypatch.setattr(
        runtime_service_module,
        "load_act_runtime_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "_load_motion_deployment_verifier",
        lambda: pytest.fail(
            "alternate backend must not use ACT deployment provenance"
        ),
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "_load_lerobot_policy_api",
        lambda: pytest.fail("alternate backend must not load LeRobot ACT"),
    )
    monkeypatch.setattr(runtime_service_module, "UvcCamera", _Camera)
    monkeypatch.setattr(runtime_service_module, "ActRuntimeService", _Service)
    monkeypatch.setitem(
        sys.modules,
        "serial",
        SimpleNamespace(Serial=lambda *_args, **_kwargs: _HardwareSerial()),
    )

    def provide_policy(_config, runtime_mode):
        events.append(("provider_mode", runtime_mode.value))
        return DigPolicyFactory({"diffusion_policy": _AlternatePolicy})

    with caplog.at_level("INFO", logger="excavator_il.act_runtime"):
        runtime_service_module.run_act_runtime(
            "/unused/runtime.json",
            motion_authorization=motion_authorization,
            dig_policy_provider=provide_policy,
        )

    assert events == [
        ("provider_mode", expected_mode),
        "warmup",
        "diffusion_policy",
        expected_mode,
        "run",
        "camera_close",
        "serial_close",
    ]
    assert (
        "Dig policy selected: backend=diffusion_policy "
        "implementation=tests.AlternatePolicy"
    ) in caplog.messages


def test_standard_lerobot_act_runtime_preserves_provenance_gates_and_warmup(
    monkeypatch,
    tmp_path,
):
    import excavator_il.act_policy_provider as act_policy_provider_module
    import excavator_il.act_runtime_service as runtime_service_module

    events = []

    class _Policy:
        def __init__(self):
            self.config = SimpleNamespace(device="cpu")

        def to(self, device):
            events.append(("to", device))

    class _PolicyClass:
        @staticmethod
        def from_pretrained(path):
            events.append(("load", path))
            return _Policy()

    class _ActAdapter:
        descriptor = DigPolicyDescriptor(
            backend_id="lerobot_act",
            implementation="tests.ActAdapter",
        )

        def select_action(self, _observation):
            return (0.1, -0.2, 0.3, 0.0)

        def warmup(self):
            events.append("warmup")
            return (0.1, -0.2, 0.3, 0.0)

        def reset(self):
            pass

    config = SimpleNamespace(
        dig_policy_backend="lerobot_act",
        checkpoint_path=tmp_path / "checkpoint",
        deployment_manifest_path=tmp_path / "deployment.json",
        machine_profile_path=tmp_path / "machine.json",
        device="cuda",
        log_root=tmp_path / "logs",
        serial=SimpleNamespace(port="/dev/test", baudrate=460800),
        camera=SimpleNamespace(
            device="/dev/video-test", width=640, height=480, nominal_fps=30
        ),
        max_inference_state_age_ms=100.0,
        state_silence_timeout_ms=250.0,
        max_camera_age_ms=120.0,
        max_inference_ms=100.0,
    )

    class _HardwareSerial:
        def close(self):
            events.append("serial_close")

    class _Camera:
        def __init__(self, _config):
            pass

        def close(self):
            events.append("camera_close")

    class _Service:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            events.append("run")

        def request_stop(self):
            pass

    monkeypatch.setattr(
        runtime_service_module,
        "load_act_runtime_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "_verify_checkpoint",
        lambda loaded: events.append(("checkpoint", loaded.checkpoint_path)),
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "_load_motion_deployment_verifier",
        lambda: lambda **kwargs: events.append(
            ("manifest", kwargs["manifest_path"])
        ),
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "_load_lerobot_policy_api",
        lambda: (
            lambda name: (
                events.append(("policy_class", name)),
                _PolicyClass,
            )[1],
            lambda *_args, **_kwargs: (
                events.append("processors"),
                (object(), object()),
            )[1],
        ),
    )
    monkeypatch.setattr(
        act_policy_provider_module,
        "ActPolicySession",
        lambda **_kwargs: (events.append("adapter"), _ActAdapter())[1],
    )
    monkeypatch.setattr(runtime_service_module, "UvcCamera", _Camera)
    monkeypatch.setattr(runtime_service_module, "ActRuntimeService", _Service)
    monkeypatch.setitem(
        sys.modules,
        "serial",
        SimpleNamespace(Serial=lambda *_args, **_kwargs: _HardwareSerial()),
    )

    runtime_service_module.run_act_runtime(
        "/unused/runtime.json",
        motion_authorization=REQUIRED_MOTION_AUTHORIZATION,
    )

    assert events == [
        ("checkpoint", config.checkpoint_path),
        ("manifest", config.deployment_manifest_path),
        ("policy_class", "act"),
        ("load", config.checkpoint_path),
        ("to", "cuda"),
        "processors",
        "adapter",
        ("checkpoint", config.checkpoint_path),
        ("manifest", config.deployment_manifest_path),
        "warmup",
        "run",
        "camera_close",
        "serial_close",
    ]
