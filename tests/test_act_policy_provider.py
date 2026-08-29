import hashlib
import sys
from types import SimpleNamespace

import pytest

import excavator_il.act_policy_provider as provider_module
from excavator_il.act_policy_provider import build_commissioned_lerobot_act_factory
from excavator_il.act_runtime import RuntimeMode
from excavator_il.dig_policy import DigPolicyDescriptor


class _Adapter:
    descriptor = DigPolicyDescriptor(
        backend_id="lerobot_act",
        implementation="tests.ActAdapter",
    )

    def select_action(self, _observation):
        return (0.0, 0.0, 0.0, 0.0)

    def warmup(self):
        return (0.0, 0.0, 0.0, 0.0)

    def reset(self):
        pass


def _config(checkpoint_path, file_hashes):
    return SimpleNamespace(
        checkpoint_path=checkpoint_path,
        checkpoint_files_sha256=file_hashes,
        deployment_manifest_path=checkpoint_path.parent / "deployment.json",
        machine_profile_path=checkpoint_path.parent / "machine.json",
        device="cuda",
    )


def _install_lerobot_boundary(monkeypatch):
    policy = SimpleNamespace(
        config=SimpleNamespace(device="cpu"),
        to=lambda _device: None,
    )
    monkeypatch.setattr(
        provider_module,
        "_load_lerobot_policy_api",
        lambda: (
            lambda _name: SimpleNamespace(from_pretrained=lambda _path: policy),
            lambda *_args, **_kwargs: (object(), object()),
        ),
    )
    monkeypatch.setattr(
        provider_module,
        "ActPolicySession",
        lambda **_kwargs: _Adapter(),
    )


def test_commissioned_shadow_factory_validates_checkpoint_without_motion_manifest(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"commissioned-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    _install_lerobot_boundary(monkeypatch)
    monkeypatch.setattr(
        provider_module,
        "_load_motion_deployment_verifier",
        lambda: pytest.fail("shadow mode must not require a motion manifest"),
    )

    factory = build_commissioned_lerobot_act_factory(
        _config(checkpoint, {model.name: digest}),
        mode=RuntimeMode.SHADOW,
    )

    assert factory.create("lerobot_act").descriptor.backend_id == "lerobot_act"


def test_commissioned_factory_preloads_packaging_version_for_lerobot(
    monkeypatch,
    tmp_path,
):
    import packaging

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model = checkpoint / "model.safetensors"
    model.write_bytes(b"commissioned-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    policy = SimpleNamespace(
        config=SimpleNamespace(device="cpu"),
        to=lambda _device: None,
    )
    monkeypatch.delattr(packaging, "version", raising=False)
    monkeypatch.delitem(sys.modules, "packaging.version", raising=False)

    def load_policy(_path):
        assert hasattr(packaging, "version")
        return policy

    monkeypatch.setattr(
        provider_module,
        "_load_lerobot_policy_api",
        lambda: (
            lambda _name: SimpleNamespace(from_pretrained=load_policy),
            lambda *_args, **_kwargs: (object(), object()),
        ),
    )
    monkeypatch.setattr(
        provider_module,
        "ActPolicySession",
        lambda **_kwargs: _Adapter(),
    )

    factory = build_commissioned_lerobot_act_factory(
        _config(checkpoint, {model.name: digest}),
        mode=RuntimeMode.SHADOW,
    )

    assert factory.create("lerobot_act").descriptor.backend_id == "lerobot_act"


def test_commissioned_factory_rejects_a_checkpoint_file_set_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "unexpected.bin").write_bytes(b"unexpected")
    factory = build_commissioned_lerobot_act_factory(
        _config(checkpoint, {"model.safetensors": "0" * 64}),
        mode=RuntimeMode.SHADOW,
    )

    with pytest.raises(ValueError, match="file set"):
        factory.create("lerobot_act")


def test_commissioned_factory_rejects_a_checkpoint_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"wrong-model")
    factory = build_commissioned_lerobot_act_factory(
        _config(checkpoint, {"model.safetensors": "0" * 64}),
        mode=RuntimeMode.SHADOW,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        factory.create("lerobot_act")
