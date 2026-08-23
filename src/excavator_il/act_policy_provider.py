"""Commissioned LeRobot ACT Adapter provider for digging runtimes.

The Runtime service depends only on :class:`DigPolicyFactory`.  This Module is
the composition root for the currently commissioned ACT backend and owns its
checkpoint/deployment provenance gates.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from .act_runtime import ActPolicySession, RuntimeMode
from .act_runtime_config import ActRuntimeConfig
from .dig_policy import DigPolicy, DigPolicyFactory


def _load_lerobot_policy_api() -> tuple[
    Callable[[str], Any],
    Callable[..., tuple[Any, Any]],
]:
    """Resolve the optional LeRobot API only on the commissioned load path."""

    from lerobot.policies import get_policy_class, make_pre_post_processors

    return get_policy_class, make_pre_post_processors


def _load_motion_deployment_verifier() -> Callable[..., dict[str, Any]]:
    """Keep LeRobot-backed evaluation imports out of non-motion seams."""

    from .act_deployment import verify_deployment_manifest

    return verify_deployment_manifest


def build_commissioned_lerobot_act_factory(
    config: ActRuntimeConfig,
    *,
    mode: RuntimeMode,
) -> DigPolicyFactory:
    """Return the sole commissioned backend without loading it eagerly."""

    return DigPolicyFactory(
        {
            "lerobot_act": lambda: _load_commissioned_lerobot_act(
                config,
                mode=mode,
            ),
        }
    )


def _load_commissioned_lerobot_act(
    config: ActRuntimeConfig,
    *,
    mode: RuntimeMode,
) -> DigPolicy:
    _verify_checkpoint(config)
    _verify_motion_deployment(config, mode=mode)
    get_policy_class, make_pre_post_processors = _load_lerobot_policy_api()
    policy_class = get_policy_class("act")
    policy = policy_class.from_pretrained(config.checkpoint_path)
    policy.to(config.device)
    policy.config.device = config.device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(config.checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
        postprocessor_overrides={"device_processor": {"device": config.device}},
    )
    session = ActPolicySession(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=config.device,
    )
    # Detect checkpoint replacement during the comparatively expensive load.
    _verify_checkpoint(config)
    _verify_motion_deployment(config, mode=mode)
    return session


def _verify_motion_deployment(
    config: ActRuntimeConfig,
    *,
    mode: RuntimeMode,
) -> None:
    if mode is not RuntimeMode.MOTION:
        return
    verify_deployment_manifest = _load_motion_deployment_verifier()
    verify_deployment_manifest(
        manifest_path=config.deployment_manifest_path,
        checkpoint_path=config.checkpoint_path,
        machine_profile_path=config.machine_profile_path,
    )


def _verify_checkpoint(config: ActRuntimeConfig) -> None:
    actual_names = {
        path.name for path in config.checkpoint_path.iterdir() if path.is_file()
    }
    expected_names = set(config.checkpoint_files_sha256)
    if actual_names != expected_names:
        raise ValueError("ACT checkpoint file set does not match runtime provenance")
    for name, expected in config.checkpoint_files_sha256.items():
        digest = hashlib.sha256(
            (config.checkpoint_path / name).read_bytes()
        ).hexdigest()
        if digest != expected:
            raise ValueError(f"ACT checkpoint SHA-256 mismatch: {name}")
