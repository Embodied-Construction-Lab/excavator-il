"""Commissioned ACT policy loading kept outside the resident control loop."""

from __future__ import annotations

from typing import Any, Callable

from .act_deployment import verify_deployment_manifest
from .act_runtime import ActPolicySession
from .dig_policy import DigPolicyFactory


def _emit_lifecycle(message: str) -> None:
    print(message, flush=True)


def load_commissioned_lerobot_act_session(config: Any) -> Any:
    """Load a commissioned LeRobot ACT adapter with deployment rechecks."""

    from lerobot.policies import get_policy_class, make_pre_post_processors

    provenance = {
        "manifest_path": config.deployment_manifest_path,
        "checkpoint_path": config.checkpoint_path,
        "machine_profile_path": config.machine_profile_path,
    }
    verify_deployment_manifest(**provenance)
    policy_class = get_policy_class("act")
    _emit_lifecycle("ACT resident build: policy load starting")
    policy = policy_class.from_pretrained(config.checkpoint_path)
    _emit_lifecycle("ACT resident build: policy load passed")
    policy.to(config.device)
    _emit_lifecycle("ACT resident build: CUDA transfer passed")
    policy.config.device = config.device
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(config.checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": config.device}},
        postprocessor_overrides={"device_processor": {"device": config.device}},
    )
    _emit_lifecycle("ACT resident build: processors ready")
    session = ActPolicySession(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=config.device,
        state_fields=config.policy_state_fields,
    )
    verify_deployment_manifest(**provenance)
    _emit_lifecycle("ACT resident build: deployment recheck passed")
    return session


def build_commissioned_dig_policy_factory(
    config: Any,
    *,
    commissioned_lerobot_act_loader: Callable[[Any], Any],
) -> DigPolicyFactory:
    return DigPolicyFactory(
        {"lerobot_act": lambda: commissioned_lerobot_act_loader(config)}
    )
