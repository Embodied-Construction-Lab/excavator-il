from types import MappingProxyType

import numpy as np
import pytest

from excavator_il.dig_policy import (
    ACTION_ORDER,
    DigPolicyDescriptor,
    DigPolicyFactory,
    DigPolicyObservation,
)


class _ConstantAdapter:
    def __init__(self, backend_id: str, action: tuple[float, ...]) -> None:
        self.descriptor = DigPolicyDescriptor(
            backend_id=backend_id,
            implementation="test.constant",
        )
        self._action = action
        self.reset_count = 0
        self.seen = []

    def select_action(self, observation: DigPolicyObservation) -> tuple[float, ...]:
        self.seen.append(observation)
        return self._action

    def warmup(self) -> tuple[float, ...]:
        return self._action

    def reset(self) -> None:
        self.reset_count += 1


def _observation() -> DigPolicyObservation:
    return DigPolicyObservation(
        state_by_name=MappingProxyType(
            {f"state_{index}": float(index) for index in range(11)}
        ),
        rgb_by_role=MappingProxyType(
            {"front": np.zeros((2, 3, 3), dtype=np.uint8)}
        ),
        state_monotonic_ns=2_000,
        camera_monotonic_ns_by_role=MappingProxyType({"front": 1_900}),
    )


def test_factory_selects_interchangeable_act_and_diffusion_adapters():
    created = {}

    def build(backend_id, action):
        def builder():
            adapter = _ConstantAdapter(backend_id, action)
            created[backend_id] = adapter
            return adapter

        return builder

    factory = DigPolicyFactory(
        {
            "lerobot_act": build("lerobot_act", (0.1, -0.2, 0.3, -0.4)),
            "diffusion_policy": build(
                "diffusion_policy", (-0.4, 0.3, -0.2, 0.1)
            ),
        }
    )

    act = factory.create("lerobot_act")
    diffusion = factory.create("diffusion_policy")
    observation = _observation()

    assert act.descriptor.action_order == ACTION_ORDER
    assert diffusion.descriptor.action_order == ACTION_ORDER
    assert act.select_action(observation) == pytest.approx((0.1, -0.2, 0.3, -0.4))
    assert diffusion.select_action(observation) == pytest.approx(
        (-0.4, 0.3, -0.2, 0.1)
    )
    assert created["lerobot_act"].seen == [observation]
    assert created["diffusion_policy"].seen == [observation]


def test_factory_rejects_unknown_or_mislabelled_backends():
    adapter = _ConstantAdapter("not_act", (0.0, 0.0, 0.0, 0.0))
    factory = DigPolicyFactory({"lerobot_act": lambda: adapter})

    with pytest.raises(ValueError, match="unknown dig policy backend"):
        factory.create("typo")
    with pytest.raises(ValueError, match="descriptor backend_id"):
        factory.create("lerobot_act")


def test_checked_policy_saturates_finite_actions_to_normalized_manual_contract():
    factory = DigPolicyFactory(
        {
            "lerobot_act": lambda: _ConstantAdapter(
                "lerobot_act", (1.01, -1.024, 0.25, -0.5)
            )
        }
    )

    policy = factory.create("lerobot_act")

    assert policy.select_action(_observation()) == pytest.approx(
        (1.0, -1.0, 0.25, -0.5)
    )


def test_checked_policy_rejects_gross_finite_action_outliers():
    factory = DigPolicyFactory(
        {
            "lerobot_act": lambda: _ConstantAdapter(
                "lerobot_act", (1.26, 0.0, 0.0, 0.0)
            )
        }
    )

    with pytest.raises(ValueError, match="normalized manual action"):
        factory.create("lerobot_act").select_action(_observation())
