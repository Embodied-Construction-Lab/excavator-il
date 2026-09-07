"""Lightweight online phase contract for the exploratory full-action ACT policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .act_runtime import ActObservation


PHASE_SCHEDULE_SCHEMA_VERSION = "excavator_act_phase_schedule.v1"
PHASE_CONDITIONED_DATASET_SCHEMA_VERSION = (
    "excavator_act_phase_conditioned_dataset.v1"
)
PHASE_ORDER = ("DIG", "TRANSPORT", "DUMP")
PHASE_FEATURE_NAMES = (
    "phase_is_dig",
    "phase_is_transport",
    "phase_is_dump",
)


@dataclass(frozen=True)
class ActPhaseSchedule:
    """Map completed ACT steps to one immutable three-way phase input."""

    dig_to_transport_step: int
    transport_to_dump_step: int

    def __post_init__(self) -> None:
        boundaries = (self.dig_to_transport_step, self.transport_to_dump_step)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in boundaries
        ):
            raise ValueError("ACT phase boundaries must be integers")
        if not 0 < self.dig_to_transport_step < self.transport_to_dump_step:
            raise ValueError("ACT phase boundaries must be strictly increasing")

    def phase_index(self, completed_steps: int) -> int:
        if (
            isinstance(completed_steps, bool)
            or not isinstance(completed_steps, int)
            or completed_steps < 0
        ):
            raise ValueError("completed ACT steps must be a non-negative integer")
        if completed_steps < self.dig_to_transport_step:
            return 0
        if completed_steps < self.transport_to_dump_step:
            return 1
        return 2

    def phase_name(self, completed_steps: int) -> str:
        return PHASE_ORDER[self.phase_index(completed_steps)]

    def state_by_name(self, completed_steps: int) -> dict[str, float]:
        active = self.phase_index(completed_steps)
        return {
            name: 1.0 if index == active else 0.0
            for index, name in enumerate(PHASE_FEATURE_NAMES)
        }

    def condition(
        self,
        observation: ActObservation,
        *,
        completed_steps: int,
    ) -> ActObservation:
        if observation.extra_state_by_name:
            raise ValueError("ACT observation already contains extra state")
        return replace(
            observation,
            extra_state_by_name=self.state_by_name(completed_steps),
        )
