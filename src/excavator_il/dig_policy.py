"""Algorithm seam for normalized digging policies.

The resident Mission owns motion authority.  A :class:`DigPolicy` only maps a
named observation to one normalized manual action in the authoritative axis
order; it never owns a camera, serial port, or Mission lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


ACTION_ORDER = ("boom", "stick", "bucket", "swing")
OUTPUT_SEMANTICS = "manual_action_normalized"
MAX_TOLERATED_NORMALIZED_MAGNITUDE = 1.25


@dataclass(frozen=True)
class DigPolicyDescriptor:
    """Stable identity and output contract for one digging policy Adapter."""

    backend_id: str
    implementation: str
    action_order: tuple[str, str, str, str] = ACTION_ORDER
    output_semantics: str = OUTPUT_SEMANTICS

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError("dig policy backend_id must be non-empty text")
        if not isinstance(self.implementation, str) or not self.implementation.strip():
            raise ValueError("dig policy implementation must be non-empty text")
        if tuple(self.action_order) != ACTION_ORDER:
            raise ValueError("dig policy action_order is not authoritative")
        if self.output_semantics != OUTPUT_SEMANTICS:
            raise ValueError("dig policy output_semantics must be normalized manual action")


@dataclass(frozen=True)
class DigPolicyObservation:
    """Named, backend-independent observation presented to digging policies."""

    state_by_name: Mapping[str, float]
    rgb_by_role: Mapping[str, Any]
    state_monotonic_ns: int
    camera_monotonic_ns_by_role: Mapping[str, int]

    def __post_init__(self) -> None:
        state = dict(self.state_by_name)
        images = dict(self.rgb_by_role)
        camera_stamps = dict(self.camera_monotonic_ns_by_role)
        if not state or not all(
            isinstance(name, str)
            and name
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for name, value in state.items()
        ):
            raise ValueError("dig policy named state must contain finite numeric values")
        if not images or set(images) != set(camera_stamps):
            raise ValueError("dig policy RGB roles and camera timestamps must match")
        if any(not isinstance(role, str) or not role for role in images):
            raise ValueError("dig policy RGB roles must be non-empty text")
        if (
            isinstance(self.state_monotonic_ns, bool)
            or not isinstance(self.state_monotonic_ns, int)
            or self.state_monotonic_ns < 0
        ):
            raise ValueError("dig policy state timestamp must be nonnegative")
        if any(
            isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0
            for stamp in camera_stamps.values()
        ):
            raise ValueError("dig policy camera timestamps must be nonnegative")
        object.__setattr__(
            self,
            "state_by_name",
            MappingProxyType({name: float(value) for name, value in state.items()}),
        )
        object.__setattr__(self, "rgb_by_role", MappingProxyType(images))
        object.__setattr__(
            self,
            "camera_monotonic_ns_by_role",
            MappingProxyType(camera_stamps),
        )

    @property
    def state(self) -> tuple[float, ...]:
        """Compatibility view; semantic consumers should use ``state_by_name``."""

        return tuple(self.state_by_name.values())

    @property
    def front_rgb(self) -> Any:
        """Compatibility view for the current single-camera ACT Adapter."""

        try:
            return self.rgb_by_role["front"]
        except KeyError as exc:
            raise ValueError("dig policy observation is missing front RGB") from exc

    @property
    def camera_monotonic_ns(self) -> int:
        """Compatibility timestamp for the front camera."""

        try:
            return self.camera_monotonic_ns_by_role["front"]
        except KeyError as exc:
            raise ValueError("dig policy observation is missing front timestamp") from exc


class DigPolicy(Protocol):
    """Algorithm Interface used by ACT and future Diffusion Policy Adapters."""

    @property
    def descriptor(self) -> DigPolicyDescriptor:
        ...

    def select_action(
        self, observation: DigPolicyObservation
    ) -> tuple[float, float, float, float]:
        ...

    def warmup(self) -> tuple[float, float, float, float]:
        ...

    def reset(self) -> None:
        ...


class _CheckedDigPolicy:
    """Keep backend-specific failures behind one semantic Interface."""

    def __init__(self, adapter: DigPolicy) -> None:
        self._adapter = adapter

    @property
    def descriptor(self) -> DigPolicyDescriptor:
        return self._adapter.descriptor

    def select_action(
        self, observation: DigPolicyObservation
    ) -> tuple[float, float, float, float]:
        return saturate_normalized_action(self._adapter.select_action(observation))

    def warmup(self) -> tuple[float, float, float, float]:
        return saturate_normalized_action(self._adapter.warmup())

    def reset(self) -> None:
        self._adapter.reset()

    def consume_new_action_chunk(
        self,
    ) -> tuple[tuple[float, float, float, float], ...] | None:
        consume = getattr(self._adapter, "consume_new_action_chunk", None)
        if not callable(consume):
            return None
        raw = consume()
        if raw is None:
            return None
        chunk = tuple(saturate_normalized_action(action) for action in raw)
        if len(chunk) != 10:
            raise ValueError("dig policy execution chunk must contain ten actions")
        return chunk


class DigPolicyFactory:
    """Strict backend selection without policy conditionals in Mission code."""

    def __init__(self, builders: Mapping[str, Callable[[], DigPolicy]]) -> None:
        copied = dict(builders)
        if not copied or any(
            not isinstance(name, str) or not name.strip() or not callable(builder)
            for name, builder in copied.items()
        ):
            raise ValueError("dig policy builders must be a non-empty named mapping")
        self._builders = MappingProxyType(copied)

    @property
    def backend_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def create(self, backend_id: str) -> DigPolicy:
        if backend_id not in self._builders:
            raise ValueError(
                "unknown dig policy backend %r; expected one of %s"
                % (backend_id, list(self.backend_ids))
            )
        adapter = self._builders[backend_id]()
        descriptor = getattr(adapter, "descriptor", None)
        if not isinstance(descriptor, DigPolicyDescriptor):
            raise ValueError("dig policy Adapter descriptor is missing or invalid")
        if descriptor.backend_id != backend_id:
            raise ValueError("dig policy descriptor backend_id does not match selection")
        return _CheckedDigPolicy(adapter)


def saturate_normalized_action(values: Any) -> tuple[float, float, float, float]:
    """Clamp small numeric overshoot while rejecting gross policy outliers."""

    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("dig policy must return a normalized manual action") from exc
    if len(raw) != len(ACTION_ORDER) or any(isinstance(value, bool) for value in raw):
        raise ValueError("dig policy must return a normalized manual action")
    converted = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("dig policy must return a normalized manual action")
    if any(abs(value) > MAX_TOLERATED_NORMALIZED_MAGNITUDE for value in converted):
        raise ValueError("dig policy must return a normalized manual action")
    return tuple(max(-1.0, min(1.0, value)) for value in converted)  # type: ignore[return-value]
