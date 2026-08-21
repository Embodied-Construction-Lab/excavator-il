"""Derive LeRobot Episode boundaries from raw collection quality events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


TRAINING_SEGMENTS_SCHEMA_VERSION = "excavator_training_segments.v1"
DEFAULT_RECOVERY_JOYSTICK_SAMPLES = 10


@dataclass(frozen=True)
class SafetyEvent:
    event_type: str
    event_stamp_monotonic_ns: int | None
    recovery_stamp_monotonic_ns: int | None

    @property
    def recovered(self) -> bool:
        return (
            self.event_stamp_monotonic_ns is not None
            and self.recovery_stamp_monotonic_ns is not None
        )


def _valid_joystick_samples(
    records: Iterable[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for record in records:
        if record.get("parse_ok") is not True:
            continue
        sequence = record.get("joystick_sample_seq")
        stamp = record.get("orin_receive_monotonic_ns")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            continue
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            continue
        samples.append((stamp, sequence))
    return sorted(samples)


def locate_joystick_timeout_events(
    command_records: Iterable[Mapping[str, Any]],
    joystick_records: Iterable[Mapping[str, Any]],
    *,
    recovery_sample_count: int = DEFAULT_RECOVERY_JOYSTICK_SAMPLES,
) -> tuple[SafetyEvent, ...]:
    """Locate timeout safe-zero events and a consecutive-packet recovery point."""
    if recovery_sample_count <= 0:
        raise ValueError("recovery_sample_count must be positive")
    timeout_records: list[tuple[int | None, bool]] = []
    for record in command_records:
        if record.get("command_kind") != "safe_zero:joystick_timeout":
            continue
        stamp = record.get("command_tx_monotonic_ns")
        timeout_records.append(
            (
                stamp
                if isinstance(stamp, int) and not isinstance(stamp, bool)
                else None,
                record.get("write_ok") is True,
            )
        )
    localized = sorted(
        (stamp, write_ok)
        for stamp, write_ok in timeout_records
        if stamp is not None
    )
    samples = _valid_joystick_samples(joystick_records)
    events: list[SafetyEvent] = []
    for index, (fault_stamp, safe_zero_written) in enumerate(localized):
        next_fault = localized[index + 1][0] if index + 1 < len(localized) else None
        run_length = 0
        previous_sequence: int | None = None
        recovery_stamp: int | None = None
        if safe_zero_written:
            for sample_stamp, sequence in samples:
                if sample_stamp <= fault_stamp:
                    continue
                if next_fault is not None and sample_stamp >= next_fault:
                    break
                run_length = (
                    run_length + 1
                    if previous_sequence is not None
                    and sequence == previous_sequence + 1
                    else 1
                )
                previous_sequence = sequence
                if run_length >= recovery_sample_count:
                    recovery_stamp = sample_stamp
                    break
        events.append(
            SafetyEvent(
                event_type="joystick_timeout",
                event_stamp_monotonic_ns=fault_stamp,
                recovery_stamp_monotonic_ns=recovery_stamp,
            )
        )
    events.extend(
        SafetyEvent(
            event_type="joystick_timeout",
            event_stamp_monotonic_ns=None,
            recovery_stamp_monotonic_ns=None,
        )
        for _ in range(len(timeout_records) - len(localized))
    )
    return tuple(events)


def state_is_quarantined(stamp_ns: int, events: Iterable[SafetyEvent]) -> bool:
    for event in events:
        fault = event.event_stamp_monotonic_ns
        if fault is None or stamp_ns < fault:
            continue
        recovery = event.recovery_stamp_monotonic_ns
        if recovery is None or stamp_ns <= recovery:
            return True
    return False


def build_training_segment_manifest(
    episode_id: str,
    rows: list[Mapping[str, Any]],
    events: tuple[SafetyEvent, ...],
    *,
    excluded_step_count: int,
    recovery_sample_count: int = DEFAULT_RECOVERY_JOYSTICK_SAMPLES,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    segment_start = 0
    localized_faults = tuple(
        event.event_stamp_monotonic_ns
        for event in events
        if event.event_stamp_monotonic_ns is not None
    )

    def append_segment(start: int, end: int) -> None:
        if start >= end:
            return
        first = rows[start]
        last = rows[end - 1]
        segments.append(
            {
                "segment_id": f"{episode_id}_segment_{len(segments):04d}",
                "start_frame_index": start,
                "end_frame_index_exclusive": end,
                "step_count": end - start,
                "start_state_receive_monotonic_ns": int(
                    first["state_receive_monotonic_ns"]
                ),
                "end_state_receive_monotonic_ns": int(
                    last["state_receive_monotonic_ns"]
                ),
            }
        )

    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]
        previous_stamp = int(previous["state_receive_monotonic_ns"])
        current_stamp = int(current["state_receive_monotonic_ns"])
        sequence_gap = int(current["state_seq"]) != int(previous["state_seq"]) + 1
        crosses_fault = any(
            previous_stamp < fault_stamp <= current_stamp
            for fault_stamp in localized_faults
        )
        if sequence_gap or crosses_fault:
            append_segment(segment_start, index)
            segment_start = index
    append_segment(segment_start, len(rows))

    return {
        "schema_version": TRAINING_SEGMENTS_SCHEMA_VERSION,
        "parent_episode_id": episode_id,
        "strategy": "lerobot_episode_boundaries",
        "recovery_joystick_sample_count": recovery_sample_count,
        "fault_events": [
            {**asdict(event), "recovered": event.recovered} for event in events
        ],
        "unresolved_safety_event_count": sum(
            event.event_stamp_monotonic_ns is None or not event.recovered
            for event in events
        ),
        "excluded_training_step_count": excluded_step_count,
        "segments": segments,
    }
