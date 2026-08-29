"""Supervisor-to-evidence lifecycle translation for hybrid Missions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .hybrid_mission import HybridMissionSegment


class HybridMissionEvidenceLifecycle:
    """Translate supervisor lifecycle changes into one append-only evidence run."""

    def __init__(
        self,
        run: Any | None,
        *,
        cycle_targets: tuple[str, ...] = (),
    ) -> None:
        self._run = run
        self._cycle_targets = cycle_targets
        self._active_phase: tuple[int, str] | None = None
        self._active_cycle: int | None = None
        self._error = ""
        self._pending_finalization: tuple[
            str, Mapping[str, Any], str | None, bool
        ] | None = None

    @property
    def run_id(self) -> str:
        if self._run is None:
            return ""
        value = getattr(self._run, "run_id", None)
        if not isinstance(value, str) or not value:
            raise ValueError("evidence run must expose a non-empty run_id")
        return value

    @property
    def error(self) -> str:
        return self._error

    @property
    def finalization_pending(self) -> bool:
        return self._run is not None and self._pending_finalization is not None

    def record(self, event_type: str, payload: Mapping[str, Any]) -> bool:
        if self._run is None:
            return True
        try:
            self._run.append_event(
                event_type,
                {"run_id": self.run_id, **payload},
            )
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def start_mission(
        self,
        *,
        automatic: bool,
        requested_cycles: int,
        dig_target_id: str,
        dig_group_id: str = "all",
    ) -> None:
        self.record(
            "mission_started",
            {
                "automatic": automatic,
                "requested_cycles": requested_cycles,
                "dig_target_id": dig_target_id,
                "dig_group_id": dig_group_id,
            },
        )
        self.start_cycle(0)

    def start_cycle(self, cycle_index: int) -> None:
        if self._run is None or not 0 <= cycle_index < len(self._cycle_targets):
            return
        recorded = self.record(
            "cycle_started",
            {
                "cycle_index": cycle_index,
                "dig_target_id": self._cycle_targets[cycle_index],
            },
        )
        self._active_cycle = cycle_index if recorded else None

    def complete_cycle(self, outcome: str, *, completed_cycles: int) -> None:
        cycle_index = self._active_cycle
        if cycle_index is None:
            return
        self.record(
            "cycle_completed",
            {
                "cycle_index": cycle_index,
                "dig_target_id": self._cycle_targets[cycle_index],
                "outcome": outcome,
                "completed_cycles": completed_cycles,
            },
        )
        self._active_cycle = None

    def start_phase(self, stage: str, dig_target_id: str) -> None:
        if not stage.startswith("running_"):
            return
        phase = stage.removeprefix("running_")
        try:
            HybridMissionSegment(phase)
        except ValueError:
            return
        self.complete_phase("success")
        cycle_index = self._active_cycle
        if cycle_index is None:
            return
        recorded = self.record(
            "phase_started",
            {
                "cycle_index": cycle_index,
                "phase": phase,
                "dig_target_id": dig_target_id,
            },
        )
        self._active_phase = (cycle_index, phase) if recorded else None

    def complete_phase(self, outcome: str) -> None:
        active = self._active_phase
        if active is None:
            return
        cycle_index, phase = active
        self.record(
            "phase_completed",
            {
                "cycle_index": cycle_index,
                "phase": phase,
                "outcome": outcome,
            },
        )
        self._active_phase = None

    def finish(
        self,
        *,
        stage: str,
        error: str,
        requested_cycles: int,
        completed_cycles: int,
        automatic: bool,
    ) -> None:
        run = self._run
        if run is None:
            return
        if self._pending_finalization is not None:
            self.retry_finalize()
            return
        outcome = {
            "completed": "success",
            "cancelled": "cancelled",
        }.get(stage, "failure")
        self.complete_phase(outcome)
        self.complete_cycle(outcome, completed_cycles=completed_cycles)
        self.record(
            {
                "completed": "mission_completed",
                "cancelled": "mission_cancelled",
            }.get(stage, "mission_failed"),
            {
                "terminal_stage": stage,
                "requested_cycles": requested_cycles,
                "completed_cycles": completed_cycles,
                "error": error,
            },
        )
        intended_status = "success" if stage == "completed" else "failure"
        metrics = {
            "requested_cycles": requested_cycles,
            "completed_cycles": completed_cycles,
            "automatic": automatic,
            "terminal_stage": stage,
        }
        summary = f"hybrid Mission {stage}"
        preserve_error = bool(self._error)
        final_status = intended_status
        if self._error:
            final_status = "failure"
            metrics = {
                **metrics,
                "evidence_complete": False,
            }
            if intended_status == "success":
                metrics["intended_status"] = intended_status
            summary += f"; evidence recording failed: {self._error}"
        self._pending_finalization = (
            final_status,
            metrics,
            summary,
            preserve_error,
        )
        self._attempt_finalize()

    def retry_finalize(self) -> None:
        """Retry only publication; terminal events are never appended twice."""

        if self._pending_finalization is None or self._run is None:
            return
        self._attempt_finalize()

    def _attempt_finalize(self) -> None:
        run = self._run
        pending = self._pending_finalization
        if run is None or pending is None:
            return
        status, metrics, summary, preserve_error = pending
        try:
            run.finalize(status, metrics=metrics, summary=summary)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            if getattr(exc, "finalized", False) is True:
                self._run = None
                self._pending_finalization = None
        else:
            self._run = None
            self._pending_finalization = None
            if not preserve_error:
                self._error = ""
        finally:
            self._active_phase = None
            self._active_cycle = None
