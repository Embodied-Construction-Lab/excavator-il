"""Bounded PC Mission lease heartbeat for the resident Orin owner."""

from __future__ import annotations

import math
import threading
from typing import Callable


class ResidentMissionLeaseHeartbeat:
    """Renew one Orin-side lease while a PC Hybrid Mission is alive.

    The heartbeat owns no motion state.  Missing heartbeats are handled by the
    Orin owner, which terminally disarms after its bounded lease expires.
    """

    def __init__(
        self,
        renew_lease: Callable[[], object],
        *,
        interval_s: float = 0.4,
    ) -> None:
        if not callable(renew_lease):
            raise ValueError("renew_lease must be callable")
        if (
            isinstance(interval_s, bool)
            or not isinstance(interval_s, (int, float))
            or not math.isfinite(float(interval_s))
            or not 0.1 <= float(interval_s) <= 0.5
        ):
            raise ValueError("interval_s must be finite and within [0.1, 0.5]")
        self._renew_lease = renew_lease
        self._interval_s = float(interval_s)
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Synchronously arm the first lease, then renew it in the background."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._raise_failure_locked()
                return
            self._failure = None

        self._renew_lease()
        stop = threading.Event()
        thread = threading.Thread(
            target=self._run,
            args=(stop,),
            name="resident-mission-lease",
            daemon=True,
        )
        with self._lock:
            self._stop = stop
            self._thread = thread
        thread.start()

    def require_healthy(self) -> None:
        with self._lock:
            self._raise_failure_locked()
            if self._thread is None or not self._thread.is_alive():
                raise RuntimeError("resident Mission lease heartbeat is not running")

    def request_stop(self) -> None:
        """Prevent any future renewal without waiting for an in-flight RPC.

        Safe-stop calls this before requesting terminal disarm.  An already
        running renewal may finish, but the Orin terminal disarm is
        irreversible and the heartbeat will not start another request.
        """

        with self._lock:
            stop = self._stop
        if stop is not None:
            stop.set()

    def stop(self) -> None:
        self.request_stop()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self._interval_s * 4.0))
            if thread.is_alive():
                raise RuntimeError("resident Mission lease heartbeat did not stop")
        with self._lock:
            if self._thread is thread:
                self._stop = None
                self._thread = None

    def _run(self, stop: threading.Event) -> None:
        while not stop.wait(self._interval_s):
            try:
                self._renew_lease()
            except BaseException as exc:
                with self._lock:
                    self._failure = exc
                return

    def _raise_failure_locked(self) -> None:
        failure = self._failure
        if failure is not None:
            raise RuntimeError("resident Mission lease heartbeat failed") from failure
