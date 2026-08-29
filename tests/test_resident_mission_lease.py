import threading
import time

import pytest

from excavator_il.resident_mission_lease import ResidentMissionLeaseHeartbeat


def test_first_lease_is_armed_synchronously_and_periodically_renewed():
    calls = []
    renewed = threading.Event()

    def renew():
        calls.append(time.monotonic())
        if len(calls) >= 2:
            renewed.set()

    heartbeat = ResidentMissionLeaseHeartbeat(renew, interval_s=0.1)
    heartbeat.start()
    try:
        assert len(calls) == 1
        assert renewed.wait(0.5)
        heartbeat.require_healthy()
    finally:
        heartbeat.stop()


def test_renewal_failure_is_reported_without_retrying_forever():
    calls = []
    failed = threading.Event()

    def renew():
        calls.append(len(calls))
        if len(calls) > 1:
            failed.set()
            raise OSError("control link lost")

    heartbeat = ResidentMissionLeaseHeartbeat(renew, interval_s=0.1)
    heartbeat.start()
    assert failed.wait(0.5)
    deadline = time.monotonic() + 0.5
    while heartbeat.running and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        heartbeat.require_healthy()
    heartbeat.stop()
    assert len(calls) == 2


def test_one_transient_renewal_failure_is_retried_without_losing_the_heartbeat():
    calls = []
    recovered = threading.Event()

    def renew():
        calls.append(len(calls))
        if len(calls) == 2:
            raise OSError("one delayed SSH renewal")
        if len(calls) >= 3:
            recovered.set()

    heartbeat = ResidentMissionLeaseHeartbeat(
        renew,
        interval_s=0.1,
        failure_grace_s=0.4,
    )
    heartbeat.start()
    try:
        assert recovered.wait(0.8)
        heartbeat.require_healthy()
    finally:
        heartbeat.stop()


def test_start_failure_never_claims_a_running_heartbeat():
    heartbeat = ResidentMissionLeaseHeartbeat(
        lambda: (_ for _ in ()).throw(OSError("cannot arm")),
        interval_s=0.1,
    )

    with pytest.raises(OSError, match="cannot arm"):
        heartbeat.start()

    assert not heartbeat.running


def test_request_stop_prevents_future_renewals_without_waiting_for_inflight():
    calls = []
    renewal_entered = threading.Event()
    release_renewal = threading.Event()

    def renew():
        calls.append(len(calls))
        if len(calls) == 2:
            renewal_entered.set()
            assert release_renewal.wait(1.0)

    heartbeat = ResidentMissionLeaseHeartbeat(renew, interval_s=0.1)
    heartbeat.start()
    assert renewal_entered.wait(0.5)

    heartbeat.request_stop()
    release_renewal.set()
    heartbeat.stop()
    time.sleep(0.15)

    assert len(calls) == 2


@pytest.mark.parametrize("interval", [True, 0.09, 0.51, float("nan")])
def test_interval_is_bounded_below_the_orin_lease_timeout(interval):
    with pytest.raises(ValueError, match="interval_s"):
        ResidentMissionLeaseHeartbeat(lambda: None, interval_s=interval)
