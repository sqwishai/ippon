"""The pressure monitor uses fake readings, so tests never make the host short of memory."""

from __future__ import annotations

import time

import pytest

from ippon.pressure import Level, Monitor, Reading, Thresholds


@pytest.mark.parametrize("gib", [8, 16, 48, 128])
def test_pressure_thresholds_hysteresis_and_kernel_warnings(gib: int) -> None:
    memory = gib << 30
    limits = Thresholds.of(memory)
    reading = Reading(100, memory, limits.clear, 200, 1)
    monitor = Monitor(lambda: reading)
    assert monitor.sample() is Level.NORMAL
    for available, kernel, expected in [
        (limits.soft - 1, 1, Level.SOFT),
        (limits.soft + 1, 1, Level.SOFT),
        (limits.clear, 1, Level.NORMAL),
        (limits.clear, 2, Level.SOFT),
        (limits.clear, 4, Level.HARD),
        (limits.clear, 1, Level.NORMAL),
        (limits.hard - 1, 1, Level.HARD),
    ]:
        reading = Reading(100, memory, available, 200, kernel)
        assert monitor.sample() is expected
    assert monitor.status()["compressed"] == 200


def test_monitor_retries_failed_readings_and_keeps_polling_through_a_long_request() -> None:
    """A request still running past the recent-activity window keeps readings at the busy cadence (an idle monitor
    would read once in that time); a closed monitor reads no more."""
    attempts = 0

    def read() -> Reading:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("memory reading unavailable")
        return Reading(100, 16 << 30, 8 << 30, 200, 1)

    monitor = Monitor(read, busy=0.01, idle=0.3)
    assert monitor.start()
    try:
        assert monitor.status()["pressure"] == "unknown"
        assert monitor.status()["monitor_error"] == "memory reading unavailable"
        monitor.begin()
        time.sleep(0.4)  # the accepted request lasts longer than the recent-activity window
        before = monitor.readings
        time.sleep(0.15)
        assert monitor.readings - before >= 3
        assert monitor.status()["pressure"] == "normal"
        assert monitor.status()["monitor_error"] is None
        monitor.end()
    finally:
        monitor.close()
    stopped = monitor.readings
    time.sleep(0.05)
    assert monitor.readings == stopped
