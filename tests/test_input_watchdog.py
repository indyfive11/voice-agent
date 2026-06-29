"""Offline tests for the input-stall watchdog — no audio device, no running pipeline.

Drives the detector's `_tick`/`_is_silent`/`_on_stall` directly (like the turn_cap tests drive `_cap`),
so the no-frames / frames-but-silent detection + best-effort restart logic is verified without the
asyncio sleep loop or a real transport. Captures the loguru transcript sink to assert the log lines.
"""

import asyncio
import time

import numpy as np

from input_watchdog import InputStallDetector


def _capture_transcript():
    from loguru import logger

    lines: list[str] = []
    sink_id = logger.add(
        lambda m: lines.append(str(m)),
        filter=lambda r: r["extra"].get("transcript"),
        format="{message}",
    )
    return lines, lambda: logger.remove(sink_id)


def _pcm(amplitude: int, n: int = 320) -> bytes:
    return (np.full(n, amplitude, dtype=np.int16)).tobytes()


def test_is_silent_discriminates_dead_source_from_quiet_room():
    d = InputStallDetector(silence_eps=4)
    assert d._is_silent(b"") is True
    assert d._is_silent(_pcm(0)) is True          # dead source: exact zeros
    assert d._is_silent(_pcm(2)) is True          # below epsilon: still "silent"
    assert d._is_silent(_pcm(50)) is False        # quiet-room noise floor: NOT silent
    assert d._is_silent(_pcm(8000)) is False      # speech


def test_no_frames_stall_triggers_restart():
    calls = []

    async def restart(start_frame):
        calls.append(start_frame)
        return True

    d = InputStallDetector(stall_secs=5.0, restart=restart)
    d._armed = True
    now = time.monotonic()
    d._last = now - 10            # no frames for 10s
    d._last_nonsilent = now - 10
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()

    assert any("INPUT STALL | no mic frames" in ln for ln in lines)
    assert len(calls) == 1        # one restart attempt


def test_silent_capture_stall_triggers_restart():
    # The 2026-06-03 failure: frames KEEP arriving (so _last is fresh) but they're silent.
    calls = []

    async def restart(_):
        calls.append(1)
        return True

    d = InputStallDetector(stall_secs=5.0, silent_secs=8.0, restart=restart)
    d._armed = True
    now = time.monotonic()
    d._last = now                 # frames flowing
    d._last_nonsilent = now - 12  # but silent for 12s
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()

    assert any("INPUT STALL | frames but silent" in ln for ln in lines)
    assert len(calls) == 1


def test_not_armed_no_warning_before_first_frame():
    d = InputStallDetector(stall_secs=5.0, restart=None)
    # never armed (no frame seen) → a tick must do nothing even with stale clocks
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()
    assert lines == []


def test_no_first_frame_stall_triggers_restart():
    # The 2026-06-28 EM blind spot: StartFrame arrives, capture is claimed, but NO audio frame ever
    # comes (device opened/dead). Before this fix the watch loop never started → deaf forever.
    calls = []

    async def restart(_):
        calls.append(1)
        return True

    d = InputStallDetector(stall_secs=5.0, first_frame_secs=15.0, restart=restart)
    # StartFrame seen (arms the deadline) but never armed by an audio frame.
    d._started_at = time.monotonic() - 20  # 20s since StartFrame, no first frame
    assert d._armed is False
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()

    assert any("INPUT STALL | no first mic frame" in ln for ln in lines)
    assert len(calls) == 1


def test_no_first_frame_within_grace_does_not_warn():
    # Within the first_frame grace window, a not-yet-armed detector must stay quiet (normal startup).
    d = InputStallDetector(stall_secs=5.0, first_frame_secs=15.0, restart=None)
    d._started_at = time.monotonic() - 3  # only 3s since StartFrame
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()
    assert lines == []


def test_no_first_frame_escalates_after_restarts_exhausted():
    # A claimed-but-dead capture that never sends a first frame must escalate (exit→systemd re-init),
    # not latch inert — the actual EM failure mode (deaf 3.5 days across 4 restarts).
    restarts, escalations = [], []

    async def restart(_):
        restarts.append(1)
        return True

    d = InputStallDetector(stall_secs=5.0, first_frame_secs=15.0, restart=restart, max_restarts=2,
                           on_unrecoverable=lambda: escalations.append(1))
    d._started_at = time.monotonic() - 20  # past the deadline, never arms
    lines, remove = _capture_transcript()
    try:
        async def go():
            for _ in range(6):
                await d._tick()
        asyncio.run(go())
    finally:
        remove()

    assert len(restarts) == 2
    assert len(escalations) == 1
    assert any("escalating" in ln for ln in lines)


def test_gives_up_after_max_restarts():
    calls = []

    async def restart(_):
        calls.append(1)
        return True

    d = InputStallDetector(stall_secs=5.0, restart=restart, max_restarts=2)
    d._armed = True
    now = time.monotonic()
    d._last = now - 10
    d._last_nonsilent = now - 10

    lines, remove = _capture_transcript()
    try:
        async def go():
            for _ in range(5):     # tick repeatedly while still stalled
                await d._tick()
        asyncio.run(go())
    finally:
        remove()

    assert len(calls) == 2        # bounded at max_restarts
    assert any("still dead after 2 restart attempts" in ln for ln in lines)


def test_escalates_once_after_restarts_exhausted():
    # When in-process kicks are exhausted on a source that won't revive, the watchdog hands off to
    # on_unrecoverable (main.py exits → systemd re-inits) — exactly once, at the give-up point.
    restarts, escalations = [], []

    async def restart(_):
        restarts.append(1)
        return True

    d = InputStallDetector(stall_secs=5.0, restart=restart, max_restarts=2,
                           on_unrecoverable=lambda: escalations.append(1))
    d._armed = True
    now = time.monotonic()
    d._last = now - 10            # no frames → stalled, never resumes
    d._last_nonsilent = now - 10

    lines, remove = _capture_transcript()
    try:
        async def go():
            for _ in range(6):    # tick past the restart cap
                await d._tick()
        asyncio.run(go())
    finally:
        remove()

    assert len(restarts) == 2          # bounded at max_restarts
    assert len(escalations) == 1       # escalated exactly once, not every tick after
    assert any("escalating" in ln for ln in lines)


def test_no_escalation_when_handler_is_none():
    # Default / INPUT_STALL_EXIT_ON_FAIL=0: log-only, no crash, no escalation (bare/interactive runs).
    async def restart(_):
        return True

    d = InputStallDetector(stall_secs=5.0, restart=restart, max_restarts=1, on_unrecoverable=None)
    d._armed = True
    now = time.monotonic()
    d._last = now - 10
    d._last_nonsilent = now - 10

    lines, remove = _capture_transcript()
    try:
        async def go():
            for _ in range(4):
                await d._tick()
        asyncio.run(go())
    finally:
        remove()

    assert any("still dead after 1 restart attempts" in ln for ln in lines)
    assert not any("escalating" in ln for ln in lines)


def test_no_escalation_before_restarts_exhausted():
    # A single stall tick that still has restart budget must NOT escalate.
    d = InputStallDetector(stall_secs=5.0, restart=(lambda _: _async_true()), max_restarts=3,
                           on_unrecoverable=lambda: (_ for _ in ()).throw(AssertionError("escalated early")))
    d._armed = True
    now = time.monotonic()
    d._last = now - 10
    d._last_nonsilent = now - 10
    asyncio.run(d._tick())        # one attempt; budget remains → no escalation (handler would raise)


async def _async_true():
    return True


def test_heartbeat_emits_frame_and_gate_state():
    g = {"gated": True, "open": False, "wake_peak": 0.18}
    d = InputStallDetector(stall_secs=5.0, heartbeat_secs=10.0, gate_state=lambda: g)
    d._armed = True
    now = time.monotonic()
    d._last = now
    d._last_nonsilent = now
    d._frames = 120
    d._silent = 5
    d._hb_at = now - 11        # a beat is due
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()
    hb = [ln for ln in lines if ln.startswith("INPUT | hb")]
    assert len(hb) == 1
    assert "frames=+120" in hb[0] and "silent=+5" in hb[0]
    assert "gated=True open=False wake_peak=0.18" in hb[0]   # lockout signature visible


def test_heartbeat_off_when_disabled():
    d = InputStallDetector(stall_secs=5.0, heartbeat_secs=0)  # disabled
    d._armed = True
    now = time.monotonic()
    d._last = now
    d._last_nonsilent = now
    d._hb_at = now - 100
    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()
    assert not any(ln.startswith("INPUT | hb") for ln in lines)


def test_resume_logs_and_resets():
    d = InputStallDetector(stall_secs=5.0, silent_secs=8.0, restart=None)
    d._armed = True
    d._stalled = True             # was stalled
    d._restarts = 3
    now = time.monotonic()
    d._last = now                 # frames flowing AND
    d._last_nonsilent = now       # non-silent → recovered

    lines, remove = _capture_transcript()
    try:
        asyncio.run(d._tick())
    finally:
        remove()

    assert any("INPUT RESUMED" in ln for ln in lines)
    assert d._stalled is False
    assert d._restarts == 0
