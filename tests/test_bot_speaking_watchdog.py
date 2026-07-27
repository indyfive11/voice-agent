"""BotSpeakingWatchdog — verify the co-signal-gated loop-break for the "broken record" incident.

The reSpeaker C-layer wedge can't be reproduced on demand, so these tests drive the escalation LOGIC
deterministically (the reason `_tick_once`/`_stalled_open`/`_frame_audio_secs` are extracted from the
sleep loop): no hardware, no live pipeline. They pin exactly the safety property GA's review demanded —
break the loop ONLY when the shared device is independently confirmed wedged, so a live narration is
never cut.
"""
import asyncio
import time

import pytest

from pipecat.frames.frames import TTSAudioRawFrame

import bot_speech
from bot_speaking_watchdog import BotSpeakingWatchdog, _frame_audio_secs


def _make(*, wedged=True, **kw):
    """A watchdog with broadcast_interruption stubbed (an unlinked processor has no pipeline).

    `wedged` seeds the input-stall co-signal (a plain callable; flip .value mid-test to simulate the
    device stalling / clearing). `calls` records interrupts AND fast-exit escalations.
    """
    signal = {"value": wedged}
    calls = {"interrupts": 0, "escalations": []}
    wd = BotSpeakingWatchdog(
        device_wedged=lambda: signal["value"],
        escalate_recovery=lambda reason: calls["escalations"].append(reason),
        tick_secs=0.0,
        **kw,
    )

    async def _spy_interrupt():
        calls["interrupts"] += 1

    wd.broadcast_interruption = _spy_interrupt  # type: ignore[method-assign]
    return wd, calls, signal


def _stuck(wd, *, produced=4.0, started_ago=30.0, last_audio_ago=30.0):
    """Put the watchdog in a 'bot-speaking hung open, audio long ceased' state."""
    now = time.monotonic()
    wd._speaking = True
    wd._produced_secs = produced
    wd._started_at = now - started_ago
    wd._last_audio_at = now - last_audio_ago


# ---- produced-audio accounting -------------------------------------------------------------------

def test_frame_audio_secs_pcm16_mono():
    f = TTSAudioRawFrame(audio=b"\x00\x00" * 16000, sample_rate=16000, num_channels=1)
    assert _frame_audio_secs(f) == pytest.approx(1.0)  # 16000 samples @ 16 kHz = 1.0 s


def test_frame_audio_secs_stereo_and_zero_rate():
    f = TTSAudioRawFrame(audio=b"\x00\x00" * 16000, sample_rate=16000, num_channels=2)
    assert _frame_audio_secs(f) == pytest.approx(0.5)
    assert _frame_audio_secs(TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=0, num_channels=1)) == 0.0


# ---- self-tuning deadline ------------------------------------------------------------------------

def test_deadline_floors_at_min():
    wd, _, _ = _make(min_deadline_secs=8.0, grace_secs=6.0)
    wd._produced_secs = 0.0
    assert wd._deadline() == 8.0


def test_deadline_extends_with_produced_audio():
    wd, _, _ = _make(min_deadline_secs=8.0, grace_secs=6.0)
    wd._produced_secs = 40.0
    assert wd._deadline() == 46.0  # a long reply's deadline rides its produced audio + grace


# ---- the co-signal gate (GA's Hole 1) ------------------------------------------------------------

def test_breaks_loop_only_during_a_confirmed_wedge(monkeypatch):
    """Stuck bot-speaking + device wedged → fast-track a clean restart (the real stop) + interrupt/unstick."""
    flags = []
    monkeypatch.setattr(bot_speech, "set_bot_speaking", lambda v: flags.append(v))
    wd, calls, _ = _make(wedged=True, grace_secs=6.0, min_deadline_secs=8.0)
    _stuck(wd)
    asyncio.run(wd._tick_once())
    assert len(calls["escalations"]) == 1   # the fast-exit — the action that actually stops the loop
    assert calls["interrupts"] == 1          # best-effort state hygiene alongside it
    assert flags == [False]                  # brain duck watchdog unstuck
    assert wd._broke is True


def test_no_action_without_the_wedge_cosignal():
    """The SAME stuck-open state, but the device is NOT wedged (a long narration's audio gap) → never
    touched (no restart, no interrupt). This is the property that stops us cutting a live reply."""
    wd, calls, _ = _make(wedged=False, grace_secs=6.0, min_deadline_secs=8.0)
    _stuck(wd)
    asyncio.run(wd._tick_once())
    assert calls["interrupts"] == 0 and calls["escalations"] == [] and wd._broke is False


def test_inert_when_no_cosignal_available():
    """No device_wedged callable (input watchdog disabled) ⇒ belt stays inert even if stuck."""
    wd = BotSpeakingWatchdog(device_wedged=None, tick_secs=0.0, grace_secs=6.0, min_deadline_secs=8.0)
    wd.broadcast_interruption = lambda: asyncio.sleep(0)  # type: ignore[method-assign]
    _stuck(wd)
    assert wd._stalled_open() is False


# ---- never false-fire on a live reply ------------------------------------------------------------

def test_no_fire_within_drain_window():
    """Inside produced+grace (audio may still be draining) → no action even if wedged co-signal is set."""
    wd, calls, _ = _make(wedged=True, grace_secs=6.0, min_deadline_secs=8.0)
    now = time.monotonic()
    wd._speaking = True
    wd._produced_secs = 5.0
    wd._started_at = now - 2.0        # just 2s in
    wd._last_audio_at = now - 2.0
    asyncio.run(wd._tick_once())
    assert calls["interrupts"] == 0


def test_streaming_audio_never_fires():
    """A reply still emitting audio (last_audio recent) → audio_ceased False → no action, even if wedged."""
    wd, calls, _ = _make(wedged=True, grace_secs=6.0, min_deadline_secs=8.0)
    now = time.monotonic()
    wd._speaking = True
    wd._produced_secs = 39.0
    wd._started_at = now - 40.0
    wd._last_audio_at = now - 1.0     # audio arrived 1s ago → still flowing
    asyncio.run(wd._tick_once())
    assert calls["interrupts"] == 0


# ---- one-shot + disarm ---------------------------------------------------------------------------

def test_breaks_once_per_span(monkeypatch):
    monkeypatch.setattr(bot_speech, "set_bot_speaking", lambda v: None)
    wd, calls, _ = _make(wedged=True, grace_secs=6.0, min_deadline_secs=8.0)
    _stuck(wd)
    asyncio.run(wd._tick_once())
    asyncio.run(wd._tick_once())      # still stuck+wedged, but already broke → no repeat interrupt
    assert calls["interrupts"] == 1


def test_wedge_clearing_before_deadline_no_action():
    """Device stalls then clears (usb-reset succeeds) before the drain deadline → belt never fires."""
    wd, calls, sig = _make(wedged=True, grace_secs=6.0, min_deadline_secs=8.0)
    now = time.monotonic()
    wd._speaking = True
    wd._produced_secs = 4.0
    wd._started_at = now - 2.0
    wd._last_audio_at = now - 2.0
    sig["value"] = False              # input recovered
    asyncio.run(wd._tick_once())
    assert calls["interrupts"] == 0


def test_disabled_is_inert():
    wd, _, _ = _make(enabled=False)
    assert wd._enabled is False
