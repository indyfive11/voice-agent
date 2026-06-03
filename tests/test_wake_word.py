"""Offline tests for the wake-word gate state machine — no audio, no openWakeWord model needed.

Drives WakeWordGate's internals directly (like the media-duck tests) with a fake oww model and a
FakeBrainClient, so the open/close/pre-duck/debounce/media-aware logic is verified without a running
pipeline. asyncio.run drives the async bodies (plain pytest, no pytest-asyncio).
"""

import asyncio

from brains.brain_client import FakeBrainClient
from wake_word import WakeWordGate, _CHUNK_BYTES


class FakeOWW:
    """Stand-in for an openWakeWord Model: .models keys + a settable .predict score."""

    def __init__(self):
        self.models = {"hey_jarvis_v0.1": None}
        self.score = 0.0

    def predict(self, pcm):
        return {"hey_jarvis_v0.1": self.score}


def _gate(client=None, **kw):
    kw.setdefault("threshold", 0.5)
    kw.setdefault("window_secs", 15.0)
    return WakeWordGate(FakeOWW(), brain_client=client, session_id="sess-test", **kw)


_CHUNK = b"\x00" * _CHUNK_BYTES  # one 80ms chunk (content irrelevant — fake oww ignores it)


def test_wake_opens_and_preducks():
    client = FakeBrainClient()
    g = _gate(client)
    g._oww.score = 0.9  # wake word present

    asyncio.run(g._feed(_CHUNK))
    assert g._open is True
    assert client.duck_calls == [("sess-test", True)]  # pre-duck fired on wake


def test_no_wake_stays_closed():
    client = FakeBrainClient()
    g = _gate(client)
    g._oww.score = 0.1  # below threshold

    asyncio.run(g._feed(_CHUNK))
    assert g._open is False
    assert client.duck_calls == []


def test_wake_is_debounced_within_refractory():
    client = FakeBrainClient()
    g = _gate(client, refractory_secs=1.5)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)  # wake
        await g._feed(_CHUNK)  # same utterance, within refractory → no second wake/duck

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # exactly one


def test_window_closes_and_restores():
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.01)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)          # open + pre-duck
        await asyncio.sleep(0.05)      # window elapses → close + restore

    asyncio.run(go())
    assert g._open is False
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_aware_gating_open_mic_when_quiet():
    # media_only: gate is required only while media plays. Quiet → not gated (open-mic).
    g_quiet = _gate(media_only=True, is_media_playing=lambda: False)
    g_playing = _gate(media_only=True, is_media_playing=lambda: True)
    assert asyncio.run(g_quiet._gated_now()) is False
    assert asyncio.run(g_playing._gated_now()) is True


def test_always_gated_when_not_media_only():
    g = _gate(media_only=False, is_media_playing=None)
    assert asyncio.run(g._gated_now()) is True
