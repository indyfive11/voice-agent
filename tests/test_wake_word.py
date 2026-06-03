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
    g_quiet = _gate(media_only=True, media_status=lambda: {"playing": False, "kind": None})
    g_playing = _gate(media_only=True, media_status=lambda: {"playing": True, "kind": "audio"})
    assert asyncio.run(g_quiet._gated_now()) is False
    assert asyncio.run(g_playing._gated_now()) is True


def test_video_is_not_gated_by_default():
    # An active movie the user is watching → pass-through (they pause by hand); audio music → gated.
    g_video = _gate(media_only=True, media_status=lambda: {"playing": True, "kind": "video"})
    g_audio = _gate(media_only=True, media_status=lambda: {"playing": True, "kind": "audio"})
    assert asyncio.run(g_video._gated_now()) is False
    assert asyncio.run(g_audio._gated_now()) is True
    # opt-in: gate_video=True forces gating even for video
    g_forced = _gate(media_only=True, gate_video=True,
                     media_status=lambda: {"playing": True, "kind": "video"})
    assert asyncio.run(g_forced._gated_now()) is True


def test_always_gated_when_not_media_only():
    g = _gate(media_only=False, media_status=None)
    assert asyncio.run(g._gated_now()) is True


def test_escape_hatch_opens_after_repeated_near_misses():
    # Over loud media the weak model peaks 0.22-0.43 (< threshold) — the user is clearly trying but
    # locked out. After escape_count sub-threshold bursts the gate must open anyway.
    client = FakeBrainClient()
    g = _gate(client, escape_count=3, escape_floor=0.15)

    async def burst(peak):
        g._oww.score = peak       # ≥ escape_floor, < threshold
        await g._feed(_CHUNK)     # peak tracked
        g._oww.score = 0.0
        await g._feed(_CHUNK)     # burst ends → one hit recorded

    async def go():
        await burst(0.30)
        assert g._open is False   # 1 hit
        await burst(0.42)
        assert g._open is False   # 2 hits
        await burst(0.22)         # 3rd hit → escape

    asyncio.run(go())
    assert g._open is True
    assert client.duck_calls == [("sess-test", True)]  # escape pre-ducks like a real wake


def test_speex_ns_falls_back_when_unavailable(monkeypatch, tmp_path):
    # WAKE_WORD_SPEEX_NS=1 but the speexdsp-ns wheel is missing → build the gate WITHOUT it, don't crash.
    import openwakeword.model as owm
    import config

    class FakeModel:
        def __init__(self, *a, enable_speex_noise_suppression=False, **k):
            if enable_speex_noise_suppression:
                raise ImportError("speexdsp_ns not installed")
            self.models = {"aria": None}

    monkeypatch.setattr(owm, "Model", FakeModel)
    model_path = tmp_path / "aria.onnx"
    model_path.write_bytes(b"stub")
    monkeypatch.setenv("WAKE_WORD", str(model_path))
    monkeypatch.setenv("WAKE_WORD_SPEEX_NS", "1")

    gate = config.build_wake_word_gate(object())  # non-brain llm
    assert gate is not None  # fell back to no-speex instead of crashing


def test_hb_state_reports_peak_and_resets():
    g = _gate()
    g._oww.score = 0.33
    asyncio.run(g._feed(_CHUNK))          # sub-threshold, but the heartbeat peak still records it
    st = g.hb_state()
    assert st["wake_peak"] == 0.33
    assert st["open"] is False
    assert g.hb_state()["wake_peak"] == 0.0  # window reset after read


def test_escape_logs_each_hit():
    client = FakeBrainClient()
    g = _gate(client, escape_count=3, escape_floor=0.15)
    lines, remove = _capture_transcript()

    async def burst(peak):
        g._oww.score = peak
        await g._feed(_CHUNK)
        g._oww.score = 0.0
        await g._feed(_CHUNK)

    try:
        asyncio.run(burst(0.30))
        asyncio.run(burst(0.30))
    finally:
        remove()

    hits = [ln for ln in lines if "escape-hit" in ln]
    assert len(hits) == 2
    assert "escape-hit 1/3" in hits[0] and "escape-hit 2/3" in hits[1]


def test_single_near_miss_does_not_escape():
    client = FakeBrainClient()
    g = _gate(client, escape_count=3)

    async def go():
        g._oww.score = 0.30
        await g._feed(_CHUNK)
        g._oww.score = 0.0
        await g._feed(_CHUNK)     # only one burst

    asyncio.run(go())
    assert g._open is False
    assert client.duck_calls == []


def test_hold_keeps_window_open_past_idle_then_releases():
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.01)

    async def go():
        g._set_hold(True)                 # confirm pending → open + hold
        assert g._open is True
        await asyncio.sleep(0.05)         # well past the idle window
        assert g._open is True            # still open — held, not closed
        g._set_hold(False)                # answer received → release
        await asyncio.sleep(0.05)         # idle window now elapses
        assert g._open is False           # closed normally

    asyncio.run(go())


def test_escape_count_zero_disables_hatch():
    client = FakeBrainClient()
    g = _gate(client, escape_count=0)

    async def go():
        for _ in range(6):        # many bursts, but escape disabled
            g._oww.score = 0.30
            await g._feed(_CHUNK)
            g._oww.score = 0.0
            await g._feed(_CHUNK)

    asyncio.run(go())
    assert g._open is False


def _capture_transcript():
    """Capture loguru lines bound transcript=True (WAKE/USER/BOT…). Returns (lines, remove_fn)."""
    from loguru import logger

    lines: list[str] = []
    sink_id = logger.add(
        lambda m: lines.append(str(m)),
        filter=lambda r: r["extra"].get("transcript"),
        format="{message}",
    )
    return lines, lambda: logger.remove(sink_id)


def test_debug_logs_nearmiss_peak_once_per_burst():
    # A sub-threshold "Aria" (peaks 0.42, under the 0.5 bar) should emit ONE near-miss line at the peak.
    g = _gate(debug=True, debug_floor=0.2)
    lines, remove = _capture_transcript()

    async def go():
        for s in (0.25, 0.42, 0.30):  # one utterance: rises, peaks, falls — still above floor
            g._oww.score = s
            await g._feed(_CHUNK)
        g._oww.score = 0.0  # burst ends (drops below floor) → emit the peak
        await g._feed(_CHUNK)

    try:
        asyncio.run(go())
    finally:
        remove()

    nearmiss = [ln for ln in lines if "near-miss" in ln]
    assert len(nearmiss) == 1
    assert "0.42" in nearmiss[0]
    assert g._open is False  # never crossed threshold


def test_debug_off_emits_nothing():
    g = _gate(debug=False)  # default
    lines, remove = _capture_transcript()

    async def go():
        g._oww.score = 0.42  # sub-threshold
        await g._feed(_CHUNK)
        g._oww.score = 0.0
        await g._feed(_CHUNK)

    try:
        asyncio.run(go())
    finally:
        remove()

    assert not any("near-miss" in ln for ln in lines)


def test_real_wake_emits_no_nearmiss():
    # A clean wake (0.9) must not also log a near-miss — the peak resets on wake.
    g = _gate(debug=True, debug_floor=0.2)
    lines, remove = _capture_transcript()

    async def go():
        g._oww.score = 0.9
        await g._feed(_CHUNK)   # wake
        g._oww.score = 0.0
        await g._feed(_CHUNK)

    try:
        asyncio.run(go())
    finally:
        remove()

    assert g._open is True
    assert not any("near-miss" in ln for ln in lines)
