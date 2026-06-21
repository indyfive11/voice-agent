"""Item-C tests: out-of-band acoustic wake signal (producer side).

Covers the three producer pieces without a live pipeline:
  1. fusion (_fuse_bare_wake_likelihood) — duration-dominant, score-gated, monotonic;
  2. wake gate emits WakeEventFrame DOWNSTREAM on a FRESH wake (only when enabled);
  3. BrainLLMService stamps it, fuses + forwards a `wake` obj ONLY on the strip-failed path,
     consumes-and-clears per turn, and stays back-compat when no wake fired.
"""
import asyncio

from pipecat.frames.frames import (
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from brains.brain_client import BrainEvent, FakeBrainClient
from brains.brain_llm_service import BrainLLMService, _fuse_bare_wake_likelihood
from wake_word import WakeWordGate, WakeEventFrame, _CHUNK_BYTES

_CHUNK = b"\x00" * _CHUNK_BYTES


# ── 1. fusion (post-wake-dominant; default band post 250..1000, dur fallback 400..1500) ──────────
def _fuse(score, post, dur):
    return _fuse_bare_wake_likelihood(score, post, dur, post_floor=250, post_ceil=1000,
                                      dur_floor=400, dur_ceil=1500)


def test_fusion_small_postwake_is_high():
    # bare wake: wake fires then ~nothing (post ~130ms) → above the brain's 0.8 suppress floor.
    assert _fuse(0.99, 130, 800) >= 0.8


def test_fusion_large_postwake_is_low():
    # command followed (post ~1000ms+) → low, even if total duration is short.
    assert _fuse(0.99, 1000, 1200) < 0.2


def test_fusion_postwake_dominates_total_duration():
    # The inversion that broke total-duration: a SLOW bare wake (long total) with small post still scores
    # HIGH, while a FAST command (short total) with large post scores LOW. post wins.
    slow_bare = _fuse(0.99, 150, 1300)   # long total, tiny post → bare
    fast_cmd = _fuse(0.99, 900, 950)     # short total, big post → command
    assert slow_bare >= 0.8 > fast_cmd


def test_fusion_monotonic_in_postwake():
    f = lambda p: _fuse(1.0, p, 1000)
    assert f(200) > f(450) > f(700) > f(950)


def test_fusion_score_gated():
    assert _fuse(0.6, 150, 800) < _fuse(0.99, 150, 800)


def test_fusion_falls_back_to_duration_when_post_unmeasurable():
    # post=None (name trailing a sentence): a long total → low (it's a command/question, not a bare wake).
    assert _fuse(1.0, None, 3600) < 0.2
    # a short total with no post → bare-ish (the old duration heuristic).
    assert _fuse(1.0, None, 600) >= 0.8


def test_fusion_neither_measurable_is_uncertain_below_floor():
    assert _fuse(1.0, None, None) < 0.8


# ── 2. gate emission ─────────────────────────────────────────────────────────────────────────────
class _FakeOWW:
    def __init__(self):
        self.models = {"hey_jarvis_v0.1": None}
        self.score = 0.0

    def predict(self, pcm):
        return {"hey_jarvis_v0.1": self.score}


def _gate(**kw):
    kw.setdefault("threshold", 0.5)
    g = WakeWordGate(_FakeOWW(), brain_client=FakeBrainClient(), session_id="sess-test", **kw)
    pushed = []

    async def _spy(frame, direction=None):
        pushed.append((type(frame).__name__, direction))

    g.push_frame = _spy  # type: ignore[assignment]
    return g, pushed


def test_gate_emits_wake_event_on_fresh_wake_when_enabled():
    g, pushed = _gate(emit_wake_events=True)
    g._oww.score = 0.9
    asyncio.run(g._feed(_CHUNK))
    assert g._open is True
    assert ("WakeEventFrame", FrameDirection.DOWNSTREAM) in pushed


def test_gate_silent_when_disabled():
    g, pushed = _gate(emit_wake_events=False)
    g._oww.score = 0.9
    asyncio.run(g._feed(_CHUNK))
    assert g._open is True
    assert not any(n == "WakeEventFrame" for n, _ in pushed)


def test_gate_carries_score():
    g, pushed = _gate(emit_wake_events=True)
    captured = []

    async def _spy(frame, direction=None):
        captured.append(frame)

    g.push_frame = _spy  # type: ignore[assignment]
    g._oww.score = 0.87
    asyncio.run(g._feed(_CHUNK))
    ev = next(f for f in captured if isinstance(f, WakeEventFrame))
    assert abs(ev.score - 0.87) < 1e-6


def test_gate_emits_in_open_mic_without_opening_window():
    # Open-mic (media idle, gated=False): a wake must still EMIT the signal but NOT open a command window or
    # duck — the mic is already passing through. Closes the open-mic leak (item C re-drive).
    g, pushed = _gate(emit_wake_events=True)
    g._oww.score = 0.9
    asyncio.run(g._feed(_CHUNK, gated=False))
    assert any(n == "WakeEventFrame" for n, _ in pushed)  # signal fired
    assert g._open is False  # but NO command window opened in open-mic
    assert g._client.duck_calls == []  # and NO pre-duck (nothing playing)


def test_gate_still_opens_window_when_gated():
    # Regression: while gating (media playing) a wake still opens the window + pre-ducks AND emits the signal.
    g, pushed = _gate(emit_wake_events=True)
    g._oww.score = 0.9
    asyncio.run(g._feed(_CHUNK, gated=True))
    assert g._open is True
    assert any(n == "WakeEventFrame" for n, _ in pushed)


def test_gate_open_mic_silent_when_disabled():
    # Open-mic + WAKE_SIGNAL_FORWARD off → no frame, and still no window (no behavior change in open mic).
    g, pushed = _gate(emit_wake_events=False)
    g._oww.score = 0.9
    asyncio.run(g._feed(_CHUNK, gated=False))
    assert not any(n == "WakeEventFrame" for n, _ in pushed)
    assert g._open is False


def test_gate_no_emit_on_keepalive_reopen():
    # A media-keepalive re-open is NOT a fresh wake → no WakeEventFrame (it goes via _open_window, not _on_wake).
    g, pushed = _gate(emit_wake_events=True)

    async def go():
        g._open_window("WAKE  | keepalive reopen")  # sync, but needs a running loop for its create_task

    asyncio.run(go())
    assert not any(n == "WakeEventFrame" for n, _ in pushed)


# ── 3. brain stamp / fuse / forward / consume ───────────────────────────────────────────────────
def _svc(client):
    svc = BrainLLMService(client, session_id="sess-test")

    async def _rec(frame, direction=None):
        pass

    svc.push_frame = _rec
    return svc


def _ctx(text):
    from pipecat.processors.aggregators.llm_context import LLMContext
    return LLMContext(messages=[{"role": "user", "content": text}])


def _stamp_wake(svc, *, score=0.99, dur_ms=600, post_ms=130):
    """Simulate the producer for a BARE wake: a fresh wake fires near the END of the span (small post-wake),
    so the post-dominant fusion reads it as bare. dur_ms = total span; post_ms = speech after the wake."""
    asyncio.run(svc.process_frame(WakeEventFrame(score=score), FrameDirection.DOWNSTREAM))
    import time
    now = time.monotonic()
    svc._vad_start_ts = now - dur_ms / 1000.0
    svc._vad_stop_ts = now
    svc._pending_wake["ts"] = now - post_ms / 1000.0  # wake fired post_ms before the stop → small post = bare


def test_vad_span_uses_first_onset_through_last_stop():
    # Bug 2026-06-20: resetting _vad_start_ts on every onset measured only the LAST fragment of a
    # multi-segment utterance ("Play some Led Zeppelin" → 227ms). Capture FIRST onset … LAST stop instead.
    svc = _svc(FakeBrainClient())
    import time
    t0 = time.monotonic()

    async def go():
        await svc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)  # first onset
        await svc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)  # mid pause
        await svc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)  # 2nd onset (flicker)
        await svc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)  # last stop

    asyncio.run(go())
    # first onset preserved (not overwritten by the 2nd), last stop wins → span covers the whole utterance.
    assert svc._vad_start_ts is not None and svc._vad_stop_ts is not None
    assert svc._vad_start_ts <= t0 + 0.05  # ~the first onset, not the later one
    assert svc._vad_stop_ts >= svc._vad_start_ts


def test_post_wake_clamped_to_speech_dur():
    # GA's catch: frame jitter can put the wake frame a few ms before the VAD onset → post_wake > speech_dur,
    # which is impossible (post is a subset). Clamp post_wake <= speech_dur.
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    asyncio.run(svc.process_frame(WakeEventFrame(score=1.0), FrameDirection.DOWNSTREAM))
    import time
    now = time.monotonic()
    svc._vad_start_ts = now - 0.126          # 126ms span
    svc._vad_stop_ts = now
    svc._pending_wake["ts"] = now - 0.129    # wake 3ms before the onset (jitter) → raw post=129 > dur=126
    asyncio.run(svc._process_context(_ctx("how are you")))
    w = client.respond_wake[-1]
    assert w["post_wake_voiced_ms"] <= w["speech_dur_ms"]  # clamped


def test_vad_span_resets_between_turns():
    # The span must not bleed across turns — reset at the _process_context boundary.
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    asyncio.run(svc.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    asyncio.run(svc.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    asyncio.run(svc._process_context(_ctx("play jazz")))
    assert svc._vad_start_ts is None and svc._vad_stop_ts is None


def test_brain_stamps_wake_event():
    svc = _svc(FakeBrainClient())
    asyncio.run(svc.process_frame(WakeEventFrame(score=0.9), FrameDirection.DOWNSTREAM))
    assert svc._pending_wake is not None and abs(svc._pending_wake["score"] - 0.9) < 1e-6


def test_forwards_wake_on_strip_failed_path():
    # The bug: STT lost the name ("how are you"), strip fails (had_wake=False) → forward the signal.
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    _stamp_wake(svc, score=0.99, dur_ms=600)
    asyncio.run(svc._process_context(_ctx("how are you")))
    assert client.respond_wake[-1] is not None
    assert client.respond_wake[-1]["bare_wake_likelihood"] >= 0.8
    assert client.respond_wake[-1]["confidence"] == 0.99
    assert client.respond_wake[-1]["speech_dur_ms"] == 600
    assert svc._pending_wake is None  # consumed


def test_no_wake_on_clean_command():
    # Clean wake+command ("aria play jazz") → strip succeeds, name survived → no out-of-band signal.
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    _stamp_wake(svc)
    asyncio.run(svc._process_context(_ctx("aria play jazz")))
    assert client.respond_calls[-1][1] == "play jazz"  # wake stripped
    assert client.respond_wake[-1] is None
    assert svc._pending_wake is None  # cleared even though not forwarded


def test_back_compat_no_signal_when_no_wake_fired():
    # No producer (no WakeEventFrame) → strip-failed text forwards wake=None (exact prior behavior).
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    asyncio.run(svc._process_context(_ctx("how are you")))
    assert client.respond_wake[-1] is None


def test_consume_and_clear_one_turn_only():
    # A wake initiates exactly ONE turn; a follow-up with no fresh wake carries no signal.
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    _stamp_wake(svc)
    asyncio.run(svc._process_context(_ctx("how are you")))
    assert client.respond_wake[-1] is not None
    asyncio.run(svc._process_context(_ctx("what about now")))
    assert client.respond_wake[-1] is None  # no fresh wake → no signal


def test_stale_wake_dropped():
    # A wake that produced no prompt turn within the max-age window is ignored.
    client = FakeBrainClient(respond_events=[BrainEvent("done")])
    svc = _svc(client)
    _stamp_wake(svc)
    svc._pending_wake["ts"] -= svc._wake_signal_max_age + 5  # age it out
    asyncio.run(svc._process_context(_ctx("how are you")))
    assert client.respond_wake[-1] is None
