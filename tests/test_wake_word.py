"""Offline tests for the wake-word gate state machine — no audio, no openWakeWord model needed.

Drives WakeWordGate's internals directly (like the media-duck tests) with a fake oww model and a
FakeBrainClient, so the open/close/pre-duck/debounce/media-aware logic is verified without a running
pipeline. asyncio.run drives the async bodies (plain pytest, no pytest-asyncio).
"""

import asyncio

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

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


def test_wake_window_mutes_media_by_default():
    # On wake-window-open the gate sends /media/duck {mute:true} → full 0% for the command window.
    client = FakeBrainClient()
    g = _gate(client)
    g._oww.score = 0.9

    asyncio.run(g._feed(_CHUNK))
    assert g._open is True
    assert client.duck_mute_calls == [("sess-test", True, True)]  # on=True, mute=True


def test_wake_window_mute_off_when_disabled():
    # WAKE_WINDOW_MUTE off → a plain pre-duck (mute=False), not a full mute.
    client = FakeBrainClient()
    g = _gate(client, window_mute=False)
    g._oww.score = 0.9

    asyncio.run(g._feed(_CHUNK))
    assert g._open is True
    assert client.duck_mute_calls == [("sess-test", True, False)]


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


def test_sustained_wake_needs_consecutive_frames():
    # consec_frames=3: a single over-threshold frame must NOT open; three in a row must.
    client = FakeBrainClient()
    g = _gate(client, consec_frames=3)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)          # consec=1
        assert g._open is False
        await g._feed(_CHUNK)          # consec=2
        assert g._open is False
        await g._feed(_CHUNK)          # consec=3 → wake

    asyncio.run(go())
    assert g._open is True
    assert client.duck_calls == [("sess-test", True)]


def test_brief_spike_rejected_and_consec_resets_on_gap():
    # A 1-frame music spike (then a sub-threshold frame) never reaches the requirement.
    client = FakeBrainClient()
    g = _gate(client, consec_frames=3)

    async def go():
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=2
        g._oww.score = 0.1; await g._feed(_CHUNK)   # below → resets to 0
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=2 (never 3 in a row)

    asyncio.run(go())
    assert g._open is False
    assert client.duck_calls == []


def test_consec_two_recovers_real_wake_rejects_single_frame():
    # The 2026-06-14 fix: consec_frames=2 must FIRE on two consecutive frames (real "Aria" over a movie
    # flickered and missed the 3-in-a-row bar) while still rejecting an isolated 1-frame phantom.
    client = FakeBrainClient()
    g = _gate(client, consec_frames=2)

    async def go():
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1 — must NOT fire yet
        assert g._open is False
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=2 → wake

    asyncio.run(go())
    assert g._open is True
    assert client.duck_calls == [("sess-test", True)]


def test_consec_two_single_spike_rejected():
    # A lone 1-frame spike (the asleep phantom) followed by a sub-threshold frame never reaches 2.
    client = FakeBrainClient()
    g = _gate(client, consec_frames=2)

    async def go():
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1
        g._oww.score = 0.0; await g._feed(_CHUNK)   # resets to 0

    asyncio.run(go())
    assert g._open is False
    assert client.duck_calls == []


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


def test_force_gated_while_asleep_overrides_media_state():
    # While asleep the gate is FORCED active even with nothing playing, so ambient TV never reaches STT
    # and only an acoustic "hey aria" can wake her. A WakeSleepFrame(asleep=True) flips it; (False) clears.
    from pipecat.processors.frame_processor import FrameDirection
    from wake_word import WakeSleepFrame

    g = _gate(media_only=True, media_status=lambda: {"playing": False, "kind": None})
    assert asyncio.run(g._gated_now()) is False                  # quiet room → open mic normally

    asyncio.run(g.process_frame(WakeSleepFrame(asleep=True), FrameDirection.UPSTREAM))
    assert g._force_gated is True
    assert asyncio.run(g._gated_now()) is True                   # forced gated despite nothing playing

    asyncio.run(g.process_frame(WakeSleepFrame(asleep=False), FrameDirection.UPSTREAM))
    assert g._force_gated is False
    assert asyncio.run(g._gated_now()) is False                  # awake → media-aware gating restored


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


def test_escape_ignores_single_frame_spikes_when_sustain_required():
    # consec_frames=2 → an escape-hit needs a burst of >=2 frames. A 1-frame music spike (even at 0.99)
    # must NOT count; only a sustained sub-threshold burst (a real clamped "hey aria") does.
    client = FakeBrainClient()
    g = _gate(client, escape_count=2, escape_floor=0.15, consec_frames=2)

    async def spike(peak):           # 1 frame above floor → 1-frame burst (music blip)
        g._oww.score = peak; await g._feed(_CHUNK)
        g._oww.score = 0.0;  await g._feed(_CHUNK)

    async def sustained(peak):       # 2 frames above floor → 2-frame burst (genuine attempt)
        g._oww.score = peak; await g._feed(_CHUNK)
        g._oww.score = peak; await g._feed(_CHUNK)
        g._oww.score = 0.0;  await g._feed(_CHUNK)

    async def go():
        await spike(0.99); await spike(0.99); await spike(0.99)  # 3 music blips → zero hits
        assert g._open is False
        await sustained(0.40)        # 1 sustained hit
        assert g._open is False
        await sustained(0.40)        # 2nd → escape opens

    asyncio.run(go())
    assert g._open is True


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
    # Pin the engine: speex fallback is openWakeWord-only. Without this the ambient .env
    # (WAKE_WORD_ENGINE=nano) hijacks the build down the nano path → loads the stub as a nano
    # model → InvalidProtobuf. Pinning makes the test deterministic regardless of .env / ordering.
    monkeypatch.setenv("WAKE_WORD_ENGINE", "oww")
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


# --- F3 gate hardening: in-flight guard + min-dwell hysteresis ---------------
class _FakeClock:
    """Injectable monotonic clock so the dwell timer is deterministic (no real sleeps)."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_inflight_guard_blocks_gating_flip_midutterance():
    # Gate is pass-through; media flips to "playing" WHILE an utterance is in flight → the gate must NOT
    # start gating (that would swallow the command). It flips only once the utterance ends.
    clk = _FakeClock()
    g = _gate(min_dwell_secs=0.0, guard_inflight=True, time_source=clk)
    assert g._effective_gated(False) is False        # seed committed = pass-through
    g._speech_in_flight = True
    assert g._effective_gated(True) is False          # guarded — stays pass-through
    assert g._effective_gated(True) is False
    g._speech_in_flight = False                       # VADUserStoppedSpeaking
    assert g._effective_gated(True) is True            # now it may gate (min_dwell=0 → instant)


def test_inflight_guard_off_allows_midutterance_flip():
    clk = _FakeClock()
    g = _gate(min_dwell_secs=0.0, guard_inflight=False, time_source=clk)
    assert g._effective_gated(False) is False
    g._speech_in_flight = True
    assert g._effective_gated(True) is True             # not guarded


def test_no_vad_frames_behaves_like_today():
    # If VAD frames never flow (some transports), _speech_in_flight stays False → no regression: the gate
    # flips on the raw decision exactly as before.
    clk = _FakeClock()
    g = _gate(min_dwell_secs=0.0, guard_inflight=True, time_source=clk)
    assert g._effective_gated(False) is False
    assert g._effective_gated(True) is True


def test_min_dwell_blocks_transient_blip():
    # A media-state blip shorter than the dwell must NOT toggle the effective gate.
    clk = _FakeClock()
    g = _gate(min_dwell_secs=2.0, guard_inflight=False, time_source=clk)
    assert g._effective_gated(False) is False          # seed
    assert g._effective_gated(True) is False            # t=0: pending, not committed
    clk.advance(1.0)
    assert g._effective_gated(True) is False            # t=1 < 2: still pending
    assert g._effective_gated(False) is False           # blip ends before dwell → pending cleared
    clk.advance(5.0)
    assert g._effective_gated(False) is False           # never flipped


def test_min_dwell_commits_after_persistence():
    clk = _FakeClock()
    g = _gate(min_dwell_secs=2.0, guard_inflight=False, time_source=clk)
    assert g._effective_gated(False) is False
    assert g._effective_gated(True) is False            # t=0 pending
    clk.advance(2.0)
    assert g._effective_gated(True) is True              # dwell elapsed → commit


def test_min_dwell_zero_is_instant_flip():
    clk = _FakeClock()
    g = _gate(min_dwell_secs=0.0, guard_inflight=False, time_source=clk)
    assert g._effective_gated(False) is False
    assert g._effective_gated(True) is True              # old instant behaviour


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


def test_post_wake_duplicate_labeled_refractory_not_sustain():
    # The exact 2026-06-05 log pattern: a wake fires (0.9), then a SECOND >=threshold frame lands inside the
    # refractory window. With _consec_required=1 it can't be a sustain failure — it's a refractory-swallowed
    # duplicate, and the near-miss reason must say so (not the misleading "didn't sustain 1 frames").
    g = _gate(debug=True, debug_floor=0.2, refractory_secs=2.0)
    lines, remove = _capture_transcript()

    async def go():
        g._oww.score = 0.9
        await g._feed(_CHUNK)   # real wake fires, opens window
        g._oww.score = 0.9
        await g._feed(_CHUNK)   # duplicate within refractory → near-miss burst
        g._oww.score = 0.0
        await g._feed(_CHUNK)   # burst ends → emit the line

    try:
        asyncio.run(go())
    finally:
        remove()

    nearmiss = [ln for ln in lines if "near-miss" in ln]
    assert len(nearmiss) == 1
    assert "duplicate within refractory" in nearmiss[0]
    assert "didn't sustain" not in nearmiss[0]


def test_nearmiss_reports_peaked_consec_count():
    # 2026-06-14 instrumentation: a high-peak burst that fails to sustain reports the BEST consecutive run
    # it achieved ("peaked N/M frames"), so the next session shows whether a lower consec would have fired
    # it (peaked 2/3 → yes) or the wake is flickering 1-frame-at-a-time (peaked 1/3 → needs an M-of-N window).
    g = _gate(debug=True, debug_floor=0.2, consec_frames=3)  # threshold 0.5 from _gate default
    lines, remove = _capture_transcript()

    async def go():
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=2 (best run — never reaches 3)
        g._oww.score = 0.3; await g._feed(_CHUNK)   # ≥floor, <threshold → consec resets, burst alive
        g._oww.score = 0.0; await g._feed(_CHUNK)   # <floor → burst ends, emit

    try:
        asyncio.run(go())
    finally:
        remove()

    assert g._open is False
    nearmiss = [ln for ln in lines if "near-miss" in ln]
    assert len(nearmiss) == 1
    assert "peaked 2/3 frames" in nearmiss[0]
    assert g._consec_max == 0  # reset for the next burst


# --- pre-duck release grace: drop the wake pre-duck if no speech follows ------
def test_preduck_releases_after_grace_without_speech():
    # Wake, then no command → the pre-duck releases on the short grace (not held for the whole window),
    # while the window stays OPEN (still listening for a late command).
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, preduck_grace=0.02)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)               # wake → open + pre-duck + arm the release grace
        assert g._ducked is True
        await asyncio.sleep(0.06)           # grace elapses with no speech

    asyncio.run(go())
    assert g._ducked is False               # pre-duck released early
    assert g._open is True                  # but still listening
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_preduck_held_while_user_speaking():
    # A VAD onset (user is speaking) keeps the pre-duck down past the grace — never restore mid-utterance.
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, preduck_grace=0.02)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)
        await g.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.06)           # grace passes, but speech is in flight

    asyncio.run(go())
    assert g._ducked is True
    assert client.duck_calls == [("sess-test", True)]


def test_preduck_cancelled_by_real_command():
    # A real transcription hands the duck to MediaDuckController → the gate stops its release timer and
    # the duck is held (media-duck will restore when Aria finishes).
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, preduck_grace=0.02)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)
        await g.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await g.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await g.process_frame(TranscriptionFrame("play the movie", "user", "t"), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.06)

    asyncio.run(go())
    assert g._ducked is True
    assert client.duck_calls == [("sess-test", True)]


def test_preduck_held_while_bot_speaking_then_window_resumes():
    # While Aria replies the pre-duck stays down AND the idle window is frozen; when she stops, the window
    # resumes and closes normally (restoring).
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.05, preduck_grace=0.02)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)
        await g.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.09)           # past both the grace and the window
        assert g._ducked is True            # held — Aria is speaking
        assert g._open is True              # window frozen, not closed
        await g.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.09)           # window resumes → closes

    asyncio.run(go())
    assert g._open is False
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_preduck_grace_zero_holds_until_window_close():
    # preduck_grace=0 → old behaviour: the pre-duck holds for the full command window.
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.05, preduck_grace=0.0)
    g._oww.score = 0.9

    async def go():
        await g._feed(_CHUNK)
        await asyncio.sleep(0.02)
        assert g._ducked is True            # not released early (no grace)
        await asyncio.sleep(0.06)           # window (0.05s) now elapses → close + restore

    asyncio.run(go())
    assert g._ducked is False               # released only at window close
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
