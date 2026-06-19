"""Offline tests for the wake-word gate state machine — no audio, no openWakeWord model needed.

Drives WakeWordGate's internals directly (like the media-duck tests) with a fake oww model and a
FakeBrainClient, so the open/close/pre-duck/debounce/media-aware logic is verified without a running
pipeline. asyncio.run drives the async bodies (plain pytest, no pytest-asyncio).
"""

import asyncio
import time

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


async def _call(fn, *args):
    """Invoke a sync gate method inside a running loop (so its asyncio.create_task calls have one)."""
    fn(*args)


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


def test_asleep_requires_stricter_consec():
    # While asleep (force-gated), the stricter asleep_consec_frames applies: a 2-frame burst that WOULD wake
    # when awake must NOT wake asleep — it takes the full asleep run. This rejects the brief 2-frame ambient
    # coincidences that opened the mic to room audio in the 2026-06-15 talk-radio self-wake.
    client = FakeBrainClient()
    g = _gate(client, consec_frames=2, asleep_consec_frames=3)
    g._force_gated = True  # brain asleep

    async def go():
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=2 — would wake AWAKE, must NOT while asleep
        assert g._open is False
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=3 → wake

    asyncio.run(go())
    assert g._open is True


def test_awake_uses_normal_consec_not_asleep_bar():
    # The stricter bar applies ONLY while asleep: awake, consec_frames=2 still fires on two frames.
    client = FakeBrainClient()
    g = _gate(client, consec_frames=2, asleep_consec_frames=3)  # not force-gated → awake

    async def go():
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=1
        g._oww.score = 0.9; await g._feed(_CHUNK)   # consec=2 → wake (awake bar)

    asyncio.run(go())
    assert g._open is True


def test_asleep_consec_never_weaker_than_awake():
    # Defensive clamp: an asleep bar below the awake bar is raised to the awake bar.
    g = _gate(consec_frames=3, asleep_consec_frames=2)
    assert g._asleep_consec_required == 3


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
    import aria_state
    from pipecat.processors.frame_processor import FrameDirection
    from wake_word import WakeSleepFrame

    g = _gate(media_only=True, media_status=lambda: {"playing": False, "kind": None})
    assert asyncio.run(g._gated_now()) is False                  # quiet room → open mic normally

    asyncio.run(g.process_frame(WakeSleepFrame(asleep=True), FrameDirection.UPSTREAM))
    assert g._force_gated is True
    assert asyncio.run(g._gated_now()) is True                   # forced gated despite nothing playing
    assert aria_state.resting_state() == "off"                   # eye must REST on off, not flash-then-idle

    asyncio.run(g.process_frame(WakeSleepFrame(asleep=False), FrameDirection.UPSTREAM))
    assert g._force_gated is False
    assert asyncio.run(g._gated_now()) is False                  # awake → media-aware gating restored
    assert aria_state.resting_state() == "idle"                  # awake → eye rests on idle again


def test_open_window_keeps_eye_off_while_asleep():
    # Regression (2026-06-18): while asleep the nano model false-fires on un-AEC'd room audio, opening a
    # wake-CANDIDATE window the brain then rejects (stays asleep). The window-open eye write must honor
    # resting_state() (`off` asleep) instead of hardcoding `listening`, else the eye lies — the maintainer saw
    # `listening` while genuinely asleep. (Same bug class as the 9cbd609 tts_gain clobber fix.)
    import aria_state

    g = _gate(window_secs=10.0)
    aria_state.set_resting_state("off")          # asleep
    aria_state.set_state("off")

    asyncio.run(_call(g._open_window, "test: asleep false-wake"))

    assert g._open is True
    assert aria_state.current() == "off"         # eye stayed dark — did NOT flash `listening`


def test_open_window_lights_listening_while_awake():
    # Awake behavior unchanged: a real wake still flashes `listening`.
    import aria_state

    g = _gate(window_secs=10.0)
    aria_state.set_resting_state("idle")         # awake
    aria_state.set_state("idle")

    asyncio.run(_call(g._open_window, "test: awake wake"))

    assert aria_state.current() == "listening"


def test_window_close_settles_off_when_asleep_midwindow():
    # A window open (listening, awake) at the instant she's put to sleep must idle-close to the resting
    # state (`off`), not the old hardcoded `idle`. Covers the second of the two missed writes.
    import aria_state

    g = _gate(window_secs=0.01)
    aria_state.set_resting_state("idle")
    aria_state.set_state("idle")

    async def go():
        g._open_window("test: awake wake")       # current → listening
        assert aria_state.current() == "listening"
        aria_state.set_resting_state("off")      # WakeSleepFrame(asleep=True) arrives mid-window
        await asyncio.sleep(0.05)                 # window elapses → idle close

    asyncio.run(go())
    assert g._open is False
    assert aria_state.current() == "off"         # settled on resting (off), not hardcoded `idle`


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


def test_keepalive_holds_window_for_ttl_then_idle_closes():
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.01, hold_max_secs=120.0)

    async def go():
        g._set_keepalive(0.06)            # media command → hold window ~60ms
        assert g._open is True
        await asyncio.sleep(0.03)         # past the 10ms idle window, still within the keepalive
        assert g._open is True            # held open by the keepalive
        await asyncio.sleep(0.05)         # keepalive expired → idle window re-armed and elapsed
        assert g._open is False           # closed normally after the TTL

    asyncio.run(go())


def test_keepalive_survives_idle_rearm_during_ttl():
    # 2026-06-15 live bug: BotStoppedSpeaking/transcription/VAD re-arm the short idle-close mid-keepalive and
    # truncated the hold. With _keepalive_active, _arm_window must bail for the whole TTL.
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.01, hold_max_secs=120.0)

    async def go():
        g._set_keepalive(0.06)
        g._arm_window()                   # simulate Aria finishing TTS (BotStoppedSpeaking) mid-keepalive
        await asyncio.sleep(0.03)         # well past the 10ms idle window
        assert g._open is True            # NOT truncated — the keepalive held it open
        await asyncio.sleep(0.05)         # TTL elapses → idle re-armed → closes
        assert g._open is False

    asyncio.run(go())


def test_keepalive_ttl_clamped_to_ceiling():
    g = _gate(hold_max_secs=30.0)
    assert g._clamp_ttl(10) == 10.0
    assert g._clamp_ttl(999) == 30.0     # oversized TTL clamped to the ceiling
    assert g._clamp_ttl(0) == 0.0
    assert g._clamp_ttl(-5) == 0.0
    assert g._clamp_ttl("bad") == 0.0    # non-numeric → no hold


def test_keepalive_refresh_extends_window():
    client = FakeBrainClient()
    g = _gate(client, window_secs=0.01, hold_max_secs=120.0)

    async def go():
        g._set_keepalive(0.04)
        await asyncio.sleep(0.03)
        assert g._open is True
        g._set_keepalive(0.06)            # a second media command refreshes the hold
        await asyncio.sleep(0.03)         # past the first TTL, still within the refreshed one
        assert g._open is True

    asyncio.run(go())


def test_keepalive_does_not_preduck_window_stays_open():
    # F3 option B (2026-06-16): a media keepalive must NOT pre-duck — the command already ran, so dipping the
    # song the user just asked to play/resume is the "dips the song I just started" annoyance (12:12:13). The
    # window still opens (held for follow-ups); a follow-up command ducks via the confirmed-speech path. This
    # also structurally retires the 2026-06-15 "movie stuck ducked for the TTL" regression — no preduck to strand.
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, hold_max_secs=120.0,
              preduck_grace=0.03, preduck_grace_keepalive=0.03)

    async def go():
        g._set_keepalive(5.0)             # media command → hold window, NO pre-duck
        await asyncio.sleep(0)            # let any fire-and-forget task run
        assert client.duck_calls == []    # the keepalive did not dip the just-started song
        await asyncio.sleep(0.06)         # past the grace — still nothing
        assert client.duck_calls == []
        assert g._open is True            # window STILL open — the keepalive holds it
        assert g._keepalive_active is True

    asyncio.run(go())


def test_keepalive_does_not_preduck_across_announcement():
    # Option B: with no keepalive pre-duck, Aria's announcement (BotStarted/Stopped) over the just-started
    # media fires no duck either — the song plays at full volume through her confirmation. (If that ever drowns
    # a play announcement over loud media we'd add a duck-on-bot-speech-only path; not pre-emptive dipping.)
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, hold_max_secs=120.0,
              preduck_grace=0.04, preduck_grace_keepalive=0.04)

    async def go():
        g._set_keepalive(5.0)             # NO pre-duck
        await g.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)  # announcement
        await asyncio.sleep(0.07)
        await g.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)  # she finishes
        await asyncio.sleep(0.07)
        assert client.duck_calls == []    # never ducked the song across the whole announcement
        assert g._open is True            # window still open (keepalive)

    asyncio.run(go())


def test_keepalive_releases_a_residual_fresh_wake_preduck_on_short_grace():
    # A fresh wake pre-ducks (long grace); if a media command then fires a keepalive, that residual pre-duck
    # must still release — and on the SHORT keepalive grace, not the long wake grace (the song shouldn't stay
    # dipped). This is what `preduck_grace_keepalive` still governs now that the keepalive itself doesn't duck.
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, hold_max_secs=120.0,
              preduck_grace=5.0, preduck_grace_keepalive=0.03)

    async def go():
        g._open_window("fresh wake")      # fresh-wake pre-duck ON (long grace)
        await asyncio.sleep(0)
        assert client.duck_calls == [("sess-test", True)]
        g._set_keepalive(5.0)             # media command → keepalive; residual pre-duck released on short grace
        await asyncio.sleep(0.06)
        assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
        assert g._open is True

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


# --- F1/F3: duck-over-music gating (2026-06-16) ----------------------------------------------------
def test_duck_onset_suppressed_in_keepalive_tail_over_media():
    # F1: over playing media, a raw VAD onset must NOT duck while the window is open only via a keepalive
    # (held over the song a command just started → the music's own onsets would dip the bed with no wake).
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, hold_max_secs=120.0, preduck_grace_keepalive=5.0)
    g._gated_committed = True              # media is playing (gated)

    async def go():
        g._set_keepalive(5.0)             # window opened by a media command, not a fresh wake
        assert g._open is True
        assert g._open_via_keepalive is True
        assert g.duck_onset_allowed() is False   # raw-onset duck suppressed in the keepalive tail
        assert g.duck_allowed() is True          # confirmed-speech path still ducks a real follow-up

    asyncio.run(go())


def test_duck_onset_allowed_after_fresh_wake_over_media():
    # A genuine "Hey Aria" over media opens a fresh-wake window → raw onset ducks immediately.
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0)
    g._gated_committed = True

    async def go():
        g._open_window("test wake")
        assert g._open_via_keepalive is False
        assert g.duck_onset_allowed() is True
        assert g.duck_allowed() is True

    asyncio.run(go())


def test_fresh_wake_supersedes_keepalive_for_onset_gate():
    # A keepalive tail (onset suppressed) followed by a real wake → onset allowed again.
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0, preduck_grace_keepalive=5.0)
    g._gated_committed = True

    async def go():
        g._set_keepalive(5.0)
        assert g.duck_onset_allowed() is False
        g._open_window("fresh wake during keepalive")
        assert g._open_via_keepalive is False
        assert g.duck_onset_allowed() is True

    asyncio.run(go())


def test_duck_onset_allowed_when_nothing_gated():
    # No media gated → an onset duck is a harmless no-op (skipped downstream); allow it.
    client = FakeBrainClient()
    g = _gate(client, window_secs=10.0)
    g._gated_committed = False
    assert g.duck_onset_allowed() is True


# --- V2 interrupt/stop word barge-in (WAKE_INTERRUPT) ----------------------------------------------
def _interrupt_gate(client=None, **kw):
    """A gate with the interrupt path enabled and push_frame spied (no live pipeline). _on_interrupt cuts
    TTS by pushing a StopWordFrame DOWNSTREAM — a custom frame the half-duplex user-mute does NOT drop (an
    InterruptionFrame from the gate IS dropped by that mute; BrainLLMService, below the mute, converts the
    StopWordFrame into the real interruption). 2026-06-16 live root cause. g._interrupt_calls records
    (frame_type, direction) for any StopWordFrame pushed."""
    from wake_word import StopWordFrame

    kw.setdefault("interrupt_enabled", True)
    kw.setdefault("interrupt_consec_frames", 2)
    g = _gate(client, **kw)
    calls: list = []

    async def _spy_push(frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, StopWordFrame):
            calls.append((type(frame).__name__, direction))
        # swallow (no downstream processor wired in the offline harness)

    g.push_frame = _spy_push  # type: ignore[assignment]
    g._interrupt_calls = calls
    return g


def test_interrupt_predicate_only_when_enabled_and_bot_speaking():
    # Routing predicate: detect the interrupt word ONLY when opt-in AND Aria is currently speaking.
    g = _gate(interrupt_enabled=True)
    g._bot_speaking = True
    assert g._interrupt_active() is True
    g._bot_speaking = False
    assert g._interrupt_active() is False  # not while idle (that's the wake path)
    g_off = _gate(interrupt_enabled=False)
    g_off._bot_speaking = True
    assert g_off._interrupt_active() is False  # disabled → never routes to the interrupt detector


def test_interrupt_fires_during_bot_speaking():
    # Sustained interrupt word while she speaks → cut TTS (InterruptionTaskFrame upstream) + open floor + duck.
    client = FakeBrainClient()
    g = _interrupt_gate(client, interrupt_consec_frames=2)
    g._oww.score = 0.9
    g._bot_speaking = True

    async def go():
        await g._feed_interrupt(_CHUNK)        # frame 1 → consec=1, not yet
        assert g._interrupt_calls == []
        await g._feed_interrupt(_CHUNK)        # frame 2 → consec=2 → fire

    asyncio.run(go())
    # Cut via a StopWordFrame pushed DOWNSTREAM (survives the half-duplex mute; BrainLLMService converts it
    # to the real interruption below the mute → flush TTS + /cancel).
    assert g._interrupt_calls == [("StopWordFrame", FrameDirection.DOWNSTREAM)]
    assert g._open is True                                # floor opened (stop-and-listen)
    assert client.duck_calls == [("sess-test", True)]    # pre-duck for the redirect


def test_interrupt_rejects_single_frame_spike():
    # A 1-frame TTS-bleed/music spike must NOT interrupt when consec=2 (same rejection as the wake path).
    client = FakeBrainClient()
    g = _interrupt_gate(client, interrupt_consec_frames=2)
    g._oww.score = 0.9
    g._bot_speaking = True

    asyncio.run(g._feed_interrupt(_CHUNK))     # one frame only
    assert g._interrupt_calls == []
    assert g._open is False
    assert client.duck_calls == []


def test_interrupt_arm_delay_suppresses_onset_selftrip():
    # 2026-06-16 live: Aria's own TTS onset self-trips the detector 75–159ms after she starts speaking
    # (AEC not yet converged). A sustained hit WITHIN the arm window must NOT fire.
    client = FakeBrainClient()
    g = _interrupt_gate(client, interrupt_consec_frames=2, interrupt_arm_delay_secs=0.6)
    g._oww.score = 0.9
    g._bot_speaking = True
    g._bot_speaking_since = time.monotonic()    # just started → inside the 0.6s arm window

    async def go():
        await g._feed_interrupt(_CHUNK)         # frame 1
        await g._feed_interrupt(_CHUNK)         # frame 2 → consec met, but armed-out → no fire

    asyncio.run(go())
    assert g._interrupt_calls == []
    assert g._open is False
    assert client.duck_calls == []


def test_interrupt_arm_delay_allows_after_window():
    # The same sustained hit AFTER the arm window elapses fires normally (real mid-reply interrupt preserved).
    client = FakeBrainClient()
    g = _interrupt_gate(client, interrupt_consec_frames=2, interrupt_arm_delay_secs=0.6)
    g._oww.score = 0.9
    g._bot_speaking = True
    g._bot_speaking_since = time.monotonic() - 100.0   # well past the arm window

    async def go():
        await g._feed_interrupt(_CHUNK)
        await g._feed_interrupt(_CHUNK)         # consec met AND armed → fire

    asyncio.run(go())
    assert g._interrupt_calls == [("StopWordFrame", FrameDirection.DOWNSTREAM)]
    assert g._open is True


def test_interrupt_threshold_defaults_to_wake_threshold():
    # interrupt_threshold=None → uses the wake threshold; a sub-threshold burst never fires.
    client = FakeBrainClient()
    g = _interrupt_gate(client, threshold=0.5, interrupt_threshold=None, interrupt_consec_frames=1)
    assert g._interrupt_threshold == 0.5
    g._oww.score = 0.3                          # below threshold
    g._bot_speaking = True

    asyncio.run(g._feed_interrupt(_CHUNK))
    assert g._interrupt_calls == []
    assert g._open is False
