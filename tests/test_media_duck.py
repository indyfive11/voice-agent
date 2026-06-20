"""Unit tests for MediaDuckController duck *timing* — the onset → confirm → restore state machine,
driven through the sync `_handle(frame)` entry point with a FakeBrainClient and a controllable
media-state provider, so the logic is verified without a running pipeline.

Focus: the 2026-06-14 onset→confirm FLAP fix — while media is playing, an unconfirmed onset must NOT
snap the bed back at confirm_grace (which flapped down→up→re-duck and raced the confirmed `on` against a
stale `off`); it falls back to the longer idle grace instead. Over silence, the old snap-back stands.
"""

import asyncio

from brains.brain_client import FakeBrainClient
from brains.media_duck import DuckReleaseFrame, MediaDuckController
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from turn_mute import BotThinkingFrame


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _ctrl(client, *, playing: bool, confirm_grace=0.02, restore_grace=0.20, max_hold_secs=120.0,
          convo_hold_secs=0.0, window_secs=15.0):
    state = {"playing": playing}
    g = MediaDuckController(
        client,
        session_id="sess-test",
        confirm_grace=confirm_grace,
        restore_grace=restore_grace,
        max_hold_secs=max_hold_secs,
        convo_hold_secs=convo_hold_secs,
        window_secs=window_secs,
        media_status=(lambda: dict(state)),
    )
    g._test_state = state  # tests can flip playback mid-turn
    return g


async def _addressed_turn(g):
    """Drive one full ADDRESSED turn over playing media: onset → confirmed → bot speaks → bot stops."""
    g._handle(VADUserStartedSpeakingFrame())
    await _drain()
    g._handle(_transcript("play something"))
    await _drain()
    g._handle(BotStartedSpeakingFrame())
    await _drain()
    g._handle(BotStoppedSpeakingFrame())
    await _drain()


def _transcript(text):
    return TranscriptionFrame(text=text, user_id="u", timestamp="t")


async def _drain():
    # Let fire-and-forget _fire() tasks and the just-scheduled timer tasks run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def test_unconfirmed_onset_snaps_back_quickly_when_media_stopped():
    # The snap-back branch is preserved for when media is NOT playing at confirm time: duck while a movie
    # plays, the movie stops mid-turn, the onset never confirms → restore on the SHORT confirm grace (not
    # the long idle grace), since there's no movie left to keep ducked.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, confirm_grace=0.02, restore_grace=5.0)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # duck on (media playing → real duck)
        await _drain()
        assert client.duck_calls == [("sess-test", True)]
        g._test_state["playing"] = False           # movie stopped mid-turn
        g._handle(VADUserStoppedSpeakingFrame())   # unconfirmed → arm confirm
        await asyncio.sleep(0.05)                   # confirm_grace (0.02) elapses, idle (5.0) has not
        await _drain()

    asyncio.run(go())
    # Restored on the short confirm grace, not held for the 5s idle grace.
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False


def test_unconfirmed_onset_over_movie_does_not_flap():
    # Media playing + onset never confirmed within confirm_grace → must NOT snap back. The duck stays on
    # (no `off` emitted at confirm_grace); it will restore later via the idle grace.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # duck on (onset) — media playing → real duck
        await _drain()
        assert client.duck_calls == [("sess-test", True)]
        g._handle(VADUserStoppedSpeakingFrame())   # stop, unconfirmed → arm confirm
        await asyncio.sleep(0.05)                   # confirm_grace (0.02) elapses
        await _drain()

    asyncio.run(go())
    # No restore fired at confirm_grace: still exactly one call, the original ON. No flapping `off`.
    assert client.duck_calls == [("sess-test", True)]
    assert g._ducked is True


def test_duck_gated_behind_wake_word():
    # 2026-06-15 (the maintainer): with media playing but the wake window CLOSED (no "Hey Aria" yet), a speech onset is
    # room conversation, not for Aria — it must NOT duck. should_duck models the gate: False while
    # gated+closed, True once the window opens. Applies to all media (audio + video), like the STT gate.
    client = FakeBrainClient()
    allow = {"v": False}  # gate gated+closed → duck suppressed
    g = MediaDuckController(
        client, session_id="sess-test", confirm_grace=0.02, restore_grace=0.2,
        media_status=(lambda: {"playing": True}),
        should_duck=(lambda: allow["v"]),
    )

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # gated+closed → ambient speech → suppressed
        await _drain()
        assert client.duck_calls == [], "ducked on ambient speech while gated+closed"
        assert g._ducked is False
        allow["v"] = True                          # "Hey Aria" opened the window
        g._handle(VADUserStartedSpeakingFrame())   # now a command onset → duck
        await _drain()
        assert client.duck_calls == [("sess-test", True)]

    asyncio.run(go())


def test_onset_gate_suppresses_raw_onset_but_confirmed_speech_still_ducks():
    # F1 (2026-06-16): in the keepalive/idle tail over playing media, the raw VAD onset is suppressed (the
    # media's own onsets must not dip the bed) — modelled by should_duck_onset=False — while a REAL follow-up
    # command (≥min_words transcription) still ducks via the broad should_duck path. Music never dips; a
    # genuine command does (a beat later).
    client = FakeBrainClient()
    g = MediaDuckController(
        client, session_id="sess-test", confirm_grace=0.02, restore_grace=5.0, min_words=2,
        media_status=(lambda: {"playing": True}),
        should_duck=(lambda: True),            # broad gate open (window open via keepalive)
        should_duck_onset=(lambda: False),     # raw-onset gate closed (keepalive tail → suppress onset)
    )

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # raw onset → suppressed (music tripping VAD)
        await _drain()
        assert client.duck_calls == [], "raw onset ducked in the keepalive tail"
        assert g._ducked is False
        g._handle(_transcript("turn it up"))        # ≥2 words → a real follow-up command
        await _drain()
        assert client.duck_calls == [("sess-test", True)]  # confirmed speech ducks
        assert g._ducked is True

    asyncio.run(go())


def test_onset_gate_defaults_to_should_duck_when_unset():
    # Back-compat: without should_duck_onset, the raw onset uses should_duck (no behaviour change).
    client = FakeBrainClient()
    g = MediaDuckController(
        client, session_id="sess-test", confirm_grace=0.02, restore_grace=5.0,
        media_status=(lambda: {"playing": True}),
        should_duck=(lambda: True),
    )

    async def go():
        g._handle(VADUserStartedSpeakingFrame())
        await _drain()
        assert client.duck_calls == [("sess-test", True)]

    asyncio.run(go())


def test_slow_transcript_over_movie_confirms_without_flap():
    # The real-command-over-a-movie case: onset ducks, confirm_grace elapses BEFORE the transcript (slow
    # Whisper), then the ≥min_words transcript lands. There must be NO down→up→down — only the single ON.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())
        await _drain()
        g._handle(VADUserStoppedSpeakingFrame())   # unconfirmed → arm confirm
        await asyncio.sleep(0.05)                   # confirm_grace elapses first (no snap-back: playing)
        await _drain()
        g._handle(_transcript("pause the movie"))   # ≥2 words → confirmed speech
        await _drain()

    asyncio.run(go())
    # Exactly one ON, never an OFF — the flap (and the off-after-on race) is gone.
    assert client.duck_calls == [("sess-test", True)]
    assert g._ducked is True
    assert g._confirmed is True


def test_unconfirmed_onset_over_movie_restores_via_idle_grace():
    # The fallback: a genuine non-speech onset over a movie still restores — after the longer idle grace,
    # not the short confirm grace — so a cough doesn't duck the movie forever.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, confirm_grace=0.02, restore_grace=0.05)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())
        await _drain()
        g._handle(VADUserStoppedSpeakingFrame())   # unconfirmed → arm confirm
        await asyncio.sleep(0.15)                   # past confirm_grace AND the idle restore_grace
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False


# --- automatic duck-policy: hold the duck through sustained speech (2026-06-15) ---
def _ctrl_clk(client, clk, *, sustained_secs=4.0):
    return MediaDuckController(
        client,
        session_id="sess-test",
        sustained_secs=sustained_secs,
        confirm_grace=5.0,
        restore_grace=5.0,
        media_status=(lambda: {"playing": True}),
        time_source=clk,
    )


def test_aside_held_while_user_still_speaking():
    # Aside verdict arrives WHILE the user is mid-utterance → hold the duck (don't snap music to full).
    client = FakeBrainClient()
    g = _ctrl_clk(client, _Clock())

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # duck on, speech_in_flight=True
        await _drain()
        assert client.duck_calls == [("sess-test", True)]
        g._handle(DuckReleaseFrame())              # aside while still speaking
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # HELD — no restore
    assert g._ducked is True


def test_aside_held_after_sustained_duration():
    # User stopped, but the utterance was long (> sustained_secs) → still hold; idle-grace owns restore.
    client = FakeBrainClient()
    clk = _Clock()
    g = _ctrl_clk(client, clk, sustained_secs=4.0)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # duck on at t=0
        await _drain()
        g._handle(VADUserStoppedSpeakingFrame())   # speech_in_flight=False
        clk.t = 5.0                                 # 5s ducked > 4s sustained
        g._handle(DuckReleaseFrame())
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # HELD
    assert g._ducked is True


def test_short_aside_still_restores_immediately():
    # A brief ambient aside (stopped, well under sustained_secs) → A1's immediate release still applies.
    client = FakeBrainClient()
    clk = _Clock()
    g = _ctrl_clk(client, clk, sustained_secs=4.0)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())   # duck on at t=0
        await _drain()
        g._handle(VADUserStoppedSpeakingFrame())   # speech_in_flight=False
        clk.t = 1.0                                 # brief: 1s < 4s
        g._handle(DuckReleaseFrame())
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]  # restored (A1 preserved)
    assert g._ducked is False


# --- hold the duck through the WHOLE brain think (BotThinkingFrame), with a safety ceiling (2026-06-19) ---
def test_thinking_holds_duck_past_restore_grace():
    # The bob-up fix: a confirmed command arms the short idle restore, but the brain then starts a slow
    # lookup (BotThinkingFrame active). The think-hold must cancel that restore so the bed stays ducked past
    # restore_grace — no pop to full volume mid-think.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, confirm_grace=5.0, restore_grace=0.03)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())     # duck on
        await _drain()
        g._handle(_transcript("play my playlist"))    # confirmed → arms the 0.03 idle restore
        await _drain()
        g._handle(BotThinkingFrame(active=True))      # brain working → cancel restore, hold the duck
        await asyncio.sleep(0.08)                      # well past restore_grace
        await _drain()
    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # still ducked — no mid-think restore
    assert g._ducked is True and g._bot_thinking is True


def test_thinking_end_rearms_restore_for_silent_turn():
    # A turn that thinks then produces NO bot speech and no aside (empty model turn) must not leave the bed
    # stuck ducked: active=False re-arms the idle grace, which then restores.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, confirm_grace=5.0, restore_grace=0.03)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())
        await _drain()
        g._handle(VADUserStoppedSpeakingFrame())        # user finished speaking the command
        g._handle(_transcript("what's the weather"))
        await _drain()
        g._handle(BotThinkingFrame(active=True))        # hold
        await asyncio.sleep(0.06)
        g._handle(BotThinkingFrame(active=False))        # think ended, no bot speech → re-arm restore
        await asyncio.sleep(0.06)                        # restore_grace elapses
        await _drain()
    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False and g._bot_thinking is False


def test_normal_fast_turn_restores_exactly_once():
    # No regression on a normal turn: duck → think → bot speaks → bot stops → exactly one ON and one OFF.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, confirm_grace=5.0, restore_grace=5.0)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())
        await _drain()
        g._handle(_transcript("pause the music"))
        await _drain()
        g._handle(BotThinkingFrame(active=True))        # hold through the think
        await _drain()
        g._handle(BotStartedSpeakingFrame())            # reply begins
        g._handle(BotThinkingFrame(active=False))        # think ends as she speaks (bot_spoke → no re-arm)
        await _drain()
        g._handle(BotStoppedSpeakingFrame())            # reply ends → restore
        await _drain()
    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False


def test_think_hold_ceiling_restores_after_max_hold():
    # The safety ceiling: if the brain stays "thinking" pathologically long (no reply, no aside, no error),
    # restore rather than hold the bed ducked forever. Short max_hold so the test exercises the backstop.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, confirm_grace=5.0, restore_grace=5.0, max_hold_secs=0.05)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())
        await _drain()
        g._handle(_transcript("play something"))
        await _drain()
        g._handle(BotThinkingFrame(active=True))         # hold + arm the 0.05s ceiling
        await asyncio.sleep(0.12)                          # ceiling fires (no reply ever came)
        await _drain()
    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False


# --- conversation-hold over media (2026-06-19) ------------------------------------------------------

def test_convo_hold_keeps_bed_ducked_after_addressed_reply():
    # After an addressed reply over media, the bed stays ducked (no `off` at bot-stopped) and a hold timer
    # is armed — the floor stays open for a follow-up.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=5.0)

    asyncio.run(_addressed_turn(g))
    assert client.duck_calls == [("sess-test", True)]  # ducked, never restored
    assert g._ducked is True
    assert g._convo_task is not None


def test_convo_hold_follow_up_no_bob_and_rearms():
    # A follow-up within the hold needs no re-wake: its onset cancels the hold, the bed never bobs up between
    # turns, and the second reply re-arms the hold.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=5.0)

    async def go():
        await _addressed_turn(g)                       # turn 1 → hold armed
        g._handle(VADUserStartedSpeakingFrame())       # follow-up onset (no re-wake)
        await _drain()
        assert g._convo_task is None                   # hold cancelled by the follow-up
        g._handle(_transcript("and tomorrow"))
        await _drain()
        g._handle(BotStartedSpeakingFrame())
        await _drain()
        g._handle(BotStoppedSpeakingFrame())           # turn 2 → hold re-armed
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # one duck on, no bob across both turns
    assert g._ducked is True
    assert g._convo_task is not None


def test_convo_hold_restores_on_idle_expiry():
    # The conversation goes quiet → the hold timer fires → the bed restores (the primary restore guarantee).
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=0.05)

    async def go():
        await _addressed_turn(g)
        assert g._ducked is True
        await asyncio.sleep(0.12)                       # convo idle (0.05) elapses
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False


def test_convo_hold_restores_on_sleep():
    # Going to sleep mid-hold must restore the bed (a held duck must never survive sleep).
    from wake_word import WakeSleepFrame

    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=5.0)

    async def go():
        await _addressed_turn(g)
        assert g._ducked is True
        g._handle(WakeSleepFrame(asleep=True))
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False
    assert g._convo_task is None


def test_convo_hold_restores_on_shutdown():
    # Pipeline teardown mid-hold must restore the bed (best-effort) — closes the stuck-duck-at-shutdown gap.
    from pipecat.frames.frames import EndFrame

    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=5.0)

    async def go():
        await _addressed_turn(g)
        assert g._ducked is True
        g._handle(EndFrame())
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False
    assert g._convo_task is None


def test_aside_does_not_enter_convo_hold():
    # An aside (not addressed → no bot speech) must NOT enter the hold; it restores normally.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=5.0)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())       # duck on
        await _drain()
        g._handle(VADUserStoppedSpeakingFrame())       # short utterance (not sustained)
        await _drain()
        g._handle(DuckReleaseFrame())                  # aside → immediate restore
        await _drain()

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False
    assert g._convo_task is None


def test_no_convo_hold_when_nothing_ducked():
    # Open-mic / nothing playing: the duck never engages, so bot-stopped enters no hold and nothing is stuck.
    client = FakeBrainClient()
    g = _ctrl(client, playing=False, convo_hold_secs=5.0)

    async def go():
        g._handle(VADUserStartedSpeakingFrame())       # _fire skips (nothing playing) → _ducked clears
        await _drain()
        g._handle(BotStartedSpeakingFrame())
        await _drain()
        g._handle(BotStoppedSpeakingFrame())
        await _drain()

    asyncio.run(go())
    assert g._ducked is False
    assert g._convo_task is None
    assert client.duck_calls == []


def test_convo_hold_disabled_restores_immediately():
    # convo_hold_secs=0 → the pre-2026-06-19 behavior: restore the instant Aria stops speaking.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=0.0)

    asyncio.run(_addressed_turn(g))
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]
    assert g._ducked is False
    assert g._convo_task is None


def test_convo_hold_clamps_to_window_secs():
    # A hold longer than the wake window would re-gate mid-conversation, so it's clamped to window_secs.
    client = FakeBrainClient()
    g = _ctrl(client, playing=True, convo_hold_secs=30.0, window_secs=15.0)
    assert g._convo_secs == 15.0
