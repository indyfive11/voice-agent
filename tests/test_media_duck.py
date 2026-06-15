"""Unit tests for MediaDuckController duck *timing* — the onset → confirm → restore state machine,
driven through the sync `_handle(frame)` entry point with a FakeBrainClient and a controllable
media-state provider, so the logic is verified without a running pipeline.

Focus: the 2026-06-14 onset→confirm FLAP fix — while media is playing, an unconfirmed onset must NOT
snap the bed back at confirm_grace (which flapped down→up→re-duck and raced the confirmed `on` against a
stale `off`); it falls back to the longer idle grace instead. Over silence, the old snap-back stands.
"""

import asyncio

from brains.brain_client import FakeBrainClient
from brains.media_duck import MediaDuckController
from pipecat.frames.frames import (
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)


def _ctrl(client, *, playing: bool, confirm_grace=0.02, restore_grace=0.20):
    state = {"playing": playing}
    g = MediaDuckController(
        client,
        session_id="sess-test",
        confirm_grace=confirm_grace,
        restore_grace=restore_grace,
        media_status=(lambda: dict(state)),
    )
    g._test_state = state  # tests can flip playback mid-turn
    return g


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
