"""DeferredAnnouncer — out-of-band speech at a free floor (announce.py, builder_spec.md §4.4), offline.

The announcer depends only on a `floor_free: () -> bool` callback (the floor authority is BrainLLMService;
here it's a controllable flag), so these tests need no service, no brain, no audio. Async bodies via
asyncio.run so plain pytest works.
"""

import asyncio

from pipecat.frames.frames import BotStoppedSpeakingFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection

from announce import (
    DeferredAnnouncer,
    SpeakDeferredFrame,
    PRIORITY_DEFAULT,
    PRIORITY_NUDGE,
    PRIORITY_QUESTION,
)


class _Floor:
    """A mutable floor-free flag standing in for BrainLLMService.is_floor_free."""

    def __init__(self, free=True):
        self.free = free

    def __call__(self):
        return self.free


def _announcer(free=True, started=True):
    floor = _Floor(free)
    ann = DeferredAnnouncer(floor_free=floor)
    ann._started = started  # tests default to a started pipeline; the gate tests pass started=False
    pushed = []

    async def _rec(frame, direction=None):
        pushed.append(frame)

    ann.push_frame = _rec
    return ann, pushed, floor


def _spoken(pushed):
    return [f.text for f in pushed if isinstance(f, TTSSpeakFrame)]


def _send(ann, frame):
    asyncio.run(ann.process_frame(frame, FrameDirection.DOWNSTREAM))


def test_speaks_immediately_when_floor_free():
    ann, pushed, _ = _announcer(free=True)
    _send(ann, SpeakDeferredFrame(text="Build finished: 3 files changed.", job_id="j1"))
    assert _spoken(pushed) == ["Build finished: 3 files changed."]
    assert ann._queue == []
    assert ann._in_flight is True  # latched until this item's BotStopped


def test_control_frame_consumed_not_forwarded():
    ann, pushed, _ = _announcer(free=True)
    _send(ann, SpeakDeferredFrame(text="hi", job_id="j"))
    assert not any(isinstance(f, SpeakDeferredFrame) for f in pushed)


def test_empty_text_dropped():
    ann, pushed, _ = _announcer(free=True)
    _send(ann, SpeakDeferredFrame(text="", job_id="j"))
    assert _spoken(pushed) == [] and ann._queue == []


def test_queues_while_floor_busy_then_drains_on_botstopped():
    ann, pushed, floor = _announcer(free=False)  # floor busy (a reply is playing, say)
    _send(ann, SpeakDeferredFrame(text="result A", job_id="a"))
    assert _spoken(pushed) == [] and len(ann._queue) == 1

    floor.free = True
    _send(ann, BotStoppedSpeakingFrame())
    assert _spoken(pushed) == ["result A"]
    # BotStopped is observe-and-forward — media_duck / wake gate still need it.
    assert any(isinstance(f, BotStoppedSpeakingFrame) for f in pushed)


def test_one_at_a_time():
    ann, pushed, _ = _announcer(free=True)
    _send(ann, SpeakDeferredFrame(text="first", job_id="1"))
    _send(ann, SpeakDeferredFrame(text="second", job_id="2"))
    assert _spoken(pushed) == ["first"]       # second held by the in-flight latch
    assert len(ann._queue) == 1

    _send(ann, BotStoppedSpeakingFrame())     # first finished → drain second
    assert _spoken(pushed) == ["first", "second"]
    assert ann._queue == []


def test_fifo_order():
    ann, pushed, floor = _announcer(free=False)
    for i in range(3):
        _send(ann, SpeakDeferredFrame(text=f"r{i}", job_id=str(i)))
    floor.free = True
    for _ in range(3):
        _send(ann, BotStoppedSpeakingFrame())
    assert _spoken(pushed) == ["r0", "r1", "r2"]


def test_priority_is_plumbed_but_ordering_still_fifo():
    # Priority is carried end-to-end but NOT yet used for ordering (deferred until a 2nd producer exists).
    ann, pushed, floor = _announcer(free=False)
    asyncio.run(ann.announce("low", job_id="l", priority=PRIORITY_NUDGE))
    asyncio.run(ann.announce("high", job_id="h", priority=PRIORITY_QUESTION))
    assert [f.priority for f in ann._queue] == [PRIORITY_NUDGE, PRIORITY_QUESTION]  # stored
    floor.free = True
    for _ in range(2):
        _send(ann, BotStoppedSpeakingFrame())
    assert _spoken(pushed) == ["low", "high"]  # still FIFO, not priority-ordered


def test_floor_blocked_generic():
    # All floor conditions live behind the callback — a False floor blocks regardless of which one it was.
    ann, pushed, _ = _announcer(free=False)
    _send(ann, SpeakDeferredFrame(text="x", job_id="j"))
    assert _spoken(pushed) == [] and len(ann._queue) == 1


def test_public_announce_method():
    ann, pushed, _ = _announcer(free=True)
    asyncio.run(ann.announce("via method", job_id="m1", priority=PRIORITY_DEFAULT))
    assert _spoken(pushed) == ["via method"]


def test_enqueue_deferred_alias():
    # The back-compat seam name GA's contract referenced.
    ann, pushed, _ = _announcer(free=True)
    asyncio.run(ann.enqueue_deferred("via alias", job_id="a1"))
    assert _spoken(pushed) == ["via alias"]


def test_no_producer_no_effect():
    ann, pushed, _ = _announcer(free=True)
    assert ann._queue == [] and _spoken(pushed) == []


def test_drain_delivered_empty_initially():
    ann, _, _ = _announcer(free=True)
    assert ann.drain_delivered() == []


def test_delivered_recorded_on_botstopped_then_drained_once():
    # Speak an item, finish it (BotStopped) → its job_id is available to ack exactly once.
    ann, pushed, _ = _announcer(free=True)
    _send(ann, SpeakDeferredFrame(text="done", job_id="j1"))
    assert _spoken(pushed) == ["done"]
    assert ann.drain_delivered() == []          # not delivered until it finishes speaking
    _send(ann, BotStoppedSpeakingFrame())
    assert ann.drain_delivered() == ["j1"]      # now ack-able
    assert ann.drain_delivered() == []          # drained — ack'd exactly once


def test_no_job_id_not_acked():
    # An item with no job_id is spoken but has nothing to key an ack on → never recorded.
    ann, pushed, _ = _announcer(free=True)
    _send(ann, SpeakDeferredFrame(text="anon", job_id=None))
    _send(ann, BotStoppedSpeakingFrame())
    assert _spoken(pushed) == ["anon"]
    assert ann.drain_delivered() == []


def test_delivered_fifo_across_items():
    # Speak three items, each finishing (its own BotStopped) before the next — delivered in spoken order.
    ann, pushed, _ = _announcer(free=True)
    for i in range(3):
        _send(ann, SpeakDeferredFrame(text=f"r{i}", job_id=f"j{i}"))
        _send(ann, BotStoppedSpeakingFrame())  # this item finishes before the next is sent
    assert _spoken(pushed) == ["r0", "r1", "r2"]
    assert ann.drain_delivered() == ["j0", "j1", "j2"]


def test_plain_reply_botstopped_does_not_record():
    # A BotStopped with no deferred item in flight (a normal reply finishing) records nothing to ack.
    ann, _, _ = _announcer(free=True)
    _send(ann, BotStoppedSpeakingFrame())
    assert ann.drain_delivered() == []


def test_pre_start_announce_holds_and_does_not_wedge():
    # Regression for the live-drive wedge: a producer that fires DURING startup (poll client pulled an item
    # before the pipeline's StartFrame) must NOT push a TTSSpeakFrame — push_frame would drop it and latch
    # _in_flight forever. Instead it queues and waits.
    ann, pushed, _ = _announcer(free=True, started=False)
    _send(ann, SpeakDeferredFrame(text="early result", job_id="j1"))
    assert _spoken(pushed) == []        # nothing spoken before StartFrame
    assert len(ann._queue) == 1         # held
    assert ann._in_flight is False      # the key: NOT wedged


def test_started_flush_drains_queued_items():
    # Once started (what the StartFrame branch does: set _started then _drain), the held item speaks.
    ann, pushed, _ = _announcer(free=True, started=False)
    _send(ann, SpeakDeferredFrame(text="early result", job_id="j1"))
    ann._started = True
    asyncio.run(ann._drain())
    assert _spoken(pushed) == ["early result"]


def test_other_frames_forwarded():
    # A non-owned frame passes straight through (offline: a bare BotStoppedSpeakingFrame already covered;
    # use a plain TTSSpeakFrame flowing through to confirm pass-through doesn't get consumed).
    ann, pushed, _ = _announcer(free=True)
    passthrough = TTSSpeakFrame("not ours")
    _send(ann, passthrough)
    assert passthrough in pushed


# --- wake-ack earcon (pre-rendered tone, bare OutputAudioRawFrame → no half-duplex mute) -------------
def test_play_earcon_pushes_bare_output_audio_frame():
    from pipecat.frames.frames import OutputAudioRawFrame, TTSAudioRawFrame
    from announce import _EARCON_PCM, _EARCON_RATE

    ann, pushed, _ = _announcer(started=True)
    asyncio.run(ann.play_earcon())
    audio = [f for f in pushed if isinstance(f, OutputAudioRawFrame)]
    assert len(audio) == 1
    f = audio[0]
    # MUST be a bare OutputAudioRawFrame, NOT a TTS/Speech audio frame (those would trip the bot-speaking
    # half-duplex mute and could clip a one-breath command — the whole reason we use an earcon, not a word).
    assert not isinstance(f, TTSAudioRawFrame)
    assert f.audio == _EARCON_PCM and f.sample_rate == _EARCON_RATE
    # No spoken text, no bot-speech bracket.
    assert _spoken(pushed) == []


def test_play_earcon_noop_before_start():
    ann, pushed, _ = _announcer(started=False)  # pipeline not up yet
    asyncio.run(ann.play_earcon())
    assert pushed == []  # never push pre-StartFrame (would be dropped + could wedge)
