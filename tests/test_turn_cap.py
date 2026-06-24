"""Offline tests for the max-turn-duration stop strategy — no pipeline/task-manager needed.

Drives `_cap` directly (the wall-clock force-complete) with a fake trigger, verifying it fires only when
there's transcribed text (the empty-turn guard). asyncio.run drives the async bodies.
"""

import asyncio
import time

from turn_cap import MaxTurnDurationUserTurnStopStrategy
from turn_stop import LongHoldState


def _strategy(**kw):
    kw.setdefault("max_turn_secs", 0.01)
    return MaxTurnDurationUserTurnStopStrategy(**kw)


def test_cap_force_completes_when_text_present():
    s = _strategy()
    fired = []

    async def fake_trigger():
        fired.append(True)

    s.trigger_user_turn_stopped = fake_trigger
    s._text = "play some music"

    asyncio.run(s._cap())
    assert fired == [True]  # runaway turn with content → force-completed


def test_cap_skips_when_no_text():
    s = _strategy()
    fired = []

    async def fake_trigger():
        fired.append(True)

    s.trigger_user_turn_stopped = fake_trigger
    s._text = "   "  # no real words

    asyncio.run(s._cap())
    assert fired == []  # empty turn → never POSTed to the brain


def test_cap_extends_during_long_form_hold():
    # While a long-form/dictation hold is active, the cap must NOT guillotine at the normal max_turn_secs —
    # it extends to dictation_max_turn_secs (the held turn's silence backstop normally ends it first).
    lh = LongHoldState()
    lh.active = True
    s = MaxTurnDurationUserTurnStopStrategy(max_turn_secs=0.01, long_hold=lh, dictation_max_turn_secs=0.06)
    fired = []

    async def fake_trigger():
        fired.append(True)

    s.trigger_user_turn_stopped = fake_trigger
    s._text = "start a builder task the project is x"

    start = time.monotonic()
    asyncio.run(s._cap())
    elapsed = time.monotonic() - start
    assert fired == [True]      # still force-completes eventually (runaway protection preserved)
    assert elapsed >= 0.05      # but only after extending well past the normal 0.01s cap


def test_cap_unchanged_without_long_hold():
    # No shared flag → exactly the old behavior: fire at max_turn_secs.
    s = MaxTurnDurationUserTurnStopStrategy(max_turn_secs=0.01)
    fired = []

    async def fake_trigger():
        fired.append(True)

    s.trigger_user_turn_stopped = fake_trigger
    s._text = "play some music"
    asyncio.run(s._cap())
    assert fired == [True]
