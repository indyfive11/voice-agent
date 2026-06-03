"""Offline tests for the max-turn-duration stop strategy — no pipeline/task-manager needed.

Drives `_cap` directly (the wall-clock force-complete) with a fake trigger, verifying it fires only when
there's transcribed text (the empty-turn guard). asyncio.run drives the async bodies.
"""

import asyncio

from turn_cap import MaxTurnDurationUserTurnStopStrategy


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
