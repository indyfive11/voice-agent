"""SmartTurnHonoringStopStrategy — suppress the turn-stop while SmartTurn judges the turn INCOMPLETE.

Verifies the single trigger funnel (`trigger_user_turn_stopped`) is gated on the analyzer's `speech_triggered`
(True ⟺ last verdict INCOMPLETE), so the base's transcript-fallback / STT-timeout can't cut the user off on a
natural pause — while a real COMPLETE verdict (speech_triggered cleared) or a finalized transcript still ends it.
"""

import asyncio

from turn_stop import SmartTurnHonoringStopStrategy


class _FakeAnalyzer:
    """Minimal stand-in: only `speech_triggered` matters to the strategy's gate."""

    def __init__(self, speech_triggered: bool):
        self.speech_triggered = speech_triggered
        self.params = None


def _make(speech_triggered: bool):
    strat = SmartTurnHonoringStopStrategy(turn_analyzer=_FakeAnalyzer(speech_triggered))
    fired: list = []
    strat.add_event_handler("on_user_turn_stopped", lambda *a, **k: fired.append(True))
    return strat, fired


def test_suppresses_stop_while_smart_turn_incomplete():
    # SmartTurn still mid-turn (speech_triggered) → the stop is held (the run-1 "cut me off mid-pause" case).
    strat, fired = _make(speech_triggered=True)
    asyncio.run(strat.trigger_user_turn_stopped())
    assert fired == []  # turn NOT ended


def test_allows_stop_when_smart_turn_complete():
    # SmartTurn completed (a COMPLETE verdict or the stop_secs silence clears speech_triggered) → turn ends.
    strat, fired = _make(speech_triggered=False)
    asyncio.run(strat.trigger_user_turn_stopped())
    assert fired == [True]


def test_finalized_transcript_does_not_override_incomplete():
    # Whisper marks every segment finalized=True, so finalization must NOT bypass the SmartTurn gate —
    # otherwise the guard is a no-op (the 2026-06-20 live miss). Honor SmartTurn regardless of finalized.
    strat, fired = _make(speech_triggered=True)
    strat._transcript_finalized = True
    asyncio.run(strat.trigger_user_turn_stopped())
    assert fired == []  # still held — SmartTurn INCOMPLETE wins over STT finalization
