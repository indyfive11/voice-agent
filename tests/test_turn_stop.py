"""SmartTurnHonoringStopStrategy — suppress the turn-stop while SmartTurn judges the turn INCOMPLETE.

Verifies the single trigger funnel (`trigger_user_turn_stopped`) is gated on the analyzer's `speech_triggered`
(True ⟺ last verdict INCOMPLETE), so the base's transcript-fallback / STT-timeout can't cut the user off on a
natural pause — while a real COMPLETE verdict (speech_triggered cleared) or a finalized transcript still ends it.
"""

import asyncio
import time

from loguru import logger

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


def _capture_transcript_logs():
    """Collect transcript-bound log lines (the #60 finalization instrumentation) emitted during a block."""
    msgs: list = []
    sink = logger.add(lambda m: msgs.append(m.record["message"]),
                      filter=lambda r: r["extra"].get("transcript"), level="DEBUG")
    return msgs, sink


def test_finalization_logs_prompt_complete_when_never_held():
    # A turn that ends without ever being held (SmartTurn COMPLETE at the VAD-stop) → "prompt COMPLETE".
    strat, _ = _make(speech_triggered=False)
    strat._last_vad_stop = time.monotonic()  # pretend the user just fell silent
    msgs, sink = _capture_transcript_logs()
    try:
        asyncio.run(strat.trigger_user_turn_stopped())
    finally:
        logger.remove(sink)
    line = next(m for m in msgs if "finalized" in m)
    assert "prompt COMPLETE, no hold" in line


def test_finalization_attributes_long_hold_to_stop_secs_fallback():
    # Held ~stop_secs before release ⇒ the silence fallback ended it (the SMART_TURN_STOP_SECS latency tax).
    strat, _ = _make(speech_triggered=False)
    strat._stop_secs = 4.0
    strat._hold_started = time.monotonic() - 5.0  # held ~5s, past the 0.9*stop_secs threshold
    msgs, sink = _capture_transcript_logs()
    try:
        asyncio.run(strat.trigger_user_turn_stopped())
    finally:
        logger.remove(sink)
    line = next(m for m in msgs if "finalized" in m)
    assert "stop_secs silence fallback" in line


def test_finalization_attributes_short_hold_to_complete_verdict():
    # A brief hold released well before stop_secs ⇒ a real COMPLETE verdict (honor-incomplete working).
    strat, _ = _make(speech_triggered=False)
    strat._stop_secs = 4.0
    strat._hold_started = time.monotonic() - 0.4  # held ~0.4s, far under the fallback threshold
    msgs, sink = _capture_transcript_logs()
    try:
        asyncio.run(strat.trigger_user_turn_stopped())
    finally:
        logger.remove(sink)
    line = next(m for m in msgs if "finalized" in m)
    assert "COMPLETE verdict" in line and "stop_secs" not in line
