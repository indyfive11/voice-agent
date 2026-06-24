"""SmartTurnHonoringStopStrategy — suppress the turn-stop while SmartTurn judges the turn INCOMPLETE.

Verifies the single trigger funnel (`trigger_user_turn_stopped`) is gated on the analyzer's `speech_triggered`
(True ⟺ last verdict INCOMPLETE), so the base's transcript-fallback / STT-timeout can't cut the user off on a
natural pause — while a real COMPLETE verdict (speech_triggered cleared) or a finalized transcript still ends it.
"""

import asyncio
import time

from loguru import logger

from turn_stop import LongHoldState, SmartTurnHonoringStopStrategy


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


class _FakeTaskMgr:
    """Minimal task manager: run scheduled coros as real asyncio tasks so the grace timer fires/cancels."""

    def create_task(self, coro, name=None):
        return asyncio.ensure_future(coro)

    async def cancel_task(self, task, *args, **kwargs):
        task.cancel()
        try:
            await task
        except BaseException:  # noqa: BLE001 - swallow CancelledError + any teardown error
            pass


def _make_grace(*, grace: float, max_words: int = 3, speech_triggered: bool = False):
    strat = SmartTurnHonoringStopStrategy(
        turn_analyzer=_FakeAnalyzer(speech_triggered),
        continuation_grace_secs=grace,
        continuation_max_words=max_words,
    )
    strat._task_manager = _FakeTaskMgr()  # property is read-only; set the backing field for the test
    fired: list = []
    strat.add_event_handler("on_user_turn_stopped", lambda *a, **k: fired.append(True))
    return strat, fired


def test_short_turn_schedules_grace_not_immediate():
    # #62: a SHORT COMPLETE turn ("up") is the fragment signature → hold the stop, don't forward yet.
    async def scenario():
        strat, fired = _make_grace(grace=0.05)
        strat._text = "up"
        await strat.trigger_user_turn_stopped()
        assert fired == []  # NOT forwarded immediately
        assert strat._pending_stop_task is not None
        await asyncio.sleep(0.12)  # let the grace elapse with no resume
        assert fired == [True]  # then it forwards
    asyncio.run(scenario())


def test_resume_within_grace_cancels_the_stop():
    # The user resumed during the grace → the pending stop is cancelled and the turn stays open (the
    # continuation will join this turn instead of being dropped as a next-turn fragment).
    async def scenario():
        strat, fired = _make_grace(grace=0.05)
        strat._text = "up"
        await strat.trigger_user_turn_stopped()
        assert strat._pending_stop_task is not None
        await strat._cancel_pending_stop()  # what a fresh VAD onset does in process_frame
        await asyncio.sleep(0.12)
        assert fired == []  # never forwarded — turn held open for the continuation
    asyncio.run(scenario())


def test_long_turn_forwards_immediately_no_grace():
    # A full sentence (> max_words) is not a fragment → forward at once, no latency tax.
    async def scenario():
        strat, fired = _make_grace(grace=0.05, max_words=3)
        strat._text = "tell me a new joke please"  # 6 words
        await strat.trigger_user_turn_stopped()
        assert fired == [True]
        assert strat._pending_stop_task is None
    asyncio.run(scenario())


def test_grace_off_by_default_short_turn_immediate():
    # grace=0.0 (default) ⇒ byte-identical to today: even a short turn forwards immediately.
    async def scenario():
        strat, fired = _make_grace(grace=0.0)
        strat._text = "up"
        await strat.trigger_user_turn_stopped()
        assert fired == [True]
        assert strat._pending_stop_task is None
    asyncio.run(scenario())


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


# --- Long-form / dictation hold (let a long multi-clause command survive clause-boundary pauses) ---


def _make_dictation(
    *,
    triggers=("start a builder task",),
    enders=("done", "that's all"),
    silence=10.0,           # long by default so only an explicit stop phrase ends the hold in a test
    heuristic=True,
    speech_triggered=False,
):
    long_hold = LongHoldState()
    strat = SmartTurnHonoringStopStrategy(
        turn_analyzer=_FakeAnalyzer(speech_triggered),
        dictation_triggers=triggers,
        dictation_enders=enders,
        dictation_silence_secs=silence,
        longform_heuristic=heuristic,
        long_hold=long_hold,
    )
    strat._task_manager = _FakeTaskMgr()  # property is read-only; set the backing field
    fired: list = []
    strat.add_event_handler("on_user_turn_stopped", lambda *a, **k: fired.append(True))
    return strat, fired, long_hold


def test_trigger_phrase_holds_across_pauses_until_stop_phrase():
    # The headline case: "start a builder task …" holds the turn through every clause-boundary pause
    # (each a confident SmartTurn COMPLETE) and only ends when the explicit stop phrase arrives.
    async def scenario():
        strat, fired, lh = _make_dictation()
        strat._text = "start a builder task the project is builder-test"
        await strat.trigger_user_turn_stopped()          # pause 1
        assert fired == [] and strat._held and lh.active  # held, cap-extend flag set
        strat._text += " create a file named hello.txt containing the text hi"
        await strat.trigger_user_turn_stopped()          # pause 2 — still held
        assert fired == []
        strat._text += " that's all"
        await strat.trigger_user_turn_stopped()          # stop phrase → dispatch the whole thing
        assert fired == [True]
        assert not strat._held and not lh.active
    asyncio.run(scenario())


def test_heuristic_holds_unfinished_tail():
    # Always-on heuristic (no trigger phrase needed): a tail ending on a connective looks unfinished → hold.
    async def scenario():
        strat, fired, lh = _make_dictation(triggers=())
        strat._text = "the task is"      # ends on "is" → unfinished
        await strat.trigger_user_turn_stopped()
        assert fired == [] and strat._held and lh.active
        await strat._cancel_backstop()   # cleanup the pending 10s timer
    asyncio.run(scenario())


def test_heuristic_finished_tail_forwards_immediately():
    # A complete-sounding command must NOT be held (keeps normal turns snappy) — "it" is an excluded pronoun.
    async def scenario():
        strat, fired, lh = _make_dictation(triggers=())
        strat._text = "what time is it"
        await strat.trigger_user_turn_stopped()
        assert fired == [True] and not strat._held and not lh.active
    asyncio.run(scenario())


def test_silence_backstop_ends_a_held_turn():
    # No stop phrase spoken → trailing silence ends the held turn after dictation_silence_secs.
    async def scenario():
        strat, fired, lh = _make_dictation(silence=0.05)
        strat._text = "start a builder task the project is x"
        await strat.trigger_user_turn_stopped()
        assert fired == [] and strat._held
        await asyncio.sleep(0.12)         # let the backstop elapse with no resume
        assert fired == [True]
        assert not strat._held and not lh.active
    asyncio.run(scenario())


def test_resume_cancels_backstop_and_keeps_holding():
    # A real resume (a fresh VAD onset, which process_frame turns into _cancel_backstop) must not let the
    # turn end — the held turn ends only after TRUE trailing silence.
    async def scenario():
        strat, fired, lh = _make_dictation(silence=0.05)
        strat._text = "start a builder task the project is x"
        await strat.trigger_user_turn_stopped()
        assert strat._backstop_task is not None
        await strat._cancel_backstop()   # what a fresh VAD onset does in process_frame
        await asyncio.sleep(0.12)
        assert fired == [] and strat._held  # never ended — still holding for the continuation
        await strat._end_hold_and_stop()    # cleanup
    asyncio.run(scenario())


def test_stop_phrase_matched_at_tail_only():
    # A stop phrase appearing mid-utterance ("…are you done loading") must NOT end the hold; only a trailing
    # one does. Guards against premature dispatch when the word happens to occur inside the command.
    async def scenario():
        strat, fired, lh = _make_dictation(enders=("done",))
        strat._text = "start a builder task are you done loading"  # "done" mid-sentence
        await strat.trigger_user_turn_stopped()
        assert fired == [] and strat._held                          # not ended
        strat._text += " done"                                      # now trailing
        await strat.trigger_user_turn_stopped()
        assert fired == [True] and not strat._held
    asyncio.run(scenario())


def test_dictation_disabled_forwards_immediately():
    # Empty triggers + heuristic off ⇒ byte-identical to the prior behavior: an unfinished-looking tail
    # still forwards at once, nothing is held, the shared flag stays clear.
    async def scenario():
        strat, fired, lh = _make_dictation(triggers=(), enders=(), heuristic=False)
        strat._text = "the task is"
        await strat.trigger_user_turn_stopped()
        assert fired == [True] and not strat._held and not lh.active
    asyncio.run(scenario())
