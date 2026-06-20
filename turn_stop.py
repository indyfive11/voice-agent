"""SmartTurn-honoring user-turn stop strategy — don't end a turn the model judged INCOMPLETE.

The stock `TurnAnalyzerUserTurnStopStrategy` ends a turn when `_turn_complete` AND a transcript exist. But it
sets `_turn_complete = True` in a **transcript fallback** ("assume the turn is complete when a transcript
arrives with no VAD-stop recorded yet", `turn_analyzer_user_turn_stop_strategy.py`), then arms a short
STT-p99 timeout (`p99 - vad_stop_secs`, ~0.8s with the default 1.0s p99) that fires the turn stop. Over our
always-on Silero VAD, a Whisper transcript routinely lands ~1ms around the VAD-stop frame, so that fallback
trips and the STT-timeout ends the turn **even when SmartTurn just predicted INCOMPLETE** — cutting the user
off on a natural mid-sentence pause (2026-06-20 live: "Tell me a little story about…<pause>" → answered with a
generic story before the user named the topic; SmartTurn had scored it INCOMPLETE 0.64% 0.8s earlier).

Fix: gate the single trigger funnel (`trigger_user_turn_stopped`, which every stop path in the base routes
through) on SmartTurn's own state. `BaseSmartTurn.speech_triggered` is True exactly while the model considers
the turn unfinished — it's cleared ONLY on a real COMPLETE verdict or the `stop_secs` silence timeout
(`_clear` runs on COMPLETE, never on INCOMPLETE). So: while `speech_triggered` is True, suppress the stop —
let only a genuine COMPLETE verdict or the `stop_secs` (4s) silence end it. A user who resumes ("…a dog and a
cat") fires a fresh VAD onset that resets the turn; a user who truly stopped ends via SmartTurn's 4s fallback.
The parallel `MaxTurnDurationUserTurnStopStrategy` (15s) remains the runaway backstop, so a turn can never hang.

We do NOT exempt "finalized" transcripts: Whisper (a batch STT) marks **every** segment `finalized=True`
(pipecat stt_service.py: "every transcription is inherently finalized"), so a finalized check never fires and
left this guard a no-op (2026-06-20 live: turn ended on INCOMPLETE, no hold). SmartTurn — not STT
finalization — is the turn-end authority in this pipeline, so honor it regardless of `finalized`.

Env `TURN_HONOR_INCOMPLETE=0` reverts to the stock behavior (A/B / kill-switch).
"""

from __future__ import annotations

from loguru import logger

from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


class SmartTurnHonoringStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """`TurnAnalyzerUserTurnStopStrategy` that never ends a turn while SmartTurn still judges it INCOMPLETE."""

    def __init__(self, *, turn_analyzer, **kwargs):
        super().__init__(turn_analyzer=turn_analyzer, **kwargs)
        self._suppressed_logged = False

    async def reset(self):
        await super().reset()
        self._suppressed_logged = False

    async def trigger_user_turn_stopped(self):
        # The single funnel for every stop path in the base. Suppress it while SmartTurn is mid-turn (last
        # verdict INCOMPLETE → analyzer.speech_triggered). A real COMPLETE verdict or the stop_secs silence
        # both clear speech_triggered, so legitimate turn-ends pass straight through. We deliberately do NOT
        # exempt finalized transcripts: Whisper finalizes every segment, so that exemption made this a no-op.
        if getattr(self._turn_analyzer, "speech_triggered", False):
            if not self._suppressed_logged:
                self._suppressed_logged = True
                _tlog("TURN  | holding — SmartTurn INCOMPLETE (waiting for end-of-turn, not cutting mid-pause)")
            return
        await super().trigger_user_turn_stopped()
