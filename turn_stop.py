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

import time

from loguru import logger

from pipecat.frames.frames import Frame, VADUserStoppedSpeakingFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


class SmartTurnHonoringStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """`TurnAnalyzerUserTurnStopStrategy` that never ends a turn while SmartTurn still judges it INCOMPLETE.

    Also emits a per-turn **finalization-latency** line (the #60 endpointing instrumentation): the felt lag
    between the user going quiet and the turn closing (so the brain can answer). The line attributes the
    latency to its cause so a live ear-test produces a tunable number instead of a guess:
      * ``prompt COMPLETE, no hold`` — SmartTurn fired COMPLETE right at the VAD-stop; the ``Nms from last
        speech`` figure is then mostly the SmartTurn **inference** cost (the suspect on weak HW like a Pi-4,
        where neural inference is slow — if this is large, the lever is offloading/lightening endpointing,
        NOT ``stop_secs``).
      * ``held …ms on INCOMPLETE → stop_secs silence fallback`` — SmartTurn kept the turn open and only the
        ``stop_secs`` (default 4s) silence timer ended it: this IS the ``SMART_TURN_STOP_SECS`` latency tax,
        the knob to lower (trading against mid-thought-pause fragmentation).
      * ``held …ms on INCOMPLETE → COMPLETE verdict`` — a real COMPLETE arrived during the pause well before
        ``stop_secs``: honor-incomplete did its job (don't lower ``stop_secs`` on account of these).
    """

    def __init__(self, *, turn_analyzer, stop_secs: float = 4.0, **kwargs):
        super().__init__(turn_analyzer=turn_analyzer, **kwargs)
        self._stop_secs = stop_secs
        self._suppressed_logged = False
        self._hold_started: float | None = None  # monotonic ts of the first INCOMPLETE suppression this turn
        self._last_vad_stop: float | None = None  # monotonic ts of the most recent VAD stop this turn

    async def reset(self):
        await super().reset()
        self._suppressed_logged = False
        self._hold_started = None
        self._last_vad_stop = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        # Timestamp the latest VAD-stop BEFORE delegating: the base handler can synchronously trigger the
        # turn-stop on a prompt COMPLETE verdict within this very call, so recording after super() would
        # leave _last_vad_stop=None for no-hold turns (the "?ms" gap). Set it first so the finalization
        # line reports the real end-to-end dead air (the latency a person feels).
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_vad_stop = time.monotonic()
        return await super().process_frame(frame)

    async def trigger_user_turn_stopped(self):
        # The single funnel for every stop path in the base. Suppress it while SmartTurn is mid-turn (last
        # verdict INCOMPLETE → analyzer.speech_triggered). A real COMPLETE verdict or the stop_secs silence
        # both clear speech_triggered, so legitimate turn-ends pass straight through. We deliberately do NOT
        # exempt finalized transcripts: Whisper finalizes every segment, so that exemption made this a no-op.
        if getattr(self._turn_analyzer, "speech_triggered", False):
            if self._hold_started is None:
                self._hold_started = time.monotonic()
            if not self._suppressed_logged:
                self._suppressed_logged = True
                _tlog("TURN  | holding — SmartTurn INCOMPLETE (waiting for end-of-turn, not cutting mid-pause)")
            return
        self._log_finalization()
        await super().trigger_user_turn_stopped()

    def _log_finalization(self) -> None:
        """Attribute this turn's finalization latency to its cause (see the class docstring)."""
        now = time.monotonic()
        lead = f"{(now - self._last_vad_stop) * 1000:.0f}ms" if self._last_vad_stop is not None else "?ms"
        if self._hold_started is not None:
            held_ms = (now - self._hold_started) * 1000
            # Held ~stop_secs ⇒ the silence fallback ended it (the tax); a clearly shorter hold ⇒ a real
            # COMPLETE verdict released the pause early (honor-incomplete working as intended).
            via = "stop_secs silence fallback" if held_ms >= self._stop_secs * 1000 * 0.9 else "COMPLETE verdict"
            tail = f" (held {held_ms:.0f}ms on INCOMPLETE → {via})"
        else:
            tail = " (prompt COMPLETE, no hold)"
        _tlog(f"TURN  | finalized — {lead} from last speech{tail}")
