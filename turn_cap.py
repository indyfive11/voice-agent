"""Max-turn-duration stop strategy — force-complete a runaway user turn so the agent never hangs.

pipecat ends a user turn only on `stop_secs` silence or the SmartTurn model's verdict; `max_duration_secs`
is just the SmartTurn model's audio-analysis window, NOT a turn cap. So continuous speech with sub-stop_secs
pauses can run indefinitely — live (2026-06-02) a ~40s ramble accumulated into one turn and felt "stuck"
(nothing fired until it finally ended). This adds a wall-clock cap: timed from the first VAD speech onset of
the turn, if it's still going at `max_turn_secs` and we have transcribed text, force-complete it.

Guard: never force-complete an *empty* turn (no transcript → no brain call), per the brain contract
(an empty /respond wastes a turn). Runs alongside TurnAnalyzerUserTurnStopStrategy as a parallel safety —
the SmartTurn strategy still ends normal turns; this only catches the runaway tail.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, VADUserStartedSpeakingFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


class MaxTurnDurationUserTurnStopStrategy(BaseUserTurnStopStrategy):
    """Force-complete a user turn that has run past `max_turn_secs` (so a long ramble can't hang)."""

    def __init__(self, *, max_turn_secs: float = 15.0, **kwargs):
        super().__init__(**kwargs)
        self._max_turn_secs = max_turn_secs
        self._text = ""
        self._armed = False
        self._task: asyncio.Task | None = None

    async def reset(self):
        """Reset per-turn state (called by the controller when a turn ends)."""
        await super().reset()
        self._text = ""
        self._armed = False
        await self._cancel()

    async def cleanup(self):
        await super().cleanup()
        await self._cancel()

    async def _cancel(self):
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        await super().process_frame(frame)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._armed:  # arm once, on the FIRST onset of the turn
                self._armed = True
                self._task = self.task_manager.create_task(self._cap(), f"{self}::_cap")
                await asyncio.sleep(0)  # ensure it's scheduled
        elif isinstance(frame, TranscriptionFrame):
            if frame.text:
                self._text = frame.text
        return ProcessFrameResult.CONTINUE  # additive: other stop strategies still evaluate

    async def _cap(self):
        try:
            await asyncio.sleep(self._max_turn_secs)
        except asyncio.CancelledError:
            return
        finally:
            self._task = None
        if self._text.strip():
            _tlog(f"TURN  | force-complete after {self._max_turn_secs}s cap (runaway turn)")
            await self.trigger_user_turn_stopped()
        else:
            # No transcript yet → don't fire an empty turn at the brain; let it end naturally.
            _tlog(f"TURN  | {self._max_turn_secs}s cap reached but no transcript — not force-completing")
