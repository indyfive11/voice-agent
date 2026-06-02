"""Media-duck controller — a pass-through FrameProcessor that owns `/media/duck` *timing*.

Sits right after STT in the pipeline because that's the only place that sees transcription frames:
the user aggregator consumes `TranscriptionFrame`/`InterimTranscriptionFrame` and does **not**
forward them downstream, so this logic can't live in `BrainLLMService` (which is downstream of the
aggregator). The brain owns the duck *action* (music → volume via Mopidy; Jellyfin video →
`<video>.volume`); we only decide *when* to signal on/off.

Design (the 2026-06-02 redesign — see plan + VOICE_MODE handoff):
- **Duck on *confirmed* speech, not raw VAD.** Trigger on transcribed words (≥ `min_words`), so
  non-speech playback (music beds, effects) that trips the VAD but yields no words can't self-trigger
  the duck. (Movie *dialogue* still transcribes — that residual is what the PipeWire echo-cancel
  source ultimately removes; this gating + the brain's volume-duck keep it benign meanwhile.)
- **Restore when Aria finishes** (`BotStoppedSpeakingFrame`), or after a generous quiet grace if a
  ducked turn never produces bot speech (a no-op/abandoned turn). No blind short timer driving
  re-ducks — that was the oscillation.
- **Asymmetric (quick duck, slow restore) + idempotent on/off** so it can't flap.
- **Skip when nothing is playing** (best-effort `GET /media/state`) and **while asleep**.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _tlog(message: str) -> None:
    """One line to the transcript log (greppable alongside USER/BOT/DUCK)."""
    logger.bind(transcript=True).info(message)


class MediaDuckController(FrameProcessor):
    """Drive media ducking off confirmed-speech + bot-speaking frames flowing through the pipeline."""

    def __init__(
        self,
        client,
        session_id: str,
        *,
        min_words: int = 2,
        restore_grace: float = 8.0,
        should_duck: Callable[[], bool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._session_id = session_id
        self._min_words = max(1, min_words)
        self._restore_grace = restore_grace
        # Gate (e.g. "not asleep"); default always-allow.
        self._should_duck = should_duck or (lambda: True)
        self._ducked = False
        self._bot_spoke = False  # did the bot speak during the current duck?
        self._restore_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._handle(frame)
        await self.push_frame(frame, direction)

    # --- frame handling ----------------------------------------------------
    def _handle(self, frame: Frame) -> None:
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            text = getattr(frame, "text", "") or ""
            if len(text.split()) >= self._min_words:
                self._on_confirmed_speech()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_spoke = True
            self._cancel_restore()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            _tlog(f"DUCK  | bot-stopped (ducked={self._ducked})")
            if self._ducked:
                self._restore("bot-stopped")

    def _on_confirmed_speech(self) -> None:
        if not self._should_duck():
            return
        if not self._ducked:
            self._ducked = True
            self._bot_spoke = False
            _tlog("DUCK  | on (confirmed speech)")
            self._fire(True)
        # (Re)arm the idle fallback: a ducked turn that never yields bot speech restores after the
        # grace. Re-arming on each transcription means continuous speech keeps media ducked; the
        # grace only elapses once speech truly stops with no reply, so a normal thinking gap (bot
        # replies within the grace → BotStarted cancels it) never restores prematurely.
        self._arm_restore()

    # --- duck firing -------------------------------------------------------
    def _fire(self, on: bool) -> None:
        """Fire-and-forget the brain duck call — never block the audio pipeline."""

        async def _go():
            try:
                if on and not await self._media_playing():
                    # Nothing actually playing — don't bother the brain; undo the optimistic flag.
                    self._ducked = False
                    _tlog("DUCK  | on SKIPPED (media_state: nothing playing)")
                    return
                await self._client.duck(self._session_id, on)
                _tlog(f"DUCK  | /media/duck on={on} sent")
            except Exception as e:  # noqa: BLE001 - media control must never break audio
                _tlog(f"DUCK  | /media/duck on={on} FAILED: {e}")

        asyncio.create_task(_go())

    async def _media_playing(self) -> bool:
        """Best-effort: True unless the brain reports nothing is playing. Unknown/old brain → True
        (the brain's own duck is a harmless no-op when there's no media)."""
        media_state = getattr(self._client, "media_state", None)
        if media_state is None:
            return True
        try:
            st = await media_state(self._session_id)
        except Exception:  # noqa: BLE001
            return True
        if not st:
            return True
        # Neutral shape: {"playing": bool, "state": "playing|paused|idle"}. Fall back to a value scan
        # for an older brain that might still return per-provider keys.
        if "playing" in st:
            return bool(st["playing"])
        return any(str(v).lower() == "playing" for v in st.values())

    # --- restore timer -----------------------------------------------------
    def _arm_restore(self) -> None:
        self._cancel_restore()

        async def _later():
            try:
                await asyncio.sleep(self._restore_grace)
                if self._ducked and not self._bot_spoke:
                    self._restore("idle-grace")
            except asyncio.CancelledError:
                pass

        self._restore_task = asyncio.create_task(_later())

    def _cancel_restore(self) -> None:
        if self._restore_task is not None and not self._restore_task.done():
            self._restore_task.cancel()
        self._restore_task = None

    def _restore(self, reason: str = "?") -> None:
        self._cancel_restore()
        if self._ducked:
            self._ducked = False
            _tlog(f"DUCK  | off / restore (via {reason})")
            self._fire(False)
