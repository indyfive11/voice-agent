"""Media-duck controller — a pass-through FrameProcessor that owns `/media/duck` *timing*.

Sits right after STT in the pipeline because that's the only place that sees transcription frames:
the user aggregator consumes `TranscriptionFrame`/`InterimTranscriptionFrame` and does **not**
forward them downstream, so this logic can't live in `BrainLLMService` (which is downstream of the
aggregator). The brain owns the duck *action* (music → volume via Mopidy; Jellyfin video →
`<video>.volume`); we only decide *when* to signal on/off.

Design (the 2026-06-02 low-latency revision — supersedes the duck-on-confirmed-transcription rev):
- **Duck on VAD speech *onset*** (`VADUserStartedSpeakingFrame`, ~0.2s after you start) rather than
  waiting for a finished, transcribed phrase. The old "duck on ≥min_words transcription" rule made the
  duck engage 1–3s late (it had to wait for the segment to end *and* for Whisper) — the thing the user
  heard as lag. Onset-triggering is safe now that the PipeWire echo-cancel source removes the
  speaker output from the mic, so playback no longer self-trips the VAD.
- **False-onset guard:** an onset that yields no real words (a cough, a stray VAD blip) restores
  quickly — on `VADUserStoppedSpeaking` we arm a short `confirm_grace` timer that restores unless a
  ≥`min_words` transcription confirms the onset was speech (which cancels it). Continued speech
  (another onset) also cancels it, so a multi-clause utterance never flaps.
- **Confirmed transcription** (≥`min_words`) marks the turn real, cancels the confirm timer, and arms
  the slow idle-restore — and still *triggers* the duck itself if the onset frame never arrived
  (graceful fallback to the old behavior; no regression).
- **Restore when Aria finishes** (`BotStoppedSpeakingFrame`), or after a generous quiet grace if a
  ducked turn never produces bot speech.
- **Idempotent on/off**, **skip when nothing is playing** (best-effort `GET /media/state`), and
  **skip while asleep**.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
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
        confirm_grace: float = 2.5,
        should_duck: Callable[[], bool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._session_id = session_id
        self._min_words = max(1, min_words)
        self._restore_grace = restore_grace
        # How long after speech *stops* to wait for a confirming transcription before treating the
        # onset as a false trigger and restoring. Must comfortably exceed Whisper's per-segment
        # latency so real speech confirms before it fires (else a slow transcribe would flap).
        self._confirm_grace = confirm_grace
        # Gate (e.g. "not asleep"); default always-allow.
        self._should_duck = should_duck or (lambda: True)
        self._ducked = False
        self._bot_spoke = False  # did the bot speak during the current duck?
        self._confirmed = False  # got ≥min_words this duck episode (so it's not a false onset)
        self._restore_task: asyncio.Task | None = None
        self._confirm_task: asyncio.Task | None = None
        # Short-TTL cache for the media-state gate. Onset-triggering can fire many duck attempts per
        # spoken sentence (each VAD segment), and each would otherwise hit `GET /media/state`. Cache
        # the answer for `_media_state_ttl`s so a talkative paused-media stretch doesn't spam the
        # brain; ≤TTL staleness only delays a duck by that much when media (re)starts.
        self._media_state_ttl = 1.0
        self._media_state_cache: bool | None = None
        self._media_state_at = 0.0
        self._media_state_inflight: asyncio.Future | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._handle(frame)
        await self.push_frame(frame, direction)

    # --- frame handling ----------------------------------------------------
    def _handle(self, frame: Frame) -> None:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # Speech onset — duck immediately. Cancel any pending false-onset restore (still talking).
            self._cancel_confirm()
            self._duck_on("speech onset")
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Speech segment ended. If this onset hasn't been confirmed by real words yet, start the
            # short countdown that restores it as a false trigger (cancelled if a transcription
            # confirms, or if the user resumes speaking).
            if self._ducked and not self._confirmed:
                self._arm_confirm()
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            text = getattr(frame, "text", "") or ""
            if len(text.split()) >= self._min_words:
                self._on_confirmed_speech()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_spoke = True
            self._cancel_confirm()
            self._cancel_restore()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            _tlog(f"DUCK  | bot-stopped (ducked={self._ducked})")
            if self._ducked:
                self._restore("bot-stopped")

    def _duck_on(self, reason: str) -> None:
        if not self._should_duck() or self._ducked:
            return
        self._ducked = True
        self._bot_spoke = False
        self._confirmed = False
        _tlog(f"DUCK  | on ({reason})")
        self._fire(True)

    def _on_confirmed_speech(self) -> None:
        if not self._should_duck():
            return
        # Real words → this duck episode is confirmed; drop the false-onset guard.
        self._confirmed = True
        self._cancel_confirm()
        # Fallback: if the onset frame never reached us, the transcription still triggers the duck
        # (old behavior — no regression if VAD frames don't flow for some transport).
        self._duck_on("confirmed speech")
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
        (the brain's own duck is a harmless no-op when there's no media). Debounced by a short TTL so
        rapid onsets during a talkative stretch don't spam `GET /media/state`."""
        now = time.monotonic()
        if self._media_state_cache is not None and (now - self._media_state_at) < self._media_state_ttl:
            return self._media_state_cache
        # Coalesce concurrent queries (onset-fired duck attempts overlap) onto one in-flight call.
        if self._media_state_inflight is not None:
            return await self._media_state_inflight
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._media_state_inflight = fut
        try:
            result = await self._query_media_playing()
            self._media_state_cache = result
            self._media_state_at = time.monotonic()
            fut.set_result(result)
            return result
        finally:
            self._media_state_inflight = None

    async def _query_media_playing(self) -> bool:
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

    # --- false-onset confirm timer ----------------------------------------
    def _arm_confirm(self) -> None:
        self._cancel_confirm()

        async def _later():
            try:
                await asyncio.sleep(self._confirm_grace)
                # Speech stopped and no qualifying transcription arrived → it wasn't real speech.
                if self._ducked and not self._confirmed and not self._bot_spoke:
                    self._restore("unconfirmed onset")
            except asyncio.CancelledError:
                pass

        self._confirm_task = asyncio.create_task(_later())

    def _cancel_confirm(self) -> None:
        if self._confirm_task is not None and not self._confirm_task.done():
            self._confirm_task.cancel()
        self._confirm_task = None

    def _restore(self, reason: str = "?") -> None:
        self._cancel_restore()
        self._cancel_confirm()
        if self._ducked:
            self._ducked = False
            self._confirmed = False
            _tlog(f"DUCK  | off / restore (via {reason})")
            self._fire(False)
