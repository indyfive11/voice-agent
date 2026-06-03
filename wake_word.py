"""Wake-word gate — require a wake word ("hey jarvis"/custom "Aria") before commands while media plays.

Why: with music playing, the VAD now *detects* the user's onset over the (capped) media, but Whisper
MIS-transcribes the opening words because they land before any duck takes effect — recognition, not
detection, is the bottleneck (see the 2026-06-02 tuning run). A wake word sidesteps it the way smart
speakers do: a robust *binary* wake-word detector survives over music far better than full STT, so —

    media playing → user says "hey jarvis" (detected over the music) → gate pre-ducks the media and
    opens a command window → user speaks the command into now-ducked audio → Whisper transcribes it
    cleanly → window closes, media restores.

The wake word is the pre-roll that lets the duck land *before* the command. Validated live: openWakeWord
scores 0.67–1.0 on the wake word over music with a ~0 noise floor (tools/wakeword_probe.py).

Design:
- **Opt-in** (`WAKE_WORD` env; absent → gate not built, open-mic as before).
- **Media-aware** (`WAKE_WORD_MEDIA_ONLY`, default on): pass-through (open-mic) when nothing plays;
  require the wake word only while media plays — so it's invisible in a quiet room.
- Sits right after the InputResampler (16 kHz mono). While gated+closed it swallows mic audio (STT/VAD
  see nothing). On wake it pre-ducks via the brain and opens a window (refreshed by each user turn),
  restoring on window close. Debounced with a refractory period (openWakeWord fires on several
  consecutive frames per utterance).
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_CHUNK_SAMPLES = 1280  # 80ms @16k — openWakeWord's preferred frame
_CHUNK_BYTES = _CHUNK_SAMPLES * 2  # int16 mono


def _tlog(message: str) -> None:
    """One line to the transcript log (greppable alongside USER/BOT/DUCK/WAKE)."""
    logger.bind(transcript=True).info(message)


class WakeWordGate(FrameProcessor):
    """Gate mic audio behind a wake word while media plays; pre-duck + open a command window on wake."""

    def __init__(
        self,
        oww_model,
        *,
        threshold: float = 0.5,
        window_secs: float = 15.0,
        refractory_secs: float = 1.5,
        brain_client=None,
        session_id: str = "",
        is_media_playing: Callable[[], "bool|None"] | None = None,
        media_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._oww = oww_model
        self._keys = list(oww_model.models.keys())
        self._threshold = threshold
        self._window_secs = window_secs
        self._refractory = refractory_secs
        self._client = brain_client
        self._session_id = session_id
        # Async callable → True/False/None (None = unknown). Used only when media_only.
        self._is_media_playing = is_media_playing
        self._media_only = media_only and is_media_playing is not None

        self._buf = bytearray()
        self._open = False
        self._last_wake = 0.0
        self._ducked = False
        self._window_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            gated = await self._gated_now()
            if gated:
                await self._feed(frame.audio)
                if not self._open:
                    return  # swallow: muted until the wake word
            await self.push_frame(frame, direction)
            return

        # Keep the command window alive across a multi-turn exchange.
        if isinstance(frame, TranscriptionFrame) and self._open:
            if len((getattr(frame, "text", "") or "").split()) >= 1:
                self._arm_window()

        await self.push_frame(frame, direction)

    # --- gating mode -------------------------------------------------------
    async def _gated_now(self) -> bool:
        """True when the wake word is currently required (always, or — media_only — while media plays)."""
        if not self._media_only:
            return True
        playing = self._is_media_playing()
        if asyncio.iscoroutine(playing) or asyncio.isfuture(playing):
            playing = await playing
        return bool(playing)

    # --- wake detection ----------------------------------------------------
    async def _feed(self, audio: bytes) -> None:
        self._buf.extend(audio)
        loop = asyncio.get_running_loop()
        while len(self._buf) >= _CHUNK_BYTES:
            chunk = bytes(self._buf[:_CHUNK_BYTES])
            del self._buf[:_CHUNK_BYTES]
            pcm = np.frombuffer(chunk, dtype=np.int16)
            # onnxruntime releases the GIL; run off the event loop to avoid audio hiccups.
            scores = await loop.run_in_executor(None, self._oww.predict, pcm)
            score = max((float(scores.get(k, 0.0)) for k in self._keys), default=0.0)
            now = time.monotonic()
            if score >= self._threshold and (now - self._last_wake) > self._refractory:
                self._last_wake = now
                self._on_wake(score)

    def _on_wake(self, score: float) -> None:
        if not self._open:
            self._open = True
            _tlog(f"WAKE  | wake word ({score:.2f}) — opening command window")
            self._fire_duck(True)  # pre-duck so the command lands on already-ducked media
        self._arm_window()

    # --- command window ----------------------------------------------------
    def _arm_window(self) -> None:
        self._cancel_window()

        async def _later():
            try:
                await asyncio.sleep(self._window_secs)
                if self._open:
                    self._open = False
                    _tlog("WAKE  | window closed (idle) — muting until next wake word")
                    self._fire_duck(False)
            except asyncio.CancelledError:
                pass

        self._window_task = asyncio.create_task(_later())

    def _cancel_window(self) -> None:
        if self._window_task is not None and not self._window_task.done():
            self._window_task.cancel()
        self._window_task = None

    # --- pre-duck (idempotent with MediaDuckController via the brain's idempotent /media/duck) ------
    def _fire_duck(self, on: bool) -> None:
        if self._client is None or on == self._ducked:
            return
        self._ducked = on

        async def _go():
            try:
                await self._client.duck(self._session_id, on)
            except Exception as e:  # noqa: BLE001 - media control must never break audio
                _tlog(f"WAKE  | duck on={on} FAILED: {e}")

        asyncio.create_task(_go())
