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
from dataclasses import dataclass
from typing import Callable

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame, SystemFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_CHUNK_SAMPLES = 1280  # 80ms @16k — openWakeWord's preferred frame
_CHUNK_BYTES = _CHUNK_SAMPLES * 2  # int16 mono


@dataclass
class WakeHoldFrame(SystemFrame):
    """Pushed UPSTREAM by the brain (BrainLLMService) to keep the wake gate's command window OPEN while
    a confirm is pending — so the user's yes/no answer doesn't require a fresh wake — and to release it
    after the confirm resolves. SystemFrame so it propagates promptly regardless of queueing."""

    hold: bool = True


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
        media_status: Callable[[], "dict|None"] | None = None,
        media_only: bool = True,
        gate_video: bool = False,
        debug: bool = False,
        debug_floor: float = 0.2,
        escape_floor: float = 0.15,
        escape_count: int = 3,
        escape_secs: float = 12.0,
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
        # Shared media-state reader → {"playing": bool, "kind": "audio"|"video"|None}. Used only when
        # media_only. gate_video=False → don't gate active VIDEO the user is watching (they pause by hand;
        # locking them behind a wake word the movie itself defeats is the 2026-06-03 over-movie lockout).
        self._media_status = media_status
        self._media_only = media_only and media_status is not None
        self._gate_video = gate_video
        # Near-miss diagnostic: log sub-threshold scores so the wake log isn't blind to the times the
        # user said the word but it scored just under the bar (logs otherwise show hits, never effort).
        self._debug = debug
        self._debug_floor = debug_floor
        # Escape hatch: if the user is clearly trying (≥escape_count sub-threshold bursts ≥escape_floor
        # within escape_secs) but the weak model never clears threshold over media, open the gate anyway
        # so they're never hard-locked-out (the 2026-06-03 over-movie lockout: "Aria" peaked 0.22–0.43).
        self._escape_floor = escape_floor
        self._escape_count = escape_count
        self._escape_secs = escape_secs
        self._escape_peak = 0.0
        self._escape_hits: list[float] = []

        self._buf = bytearray()
        self._open = False
        self._last_gated: bool | None = None  # last gate decision (for debug transition logging)
        self._hb_peak = 0.0  # max wake score since the last heartbeat (lockout-vs-freeze diagnosis)
        self._hold = False  # brain holds the window open while a confirm is pending (WakeHoldFrame)
        self._last_wake = 0.0
        self._ducked = False
        self._window_task: asyncio.Task | None = None
        # Peak score within the current sub-threshold burst (one line emitted per utterance, not per frame).
        self._nearmiss_peak = 0.0
        self._nearmiss_key = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            gated = await self._gated_now()
            if self._debug and gated != self._last_gated:
                self._last_gated = gated
                _tlog(
                    "GATE  | media=playing → gating (muted until wake)" if gated
                    else "GATE  | media=idle → pass-through (open mic)"
                )
            if gated:
                await self._feed(frame.audio)
                if not self._open:
                    return  # swallow: muted until the wake word
            await self.push_frame(frame, direction)
            return

        # Confirm pending (from the brain): hold the window open so the yes/no answer needs no re-wake.
        if isinstance(frame, WakeHoldFrame):
            self._set_hold(frame.hold)
            await self.push_frame(frame, direction)
            return

        # Keep the command window alive across a multi-turn exchange.
        if isinstance(frame, TranscriptionFrame) and self._open:
            if len((getattr(frame, "text", "") or "").split()) >= 1:
                self._arm_window()

        await self.push_frame(frame, direction)

    def _set_hold(self, hold: bool) -> None:
        """Brain confirm pending → keep the window open (open it if needed); release → resume idle close."""
        self._hold = hold
        if hold:
            if not self._open:
                self._open_window("GATE  | hold — confirm pending, opening window")
            else:
                self._cancel_window()  # freeze the idle timer while held
                _tlog("GATE  | hold — confirm pending, window held open")
        else:
            _tlog("GATE  | hold released — re-arming idle window")
            self._arm_window()

    def hb_state(self) -> dict:
        """Snapshot for the input-watchdog heartbeat (and reset the wake-peak window). `gated=True
        open=False` with a low `wake_peak` over several beats = a LOCKOUT (masking), distinct from a
        freeze (the heartbeat line stops entirely)."""
        peak = self._hb_peak
        self._hb_peak = 0.0
        return {"gated": self._last_gated, "open": self._open, "wake_peak": peak}

    # --- gating mode -------------------------------------------------------
    async def _gated_now(self) -> bool:
        """True when the wake word is currently required (always, or — media_only — while media plays).
        Active *video* is not gated (unless gate_video): the user is watching it and can pause by hand."""
        if not self._media_only:
            return True
        st = self._media_status()
        if asyncio.iscoroutine(st) or asyncio.isfuture(st):
            st = await st
        if not st or not st.get("playing"):
            return False
        if not self._gate_video and st.get("kind") == "video":
            return False  # don't lock the user out of a movie they're actively watching
        return True

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
            key, score = self._best(scores)
            if score > self._hb_peak:
                self._hb_peak = score  # for the heartbeat (was "Aria" even scoring, over music?)
            now = time.monotonic()
            if score >= self._threshold and (now - self._last_wake) > self._refractory:
                self._last_wake = now
                self._reset_nearmiss()
                self._on_wake(score)
            else:
                if self._debug:
                    self._track_nearmiss(key, score)
                if not self._open and self._escape_count > 0:
                    self._track_escape(score, now)  # lockout escape (independent of debug)

    def _best(self, scores) -> "tuple[str, float]":
        """The winning (key, score) across all loaded models for this frame."""
        best_k, best_s = "", 0.0
        for k in self._keys:
            s = float(scores.get(k, 0.0))
            if s >= best_s:
                best_k, best_s = k, s
        return best_k, best_s

    # --- near-miss diagnostic (WAKE_WORD_DEBUG) -----------------------------
    def _track_nearmiss(self, key: str, score: float) -> None:
        """Track the peak of a sub-threshold burst; emit one line when the burst ends.

        Frames arrive ~12.5/s, so a single spoken "Aria" spans several — logging each would spam. Instead
        we hold the burst's peak and emit it once the score falls back under the floor (utterance over).
        """
        if score >= self._debug_floor:
            if score >= self._nearmiss_peak:
                self._nearmiss_peak = score
                self._nearmiss_key = key
        elif self._nearmiss_peak > 0.0:
            _tlog(
                f"WAKE  | near-miss {self._nearmiss_key}={self._nearmiss_peak:.2f} "
                f"(< threshold {self._threshold:.2f})"
            )
            self._reset_nearmiss()

    def _reset_nearmiss(self) -> None:
        self._nearmiss_peak = 0.0
        self._nearmiss_key = ""

    # --- lockout escape hatch ----------------------------------------------
    def _track_escape(self, score: float, now: float) -> None:
        """Count sub-threshold bursts (burst-peak, like the near-miss log but at a lower floor); after
        `escape_count` of them within `escape_secs`, open the gate anyway — the user is clearly trying."""
        if score >= self._escape_floor:
            self._escape_peak = max(self._escape_peak, score)
            return
        if self._escape_peak <= 0.0:
            return
        # burst ended → record it, prune old, maybe escape
        peak = self._escape_peak
        self._escape_peak = 0.0
        self._escape_hits.append(now)
        self._escape_hits = [t for t in self._escape_hits if now - t <= self._escape_secs]
        _tlog(f"GATE  | escape-hit {len(self._escape_hits)}/{self._escape_count} (peak={peak:.2f})")
        if len(self._escape_hits) >= self._escape_count:
            self._open_window(
                f"GATE  | escape — {len(self._escape_hits)} near-misses in "
                f"{self._escape_secs:.0f}s, opening despite sub-threshold"
            )

    def _on_wake(self, score: float) -> None:
        self._open_window(f"WAKE  | wake word ({score:.2f}) — opening command window")

    def _open_window(self, log_msg: str) -> None:
        """Open the command window (idempotent): log the reason, pre-duck, arm the idle timer."""
        if not self._open:
            self._open = True
            _tlog(log_msg)
            self._fire_duck(True)  # pre-duck so the command lands on already-ducked media
        self._escape_peak = 0.0
        self._escape_hits.clear()  # fresh start once we're open
        self._arm_window()

    # --- command window ----------------------------------------------------
    def _arm_window(self) -> None:
        self._cancel_window()
        if self._hold:
            return  # held open for a pending confirm — no idle close until released

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
                _tlog(f"WAKE  | duck on={on} FAILED: {type(e).__name__}: {e}")

        asyncio.create_task(_go())
