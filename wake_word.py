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

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    SystemFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
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
        consec_frames: int = 1,
        guard_inflight: bool = True,
        min_dwell_secs: float = 2.0,
        window_mute: bool = True,
        time_source: Callable[[], float] | None = None,
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
        self._escape_run = 0  # consecutive frames in the current escape burst (must sustain to count)
        self._escape_hits: list[float] = []
        # Sustained-wake: require N consecutive frames (~80ms each) >= threshold before firing. A spoken
        # "hey aria" holds for 5-9 frames; music false-positives are isolated 1-frame spikes (2026-06-04).
        self._consec_required = max(1, consec_frames)
        self._consec = 0
        # F3 gate hardening (defense-in-depth; GA's Phase-1 already kills the remote-video kind-flap):
        #   (a) in-flight guard — never START gating (pass-through → gating) while a user utterance is in
        #       flight, so a media-state flip can't swallow a command mid-stream and starve the VAD. Tracked
        #       via the VAD start/stop frames (the same ones brains/media_duck consumes), which propagate
        #       UPSTREAM through this gate. Degrades to old behaviour if VAD frames don't flow on a transport.
        #   (b) min-dwell hysteresis — once committed to a gated state, require the opposite raw decision to
        #       persist for min_dwell_secs before flipping, so a single transient media-state blip can't toggle
        #       the gate (the flaps span the media-state 1s cache, so this is time-based, not frame-count).
        self._guard_inflight = guard_inflight
        self._speech_in_flight = False
        self._min_dwell_secs = max(0.0, min_dwell_secs)
        self._now = time_source or time.monotonic
        self._gated_committed: bool | None = None  # effective (debounced) gating state; None until 1st frame
        self._gated_pending: bool | None = None     # a raw decision awaiting the dwell to elapse
        self._gated_pending_since = 0.0

        # When the wake/command window opens, send /media/duck {mute:true} so media drops to a full 0%
        # (not just a partial duck) for the window → no music vocal bleeds into the command's STT. The
        # plain speech-duck (brains/media_duck) stays mute:false. Env: WAKE_WINDOW_MUTE (default on).
        self._window_mute = window_mute

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
        # Whether the burst's peak frame was suppressed by the post-wake refractory window (a real wake had
        # just fired) rather than by failing to sustain. Drives an accurate near-miss reason label.
        self._nearmiss_refractory = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            gated = self._effective_gated(await self._gated_now())
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

        # Track whether a user utterance is in flight (VAD onset→stop). These frames originate in the
        # user-aggregator DOWNSTREAM and are pushed UPSTREAM through this gate; the in-flight guard in
        # _effective_gated uses them to avoid clamping the mic mid-command. Always pass them through.
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_in_flight = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_in_flight = False

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
    def _effective_gated(self, raw: bool) -> bool:
        """Map the raw per-frame `_gated_now()` decision to the EFFECTIVE gating state, applying the F3
        in-flight guard + min-dwell hysteresis. Pure/synchronous + clock-injected so it's unit-testable."""
        # First frame seeds the committed state — no smoothing to apply yet.
        if self._gated_committed is None:
            self._gated_committed = raw
            self._gated_pending = None
            return raw
        committed = self._gated_committed

        # (a) In-flight guard: never flip pass-through → gating while an utterance is mid-flight (it would
        # swallow the command and starve the VAD). Only guards the →gating direction; →pass-through is fine.
        # An OPEN window already passes audio regardless, so this only matters with the window closed.
        if self._guard_inflight and self._speech_in_flight and raw and not committed:
            self._gated_pending = None  # also don't let the dwell commit a gating flip mid-utterance
            return committed

        # (b) Min-dwell hysteresis: a change must persist for min_dwell_secs before it commits.
        if raw == committed:
            self._gated_pending = None
            return committed
        if self._min_dwell_secs <= 0:
            self._gated_committed = raw
            self._gated_pending = None
            return raw
        now = self._now()
        if self._gated_pending != raw:
            self._gated_pending = raw
            self._gated_pending_since = now
            return committed
        if now - self._gated_pending_since >= self._min_dwell_secs:
            self._gated_committed = raw
            self._gated_pending = None
            return raw
        return committed

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
            # Count consecutive over-threshold frames; a single music blip won't reach _consec_required.
            self._consec = self._consec + 1 if score >= self._threshold else 0
            if self._consec >= self._consec_required and (now - self._last_wake) > self._refractory:
                self._last_wake = now
                self._consec = 0
                self._reset_nearmiss()
                self._on_wake(score)
            else:
                if self._debug:
                    # A >=threshold frame can land here for two very different reasons: it failed to sustain
                    # _consec_required frames, OR a real wake just fired and this frame is inside the refractory
                    # window (a duplicate, not a recall failure). Capture which, for an accurate near-miss label.
                    in_refractory = (now - self._last_wake) <= self._refractory
                    self._track_nearmiss(key, score, in_refractory)
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
    def _track_nearmiss(self, key: str, score: float, in_refractory: bool = False) -> None:
        """Track the peak of a sub-threshold burst; emit one line when the burst ends.

        Frames arrive ~12.5/s, so a single spoken "Aria" spans several — logging each would spam. Instead
        we hold the burst's peak and emit it once the score falls back under the floor (utterance over).
        """
        if score >= self._debug_floor:
            if score >= self._nearmiss_peak:
                self._nearmiss_peak = score
                self._nearmiss_key = key
                self._nearmiss_refractory = in_refractory
        elif self._nearmiss_peak > 0.0:
            if self._nearmiss_peak < self._threshold:
                reason = f"< threshold {self._threshold:.2f}"
            elif self._nearmiss_refractory:
                # Hit threshold but a real wake just fired — this is a duplicate the refractory swallowed,
                # NOT a recall miss. (With _consec_required=1 every post-wake threshold frame lands here.)
                reason = "duplicate within refractory of last wake"
            else:
                reason = f"spike — didn't sustain {self._consec_required} frames"
            _tlog(f"WAKE  | near-miss {self._nearmiss_key}={self._nearmiss_peak:.2f} ({reason})")
            self._reset_nearmiss()

    def _reset_nearmiss(self) -> None:
        self._nearmiss_peak = 0.0
        self._nearmiss_key = ""
        self._nearmiss_refractory = False

    # --- lockout escape hatch ----------------------------------------------
    def _track_escape(self, score: float, now: float) -> None:
        """Count *sustained* sub-threshold bursts; after `escape_count` of them within `escape_secs`, open
        the gate anyway — the user is clearly trying. A burst must last >= `_consec_required` frames to
        count: a clamped "hey aria" holds several frames at 0.3-0.5, but a music false-positive is a 1-frame
        spike (the same blip the sustained-wake gate rejects), so music never trips the escape (2026-06-04)."""
        if score >= self._escape_floor:
            self._escape_peak = max(self._escape_peak, score)
            self._escape_run += 1
            return
        if self._escape_peak <= 0.0:
            return
        # burst ended → count it only if it sustained (else it's a brief spike, likely music)
        peak, run = self._escape_peak, self._escape_run
        self._escape_peak = 0.0
        self._escape_run = 0
        if run < self._consec_required:
            return  # too brief to be a genuine attempt — ignore (this is what stops music tripping escape)
        self._escape_hits.append(now)
        self._escape_hits = [t for t in self._escape_hits if now - t <= self._escape_secs]
        _tlog(f"GATE  | escape-hit {len(self._escape_hits)}/{self._escape_count} (peak={peak:.2f}, {run} frames)")
        if len(self._escape_hits) >= self._escape_count:
            self._open_window(
                f"GATE  | escape — {len(self._escape_hits)} sustained near-misses in "
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
        self._escape_run = 0
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

        mute = on and self._window_mute  # full-mute the window on open; restore is a plain on=False

        async def _go():
            try:
                await self._client.duck(self._session_id, on, mute=mute)
            except Exception as e:  # noqa: BLE001 - media control must never break audio
                _tlog(f"WAKE  | duck on={on} mute={mute} FAILED: {type(e).__name__}: {e}")

        asyncio.create_task(_go())
