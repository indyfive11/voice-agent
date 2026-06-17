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
  (another onset) also cancels it, so a multi-clause utterance never flaps. **Exception (2026-06-14):
  while media is actually *playing*, the confirm timer does NOT snap the bed back — over a movie the mic
  is gated behind the wake word, so an onset is almost always a real command whose transcript merely lags
  `confirm_grace` (Whisper latency). Snapping back then flapped the bed down→up→re-duck and raced the
  confirmed `on` against a stale `off` (full-volume mid-command). It falls back to the longer idle grace,
  which a slow real command confirms within (the confirmed path then owns restore) and a genuine
  non-speech onset still hits eventually.** The quick snap-back stands when nothing is playing.
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
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    SystemFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class DuckReleaseFrame(SystemFrame):
    """Pushed UPSTREAM by the brain (BrainLLMService) when it classifies a turn as NOT addressed to Aria
    (an aside). The VAD-onset pre-duck has already engaged for the utterance — but an aside yields no reply,
    so there's nothing to make room for. This releases that duck immediately instead of waiting out the
    idle-restore grace (~8s). A SystemFrame so it reaches this controller through the user aggregator the
    same way WakeSleepFrame/BotThinkingFrame reach the wake gate."""


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
        sustained_secs: float = 4.0,
        should_duck: Callable[[], bool] | None = None,
        should_duck_onset: Callable[[], bool] | None = None,
        media_status: Callable[[], "dict|None"] | None = None,
        time_source: Callable[[], float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._session_id = session_id
        self._min_words = max(1, min_words)
        self._restore_grace = restore_grace
        # An aside (DuckReleaseFrame) normally releases the duck immediately — but if the user is in a
        # SUSTAINED utterance (still speaking, or the duck has been on this long), snapping the music back to
        # full mid-sentence is the "it kept playing while I talked" bug. Past this duration we HOLD the duck on
        # an aside and let the idle-grace restore own it (fires ~restore_grace after speech truly stops); the
        # immediate aside-release then applies only to SHORT asides (the ambient-blip case it was built for).
        self._sustained_secs = sustained_secs
        self._time = time_source or time.monotonic
        self._duck_started_at = 0.0
        # How long after speech *stops* to wait for a confirming transcription before treating the
        # onset as a false trigger and restoring. Must comfortably exceed Whisper's per-segment
        # latency so real speech confirms before it fires (else a slow transcribe would flap).
        self._confirm_grace = confirm_grace
        # Gate (e.g. "not asleep"); default always-allow.
        self._should_duck = should_duck or (lambda: True)
        # Stricter gate for the RAW VAD-ONSET duck (the immediate, pre-transcription dip). Over playing media
        # the held-open keepalive/idle window would otherwise let the media's own VAD onsets dip the bed with
        # no wake (2026-06-16); this gate suppresses the onset duck there while the confirmed-speech path
        # (gated by should_duck) still ducks a real follow-up command. Defaults to should_duck (no change).
        self._should_duck_onset = should_duck_onset or self._should_duck
        self._ducked = False
        self._bot_spoke = False  # did the bot speak during the current duck?
        self._confirmed = False  # got ≥min_words this duck episode (so it's not a false onset)
        # Is the user currently inside an active VAD speech segment (onset seen, no stop yet)? The
        # idle-restore must never fire while this is True — yanking the bed out mid-sentence is the
        # "music interjects during a long monologue" bug. Idle is measured from speech *stop*, not
        # from the last transcription chunk (which can lag / fall under min_words mid-utterance).
        self._speech_in_flight = False
        self._restore_task: asyncio.Task | None = None
        self._confirm_task: asyncio.Task | None = None
        # Media-state — the SHARED provider (config.build_media_state_provider), the SAME callable the
        # wake gate uses, so the gate and the duck can never disagree (they used to keep separate caches
        # with opposite fail-modes). Returns {"playing": bool, "kind": …}; the duck only needs `playing`.
        # It owns the 1s cache + coalescing. Default → always-duck if unwired (the brain's duck is a
        # harmless no-op when nothing plays).
        self._media_status = media_status or (lambda: {"playing": True})

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._handle(frame)
        await self.push_frame(frame, direction)

    # --- frame handling ----------------------------------------------------
    def _handle(self, frame: Frame) -> None:
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # Speech onset — duck immediately. Cancel any pending false-onset restore (still talking).
            self._speech_in_flight = True
            self._cancel_confirm()
            self._duck_on("speech onset", allow=self._should_duck_onset)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Speech segment ended. If this onset hasn't been confirmed by real words yet, start the
            # short countdown that restores it as a false trigger (cancelled if a transcription
            # confirms, or if the user resumes speaking).
            self._speech_in_flight = False
            if self._ducked and not self._confirmed:
                self._arm_confirm()
            elif self._ducked and self._confirmed:
                # Confirmed turn just paused — (re)start the idle grace from the actual stop point so
                # the bed only restores after a genuine silence, never mid-monologue.
                self._arm_restore()
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
        elif isinstance(frame, DuckReleaseFrame):
            # Brain says this turn was an aside (not addressed) → no reply is coming. Release the
            # pre-duck now rather than holding it for the idle grace. Keyed on the frame TYPE, which the
            # brain only ever sends on suppression (no bool to be dropped by a serializer). Guarded by
            # `not self._bot_spoke` so a stale aside-verdict can never cut the bed out from under live TTS
            # (e.g. an aside-then-command race where Aria is already replying).
            if self._ducked and not self._bot_spoke:
                sustained = (
                    self._speech_in_flight
                    or (self._time() - self._duck_started_at) >= self._sustained_secs
                )
                if sustained:
                    # Sustained utterance (dictation / long aside) — don't snap the bed back mid-sentence.
                    # Hold; the idle-grace restore (re-armed per transcription, gated on _speech_in_flight)
                    # owns it and fires only after speech truly stops. This is the automatic duck-policy.
                    _tlog("DUCK  | aside during sustained speech — holding (idle-grace owns restore)")
                else:
                    self._restore("aside")

    def _duck_on(self, reason: str, allow: Callable[[], bool] | None = None) -> None:
        # `allow` lets the raw-onset path use the stricter onset gate while the confirmed-speech path uses
        # the broad should_duck (default). Both still require something to be playing (checked in _fire).
        if not (allow or self._should_duck)() or self._ducked:
            return
        self._ducked = True
        self._bot_spoke = False
        self._confirmed = False
        self._duck_started_at = self._time()
        self._fire(True, reason)

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
    def _fire(self, on: bool, reason: str = "") -> None:
        """Fire-and-forget the brain duck call — never block the audio pipeline."""

        async def _go():
            try:
                if on and not await self._media_playing():
                    # Nothing actually playing — don't bother the brain; undo the optimistic flag.
                    # DEBUG-only: in open-mic conversation (no media) every VAD onset hits this path, so
                    # logging it to the transcript drowns USER/BOT in "on SKIPPED" noise (~40% of the
                    # transcript in the 2026-06-17 open-mic drive). The skip is a no-op; keep it in
                    # session.log for media debugging, off the transcript.
                    self._ducked = False
                    logger.debug("DUCK  | on SKIPPED (media_state: nothing playing)")
                    return
                await self._client.duck(self._session_id, on)
                _tlog(f"DUCK  | on ({reason}) sent" if on else "DUCK  | /media/duck on=False sent")
            except Exception as e:  # noqa: BLE001 - media control must never break audio
                _tlog(f"DUCK  | /media/duck on={on} FAILED: {type(e).__name__}: {e}")

        asyncio.create_task(_go())

    async def _media_playing(self) -> bool:
        """Delegate to the SHARED media-state provider (same callable the wake gate uses → no divergence).
        Accepts a sync OR async callable returning {"playing": …}; the provider owns the cache/fail-mode."""
        st = self._media_status()
        if asyncio.iscoroutine(st) or asyncio.isfuture(st):
            st = await st
        return bool(st and st.get("playing"))

    # --- restore timer -----------------------------------------------------
    def _arm_restore(self) -> None:
        self._cancel_restore()

        async def _later():
            try:
                await asyncio.sleep(self._restore_grace)
                if self._ducked and not self._bot_spoke:
                    if self._speech_in_flight:
                        # User is still mid-utterance — don't restore the bed under them. Re-arm and
                        # let the next stop (VADUserStoppedSpeaking) measure the idle window cleanly.
                        self._arm_restore()
                        return
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
                # Speech stopped and no qualifying transcription arrived within confirm_grace.
                if self._ducked and not self._confirmed and not self._bot_spoke:
                    # While media is actually playing, do NOT snap the bed back now. Over a movie the mic
                    # is gated behind the wake word, so an onset here is almost always a real command whose
                    # transcript is merely slower than confirm_grace (Whisper latency over the AEC mic). The
                    # immediate restore causes an audible flap — duck → up → re-duck — and a send-order race
                    # where its `off` can land *after* the imminent confirmed-speech `on`, leaving the movie
                    # at full volume mid-command (2026-06-14 log 19:36:34). Fall back to the longer idle
                    # grace instead: a slow real command confirms within it (the confirmed path then owns
                    # restore), while a genuine non-speech onset still restores once the idle window elapses.
                    if await self._media_playing():
                        self._arm_restore()
                        return
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
