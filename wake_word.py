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
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    SystemFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import aria_state

_CHUNK_SAMPLES = 1280  # 80ms @16k — openWakeWord's preferred frame
_CHUNK_BYTES = _CHUNK_SAMPLES * 2  # int16 mono


@dataclass
class WakeHoldFrame(SystemFrame):
    """Pushed UPSTREAM by the brain (BrainLLMService) to keep the wake gate's command window OPEN.

    Two modes:
    - Confirm hold (`ttl_secs=None`): hold indefinitely while a confirm is pending; `hold=True` opens/freezes,
      `hold=False` releases. The user's yes/no answer then needs no fresh wake.
    - Media keepalive (`ttl_secs` set, `hold=True`): hold the window open for a TTL after a media command, so a
      follow-up command needs no re-wake. Release is automatic on TTL expiry (the brain never sends a false);
      the gate clamps the TTL to its hard ceiling and self-releases if refreshes stop.
    SystemFrame so it propagates promptly regardless of queueing."""

    hold: bool = True
    ttl_secs: float | None = None


@dataclass
class WakeSleepFrame(SystemFrame):
    """Pushed UPSTREAM by the brain when its sleep state changes. While asleep the gate is FORCED active
    (mic muted, wake word required) regardless of media, so ambient TV/movie audio never reaches STT and
    only an acoustic wake ("hey aria") can wake her — closing the text-match false-wake hole where a movie
    line like "we can wake up very early" woke her. No-op when no gate is present. SystemFrame so it
    propagates promptly regardless of queueing."""

    asleep: bool = False


@dataclass
class StopWordFrame(SystemFrame):
    """Pushed DOWNSTREAM by the wake gate when the interrupt/stop word is detected while Aria is speaking.

    Why a custom frame instead of `broadcast_interruption()`: in half-duplex the user-aggregator's mute
    DROPS `InterruptionFrame` while the bot speaks (`_maybe_mute_frame`), so an interruption issued from the
    gate (upstream of the aggregator) never reaches the LLM/output — it's silently suppressed (2026-06-16
    live: "she didn't stop"). `StopWordFrame` is NOT in the mute's drop list, so it passes the muted
    aggregator; `BrainLLMService` (downstream of the mute) catches it and issues the actual interruption
    from below the mute → TTS flushes + the brain turn is cancelled. SystemFrame so it propagates promptly."""


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
        asleep_consec_frames: int = 3,
        gated_retry_count: int = 2,
        gated_retry_secs: float = 6.0,
        guard_inflight: bool = True,
        min_dwell_secs: float = 2.0,
        window_mute: bool = True,
        preduck_grace: float = 3.0,
        preduck_grace_keepalive: float = 1.0,
        hold_max_secs: float = 120.0,
        interrupt_enabled: bool = False,
        interrupt_threshold: float | None = None,
        interrupt_consec_frames: int = 2,
        interrupt_refractory_secs: float = 1.0,
        interrupt_arm_delay_secs: float = 0.0,
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
        # While ASLEEP (force-gated) require a stricter sustain than awake. Asleep is the high-cost-of-
        # false-positive state: a missed real wake just needs a repeat, but a false wake on un-AEC'd room
        # audio (a radio/TV the echo-canceller can't remove, spiking the model to ceiling) is the whole
        # "she won't stay asleep" bug (2026-06-15). A longer run rejects the brief 2-frame coincidences that
        # would otherwise open the mic to ambient dialogue. Never weaker than the awake requirement.
        self._asleep_consec_required = max(self._consec_required, asleep_consec_frames)
        # Gated-retry escape: a clean "Aria" over a movie/music sometimes flickers to 1-2 frames and never
        # clears the sustain (2026-06-19 live: asleep — 7 wakes peaked 2/3, 10 peaked 1/3; awake over music —
        # spikes peaked 1/2; all opened no window). Rather than weaken the single-shot bar further (which lets
        # a one-off media coincidence wake her), open when the user clearly tries AGAIN: `gated_retry_count`
        # separate >=threshold near-wakes that each fell short of the sustain, within `gated_retry_secs`. Media
        # rarely fires two near-perfect "Aria" bursts seconds apart; a person re-waking does. Applies WHENEVER
        # the wake is being gated — asleep OR awake-with-media-playing (both lock the mic behind the wake word,
        # so both flicker the same way); keyed on `_gated_committed`, not just `_force_gated`. 0 disables.
        self._gated_retry_count = max(0, gated_retry_count)
        self._gated_retry_secs = max(0.0, gated_retry_secs)
        self._gated_retry_hits: list[float] = []
        self._retry_peak = 0.0  # peak of the in-progress gated-retry burst (grouped by debug_floor)
        self._consec = 0
        self._consec_max = 0  # best consecutive ≥threshold run in the current near-miss burst (diagnostic)
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

        # Pre-duck release grace: the wake pre-duck holds media down so the command lands on ducked audio,
        # but if no speech actually follows the wake (an unused or phantom wake), release it after this many
        # seconds instead of holding for the whole command window. Held while the user OR Aria is speaking;
        # once a command is transcribed, MediaDuckController owns the duck lifecycle (restore on Aria-stops)
        # and this is cancelled. 0 disables (the pre-duck then holds until the window closes — old behaviour).
        self._preduck_grace = max(0.0, preduck_grace)
        # Shorter pre-duck release grace for a KEEPALIVE-origin window (a media command just opened it). The
        # keepalive pre-duck keeps media down so Aria's post-command announcement/reply is audible over the
        # song she just started — but unlike a fresh "Hey Aria" (where the long grace gives the user time to
        # start a command), a keepalive isn't waiting on user speech, so the bed should return promptly once
        # she stops talking instead of lingering the full wake grace (the "dips the song I just played for
        # ~6s" annoyance, 2026-06-16). Held through her reply (BotStarted cancels, BotStopped re-arms).
        self._preduck_grace_keepalive = max(0.0, preduck_grace_keepalive)
        self._preduck_task: asyncio.Task | None = None
        self._bot_speaking = False  # Aria is mid-reply → keep the pre-duck and freeze the idle close

        self._buf = bytearray()
        self._open = False
        self._last_gated: bool | None = None  # last gate decision (for debug transition logging)
        self._hb_peak = 0.0  # max wake score since the last heartbeat (lockout-vs-freeze diagnosis)
        self._hold = False  # brain holds the window open while a confirm is pending (WakeHoldFrame)
        # While the brain is asleep, force the gate active (require an acoustic wake) regardless of media,
        # so ambient TV never reaches STT and only "hey aria" can wake her (WakeSleepFrame). See _gated_now.
        self._force_gated = False
        self._last_wake = 0.0
        self._ducked = False
        self._window_task: asyncio.Task | None = None
        # Brain media-keepalive (WakeHoldFrame with ttl_secs): hold the command window open for a TTL after a
        # media command so a follow-up needs no re-wake. _hold_max_secs is the hard ceiling — a TTL is clamped
        # to it, and it auto-releases if the brain's refreshes stop (crash/dropped turn). See _set_keepalive.
        self._hold_max_secs = hold_max_secs
        self._keepalive_task: asyncio.Task | None = None
        # While True, idle-close is suspended for the WHOLE keepalive TTL (like _hold) so a BotStoppedSpeaking
        # / transcription / VAD re-arm of the 15s idle timer can't truncate the hold (2026-06-15 live bug).
        self._keepalive_active = False
        # True while the OPEN window originated from a media keepalive (a command the user already issued
        # opened it) rather than a fresh "Hey Aria" wake — and stays True through the post-keepalive idle
        # tail until the window actually closes or a fresh wake supersedes it. Two uses: (1) the raw-VAD-onset
        # duck is suppressed in this state (the open window is held OVER the playing media, so an onset is the
        # media tripping VAD, not a command — a real follow-up still ducks via confirmed speech); (2) the
        # pre-duck releases on the shorter keepalive grace. Both fix the 2026-06-16 duck-over-music report.
        self._open_via_keepalive = False
        # Peak score within the current sub-threshold burst (one line emitted per utterance, not per frame).
        self._nearmiss_peak = 0.0
        self._nearmiss_key = ""
        # Whether the burst's peak frame was suppressed by the post-wake refractory window (a real wake had
        # just fired) rather than by failing to sustain. Drives an accurate near-miss reason label.
        self._nearmiss_refractory = False

        # Interrupt/stop word (V2 barge-in): while Aria is speaking, run the SAME wake model on the mic
        # (AEC cancels her own referenced TTS → we see the user's residual) and on a sustained hit cut her
        # TTS (broadcast_interruption) and open the floor. Opt-in (WAKE_INTERRUPT); zero effect when off.
        # Separate buffer/counter/refractory from the wake path so the two never cross (they're mutually
        # exclusive per-frame: you interrupt while she speaks, you don't wake).
        self._interrupt_enabled = interrupt_enabled
        self._interrupt_threshold = threshold if interrupt_threshold is None else interrupt_threshold
        self._interrupt_consec_required = max(1, interrupt_consec_frames)
        self._interrupt_refractory = max(0.0, interrupt_refractory_secs)
        self._interrupt_consec = 0
        self._last_interrupt = 0.0
        self._interrupt_buf = bytearray()
        # Arm-delay: suppress the interrupt FIRE for the first N secs after BotStartedSpeaking. AEC needs a
        # beat to converge on a new TTS utterance; until it does, Aria's own voice leaks through and self-
        # trips the detector at her speech onset (live: 3 fires 75–159ms after bot-start). A real user
        # interrupt happens later (live: 7.5s in), so a short delay kills the onset self-trips losslessly.
        self._interrupt_arm_delay = max(0.0, interrupt_arm_delay_secs)
        self._bot_speaking_since = 0.0

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
            if self._interrupt_active():
                # While Aria speaks, detect the interrupt word instead of waking (mutually exclusive). Runs
                # regardless of `gated` so it works in a quiet room (no media). Routing below is unchanged.
                await self._feed_interrupt(frame.audio)
            elif gated:
                await self._feed(frame.audio)
            if gated and not self._open:
                return  # swallow: muted until the wake word
            await self.push_frame(frame, direction)
            return

        # Track whether a user utterance is in flight (VAD onset→stop). These frames originate in the
        # user-aggregator DOWNSTREAM and are pushed UPSTREAM through this gate; the in-flight guard in
        # _effective_gated uses them to avoid clamping the mic mid-command. Always pass them through.
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_in_flight = True
            if self._open:
                self._arm_preduck()  # a speech attempt → give it the grace to produce words
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_in_flight = False
            if self._open:
                self._arm_preduck()  # speech stopped; release the pre-duck unless words confirm it

        # Aria's reply brackets: keep the pre-duck down and freeze the idle close while she speaks, so the
        # media never restores out from under a reply, then resume the normal idle close when she finishes.
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._bot_speaking_since = time.monotonic()  # arm-delay reference; gates the interrupt fire
            self._interrupt_consec = 0  # fresh utterance → don't carry a stale consec into the arm window
            self._cancel_preduck()
            self._cancel_window()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._open:
                self._arm_window()
                # Aria's reply/announcement cancelled the pre-duck timer on BotStarted; re-arm it so a
                # pre-duck left on after she finishes (notably the keepalive pre-duck, whose window is held
                # open and so never idle-closes) still releases once the grace elapses with no speech.
                if self._ducked and not self._hold:
                    self._arm_preduck()
            await self.push_frame(frame, direction)
            return

        # Hold the window open (from the brain). ttl_secs set → media keepalive (timed); else confirm hold.
        if isinstance(frame, WakeHoldFrame):
            if frame.ttl_secs is not None and frame.hold:
                self._set_keepalive(frame.ttl_secs)
            else:
                self._set_hold(frame.hold)
            await self.push_frame(frame, direction)
            return

        # Sleep state (from the brain): while asleep, force acoustic gating regardless of media so STT
        # never runs on ambient TV and only "hey aria" can wake her (see _gated_now / _force_gated).
        if isinstance(frame, WakeSleepFrame):
            self._force_gated = frame.asleep
            rest = "off" if frame.asleep else "idle"
            # Record what "rest" means now so a goodbye TTS spoken on the way to sleep settles the eye
            # on `off`, not `idle` (tts_gain's BotStopped→rest write reads this); then set it live.
            aria_state.set_resting_state(rest)
            aria_state.set_state(rest)  # eye: asleep = closed, awake = idle
            if self._debug:
                _tlog(
                    "GATE  | asleep — forcing wake-word gate active (acoustic wake only)" if frame.asleep
                    else "GATE  | awake — restoring media-aware gating"
                )
            await self.push_frame(frame, direction)
            return

        # Keep the command window alive across a multi-turn exchange.
        if isinstance(frame, TranscriptionFrame) and self._open:
            if len((getattr(frame, "text", "") or "").split()) >= 1:
                # Real words → this wake led to a command; MediaDuckController now owns the duck lifecycle
                # (held while speaking, restored when Aria finishes), so stop the pre-duck release timer.
                self._cancel_preduck()
                self._arm_window()

        await self.push_frame(frame, direction)

    def _set_hold(self, hold: bool) -> None:
        """Brain confirm pending → keep the window open (open it if needed); release → resume idle close."""
        self._hold = hold
        if hold:
            self._cancel_preduck()  # a pending confirm must keep media ducked — never release on the grace
            if not self._open:
                self._open_window("GATE  | hold — confirm pending, opening window")
            else:
                self._cancel_window()  # freeze the idle timer while held
                _tlog("GATE  | hold — confirm pending, window held open")
        else:
            _tlog("GATE  | hold released — re-arming idle window")
            self._arm_window()

    def duck_allowed(self) -> bool:
        """Whether the downstream media-duck may fire on a speech onset. Gate the duck behind the wake word
        for ALL gated media (audio AND video, like the STT gate): duck only once the command window is OPEN
        (the user woke Aria → a command is coming, make room) or when media isn't being gated at all. While
        gated+closed (media playing, no wake yet) a speech onset is room conversation, not for Aria — so the
        duck is suppressed and media no longer dips every time someone talks (2026-06-15, the maintainer: gate the duck
        the same as STT, for all playback). The on-wake pre-duck is fired by the gate itself (`_open_window`),
        so a real "Hey Aria" still ducks immediately; this only suppresses the un-woken ambient-speech duck."""
        return self._open or not bool(self._gated_committed)

    def duck_onset_allowed(self) -> bool:
        """Stricter gate for the RAW VAD-onset duck (the immediate, pre-transcription dip). Over gated media,
        a raw onset only ducks while the window is open from a FRESH WAKE — never in the keepalive/idle tail
        a media command left open. There, the window is held OPEN OVER the very media the user just started,
        and that media isn't AEC-removed (only Aria's TTS is), so it trips VAD onset repeatedly: ducking on it
        is the "music dips with no wake word" bug (2026-06-16). A genuine follow-up command in the keepalive
        window still ducks — via the confirmed-speech path (`duck_allowed`), a beat later. When nothing is
        being gated (no media), an onset duck is a harmless no-op (skipped if nothing plays), so allow it."""
        if not bool(self._gated_committed):
            return True
        return self._open and not self._open_via_keepalive

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
        if self._force_gated:
            return True  # brain is asleep → require an acoustic wake regardless of media state
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
    @property
    def _consec_needed(self) -> int:
        """Consecutive ≥threshold frames required to fire — stricter while asleep (force-gated)."""
        return self._asleep_consec_required if self._force_gated else self._consec_required

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
            self._consec_max = max(self._consec_max, self._consec)  # for the near-miss "peaked N/M" label
            if self._consec >= self._consec_needed and (now - self._last_wake) > self._refractory:
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
                if not self._open and self._gated_retry_count > 0:
                    self._track_gated_retry(score, now)  # gated (asleep/media) re-wake escape (debug-independent)

    def _interrupt_active(self) -> bool:
        """Whether the interrupt detector should run on this frame: opt-in AND Aria is currently speaking.
        Pure/synchronous so the routing is unit-testable without a running pipeline. The wake and interrupt
        feeds are mutually exclusive — you interrupt while she speaks, you don't wake."""
        return self._interrupt_enabled and self._bot_speaking

    async def _feed_interrupt(self, audio: bytes) -> None:
        """Run the wake model while Aria is speaking; a sustained hit interrupts her (V2 barge-in).

        Mirrors `_feed`'s chunk→predict→consecutive-frame logic but with the interrupt threshold/consec/
        refractory and its own buffer, so the wake and interrupt paths never cross. A 1-frame TTS-bleed/
        music spike won't reach `_interrupt_consec_required`, the same way the wake path rejects blips."""
        self._interrupt_buf.extend(audio)
        loop = asyncio.get_running_loop()
        while len(self._interrupt_buf) >= _CHUNK_BYTES:
            chunk = bytes(self._interrupt_buf[:_CHUNK_BYTES])
            del self._interrupt_buf[:_CHUNK_BYTES]
            pcm = np.frombuffer(chunk, dtype=np.int16)
            scores = await loop.run_in_executor(None, self._oww.predict, pcm)
            _key, score = self._best(scores)
            if score > self._hb_peak:
                self._hb_peak = score
            now = time.monotonic()
            self._interrupt_consec = self._interrupt_consec + 1 if score >= self._interrupt_threshold else 0
            if (self._interrupt_consec >= self._interrupt_consec_required
                    and (now - self._bot_speaking_since) >= self._interrupt_arm_delay
                    and (now - self._last_interrupt) > self._interrupt_refractory):
                self._last_interrupt = now
                self._interrupt_consec = 0
                await self._on_interrupt(score)

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
                # Report how close the burst came: "peaked 1/2" = never strung two ≥threshold frames
                # together (real wake flickering → an M-of-N window is the next lever); "peaked 2/3" = a
                # lower consec_required would have fired it. Drives data-backed retuning, not a blind guess.
                reason = f"spike — peaked {self._consec_max}/{self._consec_needed} frames"
            _tlog(f"WAKE  | near-miss {self._nearmiss_key}={self._nearmiss_peak:.2f} ({reason})")
            self._reset_nearmiss()

    def _reset_nearmiss(self) -> None:
        self._consec_max = 0
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
        if run < self._consec_needed:
            return  # too brief to be a genuine attempt — ignore (this is what stops music tripping escape)
        self._escape_hits.append(now)
        self._escape_hits = [t for t in self._escape_hits if now - t <= self._escape_secs]
        _tlog(f"GATE  | escape-hit {len(self._escape_hits)}/{self._escape_count} (peak={peak:.2f}, {run} frames)")
        if len(self._escape_hits) >= self._escape_count:
            self._open_window(
                f"GATE  | escape — {len(self._escape_hits)} sustained near-misses in "
                f"{self._escape_secs:.0f}s, opening despite sub-threshold"
            )

    def _track_gated_retry(self, score: float, now: float) -> None:
        """While the wake is being GATED (asleep OR awake-with-media-playing), count separate >=threshold
        near-wakes that fell short of the sustain; after `gated_retry_count` of them within `gated_retry_secs`,
        open the gate — the user is clearly re-waking.

        A burst is grouped by `debug_floor` (the same low floor the near-miss diagnostic uses) so a single
        flickering "Aria" — which dips below threshold mid-word (the 2026-06-19 "peaked 1/3" / "1/2" pattern)
        but stays above the floor — counts as ONE attempt, not several. Only a burst whose PEAK cleared
        threshold counts: sub-threshold room/media noise never accrues, and a one-off coincidence that DID clear
        threshold needs a SECOND one seconds later to open — media rarely fires two near-perfect bursts apart, a
        person re-waking does. A burst that reaches the full sustain fires the normal wake path first (we're not
        `_open` here), so this only ever sees genuine sub-bar attempts. Runs only while gated (the mic is locked
        behind the wake word) — `_gated_committed`, which is True asleep AND awake-over-playing-media."""
        if not self._gated_committed:
            self._retry_peak = 0.0  # not gated (open mic) → no escape needed; don't carry a stale burst
            return
        if score >= self._debug_floor:
            self._retry_peak = max(self._retry_peak, score)
            return
        if self._retry_peak <= 0.0:
            return
        peak = self._retry_peak
        self._retry_peak = 0.0
        if peak < self._threshold:
            return  # never a genuine attempt (peaked under the bar) — ignore, as music/noise does
        self._gated_retry_hits.append(now)
        self._gated_retry_hits = [t for t in self._gated_retry_hits if now - t <= self._gated_retry_secs]
        _tlog(f"WAKE  | gated-retry-hit {len(self._gated_retry_hits)}/{self._gated_retry_count} "
              f"(peak={peak:.2f})")
        if len(self._gated_retry_hits) >= self._gated_retry_count:
            self._gated_retry_hits.clear()
            self._open_window(
                f"WAKE  | gated-retry — {self._gated_retry_count} near-wakes in "
                f"{self._gated_retry_secs:.0f}s, opening despite short sustain"
            )

    def _on_wake(self, score: float) -> None:
        self._open_window(f"WAKE  | wake word ({score:.2f}) — opening command window")

    async def _on_interrupt(self, score: float) -> None:
        """Confirmed interrupt word while Aria speaks → cut her TTS and hand the floor back (stop-and-listen).

        We push a `StopWordFrame` DOWNSTREAM rather than calling `broadcast_interruption()` here: in
        half-duplex the user-aggregator's mute DROPS `InterruptionFrame` during the bot turn, so an
        interruption issued from the gate (upstream of the aggregator) is silently suppressed and the TTS
        never stops (2026-06-16 live, twice). `StopWordFrame` survives the mute; `BrainLLMService` (downstream
        of it) catches it and issues the real interruption from below the mute → TTS flushes + brain turn
        cancelled (POST /cancel). Output's BotStoppedSpeaking then re-enters this gate and re-arms the
        window/preduck (benign). We open the command window so the user's redirect lands on ducked media."""
        # dt = seconds since BotStartedSpeaking. Re-test discriminator (2026-06-16, agreed w/ GA): a trip
        # past the arm window with no user speech (`vad` False — note VAD is muted during the bot turn, so
        # it's near-always False here) is a candidate sentence-boundary self-trip the once-per-turn arm-delay
        # can't cover; one at human-reaction latency is a real interrupt. dt < arm-delay should not appear.
        dt = time.monotonic() - self._bot_speaking_since
        _tlog(f"WAKE  | interrupt word ({score:.2f}, dt={dt:.2f}s, vad={self._speech_in_flight}) "
              "— cutting TTS, opening floor")
        await self.push_frame(StopWordFrame(), FrameDirection.DOWNSTREAM)
        self._bot_speaking = False  # the interruption stops her now (BotStopped will also confirm it)
        self._open_window("WAKE  | floor opened after interrupt")

    def _open_window(self, log_msg: str) -> None:
        """Open the command window (idempotent): log the reason, pre-duck, arm the idle timer."""
        self._cancel_keepalive()  # a fresh wake/confirm supersedes any media keepalive hold
        self._open_via_keepalive = False  # a fresh wake → the long preduck grace + raw-onset duck apply
        if not self._open:
            self._open = True
            _tlog(log_msg)
            self._fire_duck(True)  # pre-duck so the command lands on already-ducked media
            if not self._hold:
                self._arm_preduck()  # release the pre-duck if no speech follows (not while held for a confirm)
        self._escape_peak = 0.0
        self._escape_run = 0
        self._escape_hits.clear()  # fresh start once we're open
        self._retry_peak = 0.0
        self._gated_retry_hits.clear()
        self._arm_window()
        # eye: wake/floor open — capturing speech. But while asleep the gate still opens a wake-CANDIDATE
        # window (the nano model false-fires on un-AEC'd room audio); the brain may reject it and STAY
        # asleep, so honor the resting state (`off`) rather than lighting `listening` — contract: sleep→off.
        # A real wake-from-sleep relights via the awake-path writes once the brain confirms the wake.
        aria_state.set_state("listening" if aria_state.resting_state() != "off" else aria_state.resting_state())

    # --- command window ----------------------------------------------------
    def _arm_window(self) -> None:
        self._cancel_window()
        if self._hold or self._keepalive_active:
            return  # held open (pending confirm) or kept alive (media TTL) — no idle close until released

        async def _later():
            try:
                await asyncio.sleep(self._window_secs)
                if self._open:
                    self._open = False
                    self._open_via_keepalive = False
                    self._cancel_preduck()
                    _tlog("WAKE  | window closed (idle) — muting until next wake word")
                    self._fire_duck(False)
                    # eye: a wake with no command → back to the resting state (`idle` awake, `off`
                    # asleep). Only downgrade `listening`; never step on `speaking`/`thinking` (those own
                    # their own return to rest). Honors resting_state() so an asleep false-wake settles `off`.
                    if aria_state.current() == "listening":
                        aria_state.set_state(aria_state.resting_state())
            except asyncio.CancelledError:
                pass

        self._window_task = asyncio.create_task(_later())

    def _cancel_window(self) -> None:
        if self._window_task is not None and not self._window_task.done():
            self._window_task.cancel()
        self._window_task = None

    # --- media keepalive (brain TTL hold) ----------------------------------
    def _clamp_ttl(self, ttl_secs: float) -> float:
        """Clamp a brain-supplied keepalive TTL to [0, _hold_max_secs] (the hard ceiling). 0 = no hold."""
        try:
            ttl = float(ttl_secs)
        except (TypeError, ValueError):
            return 0.0
        if ttl <= 0:
            return 0.0
        return min(ttl, self._hold_max_secs)

    def _set_keepalive(self, ttl_secs: float) -> None:
        """Hold the command window open for a clamped TTL after a media command (brain keepalive), so a
        follow-up needs no re-wake. Refreshed by each new keepalive; on expiry it re-arms the normal idle
        close. Capped by _hold_max_secs so a missed/oversized refresh can't pin the mic open."""
        ttl = self._clamp_ttl(ttl_secs)
        if ttl <= 0:
            return
        self._cancel_keepalive()
        self._cancel_window()  # suspend the normal idle close while the keepalive holds the window open
        self._keepalive_active = True  # make _arm_window bail for the whole TTL (no idle-close truncation)
        self._open_via_keepalive = True  # window is held over the media a command started → strict onset gate
        if not self._open:
            self._open = True
            _tlog(f"GATE  | keepalive — opening window for {ttl:.0f}s")
        else:
            _tlog(f"GATE  | keepalive — window held {ttl:.0f}s")
        # F3 (2026-06-16, the maintainer = "option B"): do NOT pre-duck on a media keepalive. The command already ran;
        # pre-ducking here dips the song the user just asked to play/resume (the 12:12:13 "dips the song I
        # just started" annoyance). There's nothing to make room for yet — a follow-up command still ducks
        # via the confirmed-speech path (duck_allowed in the open window), and a raw onset is the media itself
        # (F1 suppresses it). So no `_fire_duck(True)` here. We still release any pre-duck left ON from a
        # PRIOR fresh wake on the SHORT keepalive grace (don't strand it for the TTL). Not while held for a
        # confirm. [If a play announcement ever gets drowned over loud media, revisit with a duck-on-bot-
        # speech-only path — duck exactly while Aria talks — rather than re-adding the pre-emptive dip.]
        if not self._hold and self._ducked:
            self._arm_preduck()

        async def _later():
            try:
                await asyncio.sleep(ttl)
                self._keepalive_active = False
                _tlog("GATE  | keepalive expired — re-arming idle window")
                self._arm_window()
            except asyncio.CancelledError:
                pass

        self._keepalive_task = asyncio.create_task(_later())

    def _cancel_keepalive(self) -> None:
        self._keepalive_active = False
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
        self._keepalive_task = None

    # --- pre-duck release grace --------------------------------------------
    def _arm_preduck(self) -> None:
        """Release the wake pre-duck after `_preduck_grace` UNLESS speech (user or Aria) keeps it down.

        The wake pre-ducks immediately so a command lands on already-ducked media — but an unused or phantom
        wake would otherwise hold media down for the whole command window. Re-armed on each VAD onset/stop
        (a cough that yields no words still releases), cancelled by a real transcription (the media-duck then
        owns restore) and while Aria is speaking. Never fires mid-speech (`_speech_in_flight`)."""
        # A keepalive-origin window uses the shorter grace (return the song promptly once Aria stops); a fresh
        # wake uses the long grace (give the user time to start a command after "Hey Aria").
        grace = self._preduck_grace_keepalive if self._open_via_keepalive else self._preduck_grace
        if grace <= 0:
            return
        self._cancel_preduck()

        async def _later():
            try:
                await asyncio.sleep(grace)
                if (self._open and self._ducked and not self._bot_speaking
                        and not self._hold and not self._speech_in_flight):
                    _tlog("WAKE  | pre-duck released (no speech after wake)")
                    self._fire_duck(False)
            except asyncio.CancelledError:
                pass

        self._preduck_task = asyncio.create_task(_later())

    def _cancel_preduck(self) -> None:
        if self._preduck_task is not None and not self._preduck_task.done():
            self._preduck_task.cancel()
        self._preduck_task = None

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
