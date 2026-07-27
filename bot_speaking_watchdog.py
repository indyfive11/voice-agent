"""BotSpeakingWatchdog — break a stuck-open bot-speaking "broken record" during a confirmed device wedge.

Incident 2026-07-27 (EM): a reply started playing, the reSpeaker mic wedged mid-playback, and Aria
repeated the same phrase "over and over" until the maintainer muted and killed her. Root cause, traced end to end
and cross-checked with GA:

  `LocalAudioTransport` shares ONE `pyaudio.PyAudio()` instance between the callback-mode INPUT and the
  blocking-write OUTPUT (`run_in_executor(_out_stream.write, ...)`). When the input-stall recovery calls
  PortAudio's SYNCHRONOUS `open()`/`stop_stream()` on a C-layer-wedged reSpeaker (which "blocks FOREVER
  in the C layer"), the output stream's blocking `write()` starves behind it on the shared instance. The
  trailing `TTSStoppedFrame` — queued AFTER the audio and drained THROUGH the device — then never plays
  out, so `_bot_stopped_speaking` never fires: bot-speaking hangs open AND the output stream underruns,
  looping its last buffer = the repeat. One device wedge, two faces (deaf input + broken-record output).

Design (adversarially reviewed with GA — the co-signal gate is GA's, and it's what makes this safe):
  - **Only acts during a CONFIRMED device wedge.** The escalation is AND-gated on the input-stall
    co-signal (`device_wedged`, = the input watchdog's live `stalled` state). Because it is ONE shared
    PyAudio instance, a genuine wedge stalls INPUT too (the incident logged INPUT STALL concurrently with
    the stuck output). A legitimate long narration — bot-speaking still open in an inter-sentence gap
    while the brain streams the next sentence — shows NO input stall, so it is never touched. Keying on
    a bare bot-speaking deadline alone would clean-cut a live reply; the co-signal is what prevents that.
  - **Only breaks the loop; does NOT own device recovery.** On a confirmed wedge it fires ONE
    `broadcast_interruption()` (the same path the maintainer's "Aria" barge-in uses) to stop the broken record as
    fast as possible, and unsticks the shared `bot_speaking` flag so the brain's duck watchdog isn't
    stranded. It does NOT call `os._exit`/usb-reset itself: the INPUT watchdog already owns the device
    ladder (kicks → usb power-cycle → `os._exit` → systemd clean restart, in that order), and is already
    running during the stall. Adding a second exit-caller would race it, skip the usb-reset ordering, and
    double-spend the unit's `StartLimitBurst`. So we defer recovery to the one ladder that has it right.

  Self-tuning deadline (never false-fires): the escalation requires BOTH that all produced audio has had
  time to drain (`elapsed >= max(min_deadline, produced_secs + grace)`) AND that audio flow has actually
  ceased (`now - last_audio >= grace`) AND the device co-signal. A reply that keeps streaming audio keeps
  advancing `last_audio` → never trips; only a genuinely stalled span during a confirmed wedge does.

Known limitation (documented, follow-up): an OUTPUT-only wedge with a HEALTHY input would show no
co-signal, so this belt would not act. Not observed (the whole reSpeaker-wedge history is input-driven),
and the conservative default — never risk cutting a live reply — is deliberate. The durable fix is GA's
"Hole 3": ONE device-health state machine that takes both the input-stall and the output-stall (bot-
speaking-stuck) as symptoms of the shared instance and owns a single kicks→usb-reset→exit ladder with
shared start-limit accounting. Flagged so the two watchdogs converge later instead of racing.

Env knobs (generous defaults ⇒ byte-identical to today in healthy operation, per the no-hardcodes SOP):
  BOT_SPEAKING_WATCHDOG=0        disable entirely (default on).
  BOT_SPEAKING_STALL_GRACE       seconds of no-audio-flow that count as "drained/ceased" (default 8).
  BOT_SPEAKING_STALL_MIN_DEADLINE  floor for the drain deadline (default 8).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import bot_speech


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _frame_audio_secs(frame: TTSAudioRawFrame) -> float:
    """Playout duration of a PCM16 audio frame in seconds (len / (2 bytes · channels · rate))."""
    rate = getattr(frame, "sample_rate", 0) or 0
    channels = getattr(frame, "num_channels", 1) or 1
    audio = getattr(frame, "audio", b"") or b""
    if rate <= 0:
        return 0.0
    return len(audio) / (2.0 * channels * rate)


class BotSpeakingWatchdog(FrameProcessor):
    """Break a stuck-open bot-speaking loop, but ONLY while the shared audio device is confirmed wedged.

    Placed at the TTS-gain seat so it sees `TTSAudioRawFrame` (downstream, to measure produced audio) and
    `BotStarted/StoppedSpeakingFrame` (upstream from the output transport). Pure pass-through — never
    mutates frames.
    """

    def __init__(
        self,
        *,
        device_wedged: Callable[[], bool] | None = None,
        escalate_recovery: Callable[[str], None] | None = None,
        grace_secs: float | None = None,
        min_deadline_secs: float | None = None,
        tick_secs: float = 1.0,
        enabled: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The wedge CO-SIGNAL: returns True while the input watchdog is in a stall episode. None ⇒ no
        # co-signal available ⇒ the belt stays inert (it must never act on a bot-speaking deadline alone).
        self._device_wedged = device_wedged
        # Fast-track the input watchdog's terminal recovery (usb-reset → clean restart). This is the action
        # that actually STOPS the loop: broadcast_interruption can't (asyncio can't cancel the executor
        # thread stuck in the poisoned C write()), so only the process exit reclaims the device.
        self._escalate_recovery = escalate_recovery
        # Generous defaults → never fires in healthy operation (the historical no-op).
        self._grace = _env_float("BOT_SPEAKING_STALL_GRACE", 8.0) if grace_secs is None else grace_secs
        self._min_deadline = (
            _env_float("BOT_SPEAKING_STALL_MIN_DEADLINE", 8.0)
            if min_deadline_secs is None else min_deadline_secs
        )
        self._tick = tick_secs
        self._enabled = (
            os.environ.get("BOT_SPEAKING_WATCHDOG", "1") not in ("0", "false", "False", "")
            if enabled is None else enabled
        )
        # Per-span state.
        self._speaking = False
        self._started_at = 0.0
        self._produced_secs = 0.0
        self._last_audio_at = 0.0
        self._broke = False  # fired the loop-break this span already (don't spam interruptions)
        self._task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._enabled:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._arm()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self._disarm()
            elif isinstance(frame, TTSAudioRawFrame) and self._speaking:
                self._produced_secs += _frame_audio_secs(frame)
                self._last_audio_at = time.monotonic()
        await self.push_frame(frame, direction)

    def _arm(self) -> None:
        self._speaking = True
        self._started_at = time.monotonic()
        self._produced_secs = 0.0
        self._last_audio_at = self._started_at  # so a BotStarted with NO audio at all still trips
        self._broke = False
        self._task = self.task_manager.create_task(self._watch(), f"{self}::watch")

    async def _disarm(self) -> None:
        self._speaking = False
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None

    async def cleanup(self):
        await super().cleanup()
        await self._disarm()

    def _deadline(self) -> float:
        """Wall-clock seconds from span start by which all produced audio must have drained."""
        return max(self._min_deadline, self._produced_secs + self._grace)

    async def _watch(self) -> None:
        try:
            while self._speaking:
                await self._tick_once()
                if not self._speaking:
                    return
        except asyncio.CancelledError:
            return
        finally:
            self._task = None

    def _stalled_open(self) -> bool:
        """True iff bot-speaking has hung open past its drain deadline AND audio flow has ceased AND the
        shared audio device is independently confirmed wedged (input-stall co-signal). All three required:
        the first two describe a stuck output; the co-signal is what distinguishes a WEDGE from a long
        narration's inter-sentence gap, so a live reply is never guillotined."""
        if not self._speaking:
            return False
        now = time.monotonic()
        drained = (now - self._started_at) >= self._deadline()
        audio_ceased = (now - self._last_audio_at) >= self._grace
        wedged = bool(self._device_wedged()) if self._device_wedged else False
        return drained and audio_ceased and wedged

    async def _tick_once(self) -> None:
        """One watch step (extracted so the escalation is unit-testable without driving the sleep loop)."""
        await asyncio.sleep(self._tick)
        if self._broke or not self._stalled_open():
            return
        self._broke = True
        _tlog(
            "BOT STALL | bot-speaking stuck open past drain deadline AND device wedged (input-stall "
            "co-signal) — broken-record loop on a poisoned audio instance; fast-tracking a clean restart"
        )
        # The ONLY thing that stops a poisoned-instance loop is process death (the OS reclaims the device
        # on exit): tell the input ladder to skip its remaining futile kicks and go straight to usb-reset →
        # os._exit → systemd clean restart. Single exit-caller, correct ordering — see input_watchdog.
        if self._escalate_recovery:
            self._escalate_recovery("output broken-record loop confirms the shared audio device is wedged")
        # Best-effort immediate state hygiene before the restart lands: clear pipecat's playout buffers and
        # unstick the shared bot-speaking flag so the brain's duck watchdog isn't stranded in the interim.
        # (Can't silence a poisoned output stream — that's why the exit above is the real fix.)
        await self.broadcast_interruption()
        bot_speech.set_bot_speaking(False)
