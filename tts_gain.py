"""TTSGainProcessor — attenuate Aria's TTS audio so she isn't louder than the (ducked) music.

Aria's TTS output is intentionally duck-excluded (`gabagent.duck_exclude=1`) so the brain's media duck
never silences her — but that leaves her voice at FULL output level while music is ducked (or 0% under
`mute:true`), so she comes across disproportionately loud, and turning the music up can't rebalance it
(her stream is independent + excluded). This processor scales the TTS PCM down by a fixed gain (0..1)
before the output transport, lowering her absolute level while KEEPING duck-exclude. Brain-agnostic —
no PipeWire/sink-input dependency. gain=1.0 → pass-through (no-op). Env: TTS_GAIN (default 0.6).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    SystemFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import aria_state
import bot_speech


@dataclass
class VoiceVolumeFrame(SystemFrame):
    """Adjust Aria's OWN voice (TTS) level at runtime — distinct from the media duck (which lowers OTHER
    apps). Pushed DOWNSTREAM by BrainLLMService when the brain classifies a "my-voice-volume" intent
    ("lower your voice" / "speak up" / "set your voice to half"), reaching this processor which lives in the
    TTS PCM path. `op` is "up" | "down" | "set"; `value` is a 0..1 absolute level for `set` (1.0 = full /
    no attenuation, 0.0 = silent), ignored for up/down (which step by the processor's configured step).
    A SystemFrame so it flows through KokoroTTS untouched. Brain emits the matching `voice_volume` SSE event
    (see brain_client.BrainEvent); kill-switched brain-side. Persistence is voice-side and not built yet."""

    op: str = "set"
    value: float | None = None

# Throttle the eye's amplitude (`level`) updates while speaking — TTS audio frames arrive far faster
# than the eye renders (~12 fps), so cap writes to ~20 Hz; the dedup in aria_state trims the rest.
_LEVEL_UPDATE_HZ = 20.0
_LEVEL_MIN_INTERVAL = 1.0 / _LEVEL_UPDATE_HZ


class TTSGainProcessor(FrameProcessor):
    """Scale TTSAudioRawFrame PCM by a fixed attenuation gain (0..1) before output.

    Doubles as the eye indicator's `speaking` source: sitting in the TTS PCM path, it publishes the
    `speaking` state on bot-speech start, the live audio RMS into `level` while she talks, and the
    return to `idle` when she stops (see aria_state / ~/dev/aria-eye-indicator-DESIGN.md)."""

    def __init__(self, gain: float = 0.6, *, step: float = 0.1, floor: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self._gain = max(0.0, min(1.0, gain))  # attenuate only; clamp to [0,1]
        # Runtime voice-volume control (F3): up/down step the live gain by `step`; `down` floors at `floor`
        # so repeated "quieter" can't reach inaudible (an explicit `set 0` still can — that's deliberate).
        self._step = step
        self._floor = max(0.0, min(1.0, floor))
        self._last_level_write = 0.0

    def _adjust_gain(self, op: str, value: float | None) -> None:
        """Apply a runtime voice-volume change (VoiceVolumeFrame). up/down step by `_step` (down floored at
        `_floor`, up capped at 1.0); set jumps to a clamped 0..1 absolute level."""
        if op == "up":
            new = min(1.0, self._gain + self._step)
        elif op == "down":
            new = max(self._floor, self._gain - self._step)
        elif op == "set" and value is not None:
            new = max(0.0, min(1.0, value))
        else:
            logger.debug(f"TTS gain: ignoring voice-volume op={op!r} value={value!r}")
            return
        self._gain = new
        logger.bind(transcript=True).info(f"VOICE | own volume {op} → gain {new:.2f}")

    def _apply(self, audio: bytes) -> bytes:
        """Scale int16 PCM by the gain (same byte length → num_frames unchanged). Pure/testable."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) * self._gain
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VoiceVolumeFrame):
            self._adjust_gain(frame.op, frame.value)
        elif isinstance(frame, BotStartedSpeakingFrame):
            aria_state.set_state("speaking")  # glow on; RMS fills in as audio frames arrive
            bot_speech.set_bot_speaking(True)  # → /media/state poll refreshes the brain's duck watchdog
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Return to the current resting state, not always `idle`: a goodbye spoken on the way to
            # sleep must settle the eye on `off` (the wake gate set resting=off before this TTS).
            aria_state.set_state(aria_state.resting_state())
            bot_speech.set_bot_speaking(False)
        elif isinstance(frame, TTSAudioRawFrame) and frame.audio:
            now = time.monotonic()
            if now - self._last_level_write >= _LEVEL_MIN_INTERVAL:
                self._last_level_write = now
                aria_state.set_state("speaking", aria_state.speaking_level_from_pcm(frame.audio))
            if self._gain < 1.0:
                frame.audio = self._apply(frame.audio)
        await self.push_frame(frame, direction)
