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

import numpy as np

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import aria_state

# Throttle the eye's amplitude (`level`) updates while speaking — TTS audio frames arrive far faster
# than the eye renders (~12 fps), so cap writes to ~20 Hz; the dedup in aria_state trims the rest.
_LEVEL_UPDATE_HZ = 20.0
_LEVEL_MIN_INTERVAL = 1.0 / _LEVEL_UPDATE_HZ


class TTSGainProcessor(FrameProcessor):
    """Scale TTSAudioRawFrame PCM by a fixed attenuation gain (0..1) before output.

    Doubles as the eye indicator's `speaking` source: sitting in the TTS PCM path, it publishes the
    `speaking` state on bot-speech start, the live audio RMS into `level` while she talks, and the
    return to `idle` when she stops (see aria_state / ~/dev/aria-eye-indicator-DESIGN.md)."""

    def __init__(self, gain: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self._gain = max(0.0, min(1.0, gain))  # attenuate only; clamp to [0,1]
        self._last_level_write = 0.0

    def _apply(self, audio: bytes) -> bytes:
        """Scale int16 PCM by the gain (same byte length → num_frames unchanged). Pure/testable."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) * self._gain
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            aria_state.set_state("speaking")  # glow on; RMS fills in as audio frames arrive
        elif isinstance(frame, BotStoppedSpeakingFrame):
            aria_state.set_state("idle")
        elif isinstance(frame, TTSAudioRawFrame) and frame.audio:
            now = time.monotonic()
            if now - self._last_level_write >= _LEVEL_MIN_INTERVAL:
                self._last_level_write = now
                aria_state.set_state("speaking", aria_state.speaking_level_from_pcm(frame.audio))
            if self._gain < 1.0:
                frame.audio = self._apply(frame.audio)
        await self.push_frame(frame, direction)
