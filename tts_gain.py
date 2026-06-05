"""TTSGainProcessor — attenuate Aria's TTS audio so she isn't louder than the (ducked) music.

Aria's TTS output is intentionally duck-excluded (`gabagent.duck_exclude=1`) so the brain's media duck
never silences her — but that leaves her voice at FULL output level while music is ducked (or 0% under
`mute:true`), so she comes across disproportionately loud, and turning the music up can't rebalance it
(her stream is independent + excluded). This processor scales the TTS PCM down by a fixed gain (0..1)
before the output transport, lowering her absolute level while KEEPING duck-exclude. Brain-agnostic —
no PipeWire/sink-input dependency. gain=1.0 → pass-through (no-op). Env: TTS_GAIN (default 0.6).
"""

from __future__ import annotations

import numpy as np

from pipecat.frames.frames import Frame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TTSGainProcessor(FrameProcessor):
    """Scale TTSAudioRawFrame PCM by a fixed attenuation gain (0..1) before output."""

    def __init__(self, gain: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        self._gain = max(0.0, min(1.0, gain))  # attenuate only; clamp to [0,1]

    def _apply(self, audio: bytes) -> bytes:
        """Scale int16 PCM by the gain (same byte length → num_frames unchanged). Pure/testable."""
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) * self._gain
        return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._gain < 1.0 and isinstance(frame, TTSAudioRawFrame) and frame.audio:
            frame.audio = self._apply(frame.audio)
        await self.push_frame(frame, direction)
