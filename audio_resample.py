"""Mic input resampler — a pass-through FrameProcessor that normalizes capture rate.

Why this exists: pipecat opens the mic via PyAudio at *exactly* the requested rate and does **not**
resample between the input transport and its consumers. Both Whisper STT and the Silero VAD expect
16 kHz, but some capture devices only open at their native rate — notably the PipeWire
`echo-cancel-source` (our AEC Mic), which is `float32le 2ch 48000Hz` and flatly rejects a 16 kHz
open (`-9997 Invalid sample rate`), unlike the raw ALSA mic aliases that happen to resample.

So the transport captures at the device's native rate (see `config.build_transport`) and this sits
**first in the pipeline** (right after `transport.input()`), resampling every `InputAudioRawFrame`
down to `target_rate` once. Everything downstream — STT, VAD (both pinned to `target_rate`),
MediaDuck, the aggregator — then runs at one consistent rate, exactly as it did when the mic opened
at 16 kHz directly. No-op when the device already opens at `target_rate` (rates match → passthrough).

Mono only (matches `audio_in_channels=1`); SOXR's stream resampler keeps history across chunks so
there are no click artifacts at frame boundaries.
"""

from __future__ import annotations

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class InputResampler(FrameProcessor):
    """Resample mic `InputAudioRawFrame`s to a fixed pipeline rate (default 16 kHz)."""

    def __init__(self, target_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self._target_rate = target_rate
        self._resampler = create_stream_resampler()
        self._warned_stereo = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self._maybe_resample(frame)
        await self.push_frame(frame, direction)

    async def _maybe_resample(self, frame: Frame) -> None:
        """Resample a mic frame to the target rate in place. No-op for other frames / matching rates."""
        if not isinstance(frame, InputAudioRawFrame) or frame.sample_rate == self._target_rate:
            return
        if frame.num_channels != 1:
            # SOXR stream resampler is mono-only; we never request stereo capture
            # (audio_in_channels=1), so this is a guard, not an expected path.
            if not self._warned_stereo:
                logger.warning(
                    f"InputResampler: {frame.num_channels}ch input not supported; "
                    "passing through unresampled."
                )
                self._warned_stereo = True
            return
        resampled = await self._resampler.resample(
            frame.audio, frame.sample_rate, self._target_rate
        )
        frame.audio = resampled
        frame.sample_rate = self._target_rate
        frame.num_frames = len(resampled) // 2  # int16 mono → 2 bytes/sample
