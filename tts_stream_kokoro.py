"""StreamingKokoroTTSService — local Kokoro that sub-splits a long sentence for fast first-audio.

pipecat hands ``run_tts`` ONE sentence (TextAggregationMode.SENTENCE). The stock KokoroTTSService
then calls ``kokoro.create_stream(sentence)`` which is FAKE-streaming (yields the whole sentence as
one chunk at full-synth time), so a long sentence synthesizes entirely before any audio emits — the
felt-latency cliff (a ~6s first-audio on a long run-on, measured on the i7-8565U laptop, 2026-06-26).

This subclass splits the received sentence at sub-sentence (clause) boundaries via the shared
``tts_chunk.split_for_synth`` and synthesizes each chunk with the BLOCKING ``kokoro.create`` (in a
thread, like tts_service/server.py) — NOT the fake-streaming ``create_stream`` — yielding audio per
chunk. First-audio lands on the first clause. Length-GATED: a sentence at or below
``split_threshold`` chars passes through whole (one chunk) = byte-identical synth to the base, so
short/normal sentences keep natural prosody and only long run-ons are chunked.

See the GA↔VAC collab drops 2026-06-26 (streaming_TTS thread) for the design rationale.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.settings import assert_given
from pipecat.utils.tracing.service_decorators import traced_tts

from tts_chunk import split_for_synth


class StreamingKokoroTTSService(KokoroTTSService):
    """Kokoro TTS that sub-splits a long sentence into clause chunks for fast first-audio.

    Identical to ``KokoroTTSService`` for a short sentence (one chunk, one synth); for a sentence
    longer than ``split_threshold`` it synthesizes + flushes per clause so first-audio arrives on the
    first clause instead of the whole sentence.
    """

    def __init__(
        self,
        *args,
        split_threshold: int = 140,
        max_chars: int = 160,
        min_len: int = 24,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._chunk_split_threshold = split_threshold
        self._chunk_max_chars = max_chars
        self._chunk_min_len = min_len

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize speech, sub-splitting a long sentence so first-audio lands on the first clause.

        Mirrors the base KokoroTTSService.run_tts (voice/lang resolution, resample, metrics, frame
        shape) but loops over sub-chunks with the blocking ``kokoro.create`` instead of the
        fake-streaming ``create_stream``.
        """
        chunks = split_for_synth(
            text,
            min_len=self._chunk_min_len,
            max_chars=self._chunk_max_chars,
            split_threshold=self._chunk_split_threshold,
        )
        if len(chunks) > 1:
            logger.debug(f"{self}: chunked TTS ({len(chunks)} sub-chunks) [{text}]")
        else:
            logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_tts_usage_metrics(text)

            voice = assert_given(self._settings.voice)
            if voice is None:
                raise ValueError("Kokoro TTS voice must be specified")
            lang = assert_given(self._settings.language)
            if lang is None:
                raise ValueError("Kokoro TTS language must be specified")

            first = True
            for chunk in chunks:
                # Blocking whole-chunk synth in a thread (releases the GIL during native compute) —
                # each chunk is short, so "whole chunk at once" IS fast first-audio. Avoids the
                # create_stream() fake-streaming trap entirely.
                samples, sample_rate = await asyncio.to_thread(
                    self._kokoro.create, chunk, voice=voice, lang=lang, speed=1.0
                )
                if first:
                    # First audio of the first chunk stops the TTFB clock — this is the whole point.
                    await self.stop_ttfb_metrics()
                    first = False

                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, self.sample_rate
                )
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:  # noqa: BLE001 - mirror base: surface as an ErrorFrame, never crash the pipeline
            yield ErrorFrame(error=f"Kokoro chunked TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
