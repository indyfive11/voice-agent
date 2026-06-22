"""RemoteTTSService — offload speech synthesis to a standalone Kokoro service over HTTP (for thin clients).

Drop-in for KokoroTTSService when the device can't run Kokoro fast enough locally (a Pi-4 is ~16s/utterance)
but piper's local voice quality isn't good enough. The Pi keeps wake/VAD/endpointing/audio-I/O; only TTS is
offloaded. ``run_tts`` POSTs the reply text to the EM TTS service (``tts_service/server.py``), gets back a WAV
in Kokoro's good voice, resamples it to the output rate, and yields a ``TTSAudioRawFrame`` — exactly mirroring
KokoroTTSService (which resamples its create_stream output to ``self.sample_rate``).

Async ``httpx`` (non-blocking) with a read timeout; a failure yields an ``ErrorFrame`` (the turn degrades, the
existing brain-error apology covers a dead service). ``room_id`` rides as ``X-Room-Id``; bearer as
``Authorization``. ``transport`` is injectable for tests (in-process ASGI stub, no network).
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncGenerator

import httpx
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


class RemoteTTSService(TTSService):
    """TTS via a remote Kokoro service. The thin client keeps everything else local; only TTS is offloaded."""

    def __init__(
        self,
        base_url: str,
        *,
        voice: str = "af_heart",
        room_id: str = "default",
        auth_token: str | None = None,
        timeout: float = 30.0,
        sample_rate: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        **kwargs,
    ):
        # push_start_frame/push_stop_frames=True → the base auto-pushes TTSStarted/Stopped + manages the audio
        # context, so run_tts only yields TTSAudioRawFrame (mirrors KokoroTTSService). voice rides the request;
        # model/language unused here (the remote service owns the model) → declared None to satisfy the base.
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=TTSSettings(model=None, voice=voice, language=None),
            **kwargs,
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._room_id = room_id or "default"
        headers = {"X-Room-Id": self._room_id}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        # READ timeout set (unlike the brain client): a hung TTS must fail the turn fast, not stall.
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers=headers,
            transport=transport,
        )
        self._resampler = create_stream_resampler()
        logger.info(
            f"TTS: remote ({self._base_url}, voice={self._voice}, room={self._room_id}, "
            f"auth={'on' if auth_token else 'off'}, read_timeout={timeout}s)"
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS (remote) [{text}]")
        await self.start_tts_usage_metrics(text)
        try:
            resp = await self._http.post(
                "/tts", json={"text": text, "voice": self._voice},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            wav = resp.content
        except Exception as e:  # noqa: BLE001 - network/timeout/HTTP: degrade to ErrorFrame, never stall the loop
            logger.warning(f"TTS: remote synth failed ({type(e).__name__}: {e})")
            yield ErrorFrame(f"Remote TTS unavailable: {type(e).__name__}")
            return
        # WAV is self-describing — read the rate from the header, resample to our output rate (like Kokoro).
        with wave.open(io.BytesIO(wav), "rb") as w:
            src_rate = w.getframerate()
            pcm = w.readframes(w.getnframes())
        await self.stop_ttfb_metrics()
        audio = await self._resampler.resample(pcm, src_rate, self.sample_rate)
        yield TTSAudioRawFrame(audio, self.sample_rate, 1)

    async def cleanup(self) -> None:
        await super().cleanup()
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
