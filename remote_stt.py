"""RemoteSTTService — offload speech-to-text to a standalone service over HTTP (for thin voice clients).

Drop-in for WhisperSTTService when the device can't run Whisper fast enough locally (a Pi-4 is ~40s/
utterance). It extends ``SegmentedSTTService``, so it receives the SAME complete VAD-segmented 16-bit mono
WAV that local Whisper would (pipecat buffers + frames it). ``run_stt`` POSTs that WAV to the EM STT
service (``stt_service/server.py``) and yields a ``TranscriptionFrame``.

CRITICAL: ``run_stt`` is awaited on the pipeline's main event loop, so the call MUST be non-blocking — an
async ``httpx`` request (network I/O) is naturally non-blocking. It carries a READ timeout (STT is
bounded); the brain client deliberately has none, but a hung STT must fail the turn FAST (→ ``ErrorFrame``
→ the existing brain-error apology) rather than stall the whole pipeline forever (which would revive the
turn-taking / barge-in / duck bugs).

``room_id`` rides as the ``X-Room-Id`` header (the durable per-room routing key); the bearer token as
``Authorization``. ``transport`` is injectable so tests drive an in-process ASGI stub (no network).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601


class RemoteSTTService(SegmentedSTTService):
    """STT via a remote HTTP service. The Pi keeps wake/VAD/endpointing/TTS; only STT is offloaded."""

    def __init__(
        self,
        base_url: str,
        *,
        room_id: str = "default",
        auth_token: str | None = None,
        timeout: float = 30.0,
        sample_rate: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        **kwargs,
    ):
        # Remote STT has no local model/language to configure — declare them None (not NOT_GIVEN) so the
        # base STTService.validate_complete is satisfied (it errors on uninitialized settings fields).
        super().__init__(
            sample_rate=sample_rate, settings=STTSettings(model=None, language=None), **kwargs
        )
        self._base_url = base_url.rstrip("/")
        self._room_id = room_id or "default"
        headers = {"X-Room-Id": self._room_id}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        # READ timeout IS set here (unlike the brain client, which has none): STT is a bounded call, so a
        # hung request must fail fast → ErrorFrame, not stall the pipeline loop. Keep a connect timeout too.
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers=headers,
            transport=transport,  # tests inject httpx.ASGITransport(app=stub)
        )
        logger.info(
            f"STT: remote ({self._base_url}, room={self._room_id}, "
            f"auth={'on' if auth_token else 'off'}, read_timeout={timeout}s)"
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """`audio` is a complete 16-bit mono WAV at self.sample_rate (SegmentedSTTService framed it)."""
        await self.start_processing_metrics()
        try:
            resp = await self._http.post("/stt", content=audio, headers={"Content-Type": "audio/wav"})
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
        except Exception as e:  # noqa: BLE001 - network/timeout/HTTP/parse: degrade, never stall the loop
            await self.stop_processing_metrics()
            logger.warning(f"STT: remote transcription failed ({type(e).__name__}: {e})")
            yield ErrorFrame(f"Remote STT unavailable: {type(e).__name__}")
            return
        await self.stop_processing_metrics()
        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601())

    async def cleanup(self) -> None:
        await super().cleanup()
        try:
            await self._http.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
