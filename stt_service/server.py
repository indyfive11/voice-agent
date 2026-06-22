"""Standalone STT service (run on a fast host, e.g. EM) — faster-whisper over aiohttp.

A thin voice client (e.g. a Pi-4 too slow for local Whisper — ~40s/utterance) keeps wake/VAD/TTS local
and POSTs only the wake-gated utterance (a 16-bit mono WAV at the pipeline rate) to ``POST /stt``,
getting ``{"text": ...}`` back in ~1s. The client side is ``remote_stt.RemoteSTTService`` + ``config.py``
``STT_PROVIDER=remote``.

Design (foundation for multi-room, hot-pluggable thin clients):
- **Stateless + per-room-keyed.** ``room_id`` rides as a header (``X-Room-Id``) and is echoed/logged but
  holds no server state — so a future reconciler can spawn/collapse one of these per room with zero
  protocol change (server-internal warm-model caching keyed on room_id can be added later, invisibly).
- **/health is open** (unauthenticated liveness); every other route requires ``Authorization: Bearer
  <token>`` once a token is configured (mirrors the brain's bearer seam). Loopback with no token = open.
- **Non-blocking.** The CPU-bound transcribe runs in a thread (``asyncio.to_thread``) so the event loop
  stays free; an ``asyncio.Semaphore`` (``EM_STT_MAX_CONCURRENCY``, default 1) bounds concurrent
  transcribes against one shared model (multi-room contention later needs no protocol change).
- Mirrors the in-pipeline Whisper config (``config.py`` build_stt): faster-whisper, ``STT_MODEL`` default
  ``small.en``, ``device=auto``.

Run:  ``uv run python -m stt_service.server --host 192.168.1.100 --port 8770``
Env:  ``STT_MODEL`` (small.en), ``STT_SERVICE_DEVICE`` (auto), ``STT_SERVICE_COMPUTE_TYPE`` (default),
      ``STT_SERVICE_AUTH_TOKEN``, ``EM_STT_MAX_CONCURRENCY`` (1), ``STT_SERVICE_HOST``/``STT_SERVICE_PORT``.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import os
import wave

import numpy as np
from aiohttp import web
from loguru import logger


def _load_model():
    """Construct the faster-whisper model, mirroring config.py build_stt (small.en / auto / default)."""
    from faster_whisper import WhisperModel

    model_name = os.environ.get("STT_MODEL") or "small.en"
    device = os.environ.get("STT_SERVICE_DEVICE") or "auto"
    compute_type = os.environ.get("STT_SERVICE_COMPUTE_TYPE") or "default"
    logger.info(f"STT service: loading faster-whisper model={model_name!r} device={device} compute={compute_type}")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    logger.info("STT service: model loaded")
    return model


def wav_to_float32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Parse a 16-bit mono WAV → (float32 ndarray in [-1, 1], sample_rate).

    Parsing the WAV (vs feeding raw bytes to np.frombuffer) drops the 44-byte header cleanly — the
    in-pipeline Whisper path feeds the header to the model as a tiny noise burst; we don't have to.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sample_rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def _transcribe(model, audio: np.ndarray, language: str | None) -> str:
    """Blocking transcribe (run via asyncio.to_thread). Mirrors whisper/stt.py text assembly."""
    segments, _ = model.transcribe(audio, language=language)
    return "".join(seg.text for seg in segments).strip()


class SttApp:
    """The STT request handlers. Holds the shared model + a concurrency gate; no per-request state."""

    def __init__(self, model, *, token: str = "", max_concurrency: int = 1):
        self._model = model
        self._token = (token or "").strip()
        self._expected = f"Bearer {self._token}" if self._token else ""
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    def _authorized(self, request: web.Request) -> bool:
        if not self._token:
            return True  # loopback / no-token mode: open
        presented = request.headers.get("Authorization", "")
        return bool(presented) and hmac.compare_digest(presented, self._expected)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "mode": "stt"})

    async def stt(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        room_id = request.headers.get("X-Room-Id") or "default"
        language = request.query.get("language") or None
        body = await request.read()
        if not body:
            return web.json_response({"error": "empty body"}, status=400)
        try:
            audio, sample_rate = wav_to_float32(body)
        except Exception as e:  # noqa: BLE001 - any malformed upload is a 400, not a crash
            logger.warning(f"STT[{room_id}]: bad WAV ({type(e).__name__}: {e})")
            return web.json_response({"error": "bad audio"}, status=400)
        async with self._sem:
            try:
                text = await asyncio.to_thread(_transcribe, self._model, audio, language)
            except Exception as e:  # noqa: BLE001 - a model failure is a 500, the client degrades gracefully
                logger.error(f"STT[{room_id}]: transcribe failed ({type(e).__name__}: {e})")
                return web.json_response({"error": "stt failed"}, status=500)
        dur = len(audio) / sample_rate if sample_rate else 0.0
        logger.info(f"STT[{room_id}]: {dur:.1f}s audio -> [{text}]")
        return web.json_response({"text": text, "room_id": room_id})


def build_app(model, *, token: str = "", max_concurrency: int = 1) -> web.Application:
    """Build the aiohttp app. `model` is injectable so tests pass a fake (no model load)."""
    handlers = SttApp(model, token=token, max_concurrency=max_concurrency)
    # client_max_size: utterances are small, but a long ramble at 16k mono is ~2MB/min — give headroom.
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([
        web.get("/health", handlers.health),
        web.post("/stt", handlers.stt),
    ])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone faster-whisper STT service")
    parser.add_argument("--host", default=os.environ.get("STT_SERVICE_HOST", "127.0.0.1"),
                        help="Bind address (default 127.0.0.1; set a LAN IP for thin clients).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("STT_SERVICE_PORT", "8770")))
    args = parser.parse_args()

    token = os.environ.get("STT_SERVICE_AUTH_TOKEN", "")
    max_concurrency = int(os.environ.get("EM_STT_MAX_CONCURRENCY", "1"))
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token.strip():
        logger.warning(
            f"STT service binding {args.host} (non-loopback) with NO auth token — "
            "set STT_SERVICE_AUTH_TOKEN so the LAN-reachable /stt requires a bearer token."
        )

    model = _load_model()
    app = build_app(model, token=token, max_concurrency=max_concurrency)
    logger.info(f"STT service listening on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
