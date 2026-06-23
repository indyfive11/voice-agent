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
Env:  ``STT_SERVICE_BACKEND`` (faster | whispercpp; default faster), ``WHISPER_CPP_URL``
      (http://127.0.0.1:8079, the warm GPU whisper-server when backend=whispercpp),
      ``STT_MODEL`` (small.en), ``STT_SERVICE_DEVICE`` (auto), ``STT_SERVICE_COMPUTE_TYPE`` (default),
      ``STT_SERVICE_AUTH_TOKEN``, ``EM_STT_MAX_CONCURRENCY`` (1), ``STT_SERVICE_HOST``/``STT_SERVICE_PORT``.

Backends: faster-whisper is CPU-only here (CTranslate2 has no AMD/ROCm build). For GPU on the EM
RX 7900 XT, run a loopback ``whisper-server`` (whisper.cpp, Vulkan) and set ``STT_SERVICE_BACKEND=
whispercpp`` — this service then proxies each utterance to it (warm inference ~30-50ms vs ~1.3s CPU),
staying the LAN-facing bearer-guarded, room-keyed layer. Default unchanged so CPU/single-box installs
behave exactly as before.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import os
import wave

import aiohttp
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


def float32_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 [-1, 1] mono array → a 16-bit mono WAV (for backends that want a file upload)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _transcribe(model, audio: np.ndarray, language: str | None) -> str:
    """Blocking transcribe (run via asyncio.to_thread). Mirrors whisper/stt.py text assembly."""
    segments, _ = model.transcribe(audio, language=language)
    return "".join(seg.text for seg in segments).strip()


# --- transcribe backends -----------------------------------------------------------------------
# A backend exposes ``async transcribe(audio, sample_rate, language) -> str``. Two ship today:
#   * faster-whisper (CPU) — the default; CTranslate2 has no AMD/ROCm path, so it's CPU-bound here.
#   * whisper.cpp (GPU) — proxies to a local warm ``whisper-server`` (Vulkan/ROCm). On the EM RX 7900 XT
#     this is ~25x faster (warm inference ~30-50ms vs ~1.3s CPU for a 2s utterance). Default stays
#     faster-whisper so a CPU-only / single-box install is byte-for-byte unchanged (no-hardcodes SOP).

class FasterWhisperBackend:
    """CPU faster-whisper. Runs the blocking transcribe in a thread so the event loop stays free."""

    def __init__(self, model):
        self._model = model

    async def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None) -> str:
        return await asyncio.to_thread(_transcribe, self._model, audio, language)

    async def close(self) -> None:
        pass


class WhisperCppBackend:
    """GPU whisper.cpp via a local ``whisper-server`` /inference endpoint (multipart WAV → {"text": …}).

    The whisper-server keeps the model warm on the GPU; we just forward each utterance over loopback
    HTTP (no thread needed — network I/O is naturally non-blocking). The server is loopback-only; THIS
    service remains the LAN-facing, bearer-guarded, room-keyed layer.
    """

    def __init__(self, url: str, *, session: aiohttp.ClientSession | None = None, timeout: float = 30.0):
        self._url = url.rstrip("/") + "/inference"
        self._session = session
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def transcribe(self, audio: np.ndarray, sample_rate: int, language: str | None) -> str:
        session = await self._ensure_session()
        form = aiohttp.FormData()
        form.add_field("file", float32_to_wav(audio, sample_rate),
                       filename="audio.wav", content_type="audio/wav")
        form.add_field("response_format", "json")
        if language:
            form.add_field("language", language)
        async with session.post(self._url, data=form, timeout=self._timeout) as r:
            r.raise_for_status()
            data = await r.json()
        return (data.get("text") or "").strip()

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def _load_backend():
    """Select the transcribe backend from STT_SERVICE_BACKEND (faster | whispercpp). Default faster."""
    backend = (os.environ.get("STT_SERVICE_BACKEND") or "faster").lower()
    if backend in ("whispercpp", "whisper.cpp", "whisper-cpp", "gpu"):
        url = os.environ.get("WHISPER_CPP_URL") or "http://127.0.0.1:8079"
        logger.info(f"STT service: backend=whispercpp (GPU) proxying to whisper-server {url}")
        return WhisperCppBackend(url)
    return FasterWhisperBackend(_load_model())


class SttApp:
    """The STT request handlers. Holds the shared model + a concurrency gate; no per-request state."""

    def __init__(self, backend, *, token: str = "", max_concurrency: int = 1):
        self._backend = backend
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
                text = await self._backend.transcribe(audio, sample_rate, language)
            except Exception as e:  # noqa: BLE001 - a backend failure is a 500, the client degrades gracefully
                logger.error(f"STT[{room_id}]: transcribe failed ({type(e).__name__}: {e})")
                return web.json_response({"error": "stt failed"}, status=500)
        dur = len(audio) / sample_rate if sample_rate else 0.0
        logger.info(f"STT[{room_id}]: {dur:.1f}s audio -> [{text}]")
        return web.json_response({"text": text, "room_id": room_id})


def build_app(model=None, *, token: str = "", max_concurrency: int = 1, backend=None) -> web.Application:
    """Build the aiohttp app. Pass either a faster-whisper `model` (wrapped in FasterWhisperBackend) or a
    ready `backend` directly; both are injectable so tests pass a fake (no model load / no GPU server)."""
    if backend is None:
        backend = FasterWhisperBackend(model)
    handlers = SttApp(backend, token=token, max_concurrency=max_concurrency)
    # client_max_size: utterances are small, but a long ramble at 16k mono is ~2MB/min — give headroom.
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([
        web.get("/health", handlers.health),
        web.post("/stt", handlers.stt),
    ])
    app.on_cleanup.append(lambda _app: backend.close())  # close any backend HTTP session on shutdown
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

    backend = _load_backend()
    app = build_app(backend=backend, token=token, max_concurrency=max_concurrency)
    logger.info(f"STT service listening on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
