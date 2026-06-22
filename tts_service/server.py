"""Standalone TTS service (run on a fast host, e.g. EM) — kokoro-onnx over aiohttp.

Symmetric with stt_service/server.py. A thin voice client too weak for local Kokoro (a Pi-4 is ~16s/
utterance; piper is fast there but lower voice quality) keeps wake/VAD/audio-I/O local and POSTs reply
text to ``POST /tts``, getting back a WAV in Kokoro's good voice (~1s on EM x86). The client side is
``remote_tts.RemoteTTSService`` + ``config.py`` ``TTS_PROVIDER=remote``.

Design (same foundation as the STT service):
- **Stateless + per-room-keyed** (``X-Room-Id`` header, echoed/logged, no server state) → reconciler can
  spawn/collapse one per room later with zero protocol change.
- **/health open**; everything else needs ``Authorization: Bearer <token>`` once a token is set.
- **Non-blocking**: Kokoro ``create`` is sync/CPU-bound → run in a thread (``asyncio.to_thread``); an
  ``asyncio.Semaphore`` (``EM_TTS_MAX_CONCURRENCY``, default 1) bounds concurrent synths on one model.
- Reuses pipecat's Kokoro model resolution (``KOKORO_CACHE_DIR`` + ``_ensure_model_files``) so the model
  matches the in-pipeline Kokoro and auto-downloads on a fresh host.
- Returns a self-describing **WAV** (16-bit mono at Kokoro's native rate); the client resamples to its
  output rate (mirrors how KokoroTTSService resamples create_stream output to self.sample_rate).

Run:  ``uv run python -m tts_service.server --host 192.168.1.100 --port 8771``
Env:  ``TTS_VOICE`` (default af_heart), ``TTS_SERVICE_AUTH_TOKEN``, ``EM_TTS_MAX_CONCURRENCY`` (1),
      ``TTS_SERVICE_HOST``/``TTS_SERVICE_PORT``.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import os
import tempfile
import wave

# Kokoro's phonemizer (espeak-ng) copies libespeak-ng.so into a tempdir and dlopen()s it; on a host that
# mounts /tmp `noexec` (e.g. CachyOS) that map fails ("failed to map segment from shared object"). Redirect
# TMPDIR to an exec-friendly $HOME dir BEFORE anything touches tempfile — mirrors main.py (this standalone
# service doesn't import main.py, so it needs its own copy).
_EXEC_TMP = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "voice-agent", "tmp"
)
os.makedirs(_EXEC_TMP, exist_ok=True)
os.environ["TMPDIR"] = _EXEC_TMP
tempfile.tempdir = None  # drop any cached value so TMPDIR is re-read

import numpy as np  # noqa: E402 - after the TMPDIR redirect above
from aiohttp import web  # noqa: E402
from loguru import logger  # noqa: E402


def _load_kokoro():
    """Construct kokoro-onnx, reusing pipecat's cache-dir + ensure-files (matches the in-pipeline Kokoro)."""
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    model = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model, voices)  # downloads on a fresh host; no-op if present
    logger.info(f"TTS service: loading Kokoro model={model}")
    kokoro = Kokoro(str(model), str(voices))
    logger.info("TTS service: Kokoro loaded")
    return kokoro


def synth_wav(kokoro, text: str, voice: str) -> bytes:
    """Blocking synth (run via asyncio.to_thread) → a 16-bit mono WAV at Kokoro's native rate."""
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setsampwidth(2)
        w.setnchannels(1)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class TtsApp:
    """The TTS request handlers. Holds the shared Kokoro + a concurrency gate; no per-request state."""

    def __init__(self, kokoro, *, token: str = "", default_voice: str = "af_heart", max_concurrency: int = 1):
        self._kokoro = kokoro
        self._token = (token or "").strip()
        self._expected = f"Bearer {self._token}" if self._token else ""
        self._default_voice = default_voice
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    def _authorized(self, request: web.Request) -> bool:
        if not self._token:
            return True
        presented = request.headers.get("Authorization", "")
        return bool(presented) and hmac.compare_digest(presented, self._expected)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "mode": "tts"})

    async def tts(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        room_id = request.headers.get("X-Room-Id") or "default"
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - malformed JSON → 400, not a crash
            return web.json_response({"error": "bad json"}, status=400)
        text = (body.get("text") or "").strip()
        voice = body.get("voice") or self._default_voice
        if not text:
            return web.json_response({"error": "empty text"}, status=400)
        async with self._sem:
            try:
                wav = await asyncio.to_thread(synth_wav, self._kokoro, text, voice)
            except Exception as e:  # noqa: BLE001 - a synth failure is a 500; the client degrades gracefully
                logger.error(f"TTS[{room_id}]: synth failed ({type(e).__name__}: {e})")
                return web.json_response({"error": "tts failed"}, status=500)
        logger.info(f"TTS[{room_id}]: {len(text)} chars voice={voice} -> {len(wav)} bytes")
        return web.Response(body=wav, content_type="audio/wav", headers={"X-Room-Id": room_id})


def build_app(kokoro, *, token: str = "", default_voice: str = "af_heart", max_concurrency: int = 1) -> web.Application:
    """Build the aiohttp app. `kokoro` is injectable so tests pass a fake (no model load)."""
    handlers = TtsApp(kokoro, token=token, default_voice=default_voice, max_concurrency=max_concurrency)
    app = web.Application()
    app.add_routes([
        web.get("/health", handlers.health),
        web.post("/tts", handlers.tts),
    ])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone kokoro-onnx TTS service")
    parser.add_argument("--host", default=os.environ.get("TTS_SERVICE_HOST", "127.0.0.1"),
                        help="Bind address (default 127.0.0.1; set a LAN IP for thin clients).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_SERVICE_PORT", "8771")))
    args = parser.parse_args()

    token = os.environ.get("TTS_SERVICE_AUTH_TOKEN", "")
    default_voice = os.environ.get("TTS_VOICE") or "af_heart"
    max_concurrency = int(os.environ.get("EM_TTS_MAX_CONCURRENCY", "1"))
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token.strip():
        logger.warning(
            f"TTS service binding {args.host} (non-loopback) with NO auth token — "
            "set TTS_SERVICE_AUTH_TOKEN so the LAN-reachable /tts requires a bearer token."
        )

    kokoro = _load_kokoro()
    app = build_app(kokoro, token=token, default_voice=default_voice, max_concurrency=max_concurrency)
    logger.info(f"TTS service listening on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
