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
- **POST /tts** returns a self-describing **WAV** (16-bit mono at Kokoro's native rate); the client
  resamples to its output rate. **POST /tts/stream** streams raw 16-bit mono PCM as Kokoro synthesizes it
  sentence-by-sentence (chunked transfer, rate in ``X-Sample-Rate``) so first-audio lands on sentence 1
  instead of the full reply — the satellite's biggest latency lever (token→audio: full-reply→first-sentence).

Run:  ``uv run python -m tts_service.server --host 192.168.1.100 --port 8771``
Env:  ``TTS_VOICE`` (default af_heart), ``TTS_SERVICE_AUTH_TOKEN``, ``EM_TTS_MAX_CONCURRENCY`` (1),
      ``TTS_SERVICE_HOST``/``TTS_SERVICE_PORT``, ``TTS_CACHE_SIZE`` (64; 0 disables the synth cache),
      ``TTS_PREWARM_PHRASES`` ('|'-separated fixed phrases synthesized into the cache at startup).

Synth cache (lever #4 "pre-render"): identical (text, voice) requests replay cached PCM with no Kokoro
call — so a REPEATED fixed phrase (the brain-down/error apologies, "Yes?" if re-enabled, a repeated
"pause") is instant. Cache-on-first-use by default; TTS_PREWARM_PHRASES makes the first occurrence instant
too. It's a server-side response cache (no pipeline-frame risk), broader than just greetings.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import io
import os
import re
import tempfile
import wave
from collections import OrderedDict

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


def _f32_to_pcm16(samples: "np.ndarray") -> bytes:
    """Float32 [-1, 1] → 16-bit little-endian PCM bytes."""
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


from tts_chunk import split_for_synth  # noqa: E402 - shared chunker (one impl, no drift); pure-stdlib, no TMPDIR concern


def _split_sentences(text: str, *, min_len: int = 24) -> list[str]:
    """Split a reply into chunks for incremental synth — delegates to the shared chunker so there is
    ONE implementation across the in-pipeline subclass and this service (no drift).

    Defaults reproduce the historical behavior byte-for-byte: ``clause=False`` (sentence terminals
    ``.!?`` only), no max-char cap, no length gate. To give THIS host (the Pi) the sub-sentence
    cliff fix too, pass ``clause=True`` + ``split_threshold``/``max_chars`` (deferred until a Pi
    live-verify — see the GA↔VAC streaming_TTS collab, 2026-06-26)."""
    return split_for_synth(text, min_len=min_len, clause=False, max_chars=0, split_threshold=0)


def synth_pcm(kokoro, text: str, voice: str) -> tuple[bytes, int]:
    """Blocking whole-utterance synth → (16-bit mono PCM bytes, sample_rate). The cache stores this."""
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    return _f32_to_pcm16(samples), sample_rate


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap 16-bit mono PCM in a self-describing WAV (for the /tts whole-utterance response)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setsampwidth(2)
        w.setnchannels(1)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def synth_wav(kokoro, text: str, voice: str) -> bytes:
    """Blocking synth (run via asyncio.to_thread) → a 16-bit mono WAV at Kokoro's native rate."""
    pcm, sample_rate = synth_pcm(kokoro, text, voice)
    return pcm_to_wav(pcm, sample_rate)


class TtsApp:
    """The TTS request handlers. Holds the shared Kokoro + a concurrency gate; no per-request state."""

    def __init__(self, kokoro, *, token: str = "", default_voice: str = "af_heart",
                 max_concurrency: int = 1, cache_size: int = 64):
        self._kokoro = kokoro
        self._token = (token or "").strip()
        self._expected = f"Bearer {self._token}" if self._token else ""
        self._default_voice = default_voice
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        # Synth cache: identical (text, voice) → cached PCM, so a REPEATED fixed phrase (the brain-down/error
        # apologies, "Yes?" if re-enabled, a repeated "pause"/"next") replays instantly with no Kokoro call.
        # LRU, bounded. cache_size=0 disables. Realizes lever #4 (pre-render) as a server-side response cache —
        # broader than just greetings and with no pipeline-frame risk.
        self._cache_size = max(0, cache_size)
        self._cache: "OrderedDict[tuple[str, str], tuple[bytes, int]]" = OrderedDict()

    def _cache_get(self, key: tuple[str, str]):
        if self._cache_size == 0:
            return None
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def _cache_put(self, key: tuple[str, str], val: tuple[bytes, int]) -> None:
        if self._cache_size == 0:
            return
        self._cache[key] = val
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def prewarm(self, phrases, voice: str) -> int:
        """Synthesize fixed phrases into the cache at startup so even the FIRST occurrence is instant
        (e.g. the error apologies). Blocking; called once before serving. Returns how many were warmed."""
        n = 0
        for p in phrases:
            text = (p or "").strip()
            if not text:
                continue
            try:
                self._cache_put((text, voice), synth_pcm(self._kokoro, text, voice))
                n += 1
            except Exception as e:  # noqa: BLE001 - a bad prewarm phrase must never block startup
                logger.warning(f"TTS prewarm skipped {text!r} ({type(e).__name__}: {e})")
        return n

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
        cached = self._cache_get((text, voice))
        if cached is not None:
            pcm, rate = cached
            hit = "cache"
        else:
            async with self._sem:
                try:
                    pcm, rate = await asyncio.to_thread(synth_pcm, self._kokoro, text, voice)
                except Exception as e:  # noqa: BLE001 - a synth failure is a 500; the client degrades gracefully
                    logger.error(f"TTS[{room_id}]: synth failed ({type(e).__name__}: {e})")
                    return web.json_response({"error": "tts failed"}, status=500)
            self._cache_put((text, voice), (pcm, rate))
            hit = "synth"
        wav = pcm_to_wav(pcm, rate)
        logger.info(f"TTS[{room_id}]: {len(text)} chars voice={voice} -> {len(wav)} bytes ({hit})")
        return web.Response(body=wav, content_type="audio/wav", headers={"X-Room-Id": room_id})

    async def tts_stream(self, request: web.Request) -> web.StreamResponse:
        """Stream raw 16-bit mono PCM as Kokoro synthesizes it, sentence by sentence (chunked transfer).

        First-audio lands when sentence 1 finishes synthesizing, not the whole reply — this is the Pi's
        biggest latency lever (token→audio: full-reply → first-sentence). Rate rides in X-Sample-Rate; the
        client resamples to its device rate. /tts (whole WAV) stays for back-compat / non-streaming callers.
        """
        if not self._authorized(request):
            return web.json_response({"error": "unauthorized"}, status=401)
        room_id = request.headers.get("X-Room-Id") or "default"
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "bad json"}, status=400)
        text = (body.get("text") or "").strip()
        voice = body.get("voice") or self._default_voice
        if not text:
            return web.json_response({"error": "empty text"}, status=400)

        # Cache hit (a repeated fixed phrase): the whole PCM is ready → write it at once. First-audio is
        # instant (no Kokoro call) — this is the #4 pre-render win for the error apologies etc.
        cached = self._cache_get((text, voice))
        if cached is not None:
            pcm, rate = cached
            resp = web.StreamResponse(
                status=200,
                headers={"X-Room-Id": room_id, "X-Sample-Rate": str(rate),
                         "Content-Type": "application/octet-stream"},
            )
            await resp.prepare(request)
            await resp.write(pcm)
            await resp.write_eof()
            logger.info(f"TTS[{room_id}] stream: {len(text)} chars voice={voice} -> cache ({len(pcm)}B @ {rate}Hz)")
            return resp

        # Miss: kokoro-onnx's create_stream() does NOT actually stream (it yields the whole reply as one
        # chunk at the end — verified). So we chunk ourselves: split into sentences, synth one at a time,
        # flush each as it finishes (first-audio at first-sentence ~0.3s, not full-reply ~1.1s+), and
        # accumulate the full PCM to cache so a repeat of this exact text is instant next time.
        sentences = _split_sentences(text)
        chunks = 0
        rate = 24000
        collected = bytearray()
        ok = True
        resp: web.StreamResponse | None = None
        async with self._sem:
            try:
                for sent in sentences:
                    samples, sr = await asyncio.to_thread(self._kokoro.create, sent, voice)
                    pcm = _f32_to_pcm16(samples)
                    collected += pcm
                    if resp is None:  # first sentence → now we know the rate; open the streamed response
                        rate = sr
                        resp = web.StreamResponse(
                            status=200,
                            headers={"X-Room-Id": room_id, "X-Sample-Rate": str(rate),
                                     "Content-Type": "application/octet-stream"},
                        )
                        await resp.prepare(request)
                    await resp.write(pcm); chunks += 1
            except Exception as e:  # noqa: BLE001
                ok = False
                if resp is None:  # failed before any audio → clean 500 (headers not yet sent)
                    logger.error(f"TTS[{room_id}] stream: synth failed ({type(e).__name__}: {e})")
                    return web.json_response({"error": "tts failed"}, status=500)
                logger.error(f"TTS[{room_id}] stream: failed mid-synth after {chunks} chunks ({type(e).__name__}: {e})")
            if resp is None:  # no sentences produced audio → empty 200
                resp = web.StreamResponse(
                    status=200,
                    headers={"X-Room-Id": room_id, "X-Sample-Rate": str(rate),
                             "Content-Type": "application/octet-stream"},
                )
                await resp.prepare(request)
            await resp.write_eof()
        if ok and collected:  # cache only a fully-synthesized reply (never a partial)
            self._cache_put((text, voice), (bytes(collected), rate))
        logger.info(f"TTS[{room_id}] stream: {len(text)} chars voice={voice} -> {chunks} sentence-chunks @ {rate}Hz")
        return resp


def _bind_is_loopback(host: str) -> bool:
    """True if ``host`` is a loopback bind (safe to run without a token). See the STT twin: ``0.0.0.0``/
    ``::`` bind every interface and are never loopback, and an unparseable bind is treated as
    non-loopback so it cannot claim the no-token exemption."""
    import ipaddress

    h = (host or "").strip().lower()
    if h in ("localhost", ""):
        return h == "localhost"
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def build_app(kokoro, *, token: str = "", default_voice: str = "af_heart", max_concurrency: int = 1,
              cache_size: int = 64, prewarm_phrases=()) -> web.Application:
    """Build the aiohttp app. `kokoro` is injectable so tests pass a fake (no model load)."""
    handlers = TtsApp(kokoro, token=token, default_voice=default_voice,
                      max_concurrency=max_concurrency, cache_size=cache_size)
    if prewarm_phrases:
        warmed = handlers.prewarm(prewarm_phrases, default_voice)
        logger.info(f"TTS service: prewarmed {warmed} fixed phrase(s) into the synth cache")
    app = web.Application()
    app.add_routes([
        web.get("/health", handlers.health),
        web.post("/tts", handlers.tts),
        web.post("/tts/stream", handlers.tts_stream),
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
    cache_size = int(os.environ.get("TTS_CACHE_SIZE", "64"))
    # Optional: phrases to synth into the cache at startup (so even the FIRST occurrence is instant — e.g.
    # the brain-down/error apologies). '|'-separated. Unset = cache-on-first-use only (default behavior).
    prewarm = [p for p in (os.environ.get("TTS_PREWARM_PHRASES", "") or "").split("|") if p.strip()]
    # See the STT twin: a warning here is indistinguishable from a secured deploy, so refuse instead.
    if not _bind_is_loopback(args.host) and not token.strip():
        if os.environ.get("TTS_SERVICE_ALLOW_OPEN", "") in ("1", "true", "True"):
            logger.warning(
                f"TTS service binding {args.host} (non-loopback) with NO auth token — "
                "running OPEN because TTS_SERVICE_ALLOW_OPEN is set. Anyone on the LAN can synthesize."
            )
        else:
            raise SystemExit(
                f"REFUSING to bind {args.host}:{args.port} with no auth token — /tts would be open to "
                "the whole LAN. Set TTS_SERVICE_AUTH_TOKEN (recommended), bind 127.0.0.1, or set "
                "TTS_SERVICE_ALLOW_OPEN=1 if an open service is genuinely what you want."
            )

    kokoro = _load_kokoro()
    app = build_app(kokoro, token=token, default_voice=default_voice, max_concurrency=max_concurrency,
                    cache_size=cache_size, prewarm_phrases=prewarm)
    logger.info(f"TTS service listening on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
