"""Tests for RemoteTTSService — TTS offloaded over HTTP (streaming), driven against an in-process ASGI stub.

Mirrors test_remote_stt. The service now consumes POST /tts/stream (chunked raw PCM + X-Sample-Rate): a 200
yields TTSAudioRawFrame(s) at the output rate (resampled from the stream's rate); a failure yields an
ErrorFrame (never raises/stalls); room_id (X-Room-Id) + bearer + the {text, voice} body ride the request.
"""

import asyncio
import json

import httpx

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

from remote_tts import RemoteTTSService


def _pcm(nframes: int = 2400) -> bytes:
    return b"\x00\x00" * nframes  # 16-bit mono silence


def _stub(status: int = 200, pcm: bytes | None = None, rate: int = 24000,
          nchunks: int = 2, record: dict | None = None):
    """ASGI app that streams raw PCM in `nchunks` body parts (simulates sentence-by-sentence delivery)."""
    body = pcm if pcm is not None else _pcm()

    async def app(scope, receive, send):
        if record is not None:
            record["path"] = scope["path"]
            record["headers"] = {k.decode(): v.decode() for k, v in scope.get("headers") or ()}
        buf = b""
        while True:
            msg = await receive()
            buf += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        if record is not None:
            record["body"] = buf
        if status == 200:
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"application/octet-stream"),
                                    (b"x-sample-rate", str(rate).encode())]})
            # split the body across nchunks http.response.body messages
            step = max(1, len(body) // nchunks)
            for i in range(0, len(body), step):
                part = body[i:i + step]
                await send({"type": "http.response.body", "body": part,
                            "more_body": i + step < len(body)})
        else:
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"error":"boom"}'})

    return app


def _svc(app, *, out_rate: int = 16000, **kwargs) -> RemoteTTSService:
    svc = RemoteTTSService("http://tts.test", transport=httpx.ASGITransport(app=app), **kwargs)
    svc._sample_rate = out_rate  # normally set by start(); set directly for the no-pipeline test
    async def _noop(*a, **k):
        return None
    svc.start_tts_usage_metrics = _noop
    svc.stop_ttfb_metrics = _noop
    return svc


def _collect(factory):
    async def go():
        out = []
        async for f in factory():
            out.append(f)
        return out
    return asyncio.run(go())


def test_success_streams_audio_at_output_rate_and_sends_request():
    rec: dict = {}
    svc = _svc(_stub(200, _pcm(2400), rate=24000, nchunks=2, record=rec), out_rate=16000,
               voice="af_heart", room_id="kitchen", auth_token="tok")
    frames = _collect(lambda: svc.run_tts("hello there", "ctx1"))
    audio = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio and audio[0].sample_rate == 16000  # resampled from the stream's 24k to our output rate
    assert rec["path"] == "/tts/stream"
    assert rec["headers"].get("authorization") == "Bearer tok"
    assert rec["headers"].get("x-room-id") == "kitchen"
    body = json.loads(rec["body"])
    assert body["text"] == "hello there" and body["voice"] == "af_heart"
    asyncio.run(svc.cleanup())


def test_http_error_yields_errorframe_never_raises():
    svc = _svc(_stub(500), out_rate=16000, room_id="kitchen")
    frames = _collect(lambda: svc.run_tts("x", "ctx"))
    assert any(isinstance(f, ErrorFrame) for f in frames)
    asyncio.run(svc.cleanup())


def test_no_token_omits_authorization_header():
    rec: dict = {}
    svc = _svc(_stub(200, _pcm(), record=rec), out_rate=16000, room_id="den")  # no auth_token
    _collect(lambda: svc.run_tts("hi", "ctx"))
    assert "authorization" not in rec["headers"]
    assert rec["headers"].get("x-room-id") == "den"
    asyncio.run(svc.cleanup())


def test_odd_byte_chunk_boundaries_dont_corrupt_or_crash():
    # PCM is 16-bit; if the transport splits mid-sample the client must carry the odd byte, not crash.
    rec: dict = {}
    svc = _svc(_stub(200, _pcm(2400), rate=24000, nchunks=7, record=rec), out_rate=16000)
    frames = _collect(lambda: svc.run_tts("a longer sentence here", "ctx"))
    assert any(isinstance(f, TTSAudioRawFrame) for f in frames)
    asyncio.run(svc.cleanup())
