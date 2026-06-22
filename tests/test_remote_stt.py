"""Tests for RemoteSTTService — STT offloaded over HTTP, driven against an in-process ASGI stub.

Mirrors the brain-client test style (httpx.ASGITransport, no network). The key behaviors: a 200 with text
yields ONE TranscriptionFrame; any failure (HTTP error / network) yields an ErrorFrame (never raises, never
stalls); empty text yields nothing; and the room_id (X-Room-Id) + bearer headers ride every request.
"""

import asyncio
import json

import httpx

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame

from remote_stt import RemoteSTTService


def _stub(status: int = 200, text: str = "hello world", record: dict | None = None):
    async def app(scope, receive, send):
        assert scope["type"] == "http"
        if record is not None:
            record["path"] = scope["path"]
            record["headers"] = {k.decode(): v.decode() for k, v in scope.get("headers") or ()}
        while True:  # drain body
            msg = await receive()
            if not msg.get("more_body"):
                break
        body = json.dumps({"text": text} if status == 200 else {"error": "boom"}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
    return app


def _svc(app, **kwargs) -> RemoteSTTService:
    svc = RemoteSTTService("http://stt.test", transport=httpx.ASGITransport(app=app), **kwargs)
    async def _noop(*a, **k):  # metrics need a running pipeline; stub them out (like test_brain_llm)
        return None
    svc.start_processing_metrics = _noop
    svc.stop_processing_metrics = _noop
    return svc


def _collect(factory):
    async def go():
        out = []
        async for f in factory():
            out.append(f)
        return out
    return asyncio.run(go())


def test_success_yields_one_transcription_and_sends_headers():
    rec: dict = {}
    svc = _svc(_stub(200, "play some jazz", rec), room_id="kitchen", auth_token="tok")
    frames = _collect(lambda: svc.run_stt(b"RIFF....fake-wav-bytes"))
    assert len(frames) == 1 and isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "play some jazz"
    assert rec["path"] == "/stt"
    assert rec["headers"].get("authorization") == "Bearer tok"      # bearer rides every request
    assert rec["headers"].get("x-room-id") == "kitchen"             # durable room key rides every request
    asyncio.run(svc.cleanup())


def test_http_error_yields_errorframe_never_raises():
    svc = _svc(_stub(500), room_id="kitchen")
    frames = _collect(lambda: svc.run_stt(b"x"))
    assert len(frames) == 1 and isinstance(frames[0], ErrorFrame)  # degrade, never stall the pipeline loop
    asyncio.run(svc.cleanup())


def test_empty_text_yields_nothing():
    svc = _svc(_stub(200, ""), room_id="kitchen")
    assert _collect(lambda: svc.run_stt(b"x")) == []
    asyncio.run(svc.cleanup())


def test_no_token_omits_authorization_header():
    rec: dict = {}
    svc = _svc(_stub(200, "hi", rec), room_id="den")  # no auth_token
    _collect(lambda: svc.run_stt(b"x"))
    assert "authorization" not in rec["headers"]
    assert rec["headers"].get("x-room-id") == "den"
    asyncio.run(svc.cleanup())
