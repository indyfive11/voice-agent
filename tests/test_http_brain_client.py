"""Tests for HttpBrainClient against an in-process ASGI stub (no gabagent, no network).

Uses httpx.ASGITransport (built into httpx) to drive the client over a tiny raw-ASGI app that
emits SSE the way gabagent's voice server does: `data: {json}\\n\\n` frames, /respond ending after
a confirm, /confirm returning the continuation as a fresh SSE, /cancel and /health as JSON.
"""

import asyncio
import json

import httpx

from brains.brain_client import BrainEvent
from brains.http_brain_client import HttpBrainClient


# --- a minimal gabagent-voice-server stub --------------------------------------------------
def make_stub():
    """Returns (asgi_app, calls) where calls records posted bodies per path."""
    calls = {"respond": [], "confirm": [], "cancel": []}

    def sse(*events: dict) -> bytes:
        return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        path = scope["path"]
        # drain request body
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        payload = json.loads(body) if body else {}

        async def json_resp(obj, status=200):
            data = json.dumps(obj).encode()
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": data})

        async def sse_resp(blob: bytes):
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body": blob})

        if path == "/health":
            await json_resp({"status": "ok", "mode": "voice"})
        elif path == "/respond":
            calls["respond"].append(payload)
            # token, token, then a confirm that ENDS the stream (no done).
            await sse_resp(sse(
                {"type": "token", "text": "Working on it. "},
                {"type": "confirm", "id": "c1", "tier": 2, "method": "spoken_yesno",
                 "summary": "edit the README"},
            ))
        elif path == "/confirm":
            calls["confirm"].append(payload)
            await sse_resp(sse(
                {"type": "token", "text": "Edited the README."},
                {"type": "done"},
            ))
        elif path == "/cancel":
            calls["cancel"].append(payload)
            await json_resp({"ok": True})
        else:
            await json_resp({"error": "not found"}, status=404)

    return app, calls


def _client(app):
    return HttpBrainClient("http://brain.test", transport=httpx.ASGITransport(app=app))


def _collect(agen_factory):
    async def go():
        out = []
        async for ev in agen_factory():
            out.append(ev)
        return out
    return asyncio.run(go())


def test_respond_streams_events_and_ends_on_confirm():
    app, calls = make_stub()
    client = _client(app)
    evs = _collect(lambda: client.respond("s1", "update the readme"))
    assert [e.type for e in evs] == ["token", "confirm"]
    assert evs[0].text == "Working on it. "
    assert evs[1].id == "c1" and evs[1].method == "spoken_yesno" and evs[1].tier == 2
    assert calls["respond"] == [{"session_id": "s1", "text": "update the readme", "override_token": None}]
    asyncio.run(client.aclose())


def test_respond_omits_wake_when_absent():
    # Item C: no wake signal → payload is exactly the pre-item-C shape (back-compat).
    app, calls = make_stub()
    client = _client(app)
    _collect(lambda: client.respond("s1", "play jazz"))
    assert "wake" not in calls["respond"][0]
    asyncio.run(client.aclose())


def test_respond_includes_wake_when_present():
    # Item C: the out-of-band acoustic wake signal rides the /respond POST body.
    app, calls = make_stub()
    client = _client(app)
    wake = {"bare_wake_likelihood": 0.81, "confidence": 0.99, "speech_dur_ms": 600, "post_wake_voiced_ms": 120}
    _collect(lambda: client.respond("s1", "how are you", wake=wake))
    assert calls["respond"][0]["wake"] == wake
    assert calls["respond"][0]["text"] == "how are you"
    asyncio.run(client.aclose())


def test_confirm_returns_continuation_sse():
    app, calls = make_stub()
    client = _client(app)
    evs = _collect(lambda: client.confirm("s1", "c1", True))
    assert [e.type for e in evs] == ["token", "done"]
    assert evs[0].text == "Edited the README."
    assert calls["confirm"] == [{"session_id": "s1", "id": "c1", "approved": True, "passphrase": None}]
    asyncio.run(client.aclose())


def test_field_omission_parsing_is_safe():
    # gabagent omits empty/zero fields; BrainEvent(**json) must tolerate missing keys + extras.
    app, _ = make_stub()
    client = _client(app)
    evs = _collect(lambda: client.respond("s1", "x"))
    assert all(isinstance(e, BrainEvent) for e in evs)
    assert evs[0].id is None and evs[0].reason is None  # absent fields default to None
    asyncio.run(client.aclose())


def test_cancel_posts_to_cancel_endpoint():
    app, calls = make_stub()
    client = _client(app)
    _collect(lambda: client.respond("s9", "hi"))  # sets active session
    asyncio.run(client.cancel("s9"))
    assert calls["cancel"] == [{"session_id": "s9"}]
    asyncio.run(client.aclose())


def test_start_attaches_when_brain_already_healthy():
    # A launch cmd that would fail if executed — proves connect-or-spawn ATTACHES, not spawns.
    app, _ = make_stub()
    client = HttpBrainClient(
        "http://brain.test",
        launch=["/nonexistent/gab", "--voice-serve"],
        transport=httpx.ASGITransport(app=app),
    )
    asyncio.run(client.start())
    assert client._proc is None  # attached to the existing /health, never spawned
    asyncio.run(client.aclose())


def test_aclose_cancels_active_session():
    app, calls = make_stub()
    client = _client(app)
    _collect(lambda: client.respond("s9", "hi"))
    asyncio.run(client.aclose())
    assert calls["cancel"] == [{"session_id": "s9"}]  # aclose cancels the active turn
