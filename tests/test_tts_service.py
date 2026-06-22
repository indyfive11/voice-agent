"""Tests for the standalone EM TTS service (tts_service/server.py).

Drives the real aiohttp app (TestServer/TestClient) with a fake Kokoro (no model load). Covers /health open,
/tts bearer-enforced, a valid request → a WAV, bad/empty input → 400, and the synth helper.
"""

import asyncio
import io
import wave

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from tts_service.server import build_app, synth_wav


class _FakeKokoro:
    """Stand-in for kokoro_onnx.Kokoro — returns canned float samples at 24kHz, records calls."""

    def __init__(self):
        self.calls = []

    def create(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        self.calls.append((text, voice))
        return np.zeros(2400, dtype=np.float32), 24000


def _run(coro):
    return asyncio.run(coro)


def test_synth_wav_produces_valid_wav():
    wav = synth_wav(_FakeKokoro(), "hello", "af_heart")
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 2400


def test_health_open_and_tts_requires_bearer():
    async def go():
        fake = _FakeKokoro()
        app = build_app(fake, token="s3cret")
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/health")).status == 200            # open probe
            r = await client.post("/tts", json={"text": "hi"})            # no token → 401
            assert r.status == 401
            r = await client.post("/tts", json={"text": "hi"},
                                   headers={"Authorization": "Bearer wrong"})
            assert r.status == 401
            r = await client.post("/tts", json={"text": "play jazz"},
                                   headers={"Authorization": "Bearer s3cret"})
            assert r.status == 200
            assert r.headers["Content-Type"] == "audio/wav"
            wav = await r.read()
            with wave.open(io.BytesIO(wav), "rb") as w:
                assert w.getframerate() == 24000
            assert fake.calls and fake.calls[-1][0] == "play jazz"
    _run(go())


def test_tts_open_when_no_token_and_bad_input_is_400():
    async def go():
        app = build_app(_FakeKokoro(), token="")  # loopback / no-token mode: open
        async with TestClient(TestServer(app)) as client:
            assert (await client.post("/tts", json={"text": "ok"})).status == 200  # no header needed
            assert (await client.post("/tts", json={})).status == 400              # empty text → 400
            assert (await client.post("/tts", data=b"not json")).status == 400     # malformed → 400
    _run(go())


def test_voice_and_room_passthrough():
    async def go():
        fake = _FakeKokoro()
        app = build_app(fake, token="", default_voice="af_heart")
        async with TestClient(TestServer(app)) as client:
            r = await client.post("/tts", json={"text": "hi", "voice": "am_michael"},
                                  headers={"X-Room-Id": "kitchen"})
            assert r.status == 200 and r.headers.get("X-Room-Id") == "kitchen"
            assert fake.calls[-1][1] == "am_michael"  # requested voice used
    _run(go())
