"""Tests for the standalone EM STT service (stt_service/server.py).

Drives the real aiohttp app (TestServer/TestClient, no model load — a fake model is injected). Covers:
/health open, /stt bearer-enforced (401 without, 200 with), a valid WAV → text, a bad upload → 400, and
the WAV parser. Mirrors the project's no-network test ethos; the model is faked so there's no Whisper load.
"""

import asyncio
import io
import wave

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from stt_service.server import (
    WhisperCppBackend,
    build_app,
    float32_to_wav,
    wav_to_float32,
)


class _FakeSeg:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    """Stand-in for faster-whisper.WhisperModel — returns canned segments, records the audio it saw."""

    def __init__(self, text="hello there"):
        self._text = text
        self.calls = 0

    def transcribe(self, audio, language=None):
        self.calls += 1
        assert isinstance(audio, np.ndarray)  # the handler hands us a float32 array, not raw bytes
        return ([_FakeSeg(self._text)], None)


def _wav_bytes(seconds=0.1, rate=16000) -> bytes:
    n = int(seconds * rate)
    pcm = (np.zeros(n, dtype=np.int16)).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setsampwidth(2)
        w.setnchannels(1)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _run(coro):
    return asyncio.run(coro)


def test_wav_to_float32_roundtrip():
    audio, sr = wav_to_float32(_wav_bytes(seconds=0.05, rate=16000))
    assert sr == 16000
    assert audio.dtype == np.float32
    assert len(audio) == int(0.05 * 16000)


def test_health_open_and_stt_requires_bearer():
    async def go():
        app = build_app(_FakeModel("play jazz"), token="s3cret")
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/health")).status == 200          # open liveness probe
            r = await client.post("/stt", data=_wav_bytes())            # no token → 401
            assert r.status == 401
            r = await client.post("/stt", data=_wav_bytes(),
                                   headers={"Authorization": "Bearer wrong"})
            assert r.status == 401
            r = await client.post("/stt", data=_wav_bytes(),
                                   headers={"Authorization": "Bearer s3cret"})
            assert r.status == 200
            assert (await r.json())["text"] == "play jazz"
    _run(go())


def test_stt_open_when_no_token_and_bad_audio_is_400():
    async def go():
        app = build_app(_FakeModel(), token="")  # loopback / no-token mode: open
        async with TestClient(TestServer(app)) as client:
            r = await client.post("/stt", data=_wav_bytes())            # no header needed
            assert r.status == 200
            r = await client.post("/stt", data=b"not a wav")            # malformed → 400, not a crash
            assert r.status == 400
            r = await client.post("/stt", data=b"")                     # empty → 400
            assert r.status == 400
    _run(go())


def test_room_id_echoed():
    async def go():
        app = build_app(_FakeModel("ok"), token="")
        async with TestClient(TestServer(app)) as client:
            r = await client.post("/stt", data=_wav_bytes(), headers={"X-Room-Id": "kitchen"})
            assert (await r.json())["room_id"] == "kitchen"
    _run(go())


def test_float32_wav_roundtrip():
    audio, sr = wav_to_float32(_wav_bytes(seconds=0.05, rate=16000))
    audio2, sr2 = wav_to_float32(float32_to_wav(audio, sr))
    assert sr2 == 16000 and len(audio2) == len(audio)


class _FakeBackend:
    """A ready backend (the build_app `backend=` seam) — async transcribe, records the audio it saw."""

    def __init__(self, text="gpu text"):
        self._text = text
        self.calls = 0

    async def transcribe(self, audio, sample_rate, language):
        self.calls += 1
        assert isinstance(audio, np.ndarray) and sample_rate == 16000
        return self._text

    async def close(self):
        pass


def test_build_app_accepts_a_backend_directly():
    async def go():
        be = _FakeBackend("from gpu")
        app = build_app(backend=be, token="")
        async with TestClient(TestServer(app)) as client:
            r = await client.post("/stt", data=_wav_bytes())
            assert r.status == 200
            assert (await r.json())["text"] == "from gpu"
            assert be.calls == 1
    _run(go())


def test_whispercpp_backend_proxies_to_inference():
    """WhisperCppBackend POSTs a WAV to whisper-server /inference and returns the JSON text."""
    seen = {}

    class _FakeResp:
        def raise_for_status(self): pass
        async def json(self): return {"text": "  hello from whisper.cpp\n"}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _FakeSession:
        closed = False
        def post(self, url, data=None, timeout=None):
            seen["url"] = url
            seen["data"] = data
            return _FakeResp()

    async def go():
        be = WhisperCppBackend("http://127.0.0.1:8079", session=_FakeSession())
        audio, sr = wav_to_float32(_wav_bytes(seconds=0.05))
        text = await be.transcribe(audio, sr, "en")
        assert text == "hello from whisper.cpp"          # stripped
        assert seen["url"].endswith("/inference")
    _run(go())
