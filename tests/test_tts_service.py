"""Tests for the standalone EM TTS service (tts_service/server.py).

Drives the real aiohttp app (TestServer/TestClient) with a fake Kokoro (no model load). Covers /health open,
/tts bearer-enforced, a valid request → a WAV, bad/empty input → 400, and the synth helper.
"""

import asyncio
import io
import wave

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from tts_service.server import _split_sentences, build_app, synth_pcm, synth_wav


class _FakeKokoro:
    """Stand-in for kokoro_onnx.Kokoro — canned float samples at 24kHz, records calls; create + create_stream."""

    def __init__(self, stream_chunks=2, chunk_frames=1200):
        self.calls = []
        self._stream_chunks = stream_chunks
        self._chunk_frames = chunk_frames

    def create(self, text, voice="af_heart", speed=1.0, lang="en-us"):
        self.calls.append((text, voice))
        return np.zeros(2400, dtype=np.float32), 24000

    async def create_stream(self, text, voice="af_heart", speed=1.0, lang="en-us", **kwargs):
        self.calls.append((text, voice))
        for _ in range(self._stream_chunks):
            yield np.zeros(self._chunk_frames, dtype=np.float32), 24000


def _run(coro):
    return asyncio.run(coro)


def test_split_sentences_first_chunk_short_rest_merged():
    # full sentences split; the short first fragment merges forward so we don't synth a 1-word clip
    out = _split_sentences("This is the first full sentence. Here is the second full sentence.")
    assert len(out) == 2
    assert out[0].startswith("This is the first")
    # tiny fragments collapse into one chunk (no choppy micro-synths)
    assert _split_sentences("Hi. Ok. Go.") == ["Hi. Ok. Go."]
    assert _split_sentences("") == []


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


def test_tts_stream_returns_chunked_pcm_per_sentence():
    async def go():
        fake = _FakeKokoro()  # create() returns 2400 frames per call regardless of text
        app = build_app(fake, token="")
        async with TestClient(TestServer(app)) as client:
            # three full-length sentences (each > the merge floor) → three per-sentence synth calls
            text = ("This is the very first sentence here. "
                    "Here is the second full sentence now. "
                    "And the third sentence follows after.")
            r = await client.post("/tts/stream", json={"text": text, "voice": "af_heart"},
                                  headers={"X-Room-Id": "den"})
            assert r.status == 200
            assert r.headers.get("X-Sample-Rate") == "24000"
            assert r.headers.get("X-Room-Id") == "den"
            data = await r.read()
            assert len(data) == 3 * 2400 * 2          # 3 sentences * 2400 frames * 2 bytes
            assert len(fake.calls) == 3               # synthesized one sentence at a time
    _run(go())


def test_synth_cache_hit_skips_kokoro_on_repeat():
    async def go():
        fake = _FakeKokoro()
        app = build_app(fake, token="", cache_size=8)
        async with TestClient(TestServer(app)) as client:
            r1 = await client.post("/tts", json={"text": "Sorry, I hit a problem."})
            assert r1.status == 200
            assert len(fake.calls) == 1                       # first time → synth
            r2 = await client.post("/tts", json={"text": "Sorry, I hit a problem."})
            assert r2.status == 200
            assert len(fake.calls) == 1                       # repeat → cache hit, no new synth
            assert (await r1.read()) == (await r2.read())     # identical audio
    _run(go())


def test_cache_size_zero_disables_cache():
    async def go():
        fake = _FakeKokoro()
        app = build_app(fake, token="", cache_size=0)
        async with TestClient(TestServer(app)) as client:
            await client.post("/tts", json={"text": "repeat me please now."})
            await client.post("/tts", json={"text": "repeat me please now."})
            assert len(fake.calls) == 2                       # no caching → synth both times
    _run(go())


def test_stream_then_cached_replay_is_instant_path():
    async def go():
        fake = _FakeKokoro()
        app = build_app(fake, token="", cache_size=8)
        async with TestClient(TestServer(app)) as client:
            text = "This is one full sentence. Here is a second full sentence."
            r1 = await client.post("/tts/stream", json={"text": text})
            assert r1.status == 200
            d1 = await r1.read()
            n_after_first = len(fake.calls)                   # 2 sentence synths on the miss
            assert n_after_first == 2
            r2 = await client.post("/tts/stream", json={"text": text})
            d2 = await r2.read()
            assert len(fake.calls) == 2                       # cache hit → no new synth
            assert d1 == d2                                   # same bytes, served from cache
    _run(go())


def test_prewarm_makes_first_request_a_cache_hit():
    async def go():
        fake = _FakeKokoro()
        app = build_app(fake, token="", cache_size=8,
                        prewarm_phrases=["Sorry, I lost my connection just then."])
        async with TestClient(TestServer(app)) as client:
            calls_after_prewarm = len(fake.calls)             # 1 synth at startup
            assert calls_after_prewarm == 1
            r = await client.post("/tts", json={"text": "Sorry, I lost my connection just then."})
            assert r.status == 200
            assert len(fake.calls) == 1                       # served from the prewarmed cache, no new synth
    _run(go())


def test_synth_pcm_returns_pcm_and_rate():
    pcm, rate = synth_pcm(_FakeKokoro(), "hi", "af_heart")
    assert rate == 24000 and len(pcm) == 2400 * 2


def test_tts_stream_bearer_and_empty_text():
    async def go():
        app = build_app(_FakeKokoro(), token="s3cret")
        async with TestClient(TestServer(app)) as client:
            assert (await client.post("/tts/stream", json={"text": "hi"})).status == 401  # no token
            ok = await client.post("/tts/stream", json={"text": "hi"},
                                   headers={"Authorization": "Bearer s3cret"})
            assert ok.status == 200
            empty = await client.post("/tts/stream", json={"text": ""},
                                      headers={"Authorization": "Bearer s3cret"})
            assert empty.status == 400
    _run(go())
