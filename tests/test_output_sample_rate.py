"""Tests for the explicit per-device output sample-rate pin (AUDIO_OUTPUT_SAMPLE_RATE).

A fixed-rate output device (the Pi's EMEET = 48 kHz-only) needs Kokoro's emit-rate AND the transport's
open-rate pinned to the SAME value, or playback chipmunks (the earlier auto-probe bug). The fix is opt-in:
unset (EM) ⇒ neither is touched ⇒ the historical known-good path. These tests assert exactly that — set ⇒
the kwarg is forwarded to both factories; unset ⇒ it's omitted. (The live "does it sound right" check is by
ear on the Pi; unit tests can't hear 1.84×.)

Both factories import their pipecat classes lazily inside the function, so we patch the class on its module
and the in-function `from … import …` picks up the fake at call time — no Kokoro model load, no PyAudio.
"""

from __future__ import annotations

import pipecat.services.kokoro.tts as kokoro_mod
import pipecat.transports.local.audio as audio_mod

import config


def _fake_kokoro(capture: dict):
    real_settings = kokoro_mod.KokoroTTSService.Settings

    class _FakeKokoro:
        Settings = real_settings  # build_tts still builds a real Settings(voice=…)

        def __init__(self, **kwargs):
            capture.clear()
            capture.update(kwargs)

    return _FakeKokoro


def test_build_tts_pins_kokoro_sample_rate_when_env_set(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr(kokoro_mod, "KokoroTTSService", _fake_kokoro(cap))
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")
    monkeypatch.setenv("AUDIO_OUTPUT_SAMPLE_RATE", "48000")
    config.build_tts()
    # The lever GA flagged: sample_rate must reach the constructor (→ base TTSService _init_sample_rate).
    assert cap.get("sample_rate") == 48000


def test_build_tts_omits_sample_rate_when_env_unset(monkeypatch):
    cap: dict = {}
    monkeypatch.setattr(kokoro_mod, "KokoroTTSService", _fake_kokoro(cap))
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")
    monkeypatch.delenv("AUDIO_OUTPUT_SAMPLE_RATE", raising=False)
    config.build_tts()
    # EM-safety: no sample_rate kwarg at all → Kokoro keeps its default emit-rate.
    assert "sample_rate" not in cap


def _patch_transport(monkeypatch, capture: dict):
    class _FakeParams:
        def __init__(self, **kwargs):
            capture.clear()
            capture.update(kwargs)

    class _FakeTransport:
        def __init__(self, params):
            self.params = params

    monkeypatch.setattr(audio_mod, "LocalAudioTransportParams", _FakeParams)
    monkeypatch.setattr(audio_mod, "LocalAudioTransport", _FakeTransport)
    # Stub the PyAudio-backed helpers so the factory runs with no hardware.
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: None)
    monkeypatch.setattr(config, "_readback_device", lambda *a, **k: None)
    monkeypatch.setattr(config, "_supported_input_rate", lambda *a, **k: config.PIPELINE_AUDIO_RATE)
    for k in ("AUDIO_INPUT_DEVICE_INDEX", "AUDIO_OUTPUT_DEVICE_INDEX"):
        monkeypatch.delenv(k, raising=False)


def test_build_transport_pins_output_rate_and_channels_when_env_set(monkeypatch):
    cap: dict = {}
    _patch_transport(monkeypatch, cap)
    monkeypatch.setenv("AUDIO_OUTPUT_SAMPLE_RATE", "48000")
    monkeypatch.setenv("AUDIO_OUTPUT_CHANNELS", "2")
    config.build_transport()
    assert cap.get("audio_out_sample_rate") == 48000
    assert cap.get("audio_out_channels") == 2


def test_build_transport_omits_output_rate_when_env_unset(monkeypatch):
    cap: dict = {}
    _patch_transport(monkeypatch, cap)
    monkeypatch.delenv("AUDIO_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.delenv("AUDIO_OUTPUT_CHANNELS", raising=False)
    config.build_transport()
    # EM-safety: the transport gets no output-rate/channels overrides → framework default.
    assert "audio_out_sample_rate" not in cap
    assert "audio_out_channels" not in cap


def test_build_transport_channels_independent_of_rate(monkeypatch):
    # GA verify-point #2: channels must be settable WITHOUT a rate (and vice-versa) so a channel tweak
    # can't drag in a rate surprise.
    cap: dict = {}
    _patch_transport(monkeypatch, cap)
    monkeypatch.delenv("AUDIO_OUTPUT_SAMPLE_RATE", raising=False)
    monkeypatch.setenv("AUDIO_OUTPUT_CHANNELS", "2")
    config.build_transport()
    assert cap.get("audio_out_channels") == 2
    assert "audio_out_sample_rate" not in cap
