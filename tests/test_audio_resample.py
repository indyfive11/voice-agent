"""Offline tests for the mic input resampler — no audio device needed.

Guards the AEC-Mic fix: the PipeWire echo-cancel source only opens at 48 kHz, so the transport
captures at 48 kHz and InputResampler must normalize to 16 kHz before STT/VAD (which assume 16 kHz).
asyncio.run drives the async bodies so plain pytest works (no pytest-asyncio).
"""

import asyncio

import numpy as np

from pipecat.frames.frames import InputAudioRawFrame, TextFrame

from audio_resample import InputResampler


def _tone_pcm(rate, secs=0.1, hz=440):
    t = np.linspace(0, secs, int(rate * secs), endpoint=False)
    return (np.sin(2 * np.pi * hz * t) * 10000).astype(np.int16).tobytes()


def test_downsamples_48k_to_16k_and_rewrites_metadata():
    r = InputResampler(target_rate=16000)
    pcm = _tone_pcm(48000, secs=0.5)  # 24000 samples
    f = InputAudioRawFrame(audio=pcm, sample_rate=48000, num_channels=1)
    asyncio.run(r._maybe_resample(f))

    assert f.sample_rate == 16000
    out_samples = len(f.audio) // 2
    # 48k→16k is 3:1; allow for the stream resampler's small filter-warmup deficit on the run.
    assert 7000 <= out_samples <= 8000, out_samples
    assert f.num_frames == out_samples


def test_noop_when_rate_already_matches():
    r = InputResampler(target_rate=16000)
    pcm = _tone_pcm(16000)
    f = InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
    asyncio.run(r._maybe_resample(f))
    assert f.sample_rate == 16000
    assert f.audio == pcm  # untouched


def test_ignores_non_audio_frames():
    r = InputResampler(target_rate=16000)
    f = TextFrame("hello")
    asyncio.run(r._maybe_resample(f))  # must not raise / mutate


def test_stereo_passes_through_unresampled():
    # Guard path: we never request stereo capture, but a 2ch frame must not crash the mono resampler.
    r = InputResampler(target_rate=16000)
    pcm = _tone_pcm(48000)
    f = InputAudioRawFrame(audio=pcm, sample_rate=48000, num_channels=2)
    asyncio.run(r._maybe_resample(f))
    assert f.sample_rate == 48000  # left as-is
    assert f.audio == pcm
