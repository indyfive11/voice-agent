"""TTSGainProcessor — verify the PCM attenuation math (the audio scaling Aria's voice down)."""
import asyncio

import numpy as np
from pipecat.frames.frames import BotStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

import aria_state
from tts_gain import TTSGainProcessor


def _pcm(samples):
    return np.array(samples, dtype=np.int16).tobytes()


def test_gain_halves_amplitude():
    p = TTSGainProcessor(gain=0.5)
    out = np.frombuffer(p._apply(_pcm([100, -200, 30000, -30000])), dtype=np.int16)
    assert list(out) == [50, -100, 15000, -15000]


def test_gain_length_preserved():
    p = TTSGainProcessor(gain=0.6)
    src = _pcm(list(range(-500, 500)))
    assert len(p._apply(src)) == len(src)  # same byte length → num_frames unchanged


def test_gain_clamped_to_unit_range():
    assert TTSGainProcessor(gain=2.0)._gain == 1.0   # no amplification
    assert TTSGainProcessor(gain=-1.0)._gain == 0.0  # no negative
    # gain=1.0 is a no-op pass-through (handled in process_frame; _apply would be identity-ish)
    p = TTSGainProcessor(gain=1.0)
    out = np.frombuffer(p._apply(_pcm([123, -456])), dtype=np.int16)
    assert list(out) == [123, -456]


def test_gain_no_overflow_at_extremes():
    # full-scale input at gain<1 must stay in range (no int16 wrap)
    p = TTSGainProcessor(gain=0.9)
    out = np.frombuffer(p._apply(_pcm([32767, -32768])), dtype=np.int16)
    assert out.min() >= -32768 and out.max() <= 32767


def test_bot_stopped_returns_to_resting_state(monkeypatch):
    # The clobber bug: a goodbye spoken on the way to sleep set the eye `off`, then this BotStopped
    # handler overwrote it with a hardcoded `idle` — so sleep never showed. It must honor the resting
    # state the wake gate recorded instead.
    writes = []
    monkeypatch.setattr(aria_state, "set_state", lambda s, level=0.0: writes.append(s))
    monkeypatch.setattr(aria_state, "_resting_state", "off")  # wake gate set rest=off (asleep)
    p = TTSGainProcessor(gain=0.6)
    asyncio.run(p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    assert writes == ["off"]  # NOT idle — the goodbye no longer clobbers the sleep state


def test_bot_stopped_rests_on_idle_when_awake(monkeypatch):
    writes = []
    monkeypatch.setattr(aria_state, "set_state", lambda s, level=0.0: writes.append(s))
    monkeypatch.setattr(aria_state, "_resting_state", "idle")  # awake → normal return-to-idle
    p = TTSGainProcessor(gain=0.6)
    asyncio.run(p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    assert writes == ["idle"]
