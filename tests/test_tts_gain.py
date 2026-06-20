"""TTSGainProcessor — verify the PCM attenuation math (the audio scaling Aria's voice down)."""
import asyncio

import numpy as np
import pytest
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

import aria_state
import bot_speech
from tts_gain import TTSGainProcessor, VoiceVolumeFrame


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
    # The clobber bug: a goodbye spoken on the way to sleep set the eye to the sleep state, then this
    # BotStopped handler overwrote it with a hardcoded `idle` — so sleep never showed. It must honor the
    # resting state the wake gate recorded instead.
    writes = []
    monkeypatch.setattr(aria_state, "set_state", lambda s, level=0.0: writes.append(s))
    monkeypatch.setattr(aria_state, "_resting_state", "sleeping")  # wake gate set rest=sleeping (asleep)
    p = TTSGainProcessor(gain=0.6)
    asyncio.run(p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    assert writes == ["sleeping"]  # NOT idle — the goodbye no longer clobbers the sleep state


def test_bot_stopped_rests_on_idle_when_awake(monkeypatch):
    writes = []
    monkeypatch.setattr(aria_state, "set_state", lambda s, level=0.0: writes.append(s))
    monkeypatch.setattr(aria_state, "_resting_state", "idle")  # awake → normal return-to-idle
    p = TTSGainProcessor(gain=0.6)
    asyncio.run(p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    assert writes == ["idle"]


def test_bot_started_sets_bot_speaking(monkeypatch):
    # BotStarted flips the live flag the /media/state poll carries to the brain's duck watchdog.
    monkeypatch.setattr(aria_state, "set_state", lambda s, level=0.0: None)
    bot_speech.set_bot_speaking(False)
    p = TTSGainProcessor(gain=0.6)
    asyncio.run(p.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    assert bot_speech.bot_speaking() is True


def test_bot_stopped_clears_bot_speaking(monkeypatch):
    monkeypatch.setattr(aria_state, "set_state", lambda s, level=0.0: None)
    bot_speech.set_bot_speaking(True)
    p = TTSGainProcessor(gain=0.6)
    asyncio.run(p.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    assert bot_speech.bot_speaking() is False


# --- F3: runtime voice-volume control --------------------------------------------------------------
def test_voice_volume_up_steps_gain():
    p = TTSGainProcessor(gain=0.5, step=0.1)
    p._adjust_gain("up", None)
    assert p._gain == pytest.approx(0.6)


def test_voice_volume_up_caps_at_one():
    p = TTSGainProcessor(gain=0.95, step=0.1)
    p._adjust_gain("up", None)
    assert p._gain == 1.0  # no boost above full


def test_voice_volume_down_steps_gain():
    p = TTSGainProcessor(gain=0.5, step=0.1)
    p._adjust_gain("down", None)
    assert p._gain == pytest.approx(0.4)


def test_voice_volume_down_floors_above_inaudible():
    p = TTSGainProcessor(gain=0.15, step=0.1, floor=0.1)
    p._adjust_gain("down", None)
    assert p._gain == pytest.approx(0.1)  # floored — repeated "quieter" can't reach inaudible


def test_voice_volume_set_absolute_clamped():
    p = TTSGainProcessor(gain=0.5)
    p._adjust_gain("set", 0.3)
    assert p._gain == pytest.approx(0.3)
    p._adjust_gain("set", 2.0)
    assert p._gain == 1.0  # clamped to full
    p._adjust_gain("set", -1.0)
    assert p._gain == 0.0  # explicit silence allowed via set


def test_voice_volume_set_without_value_is_noop():
    p = TTSGainProcessor(gain=0.5)
    p._adjust_gain("set", None)
    assert p._gain == pytest.approx(0.5)  # unchanged


def test_voice_volume_frame_adjusts_live_gain():
    p = TTSGainProcessor(gain=0.5, step=0.1)
    asyncio.run(p.process_frame(VoiceVolumeFrame(op="down", value=None), FrameDirection.DOWNSTREAM))
    assert p._gain == pytest.approx(0.4)
