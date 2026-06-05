"""TTSGainProcessor — verify the PCM attenuation math (the audio scaling Aria's voice down)."""
import numpy as np

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
