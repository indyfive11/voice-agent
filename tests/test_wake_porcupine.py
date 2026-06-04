"""Offline tests for the Porcupine adapter (wake_porcupine.PorcupineModel).

pvporcupine is mocked so these run with no access key and no native library. The adapter's job is to hide
two Porcupine quirks behind openWakeWord's Model interface: (1) binary detect → {name: 1.0/0.0}, and
(2) fixed 512-sample frames vs the gate's 1280-sample frames (internal re-chunking with carry).
"""

import sys
import types

import numpy as np

from wake_word import WakeWordGate


class _FakePV:
    """Mimics a Porcupine handle: fixed frame_length, process() returns 0 on a 'detect' frame else -1."""

    def __init__(self, frame_length=512):
        self.frame_length = frame_length
        self.sample_rate = 16000
        self._hits = set()  # indices (within a process-call sequence) that should 'detect'
        self._n = 0

    def process(self, frame):
        assert len(frame) == self.frame_length  # Porcupine requires an exact frame
        idx = self._n
        self._n += 1
        return 0 if idx in self._hits else -1


def _install_fake_pv(handle):
    mod = types.ModuleType("pvporcupine")

    def create(access_key=None, keyword_paths=None, sensitivities=None, **kw):
        handle.created_with = dict(access_key=access_key, keyword_paths=keyword_paths, sensitivities=sensitivities)
        return handle

    mod.create = create
    sys.modules["pvporcupine"] = mod


def test_requires_access_key():
    _install_fake_pv(_FakePV())
    from wake_porcupine import PorcupineModel

    try:
        PorcupineModel(["hey-aria.ppn"], access_key="", sensitivity=0.7, key="aria")
        assert False, "expected ValueError on missing access key"
    except ValueError:
        pass


def test_rechunks_and_reports_binary():
    handle = _FakePV(frame_length=512)
    _install_fake_pv(handle)
    from wake_porcupine import PorcupineModel

    model = PorcupineModel(["hey-aria.ppn"], access_key="ak-test", sensitivity=0.6, key="aria")
    assert handle.created_with["sensitivities"] == [0.6]
    assert list(model.models.keys()) == ["aria"]

    # No detections → 0.0. A 1280-sample frame yields two full 512-frames (carry 256).
    out = model.predict(np.zeros(1280, dtype=np.int16))
    assert out == {"aria": 0.0}

    # Arrange a hit on the next process-call → 1.0 for the frame it lands in.
    handle._hits = {handle._n}  # next call detects
    out = model.predict(np.zeros(1280, dtype=np.int16))
    assert out == {"aria": 1.0}


def test_gate_wakes_on_porcupine_hit():
    handle = _FakePV(frame_length=512)
    _install_fake_pv(handle)
    from wake_porcupine import PorcupineModel

    model = PorcupineModel(["hey-aria.ppn"], access_key="ak", sensitivity=0.7, key="aria")
    gate = WakeWordGate(model, threshold=0.5, media_only=False, session_id="t")
    key, score = gate._best(model.predict(np.zeros(1280, dtype=np.int16)))
    assert key == "aria" and score == 0.0
    handle._hits = {handle._n}
    key, score = gate._best(model.predict(np.zeros(1280, dtype=np.int16)))
    assert score == 1.0  # >= threshold → gate would wake
