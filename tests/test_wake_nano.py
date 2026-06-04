"""Offline tests for the nanowakeword adapter (wake_nano.NanoWakeWordModel).

NanoInterpreter is mocked so these run with no trained model and no GPU. The adapter's whole job is to
present nanowakeword through the exact interface WakeWordGate consumes from an openWakeWord Model:
a `.models` dict (keyed by name) and `predict(pcm) -> {name: score}`. We verify that shape and that the
gate drives end-to-end on top of it.
"""

import sys
import types

import numpy as np

from wake_word import WakeWordGate


class _FakeDetectionResult:
    """Mimics nanowakeword.DetectionResult: a {name: score} dict exposed as `.scores`."""

    def __init__(self, scores):
        self.scores = scores


class _FakeInterp:
    """Mimics NanoInterpreter: a `.models` dict + predict() returning a DetectionResult."""

    def __init__(self):
        self.models = {"aria": object()}  # name → onnxruntime session (only keys are read)
        self.next_score = 0.0

    def predict(self, pcm):
        return _FakeDetectionResult({"aria": self.next_score})


def _install_fake_nano(interp):
    """Inject a fake `nanowakeword` module so `from nanowakeword import NanoInterpreter` resolves to ours."""
    mod = types.ModuleType("nanowakeword")

    class NanoInterpreter:
        @staticmethod
        def load_model(model=None, **kw):
            interp.loaded_with = model
            return interp

    mod.NanoInterpreter = NanoInterpreter
    sys.modules["nanowakeword"] = mod


def test_adapter_presents_oww_interface():
    interp = _FakeInterp()
    _install_fake_nano(interp)
    from wake_nano import NanoWakeWordModel

    model = NanoWakeWordModel(["wakewords/aria_nano.onnx"])
    # load_model received the resolved path(s)
    assert interp.loaded_with == ["wakewords/aria_nano.onnx"]
    # .models is shared straight through (gate reads .keys())
    assert list(model.models.keys()) == ["aria"]
    # predict unwraps DetectionResult.scores into the {name: score} dict the gate expects
    interp.next_score = 0.73
    out = model.predict(np.zeros(1280, dtype=np.int16))
    assert out == {"aria": 0.73}


def test_gate_fires_on_nano_adapter():
    """The gate's _best/threshold path works unchanged on top of the adapter."""
    interp = _FakeInterp()
    _install_fake_nano(interp)
    from wake_nano import NanoWakeWordModel

    model = NanoWakeWordModel(["wakewords/aria_nano.onnx"])
    gate = WakeWordGate(model, threshold=0.5, media_only=False, session_id="t")
    key, score = gate._best(model.predict(np.zeros(1280, dtype=np.int16)))
    assert key == "aria" and score == 0.0
    interp.next_score = 0.9
    key, score = gate._best(model.predict(np.zeros(1280, dtype=np.int16)))
    assert key == "aria" and score == 0.9  # clears threshold → gate would wake
