# nanowakeword adapter — present a nanowakeword NanoInterpreter through the same tiny interface that
# WakeWordGate already consumes from an openWakeWord Model, so the gate (and all its hardening: escape
# hatch, near-miss diagnostics, heartbeat) is engine-agnostic.
#
# Why nanowakeword: the AEC double-talk clamp drives "Aria" to the music floor over media (see
# project-wake-over-music-rootcause). The durable fix is a model trained to fire *through* the clamp —
# nanowakeword's training takes a dedicated noise/ set (put music in it) plus phonetic adversarial
# negatives, which is exactly that recipe. Inference is local onnxruntime, same audio contract as
# openWakeWord: int16, 16 kHz, 1280-sample (80 ms) frames.
#
# The gate only ever touches two things on the model object:
#   - `.models`            : a dict keyed by model name (gate reads .keys())
#   - `.predict(pcm)`      : returns a {name: score} dict
# NanoInterpreter already exposes `.models` (name → onnxruntime session, same shape), and its
# `predict()` returns a DetectionResult whose `.scores` IS that {name: score} dict — so we share
# `.models` straight through and only unwrap the result.
from __future__ import annotations


class NanoWakeWordModel:
    """Quacks like openWakeWord's Model for WakeWordGate, backed by nanowakeword's NanoInterpreter."""

    def __init__(self, model_paths: "list[str]"):
        from nanowakeword import NanoInterpreter

        # load_model accepts a single path or a list; pass the resolved custom-model path(s).
        self._interp = NanoInterpreter.load_model(model=model_paths)
        # NanoInterpreter.models is {model_name: InferenceSession} — same shape oww exposes, so the
        # gate's `list(model.models.keys())` and `_best()` work unchanged.
        self.models = self._interp.models

    def predict(self, pcm):
        """int16 1280-sample frame → {model_name: score} (DetectionResult.scores)."""
        return self._interp.predict(pcm).scores
