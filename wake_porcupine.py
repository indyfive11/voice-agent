# Picovoice Porcupine adapter — present Porcupine through the same tiny interface WakeWordGate consumes
# from an openWakeWord Model (.models dict + predict(pcm) -> {name: score}), so the gate and all its
# hardening stay engine-agnostic.
#
# Porcupine differs from openWakeWord in two ways the gate doesn't need to know about — both hidden here:
#   1. It's a BINARY detector. process() returns a keyword index (>=0 = detected, -1 = not); there is no
#      graded per-frame score. We surface 1.0 on a hit and 0.0 otherwise, so the gate's `score >= threshold`
#      compare still fires. Consequence: the near-miss diagnostic and the escape hatch (which need a graded
#      sub-threshold score) are inert under Porcupine — which is fine, because Porcupine's robustness over
#      noise is the whole reason we'd use it: it either hears the wake word or it doesn't.
#   2. It consumes fixed frame_length frames (typically 512 samples) at 16 kHz, while the gate feeds
#      1280-sample (80 ms) frames. We buffer internally and run process() on each full sub-frame, carrying
#      the remainder to the next call. 1280 is not a multiple of 512, hence the carry.
#
# Sensitivity (0..1, higher = more eager) is Porcupine's real tuning knob and is set at create() time —
# not per frame. WAKE_WORD_SENSITIVITY maps to it; lean high for hearing over music.
from __future__ import annotations

import numpy as np


class PorcupineModel:
    """Quacks like openWakeWord's Model for WakeWordGate, backed by Picovoice Porcupine."""

    def __init__(self, keyword_paths, *, access_key: str, sensitivity: float = 0.7, key: str = "wake"):
        import pvporcupine

        if not access_key:
            raise ValueError(
                "Porcupine needs an access key — set PORCUPINE_ACCESS_KEY "
                "(free at https://console.picovoice.ai/)."
            )
        kps = list(keyword_paths)
        self._pv = pvporcupine.create(
            access_key=access_key,
            keyword_paths=kps,
            sensitivities=[sensitivity] * len(kps),
        )
        self._frame = self._pv.frame_length  # samples per process() call (e.g. 512)
        self._buf = np.empty(0, dtype=np.int16)
        self._key = key
        self.models = {key: None}  # gate only reads .models.keys()

    def predict(self, pcm):
        """int16 frame (gate feeds 1280 samples) → {name: 1.0 on detect, else 0.0}."""
        self._buf = np.concatenate((self._buf, np.asarray(pcm, dtype=np.int16)))
        detected = False
        while self._buf.shape[0] >= self._frame:
            frame = self._buf[: self._frame]
            self._buf = self._buf[self._frame :]
            if self._pv.process(frame) >= 0:
                detected = True
        return {self._key: 1.0 if detected else 0.0}

    def delete(self):
        """Release the native Porcupine handle (optional; GC also frees it)."""
        try:
            self._pv.delete()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
