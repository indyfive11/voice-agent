"""Opt-in VAD near-miss diagnostic (VAD_DEBUG=1) — instruments the "hear me over the music" ruler.

Silero AND-gates voice activity on `confidence >= VAD_CONFIDENCE` AND `volume >= VAD_MIN_VOLUME`
(pipecat `vad_analyzer.py::_run_analyzer`). When the user's voice is partly masked by playing media it
can fail *either* gate and never start a turn — but the plain logs only show the *absence* of an onset,
not *why*. This subclass logs near-misses (exactly one gate passed) with the actual confidence/volume vs
their thresholds, so tuning `VAD_CONFIDENCE` / `VAD_MIN_VOLUME` (and the brain's ambient-duck level) is
data-driven instead of blind: "volume-gated" → lower min_volume / duck media more; "confidence-gated" →
lower confidence.

Off by default (no cost); enable per tuning session. It only adds logging — all real work is delegated
to the base analyzer, so a pipecat internal change can't break audio, only this diagnostic.
"""

from __future__ import annotations

import time

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer


def _tlog(message: str) -> None:
    """One line to the transcript log (greppable alongside USER/BOT/DUCK)."""
    logger.bind(transcript=True).info(message)


class LoggingSileroVADAnalyzer(SileroVADAnalyzer):
    """SileroVADAnalyzer that logs throttled near-misses to explain missed/false onsets."""

    def __init__(self, *args, log_throttle_secs: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._dbg_conf = 0.0
        self._dbg_vol = 0.0
        self._dbg_last_log = 0.0
        self._dbg_throttle = log_throttle_secs

    # Capture the per-frame values the base computes (these run in the analyzer's executor thread;
    # the await in analyze_audio is a happens-before barrier before we read them).
    def voice_confidence(self, audio: bytes) -> float:
        self._dbg_conf = super().voice_confidence(audio)
        return self._dbg_conf

    def _get_smoothed_volume(self, audio: bytes) -> float:
        self._dbg_vol = super()._get_smoothed_volume(audio)
        return self._dbg_vol

    async def analyze_audio(self, buffer: bytes):
        state = await super().analyze_audio(buffer)
        conf, vol = self._dbg_conf, self._dbg_vol
        conf_t = self._params.confidence
        vol_t = self._params.min_volume
        # Near-miss = exactly one gate passed, and there's real signal (ignore silence).
        if vol > 0.05 and (conf >= conf_t) != (vol >= vol_t):
            now = time.monotonic()
            if now - self._dbg_last_log >= self._dbg_throttle:
                self._dbg_last_log = now
                gate = "volume-gated" if conf >= conf_t else "confidence-gated"
                _tlog(
                    f"VAD   | near-miss ({gate}): conf={conf:.2f}/{conf_t} vol={vol:.2f}/{vol_t}"
                )
        return state
