"""aria_state.py — publish Aria's semantic state to a tmpfs file for the "HAL eye" indicator.

The ONLY contract with the Conky eye panel is this file (see ~/dev/aria-eye-indicator-DESIGN.md §3):

    path : ${XDG_RUNTIME_DIR}/aria/state   (fallback: ~/.local/state/aria/state)
    JSON : {"state": <name>, "level": <0.0-1.0>, "ts": <unix epoch>}

  - `state` — one of: off / idle / listening / thinking / speaking (+ "error" reserved for later).
  - `level` — 0.0-1.0; live audio RMS during `speaking` so the eye pulses with the voice, else 0.0.
  - `ts`    — write time, so the reader fails safe to `off` if the writer dies (stale > ~5s → off).

The voice-agent is the single writer (one process owns the loop). Writes happen on a *transition*
plus throttled `level` updates while speaking. Writes are atomic (tmp file + os.replace) so the
reader never sees a half-written object, and every error is swallowed: the eye is a cosmetic
side-channel and must never crash or stall the agent. Opt out entirely with ARIA_EYE_STATE=0.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

# Semantic states the eye understands. "error" is reserved (the Conky renderer defines it) but not
# emitted by v1; kept here so adding it later is a one-liner with no contract change.
VALID_STATES = frozenset({"off", "idle", "listening", "thinking", "speaking", "error"})

_disabled = os.environ.get("ARIA_EYE_STATE", "1") == "0"

# Visibility boost for the speaking RMS: raw normalized speech RMS is small (~0.05-0.3), so the eye
# would barely move. Scale it into a livelier 0..1 swing. Tunable without a rebuild.
try:
    _level_gain = float(os.environ.get("ARIA_EYE_LEVEL_GAIN", "4.0"))
except ValueError:
    _level_gain = 4.0

_last_state: str | None = None
_last_level: float = -1.0

# The state the eye should settle on once a transient (speaking/thinking/listening) ends: `idle`
# while awake, `off` while asleep. Whoever owns a return-to-rest transition reads this instead of
# hardcoding `idle`, so a goodbye spoken on the way to sleep settles the eye on `off`, not `idle`
# (the goodbye TTS's BotStopped→rest write would otherwise clobber the earlier `off`).
_resting_state: str = "idle"


def _state_path() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) / "aria" if base else Path.home() / ".local" / "state" / "aria"
    return root / "state"


_path = _state_path()


def current() -> str | None:
    """The last state we wrote (None before the first write). Lets callers avoid clobbering a state
    they don't own — e.g. only downgrade `listening`→`idle`, never step on `speaking`."""
    return _last_state


def set_resting_state(state: str) -> None:
    """Record the state the eye should return to when no transient is active: `idle` while awake,
    `off` while asleep. Set by the sleep/wake authority (the wake gate); read via resting_state() by
    whoever owns the return-to-rest transition (tts_gain on BotStopped). Does NOT itself write the
    file — call set_state() for that."""
    global _resting_state
    _resting_state = state


def resting_state() -> str:
    """The state to settle on once a transient ends (see set_resting_state). Defaults to `idle`."""
    return _resting_state


def set_state(state: str, level: float = 0.0) -> None:
    """Atomically publish `state` (+ optional 0..1 `level`). Dedups identical writes; swallows all
    errors. No-op if ARIA_EYE_STATE=0."""
    global _last_state, _last_level
    if _disabled:
        return
    lvl = 0.0 if level <= 0.0 else (1.0 if level >= 1.0 else float(level))
    # Skip a redundant write (same state, ~same level) to keep the speaking-RMS path cheap.
    if state == _last_state and abs(lvl - _last_level) < 0.02:
        return
    try:
        d = _path.parent
        d.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"state": state, "level": round(lvl, 3), "ts": round(time.time(), 3)})
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp, _path)  # atomic on the same filesystem
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:
        return  # the eye must never break the agent
    _last_state, _last_level = state, lvl


def speaking_level_from_pcm(audio: bytes) -> float:
    """Normalized, visibility-boosted RMS (0..1) of int16 PCM, for the `speaking` `level`. Returns
    0.0 on empty/garbage input. Pure + testable (no numpy dependency required at call sites)."""
    if not audio:
        return 0.0
    try:
        import numpy as np

        s = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        if s.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(s * s)))
    except Exception:
        return 0.0
    boosted = rms * _level_gain
    return 0.0 if boosted <= 0.0 else (1.0 if boosted >= 1.0 else boosted)
