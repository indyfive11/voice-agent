"""STT decoding-accuracy levers — read faster-whisper transcribe() kwargs from env.

The in-pipeline `WhisperSTTService` (config.py) and the standalone `stt_service` both transcribe with bare
library defaults today (`model.transcribe(audio, language=...)`), so none of faster-whisper's accuracy dials
are reachable. This module is the single source of truth for those dials, read from env.

**Safe-no-op default (hardware/install-portability SOP):** every knob is UNSET by default, and an unset knob
is *omitted* from the kwargs dict — so `model.transcribe(audio, language=…, **decoding_kwargs())` with nothing
configured is byte-identical to today (pure faster-whisper defaults). An install opts into tuning via env;
an unconfigured install is never changed.

Knobs (all optional):
  STT_BEAM_SIZE (int)              → beam_size            (fw default 5)
  STT_BEST_OF (int)                → best_of              (fw default 5)
  STT_TEMPERATURE (float)          → temperature          (fw default is a fallback ladder; pinning a single
                                                            value DISABLES the ladder — usually leave unset)
  STT_CONDITION_ON_PREVIOUS (bool) → condition_on_previous_text (fw default True; set 0 for command STT —
                                                            stops prior-turn context from dragging a short
                                                            command toward a hallucinated continuation)
  STT_NO_SPEECH_THRESHOLD (float)  → no_speech_threshold
  STT_LOGPROB_THRESHOLD (float)    → log_prob_threshold
  STT_VAD_FILTER (bool)            → vad_filter           (fw default False; trims non-speech regions)
  STT_INITIAL_PROMPT (str)         → initial_prompt       (Tier-2 vocabulary biasing — domain terms, file
                                                            extensions, project names)
  STT_HOTWORDS (str)               → hotwords             (Tier-2 stronger per-term biasing, faster-whisper)

whisper.cpp (the GPU/Vulkan `whisper-server`) is reached over HTTP and exposes only a subset — see
`whispercpp_form_fields()`.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

# (env key, transcribe kwarg, parser) — order is the display order in the startup banner.
_INT = "int"
_FLOAT = "float"
_BOOL = "bool"
_STR = "str"

_KNOBS = (
    ("STT_BEAM_SIZE", "beam_size", _INT),
    ("STT_BEST_OF", "best_of", _INT),
    ("STT_TEMPERATURE", "temperature", _FLOAT),
    ("STT_CONDITION_ON_PREVIOUS", "condition_on_previous_text", _BOOL),
    ("STT_NO_SPEECH_THRESHOLD", "no_speech_threshold", _FLOAT),
    ("STT_LOGPROB_THRESHOLD", "log_prob_threshold", _FLOAT),
    ("STT_VAD_FILTER", "vad_filter", _BOOL),
    ("STT_INITIAL_PROMPT", "initial_prompt", _STR),
    ("STT_HOTWORDS", "hotwords", _STR),
)

_FALSEY = ("0", "false", "no", "off")


def _coerce(raw: str, kind: str, key: str):
    """Parse one env value; return None (skip the knob) on blank or an unparseable value (logged, never raises)."""
    raw = raw.strip()
    if raw == "":
        return None
    try:
        if kind == _INT:
            return int(raw)
        if kind == _FLOAT:
            return float(raw)
        if kind == _BOOL:
            return raw.lower() not in _FALSEY
        return raw  # _STR
    except ValueError:
        logger.warning(f"STT tuning: ignoring {key}={raw!r} — not a valid {kind}")
        return None


def decoding_kwargs(env: dict | None = None) -> dict[str, Any]:
    """faster-whisper ``transcribe()`` kwargs from env; UNSET knobs are omitted (→ library default)."""
    env = os.environ if env is None else env
    out: dict[str, Any] = {}
    for key, kwarg, kind in _KNOBS:
        if key not in env:
            continue
        val = _coerce(env[key], kind, key)
        if val is not None:
            out[kwarg] = val
    return out


def describe(kwargs: dict[str, Any]) -> str:
    """One-line, log-safe summary (truncates a long prompt) for the startup banner."""
    if not kwargs:
        return "defaults (no env tuning)"
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "…"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def wrap_whisper_service(svc, env: dict | None = None):
    """Inject the env decoding kwargs into a pipecat ``WhisperSTTService``.

    pipecat calls ``self._model.transcribe(audio, language=language)`` and exposes no hook for the other
    kwargs, so we bind them onto the underlying faster-whisper model with ``functools.partial``. ``language``
    is never in our kwargs, so there's no collision with pipecat's call. No-op if nothing is configured or the
    model isn't a faster-whisper model.
    """
    kwargs = decoding_kwargs(env)
    if not kwargs:
        logger.info("STT tuning: defaults (no env tuning)")
        return svc
    model = getattr(svc, "_model", None)
    transcribe = getattr(model, "transcribe", None)
    if transcribe is None:
        logger.warning("STT tuning: no faster-whisper model to tune — knobs ignored")
        return svc
    import functools

    model.transcribe = functools.partial(transcribe, **kwargs)
    logger.info(f"STT tuning (in-pipeline): {describe(kwargs)}")
    return svc


def whispercpp_form_fields(env: dict | None = None) -> dict[str, str]:
    """The subset of decoding levers the whisper.cpp ``whisper-server`` /inference HTTP API accepts.

    The server takes ``temperature`` and ``prompt`` (its name for ``initial_prompt``); ``beam_size`` and the
    model are set by the server's own launch flags, and ``condition_on_previous_text``/thresholds aren't
    exposed over HTTP. Returns string-valued form fields; empty if nothing applicable is configured.
    """
    env = os.environ if env is None else env
    fields: dict[str, str] = {}
    temp = _coerce(env.get("STT_TEMPERATURE", ""), _FLOAT, "STT_TEMPERATURE")
    if temp is not None:
        fields["temperature"] = str(temp)
    prompt = (env.get("STT_INITIAL_PROMPT") or "").strip()
    if prompt:
        fields["prompt"] = prompt
    return fields
