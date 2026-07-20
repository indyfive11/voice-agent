"""Output sample-rate detection — the chipmunk-guard's install-time step.

WHY THIS EXISTS (the 1.84x chipmunk-TTS regression). ``AUDIO_OUTPUT_SAMPLE_RATE`` must be UNSET on a box
whose output goes through a resampling sound server (Kokoro's 24k is converted transparently), but PINNED
to the device's native rate on a box that opens a fixed-rate hardware endpoint directly (open at the wrong
rate → PortAudio -9997 → chipmunk/silence). Picking wrong in either direction breaks audio.

The hard part is that NO probed *rate* discriminates the two — validated on real hardware 2026-07-20:

  signal                     EM (`pulse`, must stay UNSET)   Pi (`EMEET`, must PIN 48000)
  -------------------------  ------------------------------  ----------------------------
  pactl sink Sample Spec     48000 (graph rate)              48000                          ← same, useless
  PyAudio defaultSampleRate  44100                           44100 (server devices)         ← wrong on both
  ALSA /proc/asound rates    n/a (server)                    48000 (48000/1), FIXED         ← authoritative

So the discriminator is the device *KIND*, not any rate: a **resampling sound-server endpoint**
(`pulse`/`pipewire`/`default`/`jack`/…) → leave UNSET; a **specific hardware device** (a named USB card /
``hw:X,Y``) → PIN its ALSA-authoritative native playback rate. And the rate is read ONLY from the
authoritative OS source (``/proc/asound``), never from PyAudio/pactl (both proven unreliable above) and
never a per-startup probe (that per-startup probe is what chipmunked EM in the first place).

This is the SETUP-TIME detect-and-write step (portability SOP: the app reads config dumb; setup detects
once and writes; the user overrides). It RETURNS a decision; the caller writes it. Fail-safe by design:
anything it cannot classify authoritatively → UNSET (the historical no-op; a resampling server never
chipmunks, and a fixed-rate device that needs a pin is the rarer, explicitly-named case).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "SERVER_DEVICE_NAMES",
    "AudioOutputDecision",
    "classify_output_device",
    "list_alsa_cards",
    "alsa_card_index_for",
    "alsa_playback_rates",
    "detect_output_sample_rate",
]

# Named PortAudio/ALSA endpoints that are a RESAMPLING sound server, not a fixed device. Targeting any of
# these means the server converts Kokoro's rate to the hardware rate → AUDIO_OUTPUT_SAMPLE_RATE stays UNSET.
SERVER_DEVICE_NAMES = frozenset(
    {"pulse", "pipewire", "default", "sysdefault", "jack", "pulseaudio", "dmix", "dsnoop"}
)


@dataclass(frozen=True)
class AudioOutputDecision:
    """What the installer should write. ``sample_rate is None`` → leave ``AUDIO_OUTPUT_SAMPLE_RATE`` UNSET
    (the historical no-op). ``device_kind`` ∈ {``server``, ``hardware``, ``unknown``}; ``reason`` explains
    the call so the wizard can show its work."""

    sample_rate: Optional[int]
    device_kind: str
    reason: str


def classify_output_device(name: Optional[str]) -> str:
    """``server`` for a resampling sound-server endpoint (incl. empty/unset → the framework default is a
    server on any modern box), ``hardware`` for a specifically-named device."""
    if not name or not name.strip():
        return "server"  # unset → framework default (pulse/pipewire) → resamples → leave unset
    return "server" if name.strip().lower() in SERVER_DEVICE_NAMES else "hardware"


_CARD_HEADER = re.compile(r"^\s*(\d+)\s*\[([^\]]+)\]:\s*(.*)$")


def list_alsa_cards(proc_asound: str = "/proc/asound") -> list[tuple[int, str, str]]:
    """Parse ``/proc/asound/cards`` → ``[(index, id, description)]``. The description folds the header
    line's shortname and the following indented longname line, so a name match can hit either. Returns
    ``[]`` if the file is absent/unreadable (a box with no ALSA → nothing to pin → the caller unsets)."""
    try:
        text = (Path(proc_asound) / "cards").read_text()
    except OSError:
        return []
    cards: list[tuple[int, str, str]] = []
    idx: Optional[int] = None
    cid = ""
    desc_parts: list[str] = []

    def _flush() -> None:
        if idx is not None:
            cards.append((idx, cid, " ".join(desc_parts).strip()))

    for line in text.splitlines():
        m = _CARD_HEADER.match(line)
        if m:
            _flush()
            idx = int(m.group(1))
            cid = m.group(2).strip()
            desc_parts = [m.group(3).strip()]
        elif idx is not None and line.strip():
            desc_parts.append(line.strip())  # the indented longname continuation
    _flush()
    return cards


def alsa_card_index_for(
    name: str, cards: Optional[list[tuple[int, str, str]]] = None, *, proc_asound: str = "/proc/asound"
) -> Optional[int]:
    """Case-insensitive substring match of ``name`` against each card's id + description (e.g. ``EMEET``
    → card 3 ``EMEET OfficeCore M0 Plus``). Returns the card index, or ``None`` if nothing matches (→ the
    caller leaves the rate unset). A single unambiguous match wins; multiple matches → ``None`` (ambiguous
    → safe-unset rather than pin the wrong card)."""
    if cards is None:
        cards = list_alsa_cards(proc_asound)
    needle = name.strip().lower()
    if not needle:
        return None
    hits = [idx for idx, cid, desc in cards if needle in cid.lower() or needle in desc.lower()]
    return hits[0] if len(hits) == 1 else None


def alsa_playback_rates(card_index: int, proc_asound: str = "/proc/asound") -> Optional[list[int]]:
    """Read the PLAYBACK stream's advertised rates from ``/proc/asound/card<N>/stream0`` (USB-audio) —
    only the ``Playback:`` section, never the ``Capture:`` rates. Returns the sorted distinct rates, or
    ``None`` if the file/section is unreadable OR any ``Rates:`` line is a continuous RANGE (``8000 - 48000``)
    — a range means the endpoint is flexible, so there is nothing fixed to pin (→ caller unsets)."""
    try:
        text = (Path(proc_asound) / f"card{card_index}" / "stream0").read_text()
    except OSError:
        return None
    # isolate the Playback block: from "Playback:" up to the next top-level section ("Capture:").
    lines = text.splitlines()
    block: list[str] = []
    in_play = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Playback:":
            in_play = True
            continue
        if in_play and stripped in ("Capture:", "Midi:"):
            break
        if in_play:
            block.append(stripped)
    rates: set[int] = set()
    for line in block:
        if not line.lower().startswith("rates:"):
            continue
        val = line.split(":", 1)[1]
        if "-" in val:
            return None  # continuous range → not a fixed-rate endpoint → don't pin
        for tok in re.split(r"[,\s]+", val.strip()):
            if tok.isdigit():
                rates.add(int(tok))
    return sorted(rates) if rates else None


def detect_output_sample_rate(
    device_name: Optional[str], *, proc_asound: str = "/proc/asound"
) -> AudioOutputDecision:
    """The chipmunk-guard decision. Returns an :class:`AudioOutputDecision`; ``sample_rate is None`` means
    leave ``AUDIO_OUTPUT_SAMPLE_RATE`` UNSET.

    - a server endpoint (or unset name) → UNSET (the server resamples; unset is known-good).
    - a named hardware device with exactly ONE fixed playback rate → PIN that rate (the Pi EMEET → 48000).
    - a named hardware device that can't be matched, advertises multiple/ranged rates, or is unreadable →
      UNSET (fail-safe: never pin a guess).
    """
    kind = classify_output_device(device_name)
    if kind == "server":
        label = (device_name or "").strip() or "(unset → framework default)"
        return AudioOutputDecision(None, "server", f"{label} is a resampling sound-server endpoint → leave unset")

    assert device_name is not None  # classify() returns "server" for None/empty
    idx = alsa_card_index_for(device_name, proc_asound=proc_asound)
    if idx is None:
        return AudioOutputDecision(
            None, "unknown", f"no unambiguous ALSA card matches {device_name!r} → leave unset (safe)"
        )
    rates = alsa_playback_rates(idx, proc_asound=proc_asound)
    if rates and len(rates) == 1:
        return AudioOutputDecision(
            rates[0], "hardware",
            f"{device_name!r} = ALSA card {idx}, fixed playback rate {rates[0]}Hz → pin (must match Kokoro emit-rate)",
        )
    detail = "multiple rates" if rates else "no fixed rate readable"
    return AudioOutputDecision(
        None, "hardware", f"{device_name!r} = ALSA card {idx} advertises {detail} → leave unset (not fixed-rate)"
    )
