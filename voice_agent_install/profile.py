"""The satellite PROFILE — a default set validated as a set, not key-by-key.

WHY A PROFILE AND NOT "just write the role-essential keys". The reference satellite's live ``.env``
diverges from ``config.py``'s defaults in twelve places, every one of them wake or audio. Read
naively that looks like personal taste to be left alone. It is the opposite: **it is evidence that
the code defaults are not a working satellite.** Writing a dozen keys and letting the other
twenty-nine fall through to defaults that no satellite has ever run would ship a config nobody has
validated, assembled at install time, and call it provisioned.

So the profile below is a single artifact that gets reviewed, changed, and tested as a unit.

THE TWO ENTRIES THAT ARE LOAD-BEARING AND NON-OBVIOUS:

``WAKE_WORD`` — ``config.py:1494-95`` is::

    names = _env("WAKE_WORD", "")
    if not names:
        return None  # not configured → open-mic as before

An unwritten ``WAKE_WORD`` does not degrade the wake gate. It **deletes** it: the whole gate returns
None and the mic is open. There is no error and no log line saying the device is now always
listening. This key is therefore never optional, and its absence must be impossible to produce.

``WAKE_WORD_MEDIA_ONLY`` — defaults to ``"1"`` (``config.py:1507``), meaning the wake word is
required *only while media is playing*. On a quiet box the mic is open, so a freshly-provisioned
satellite answers without ever invoking the gate — which would let an acceptance test "say the wake
word, get an answer" pass green while the gate was never exercised. We write ``0`` so the gate is
always on the critical path, and the acceptance criteria can actually observe it.

WHAT IS DELIBERATELY *NOT* HERE. No wake threshold, no engine, no wake model path: those are
per-device and per-voice (the reference Pi's tuned values came out of a training run against the maintainer's
voice and its specific mic). The provisioner picks them at detection time and explains the choice —
guessing them here would be writing a value we cannot justify, which the portability SOP calls worse
than leaving it unset.
"""

from __future__ import annotations

__all__ = ["SATELLITE_PROFILE", "REQUIRED_KEYS", "profile_for"]

# Keys without which the satellite is not a satellite. The provisioner refuses to write a config
# missing any of these rather than emitting a file that starts and then misbehaves — an unwritten
# STT_REMOTE_URL is a hard startup error (config.py:130), and an unwritten GAB_HOST silently defaults
# to 127.0.0.1, which flips GAB_LAUNCH to 1 and makes the satellite try to SPAWN a brain it has no
# binary for (config.py:286,298).
REQUIRED_KEYS = (
    "BRAIN",
    "GAB_HOST",
    "GAB_PORT",
    "GAB_LAUNCH",
    "ROOM_ID",
    "STT_PROVIDER",
    "STT_REMOTE_URL",
    "TTS_PROVIDER",
    "TTS_REMOTE_URL",
    "WAKE_WORD",
    "WAKE_WORD_MEDIA_ONLY",
    "AUDIO_INPUT_DEVICE_NAME",
    "AUDIO_OUTPUT_DEVICE_NAME",
)

# The validated set. Values here are topology decisions that hold for EVERY satellite; anything that
# varies per box is filled in by detection or a prompt and is absent from this dict on purpose.
SATELLITE_PROFILE: dict[str, str] = {
    # --- brain ------------------------------------------------------------------------------------
    # A satellite delegates cognition. `BRAIN` must be written explicitly because the generated unit
    # execs the venv python directly rather than going through `run.sh gab` — dropping that wrapper
    # (to get `uv` off the boot path) also drops the argv that used to export it.
    "BRAIN": "gabagent",
    # Never spawn. `config.py:298` defaults GAB_LAUNCH to 1 on a loopback host; a satellite has no
    # `gab` binary, so an accidental loopback would fail in a way that reads like a brain outage.
    "GAB_LAUNCH": "0",
    # --- offload ----------------------------------------------------------------------------------
    # Pi-class hardware cannot run these locally at conversational latency: local Kokoro measures
    # ~16s/utterance on a Pi 4, and local Whisper would load a multi-GB model onto a 3.7GiB board.
    "STT_PROVIDER": "remote",
    "TTS_PROVIDER": "remote",
    # --- wake -------------------------------------------------------------------------------------
    # See the module docstring: 0 puts the gate on the critical path at all times. Without this a
    # satellite is an open mic whenever nothing is playing.
    "WAKE_WORD_MEDIA_ONLY": "0",
}


def profile_for(role: str) -> dict[str, str]:
    """Return the validated default set for ``role``. Unknown roles raise rather than silently
    returning ``{}`` — an empty profile would provision a box with nothing but code defaults, which
    is exactly the unvalidated config this module exists to prevent."""
    if role != "satellite":
        raise ValueError(f"no validated profile for role {role!r} (only 'satellite' exists today)")
    return dict(SATELLITE_PROFILE)
