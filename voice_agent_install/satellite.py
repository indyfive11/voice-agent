"""The satellite role provisioner — the piece that had no caller.

This is the module GA's tree-audit was pointing at: ``discover_brain``/``default_providers`` and
``detect_output_sample_rate`` were built, tested, and imported by nothing outside tests. A seam with
no caller is unbuilt work that looks shipped. This calls them.

SHAPE (3b). Credentials are supplied by prompt — three secrets, which is the honest floor. The
interactive claim handshake that removes the typing is 3c and deliberately not here: everything
below is independent of *how* credentials arrive, and it is where a clean-box install actually dies,
so it ships first rather than waiting on the pairing design.

WHAT DISCOVERY DOES AND DOES NOT BUY US. mDNS returns host and port only, and it is off by default on
a stock brain (``voice_advertise: bool = False``), so it can never be load-bearing. It is used here
purely to pre-fill a default the operator confirms. The STT/TTS URLs are COMPOSED from the brain host
— note *composed*, not "derived for free": ``_repoint_remote_services`` only rewrites an existing
absolute URL and skips on empty, so nothing in the codebase would construct these at first provision,
and an unwritten ``STT_REMOTE_URL`` is a hard startup error (``config.py:130``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from . import envfile, profile
from .paths import Layout

__all__ = ["SatelliteAnswers", "compose_env", "service_url", "DEFAULT_STT_PORT", "DEFAULT_TTS_PORT"]

# These are voice-agent's OWN default service ports, declared server-side with an env override
# (`stt_service/server.py:235`, `tts_service/server.py:303`) — structurally identical to
# `discovery.DEFAULT_BRAIN_PORT = 8765`. An application's own default port with an override is a safe
# universal default under the portability SOP, NOT an installation-specific hardcode; the values that
# would be hardcodes (which host, which token) are supplied per-install.
DEFAULT_STT_PORT = 8770
DEFAULT_TTS_PORT = 8771


@dataclass
class SatelliteAnswers:
    """Everything the provisioner needs. Assembled from detection + prompts; kept as plain data so the
    whole compose step is testable without a terminal, a network, or a sound card."""

    brain_host: str
    brain_port: int
    room_id: str
    brain_token: Optional[str]
    stt_token: Optional[str]
    tts_token: Optional[str]
    input_device: str
    output_device: str
    wake_word: str
    wake_engine: str = "oww"
    output_sample_rate: Optional[int] = None
    stt_host: Optional[str] = None
    tts_host: Optional[str] = None
    stt_port: int = DEFAULT_STT_PORT
    tts_port: int = DEFAULT_TTS_PORT
    extra: dict[str, str] = field(default_factory=dict)


def service_url(host: str, port: int) -> str:
    """Compose an absolute service URL. Always an IP/host we were given — never a ``.local`` name:
    a ``.local`` in a service URL makes every connect eat a synchronous ``getaddrinfo``/nss-mdns
    lookup (~5s with Avahi down), which lands inside the boot budget."""
    return f"http://{host}:{port}"


def compose_env(answers: SatelliteAnswers) -> dict[str, Optional[str]]:
    """Build the full key→value map for a satellite ``.env``.

    Returns values as strings (or ``None`` for "leave unset"). Every key in
    ``profile.REQUIRED_KEYS`` is guaranteed present — :func:`validate` enforces it, and the caller
    refuses to write a config that fails that check rather than emitting a file that starts and then
    misbehaves.
    """
    stt_host = answers.stt_host or answers.brain_host
    tts_host = answers.tts_host or answers.brain_host

    env: dict[str, Optional[str]] = dict(profile.profile_for("satellite"))
    env.update({
        "GAB_HOST": answers.brain_host,
        "GAB_PORT": str(answers.brain_port),
        "ROOM_ID": answers.room_id,
        "STT_REMOTE_URL": service_url(stt_host, answers.stt_port),
        "TTS_REMOTE_URL": service_url(tts_host, answers.tts_port),
        "AUDIO_INPUT_DEVICE_NAME": answers.input_device,
        "AUDIO_OUTPUT_DEVICE_NAME": answers.output_device,
        "WAKE_WORD": answers.wake_word,
        "WAKE_WORD_ENGINE": answers.wake_engine,
    })

    # Tokens: written only when we have one. An absent token is NOT the same as an empty one — the
    # services read empty as "run open" (`stt_service/server.py:183-184`), so writing `KEY=` would
    # assert an open service we have not verified is open.
    for key, value in (("GAB_AUTH_TOKEN", answers.brain_token),
                       ("STT_REMOTE_TOKEN", answers.stt_token),
                       ("TTS_REMOTE_TOKEN", answers.tts_token)):
        env[key] = value if value else None

    # The chipmunk guard's write half. `None` means leave UNSET, which is the historical no-op for a
    # resampling sound server; a fixed-rate hardware endpoint gets its ALSA-authoritative rate pinned.
    # Never a guess in either direction — an unclassifiable device yields None upstream.
    env["AUDIO_OUTPUT_SAMPLE_RATE"] = str(answers.output_sample_rate) if answers.output_sample_rate else None

    env.update(answers.extra)
    return env


def validate(env: dict[str, Optional[str]]) -> list[str]:
    """Return a list of problems, empty if the config is provisionable.

    This is the gate that stops the two silent-misconfiguration classes we know about: a missing key
    that deletes a whole subsystem (``WAKE_WORD`` → the gate returns None and the mic is open), and a
    missing key that flips a default into the wrong topology (``GAB_HOST`` → loopback → ``GAB_LAUNCH``
    → try to spawn a local brain).
    """
    problems = []
    for key in profile.REQUIRED_KEYS:
        if not (env.get(key) or "").strip():
            problems.append(f"{key} is required for a satellite but is unset")
    host = (env.get("GAB_HOST") or "").strip().lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        problems.append("GAB_HOST is loopback — a satellite must point at another box (and loopback "
                        "would flip GAB_LAUNCH to 1 and try to spawn a brain this box has no binary for)")
    if (env.get("WAKE_WORD_MEDIA_ONLY") or "") != "0":
        problems.append("WAKE_WORD_MEDIA_ONLY must be 0 on a satellite, or the mic is open whenever "
                        "nothing is playing and the wake gate is never exercised")
    return problems


def apply(layout: Layout, env: dict[str, Optional[str]], *,
          confirm: Callable[[str, dict[str, tuple[str, str]]], bool]) -> tuple[Optional[str], dict]:
    """Merge ``env`` into the layout's ``.env`` and write it, after ``confirm`` approves the diff.

    Returns ``(backup_path, changes)``. Merge — never overwrite: an operator's hand-tuned keys and the
    comments explaining them survive a re-provision, and a backup is taken before any write so a bad
    provision is always recoverable.
    """
    path = str(layout.env_path)
    try:
        with open(path, encoding="utf-8") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    if existing:
        text, changes = envfile.merge(existing, env)
    else:
        text, changes = envfile.render(env), {k: ("", v) for k, v in env.items() if v is not None}

    if not changes:
        return None, {}
    if not confirm(path, changes):
        return None, {}

    backup_path = envfile.backup(path)
    envfile.write(path, text)
    return backup_path, changes
