"""Live "is Aria's TTS playing right now?" flag — a one-bool in-memory side-channel.

The TTS output stage (`tts_gain.TTSGainProcessor`) flips this True on BotStartedSpeaking and False on
BotStoppedSpeaking; the shared `/media/state` provider (`config.build_media_state_provider`) reads it and
carries `bot_speaking` to the brain on each poll. The brain treats `bot_speaking=true` as a duck-watchdog
heartbeat **refresh** so a long narration's TTS doesn't outlive the watchdog grace and pop the ducked bed
mid-reply.

Why this exists (2026-06-20): the brain's duck watchdog only refreshes on *incoming* user speech, so it
can't see Aria's own long TTS as activity — a 21s story outran the 20s grace and `duck_watchdog auto_restore`
popped the bed ~12s in, 9s before the voice-side single-writer's correct BotStopped restore. The voice side
holds the duck correctly through the story; this flag stops the brain's backstop from racing it.

In-memory (not a tmpfs file like aria_state) because the only reader is in-process. Defaults False; a missed
flip just degrades to the brain's grace timer (the original crash-net behaviour) — never worse than today.
"""

from __future__ import annotations

_speaking = False


def set_bot_speaking(value: bool) -> None:
    """Set the live flag (True between BotStartedSpeaking and BotStoppedSpeaking)."""
    global _speaking
    _speaking = bool(value)


def bot_speaking() -> bool:
    """True while Aria's TTS is actively playing."""
    return _speaking
