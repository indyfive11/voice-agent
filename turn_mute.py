"""Mute the user through the brain's "thinking" gap (the empty-barge-in fix).

In half-duplex, `AlwaysUserMuteStrategy` mutes the user only while the bot is *speaking* — i.e. between
`BotStartedSpeakingFrame` and `BotStoppedSpeakingFrame`. But the bot's turn opens the moment its context is
pushed, and on a remote/high-latency brain (Claude/opus) there's a sizable gap between then and the first
audio frame. During that gap nothing mutes the user, so a stray VAD/transcription onset (the user's own
trailing audio, a duck-restore amplitude spike, ambient speech) starts a fresh turn and cancels the
in-flight-but-silent bot turn. That was 396/435 (91%) of interrupts in the 2026-06-07 logs, all logged
`BARGE-IN| interrupted; said so far: ''` (bot had said nothing yet).

Fix: the brain emits `BotThinkingFrame(active=True)` UPSTREAM when it begins a turn and `active=False` when
it ends (always, via its `finally`) — a `SystemFrame` so it reaches the user aggregator the same way
`BotStartedSpeakingFrame` does. `BotThinkingMuteStrategy` mutes between the two. Added ALONGSIDE
`AlwaysUserMuteStrategy` (the aggregator OR-combines mute strategies), so the user is muted across the whole
bot turn: thinking (this) OR speaking (Always). This is consistent with half-duplex's existing "no barge-in"
contract — it just plugs the unmuted hole before TTS starts. No effect in full-duplex (no mute strategies).
"""

from __future__ import annotations

from dataclasses import dataclass

from pipecat.frames.frames import Frame, SystemFrame
from pipecat.turns.user_mute.base_user_mute_strategy import BaseUserMuteStrategy


@dataclass
class BotThinkingFrame(SystemFrame):
    """Pushed UPSTREAM by the brain (BrainLLMService) to bracket a bot turn: active=True when it begins
    processing (before any TTS), active=False when it ends. Lets the user-mute strategy cover the
    first-token latency gap that `AlwaysUserMuteStrategy` (bot-speaking only) leaves open."""

    active: bool = True


class BotThinkingMuteStrategy(BaseUserMuteStrategy):
    """Mute the user while the bot is thinking — between BotThinkingFrame(active=True) and (active=False)."""

    def __init__(self):
        super().__init__()
        self._thinking = False

    async def process_frame(self, frame: Frame) -> bool:
        await super().process_frame(frame)
        if isinstance(frame, BotThinkingFrame):
            self._thinking = frame.active
        return self._thinking
