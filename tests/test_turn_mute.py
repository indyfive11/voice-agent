"""Offline tests for BotThinkingMuteStrategy — the empty-barge-in fix.

The user aggregator OR-combines mute strategies (`should_mute_next_time |= await s.process_frame(frame)`),
so BotThinkingMuteStrategy (mutes while the bot is thinking) added alongside AlwaysUserMuteStrategy (mutes
while the bot is speaking) keeps the user muted across the WHOLE bot turn. asyncio.run drives the bodies.
"""

import asyncio

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from turn_mute import BotThinkingFrame, BotThinkingMuteStrategy


def _audio():
    return InputAudioRawFrame(b"\x00\x00", sample_rate=16000, num_channels=1)


def test_mutes_between_thinking_active_true_and_false():
    s = BotThinkingMuteStrategy()

    async def go():
        assert await s.process_frame(_audio()) is False          # idle → not muted
        assert await s.process_frame(BotThinkingFrame(active=True)) is True
        assert await s.process_frame(_audio()) is True            # stays muted through the gap
        assert await s.process_frame(BotThinkingFrame(active=False)) is False
        assert await s.process_frame(_audio()) is False           # released

    asyncio.run(go())


def test_or_composes_with_always_strategy_across_whole_turn():
    # Simulate the aggregator's OR-combine over a full turn: thinking (no audio yet) → speaking → done.
    thinking = BotThinkingMuteStrategy()
    always = AlwaysUserMuteStrategy()

    async def muted(frame):
        a = await thinking.process_frame(frame)
        b = await always.process_frame(frame)
        return a or b

    async def go():
        assert await muted(_audio()) is False                     # before the turn: open
        assert await muted(BotThinkingFrame(active=True)) is True  # turn opens (thinking)
        assert await muted(_audio()) is True                       # thinking gap: muted (the fix)
        assert await muted(BotStartedSpeakingFrame()) is True      # now speaking: muted
        assert await muted(BotThinkingFrame(active=False)) is True # thinking ends but still speaking → muted
        assert await muted(_audio()) is True
        assert await muted(BotStoppedSpeakingFrame()) is False     # turn fully done → open again

    asyncio.run(go())
