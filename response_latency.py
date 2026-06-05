"""ResponseLatencyObserver — log the real user-perceived response latency per turn.

The TurnTrackingObserver `on_turn_ended` duration is a conversation-turn *span*: it runs from the user's
first vocalization (incl. transcript-less false-starts) through any think-pause to the bot's completing
reply, so it can read as "90s" when the actual interaction was fine (2026-06-04 re-test, TURN #2). That
span is NOT latency. This observer logs the thing that matters: time from the user *finishing* speaking
(`UserStoppedSpeakingFrame`) to the bot *starting* to speak (`BotStartedSpeakingFrame`) — STT-finalize +
brain round-trip + TTS TTFB, i.e. how long Aria took to respond. One clean `RESPONSE | x.xxs` line per turn.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import BotStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed


class ResponseLatencyObserver(BaseObserver):
    """Emit `RESPONSE | <secs>s` = user-stopped-speaking → bot-started-speaking, per turn."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._t_user_stop_ns: int | None = None

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, UserStoppedSpeakingFrame):
            # Latest user-stop before a bot reply wins (a turn may have intermediate stops).
            self._t_user_stop_ns = data.timestamp
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._t_user_stop_ns is not None:
                secs = (data.timestamp - self._t_user_stop_ns) / 1e9
                self._t_user_stop_ns = None
                if 0 <= secs < 600:  # ignore clock glitches / cross-turn carryover
                    logger.bind(transcript=True).info(f"RESPONSE | {secs:.2f}s (user-stop → bot-start)")
