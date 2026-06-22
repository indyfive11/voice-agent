"""ResponseLatencyObserver — log the real user-perceived response latency per turn, decomposed.

The TurnTrackingObserver `on_turn_ended` duration is a conversation-turn *span*: it runs from the user's
first vocalization (incl. transcript-less false-starts) through any think-pause to the bot's completing
reply, so it can read as "90s" when the actual interaction was fine (2026-06-04 re-test, TURN #2). That
span is NOT latency. This observer logs the thing that matters: time from the user *finishing* speaking
(`UserStoppedSpeakingFrame`) to the bot *starting* to speak (`BotStartedSpeakingFrame`) — i.e. how long
Aria took to respond.

It also DECOMPOSES that total into the two halves the felt-latency lever splits on (the #60 first-audio
breakdown — GA confirmed 2026-06-22 the brain already streams per-sentence `token` events, so the open
question is whether OUR remote-TTS path synthesizes them incrementally or buffers to the end of the reply):
  * **to-first-token** — `UserStoppedSpeakingFrame` → first `LLMTextFrame` of the reply = STT-finalize +
    brain round-trip + model TTFT (the BRAIN's contribution; the lever there is brain-side).
  * **token→audio** — first `LLMTextFrame` → `BotStartedSpeakingFrame` = OUR TTS path (sentence
    aggregation + remote synth + playout). If this ≈ the whole reply, we are buffering to `done` and
    leaving the win on the table; if it is just the first sentence's synth (~1s), the incremental path
    is working and first audio starts at sentence #1.
One clean `RESPONSE | <total>s · stt+brain <a>s · token→audio <b>s` line per turn.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class ResponseLatencyObserver(BaseObserver):
    """Emit a per-turn `RESPONSE | …` line: total user-stop→bot-start, split into brain vs. our-TTS halves."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._t_user_stop_ns: int | None = None
        self._t_first_token_ns: int | None = None  # first reply LLMTextFrame since the last user-stop

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, UserStoppedSpeakingFrame):
            # Latest user-stop before a bot reply wins (a turn may have intermediate stops); arm a fresh turn.
            self._t_user_stop_ns = data.timestamp
            self._t_first_token_ns = None
        elif isinstance(frame, LLMTextFrame):
            # First speakable token of the reply this turn — the brain→TTS handoff. Only the first counts.
            if self._t_user_stop_ns is not None and self._t_first_token_ns is None:
                self._t_first_token_ns = data.timestamp
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._t_user_stop_ns is not None:
                user_stop_ns = self._t_user_stop_ns
                first_token_ns = self._t_first_token_ns
                self._t_user_stop_ns = None
                self._t_first_token_ns = None
                total = (data.timestamp - user_stop_ns) / 1e9
                if not (0 <= total < 600):  # ignore clock glitches / cross-turn carryover
                    return
                line = f"RESPONSE | {total:.2f}s (user-stop → bot-start)"
                # Decompose only when a reply token preceded the audio sanely (a TTSSpeakFrame filler — e.g.
                # an escalation status — can start audio with no prior LLMTextFrame; skip the split then).
                if first_token_ns is not None:
                    to_token = (first_token_ns - user_stop_ns) / 1e9
                    token_to_audio = (data.timestamp - first_token_ns) / 1e9
                    if to_token >= 0 and token_to_audio >= 0:
                        line += f" · stt+brain {to_token:.2f}s · token→audio {token_to_audio:.2f}s"
                logger.bind(transcript=True).info(line)
