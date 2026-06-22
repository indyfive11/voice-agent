"""ResponseLatencyObserver — per-turn `RESPONSE | …` line + the first-audio decomposition (#60).

Verifies the total user-stop→bot-start latency, and its split into the brain half (user-stop → first
`LLMTextFrame`) and our-TTS half (first `LLMTextFrame` → `BotStartedSpeakingFrame`) — the breakdown GA
asked for (2026-06-22) to tell whether the remote-TTS path synthesizes incrementally or buffers to `done`.
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMTextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from response_latency import ResponseLatencyObserver

_S = 1_000_000_000  # ns per second


def _push(obs, frame, t_secs):
    """Feed one frame to the observer at monotonic time t_secs (ns timestamp)."""
    data = FramePushed(
        source=None,
        destination=None,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=int(t_secs * _S),
    )
    asyncio.run(obs.on_push_frame(data))


def _capture_transcript_logs():
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(m.record["message"]),
                      filter=lambda r: r["extra"].get("transcript"), level="DEBUG")
    return msgs, sink


def test_decomposes_total_into_brain_and_tts_halves():
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 10.0)
        _push(obs, LLMTextFrame("Hello"), 11.3)   # first token 1.30s after user stop (stt+brain)
        _push(obs, LLMTextFrame(" there"), 11.5)  # later tokens ignored for the split
        _push(obs, BotStartedSpeakingFrame(), 12.0)  # audio 0.70s after first token (our TTS)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "RESPONSE | 2.00s" in msgs[0]
    assert "stt+brain 1.30s" in msgs[0]
    assert "token→audio 0.70s" in msgs[0]


def test_no_split_when_audio_starts_without_a_reply_token():
    # A TTSSpeakFrame filler (e.g. an escalation status) can start audio with no prior LLMTextFrame →
    # report the total only, no (misleading) decomposition.
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 5.0)
        _push(obs, BotStartedSpeakingFrame(), 6.5)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "RESPONSE | 1.50s" in msgs[0]
    assert "stt+brain" not in msgs[0]


def test_ignores_bot_start_with_no_preceding_user_stop():
    # Cross-turn carryover / startup: a BotStartedSpeakingFrame with no armed user-stop emits nothing.
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, BotStartedSpeakingFrame(), 3.0)
    finally:
        logger.remove(sink)
    assert msgs == []


def test_first_token_only_counts_once_per_turn():
    # The split must use the FIRST reply token, not a later one, even across two user-stops in a turn.
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 0.0)
        _push(obs, UserStoppedSpeakingFrame(), 1.0)  # re-arm: latest user-stop wins, first-token resets
        _push(obs, LLMTextFrame("hi"), 2.0)          # 1.00s after the winning user-stop
        _push(obs, BotStartedSpeakingFrame(), 2.5)   # 0.50s token→audio
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "RESPONSE | 1.50s" in msgs[0]
    assert "stt+brain 1.00s" in msgs[0]
    assert "token→audio 0.50s" in msgs[0]
