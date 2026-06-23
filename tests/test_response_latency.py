"""ResponseLatencyObserver — per-turn `RESPONSE | …` line + the first-audio decomposition (#60).

Verifies the total user-stop→bot-start latency, and its split into the brain half (user-stop → first
`TTSStartedFrame`) and our-TTS half (first `TTSStartedFrame` → `BotStartedSpeakingFrame`). The boundary is
TTS-synth-start, not first-LLM-token, so a non-spoken STATUS line + the brain's tool round-trip stay in the
brain bucket (the 2026-06-22 joint-latency consensus with GA — attribution fix, not a felt-latency change).
"""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    TTSStartedFrame,
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
        _push(obs, TTSStartedFrame(), 11.3)        # synth begins 1.30s after user stop (stt+brain)
        _push(obs, TTSStartedFrame(), 11.5)        # later synth starts ignored for the split
        _push(obs, BotStartedSpeakingFrame(), 12.0)  # audio 0.70s after synth start (our TTS)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "RESPONSE | 2.00s" in msgs[0]
    assert "stt+brain 1.30s" in msgs[0]
    assert "token→audio 0.70s" in msgs[0]


def test_status_token_before_synth_stays_in_brain_bucket():
    # The 18:23:03 outlier: a non-spoken STATUS line arrives early, the brain then runs a tool call, and
    # synth (TTSStartedFrame) only starts much later. The pre-synth gap must land in stt+brain, NOT in
    # token→audio — that's the whole point of anchoring on TTS-synth-start.
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 0.0)
        _push(obs, TTSStartedFrame(), 2.94)          # synth starts only after brain finished the spoken sentence
        _push(obs, BotStartedSpeakingFrame(), 3.88)  # 0.94s real synth → token→audio
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "RESPONSE | 3.88s" in msgs[0]
    assert "stt+brain 2.94s" in msgs[0]
    assert "token→audio 0.94s" in msgs[0]


def test_no_split_when_audio_starts_without_a_synth_start():
    # Degenerate case: audio begins with no preceding TTSStartedFrame this turn → report the total only,
    # no (misleading) decomposition.
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


def _push_transcript(obs, text, t_secs):
    _push(obs, TranscriptionFrame(text, "user", "2026-06-23T00:00:00Z"), t_secs)


def test_cmd_tag_on_command_turn_when_enabled(monkeypatch):
    # #62: with RESPONSE_CMD_TAG on, a command-shaped utterance gets ` · cmd` so the drive can pick
    # turbo-eligible turns out of the Pi log and join them to the brain's `route via=turbo` receipts.
    monkeypatch.setenv("RESPONSE_CMD_TAG", "1")
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 10.0)
        _push_transcript(obs, "pause the music", 10.1)  # STT finalizes after the stop
        _push(obs, TTSStartedFrame(), 11.3)
        _push(obs, BotStartedSpeakingFrame(), 12.0)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "· cmd" in msgs[0]
    assert "stt+brain 1.30s" in msgs[0]  # decomposition still intact alongside the tag


def test_cmd_tag_remote_stt_ordering_transcript_before_user_stop(monkeypatch):
    # REGRESSION (2026-06-23 Pi drive): remote STT emits the TranscriptionFrame BEFORE
    # UserStoppedSpeakingFrame (the transcript drives end-of-turn). The observer must NOT clear the
    # transcript on user-stop, or the tag never fires for remote STT. (The local-STT-ordered test above
    # pushed transcript AFTER user-stop and missed this.)
    monkeypatch.setenv("RESPONSE_CMD_TAG", "1")
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push_transcript(obs, "turn it up", 9.9)      # transcript FIRST (remote STT)
        _push(obs, UserStoppedSpeakingFrame(), 10.0)  # ...then the stop
        _push(obs, TTSStartedFrame(), 11.3)
        _push(obs, BotStartedSpeakingFrame(), 12.0)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "· cmd" in msgs[0]


def test_cmd_tag_off_by_default(monkeypatch):
    # Default (flag unset): no tag — the feature is inert until a drive enables it.
    monkeypatch.delenv("RESPONSE_CMD_TAG", raising=False)
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 10.0)
        _push_transcript(obs, "pause the music", 10.1)
        _push(obs, TTSStartedFrame(), 11.3)
        _push(obs, BotStartedSpeakingFrame(), 12.0)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "· cmd" not in msgs[0]


def test_cmd_tag_absent_on_noncommand_turn(monkeypatch):
    # Enabled but the utterance isn't command-shaped → no tag (conversation turn).
    monkeypatch.setenv("RESPONSE_CMD_TAG", "1")
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 10.0)
        _push_transcript(obs, "what's the weather like today", 10.1)
        _push(obs, TTSStartedFrame(), 11.3)
        _push(obs, BotStartedSpeakingFrame(), 12.0)
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "· cmd" not in msgs[0]


def test_synth_start_only_counts_once_per_turn():
    # The split must use the FIRST synth start, not a later one, even across two user-stops in a turn.
    obs = ResponseLatencyObserver()
    msgs, sink = _capture_transcript_logs()
    try:
        _push(obs, UserStoppedSpeakingFrame(), 0.0)
        _push(obs, UserStoppedSpeakingFrame(), 1.0)  # re-arm: latest user-stop wins, synth-start resets
        _push(obs, TTSStartedFrame(), 2.0)           # 1.00s after the winning user-stop
        _push(obs, BotStartedSpeakingFrame(), 2.5)   # 0.50s token→audio
    finally:
        logger.remove(sink)
    assert len(msgs) == 1
    assert "RESPONSE | 1.50s" in msgs[0]
    assert "stt+brain 1.00s" in msgs[0]
    assert "token→audio 0.50s" in msgs[0]
