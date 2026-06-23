"""ResponseLatencyObserver — log the real user-perceived response latency per turn, decomposed.

The TurnTrackingObserver `on_turn_ended` duration is a conversation-turn *span*: it runs from the user's
first vocalization (incl. transcript-less false-starts) through any think-pause to the bot's completing
reply, so it can read as "90s" when the actual interaction was fine (2026-06-04 re-test, TURN #2). That
span is NOT latency. This observer logs the thing that matters: time from the user *finishing* speaking
(`UserStoppedSpeakingFrame`) to the bot *starting* to speak (`BotStartedSpeakingFrame`) — i.e. how long
Aria took to respond.

It also DECOMPOSES that total into the two halves the felt-latency lever splits on (the #60 first-audio
breakdown). The split boundary is the **first `TTSStartedFrame`** — the moment Kokoro actually begins
synthesizing spoken text — NOT the first `LLMTextFrame`:
  * **stt+brain** — `UserStoppedSpeakingFrame` → first `TTSStartedFrame` = STT-finalize + brain round-trip
    + model TTFT + any sentence streaming/aggregation and tool round-trips before a synthesizable sentence
    exists (all the BRAIN's contribution; the lever there is brain-side).
  * **token→audio** — first `TTSStartedFrame` → `BotStartedSpeakingFrame` = OUR TTS path (synth + playout).

Why the boundary is TTS-synth-start, not first-LLM-token (the 2026-06-22 joint-latency consensus with GA):
on a tool/status turn the brain streams a non-spoken `STATUS` line first (e.g. 'Trying Jellyfin…'), then
runs a tool call, THEN emits the real spoken sentence. Anchoring on the first `LLMTextFrame` (the STATUS)
absorbed that brain tool-time into `token→audio` — the 18:23:03 outlier read `token→audio 3.88s` when the
real Kokoro synth was 0.94s; the other 2.94s was the brain aggregating the spoken sentence. Kokoro only
ever synthesizes *spoken* text, so the first `TTSStartedFrame` excludes STATUS/non-spoken tokens by
construction (no brain-side frame tagging needed). This is an ATTRIBUTION fix — it does not change felt
latency, it relabels brain tool-time out of the TTS bucket so the decomposition stays honest. The output
label strings are unchanged so GA's per-turn join parser is unaffected.
One clean `RESPONSE | <total>s · stt+brain <a>s · token→audio <b>s` line per turn.
"""

from __future__ import annotations

import os
import re

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

# #62 command-turn tag (join aid for the per-room-media / turbo drive). A turbo-eligible turn is one whose
# text "looks like a command" — the brain routes those straight to the fast rung (`route via=turbo`, ~1–2s)
# and acts on the room's media. To pick those turns out of the Pi log ALONE (and join them to the brain's
# `route via=`/`gab_call ttft` receipts in voice_debug.jsonl), we tag the RESPONSE line with ` · cmd`.
# NOTE: the brain's `agent/router.py:_COMMAND_INTENT` is the AUTHORITATIVE classifier; this is a deliberately
# coarse high-level MIRROR for log-side filtering only — it may drift from the brain's exact regex and must
# never be treated as the routing decision. Off by default (`RESPONSE_CMD_TAG`), so it's inert until a drive.
_CMD_RE = re.compile(
    r"\bturn\s+(?:it|the|that|them)?\s*(?:up|down|on|off)\b"
    r"|\b(?:volume|louder|quieter|softer|mute|unmute|un-?pause|pause|resume|skip|rewind|stop|play|next|previous)\b"
    r"|\b(?:music|song|track|playlist|movie|show|video|album|tidal|jellyfin)\b",
    re.IGNORECASE,
)


class ResponseLatencyObserver(BaseObserver):
    """Emit a per-turn `RESPONSE | …` line: total user-stop→bot-start, split into brain vs. our-TTS halves."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._t_user_stop_ns: int | None = None
        self._t_synth_start_ns: int | None = None  # first TTSStartedFrame (Kokoro begins synth) since user-stop
        self._transcript: str | None = None  # latest user transcript this turn (for the #62 command tag)
        self._cmd_tag = os.environ.get("RESPONSE_CMD_TAG", "0") not in ("0", "false", "False", "")

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        if isinstance(frame, UserStoppedSpeakingFrame):
            # Latest user-stop before a bot reply wins (a turn may have intermediate stops); arm a fresh turn.
            self._t_user_stop_ns = data.timestamp
            self._t_synth_start_ns = None
            # NB: do NOT clear self._transcript here. REMOTE STT emits the TranscriptionFrame BEFORE
            # UserStoppedSpeakingFrame (the finalized transcript is what drives end-of-turn detection), so
            # clearing on stop wiped the turn's text before bot-start and the cmd tag never fired — caught on
            # the 2026-06-23 Pi drive ("turn it up"/"play me some music" went untagged). It's cleared after
            # the RESPONSE line is emitted, so each turn still uses only its own (most-recent) transcript.
        elif isinstance(frame, TranscriptionFrame):
            # The user's finalized text for this turn — kept only to tag command-shaped turns (#62 drive join).
            self._transcript = frame.text
        elif isinstance(frame, TTSStartedFrame):
            # First moment Kokoro actually synthesizes spoken text — the true brain→TTS handoff. Only the
            # first counts. Anchoring here (not the first LLMTextFrame) keeps a non-spoken STATUS line + the
            # brain's tool round-trip in the stt+brain bucket where they belong (2026-06-22 GA consensus).
            if self._t_user_stop_ns is not None and self._t_synth_start_ns is None:
                self._t_synth_start_ns = data.timestamp
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._t_user_stop_ns is not None:
                user_stop_ns = self._t_user_stop_ns
                synth_start_ns = self._t_synth_start_ns
                self._t_user_stop_ns = None
                self._t_synth_start_ns = None
                total = (data.timestamp - user_stop_ns) / 1e9
                if not (0 <= total < 600):  # ignore clock glitches / cross-turn carryover
                    return
                line = f"RESPONSE | {total:.2f}s (user-stop → bot-start)"
                # Decompose only when TTS synthesis actually started this turn (audio can begin with no prior
                # TTSStartedFrame in degenerate cases; skip the misleading split then).
                if synth_start_ns is not None:
                    to_synth = (synth_start_ns - user_stop_ns) / 1e9
                    synth_to_audio = (data.timestamp - synth_start_ns) / 1e9
                    if to_synth >= 0 and synth_to_audio >= 0:
                        line += f" · stt+brain {to_synth:.2f}s · token→audio {synth_to_audio:.2f}s"
                # #62: tag command-shaped turns so the drive can pick turbo-eligible turns out of the Pi log
                # and join them to the brain's `route via=turbo`/`gab_call ttft` receipts. Appended suffix only
                # — GA's parser keys on the unchanged `stt+brain`/`token→audio` labels, so it's join-safe.
                if self._cmd_tag and self._transcript and _CMD_RE.search(self._transcript):
                    line += " · cmd"
                self._transcript = None
                logger.bind(transcript=True).info(line)
