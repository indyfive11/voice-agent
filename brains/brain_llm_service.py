"""Pipecat LLM service that delegates to an external black-box brain.

Mirrors the OpenAI/Anthropic services' `process_frame` → `_process_context` pattern
(`pipecat/services/openai/base_llm.py:542`) but, instead of calling an LLM API, streams
text from a `BrainClient`. The brain owns its own tools/routing/state and returns only
speakable text, so Pipecat's own function-calling machinery is not used. Drops into the
pipeline exactly where the LLM goes; `main.py` and the aggregators are unchanged.

Confirmation is **turn-based** (you can't capture a spoken yes/no mid-response): on a
`confirm` event the service speaks the summary and ends the turn, remembering the pending
id; the user's next utterance is read as the decision and sent via `client.confirm(...)`,
whose continuation is streamed normally.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid

from loguru import logger

from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from brains.brain_client import BrainClient, BrainEvent

import aria_state
import bot_speech

_YES_WORDS = ("yes", "yeah", "yep", "yup", "confirm", "proceed", "go ahead", "do it", "affirmative", "sure")

# Explicit "back out entirely" phrases for the controllable-client play confirm (yes-here / no-new-window /
# cancel). A plain "no" must still mean *new window*, so these are matched separately and ONLY trigger cancel
# — never folded into the yes/no parse. Sent through the existing confirm `passphrase` field (brain accepts
# {cancel,cancelled,canceled,__cancel__}); no schema change. 2026-06-15 GA ask.
_CANCEL_PHRASES = (
    "cancel", "never mind", "nevermind", "forget it", "wrong movie", "wrong one",
    "neither", "don't play it", "dont play it", "do not play it", "stop it",
)

# Multi-word phrases only (single words like "sleep"/"wake" cause false hits, e.g. "otters sleep").
# "mute"-style requests map here too — live, the user asked to "mute voice mode" repeatedly and got
# pointed at *shutdown*; muting/"stop until I call you" IS this sleep gate. Deliberately NOT a bare
# "mute" — that collides with "mute the music" (a media command); require the voice-mode framing.
_SLEEP_PHRASES = (
    "go to sleep", "stop listening", "pause listening", "stop responding",
    "mute voice mode", "mute yourself", "mute your voice",
    "unless i call your name", "only respond when i call",
)
_WAKE_PHRASES = ("wake up", "i'm back", "are you awake", "you awake", "start listening")

# The assistant's spoken name — the PRIMARY wake while asleep. While asleep we leave sleep only on a
# transcript that names her (this vocative) or carries a wake phrase, never on bare ambient speech. A
# phantom acoustic spike on un-AEC'd room audio (a radio/TV the echo-canceller can't remove → wake_peak
# hits ceiling, which consec can't always reject) opens the mic and lets the next line of dialogue through;
# the old "any transcript while gated = wake" rule then woke her on it (2026-06-15: talk radio re-woke her
# on "Hike through the wilderness…"). Matched as a WHOLE WORD against the punctuation-stripped transcript
# (so "malaria" can't trip it). Conservative on purpose — a missed real wake costs one repeat, a false wake
# while asleep is the whole bug — so risky soundalikes ("area") are deliberately excluded. "ari" is the
# common STT clipping of "Aria" (2026-06-15 live: "Hey, Ari." left a residual that defeated the bare-
# vocative guard → an unwanted chit-chat reply); included as a name variant, not a generic soundalike.
_WAKE_VOCATIVES = ("aria", "arya", "ariya", "ari")

# Lenient vocative match — the wake-word strip and bare-wake handling, AND (since 2026-06-15) the
# asleep-wake gate. STT spells "Aria" many ways live (aria/arya/ariya/ari/ariette/ariane), so an exact
# list is whack-a-mole. A short "ari"/"ary" prefix catches the family; a denylist keeps real words
# (arid/arise/ariel/arial…) out. It does NOT loosen the asleep gate: a wake token/phrase is still REQUIRED
# there, so a NAMELESS ambient line stays asleep — the soundalike only recovers a mis-spelled name (the maintainer:
# "the wake word should not have an unnecessary response", and a real wake lost to STT noise just repeats).
_VOCATIVE_PREFIXES = ("ari", "ary")
_VOCATIVE_DENY = frozenset((
    "arid", "arise", "arises", "arisen", "arising", "aristocrat", "aristocratic",
    "arithmetic", "ariel", "arial", "aryan", "aryans",
))


def _looks_like_vocative(tok: str) -> bool:
    """Lenient: a token that is or resembles the wake name (see the note above)."""
    if tok in _WAKE_VOCATIVES:
        return True
    if tok in _VOCATIVE_DENY:
        return False
    return any(tok.startswith(p) for p in _VOCATIVE_PREFIXES)

# Clean exit of the whole voice agent. Deliberately specific to *this process* — never bare
# "shut down"/"power off", which are the brain's job (system control) and must pass through.
_SHUTDOWN_PHRASES = (
    "shut yourself down", "shut down voice mode", "shut down voice agent",
    "shut down the voice agent", "exit voice mode", "quit voice mode",
    "end voice mode", "turn yourself off", "stop the voice agent",
    # "close…" variants (live: "Close down voice mode" missed the gate). All require
    # "voice mode"/"voice agent" so they can't fire on "close the window/movie".
    "close voice mode", "close down voice mode", "close the voice agent",
    "close voice agent", "close down the voice agent",
)

# STT regularly mishears the safety-critical keyword "voice" (live: "shut down voice mode" →
# "shut down boys mode"), which silently misses the shutdown gate — the brain then can't exit and
# the agent keeps running. Canonicalize a soundalike → "voice" *only* when it directly precedes
# "mode"/"agent" (the exact shutdown/exit frame), so a bare soundalike elsewhere ("the boys are
# loud") can never trigger a false shutdown. Applied to the punctuation-normalized text.
_VOICE_SOUNDALIKE_RE = re.compile(
    r"\b(?:boys|boyce|boy s|voys|vois|voce|boise|voiced|boyz)\s+(mode|agent)\b"
)

# If the user is ASKING ABOUT a control rather than issuing it ("what's the command to make you
# stop listening?", "do you know how to turn yourself off?"), don't fire the gate — let it reach
# the brain so Aria can explain. Only guards the destructive gates (shutdown/sleep); wake stays
# permissive since a false wake is harmless.
_META_COMMAND_MARKERS = (
    "what's the command", "what is the command", "whats the command",
    "the command to", "the command for", "which command", "what command",
    "do you know how", "how do i", "how do you", "how can i", "how would i",
)

# The user is NARRATING, not commanding — a self-labelled dictation/aside. Live (2026-06-10) the maintainer
# dictated "Dictating. The last time we told you… shut down voice mode" and the trailing phrase, swept
# into one 15s runaway-turn blob, hit the shutdown gate and killed the session. The brain's intent
# filter classified the same utterance as an aside (no action), but the destructive gates run here,
# ahead of /respond, so they need their own narration guard. Markers checked against the leading clause.
_DICTATION_MARKERS = (
    "dictating", "just dictating", "dictation", "note for", "note to",
    "for the record", "transcribe", "take a note", "you don't need to respond",
    "no action needed", "don't respond to this", "ignore this",
)

# Filler/politeness/vocative tokens stripped before measuring how much of an utterance is "left over"
# around a destructive phrase. A deliberate command ("Aria, shut down voice mode please") reduces to
# ~nothing; a narration that merely CONTAINS the phrase keeps its content words ("the last time we told").
_COMMAND_FILLERS = frozenset((
    "okay", "ok", "alright", "please", "now", "then", "just", "ahead", "yeah", "yep", "yes",
    "thanks", "thank", "you", "can", "could", "would", "will", "and", "so", "hey", "aria",
    "the", "a", "to", "for", "me", "us", "it", "um", "uh",
))

# A destructive phrase buried in a long utterance fires only if the residual (after removing the matched
# phrase + fillers) is at most this many words. Tunable: raise it to allow wordier commands, lower to be
# stricter. Default 3 keeps "Aria please shut down voice mode, I'm done" firing while blocking a blob.
_DESTRUCTIVE_CMD_MAX_RESIDUAL_WORDS = int(os.environ.get("DESTRUCTIVE_CMD_MAX_RESIDUAL_WORDS", "3"))


def _matched_phrase(low_norm: str, phrases: tuple[str, ...]) -> str | None:
    """Return the first control phrase that appears in `low_norm`, else None."""
    return next((p for p in phrases if p in low_norm), None)


def _command_is_standalone(low_norm: str, phrase: str) -> bool:
    """True when `phrase` is the substance of the utterance, not buried in a longer narration blob.

    Removes the matched phrase, then strips filler/politeness/vocative tokens, and counts what's left.
    A deliberate destructive command leaves ~nothing; a dictation/runaway blob keeps its content words.
    Guards the shutdown gate so a phrase swept into a 15s force-completed turn can't kill the session.
    (Not applied to sleep, whose commands legitimately carry qualifiers — see the call site.)
    """
    residual = [w for w in low_norm.replace(phrase, " ").split() if w not in _COMMAND_FILLERS]
    return len(residual) <= _DESTRUCTIVE_CMD_MAX_RESIDUAL_WORDS


# Lead-in words that can precede the name in a wake ("hey aria", "ok aria", "yo aria"). Stripped with the
# name so the brain hears only the command. Kept short — these are conversational openers, not commands.
_WAKE_LEADINS = frozenset(("hey", "ok", "okay", "yo", "hi", "hello", "um", "uh", "so", "well"))

# Spoken only when the user says the wake word alone and never follows with a command (after a short wait).
# Natural-conversation flow: you answer the command after your name, not the name itself — and you ask
# "did you need something?" only when the name is left hanging (2026-06-15, the maintainer).
# Spoken on a bare wake ("Hey Aria" with no command) ONLY when no media is playing and BARE_WAKE_GREET=1
# — a short, prompt name-acknowledgment (the user, 2026-06-20: a quick "Yes?", not a chatty line). Override with
# BARE_WAKE_GREET_PHRASE.
_BARE_WAKE_PROMPT = os.environ.get("BARE_WAKE_GREET_PHRASE", "Yes?")

# Spoken when a turn fails because the brain dropped mid-request (crash / connection lost). Without an
# audible signal the turn just vanishes into dead air and the user waits on a reply that will never come
# (2026-06-17: gabagent died mid web-lookup — "incomplete chunked read" — and Aria went silent). TTS is
# local, so we can still speak even when the brain is unreachable.
_BRAIN_ERROR_REPLY = "Sorry, I lost my connection just then. Try that again in a moment."


def _strip_leading_wake(user_text: str, low_norm: str) -> tuple[bool, str, bool]:
    """Strip a LEADING wake trigger and return (had_wake, command, is_bare).

    A wake trigger is an optional lead-in ("hey"/"ok"/…) followed by the name (or a soundalike — see
    _looks_like_vocative), OR a wake phrase ("wake up", "are you awake") at the very start. Consecutive
    triggers are all stripped ("Hey Aria. Hey Ari." → nothing), and only LEADING ones, so a name used
    mid-sentence ("what's that opera Aria about") is left intact.

    `had_wake` is True when the utterance opened with a trigger; `command` is the remainder with original
    case preserved (the wake word is consumed by the voice-side gate and never sent to the brain). `is_bare`
    is True when, after the trigger, nothing meaningful is left — empty, or only fillers/leftover vocatives
    ("Hey Aria", "Hey Aria please", "Hey Aria. Hey Ari.") — i.e. the name was left hanging with no command.
    """
    toks = low_norm.split()
    consumed = 0
    had_wake = False
    while consumed < len(toks):
        i = consumed
        while i < len(toks) and toks[i] in _WAKE_LEADINS:
            i += 1
        if i < len(toks) and _looks_like_vocative(toks[i]):
            consumed = i + 1
        else:
            rest = " ".join(toks[i:])
            wp = next((p for p in _WAKE_PHRASES if rest == p or rest.startswith(p + " ")), None)
            if wp is None:
                break
            consumed = i + len(wp.split())
        had_wake = True
    if not had_wake:
        return False, user_text, False
    residual = [w for w in toks[consumed:] if w not in _COMMAND_FILLERS and not _looks_like_vocative(w)]
    command = " ".join(user_text.split()[consumed:])
    return True, command, not residual


def _fuse_bare_wake_likelihood(score: float, post_wake_ms: float | None, speech_dur_ms: float | None, *,
                               post_floor: float, post_ceil: float,
                               dur_floor: float, dur_ceil: float) -> float:
    """Fuse the acoustic wake into a 0..1 likelihood this utterance was NOTHING but the wake word (item C).

    POST-WAKE-DOMINANT — the discriminator the brain physically can't compute (it never sees audio): a bare
    wake = the wake fires and then ~nothing follows (small post-wake span); a command = the wake then the
    command words (large post-wake span). post_wake separates them even when TOTAL durations overlap (a slow
    bare wake can run longer than a fast command, so total duration alone inverts — bare 0.49 < command 0.67,
    live 2026-06-20). Below `post_floor` → ~bare; above `post_ceil` → ~command. Falls back to total
    speech_dur only when post can't be measured (wake at/after the VAD stop, e.g. the name trailing a
    sentence). conf gates it (a marginal wake can't yield a high likelihood). Neither measurable → 0.5*conf
    (deliberately under the brain's 0.8 suppress floor — never aggressively suppress on a blind guess).

    The band is a CROSS-HARDWARE DEFAULT — a best guess from our condenser + reSpeaker data (bare post ~130ms;
    command post ~1000-4100ms), tunable per deployment. On a clean mic this path rarely fires (STT recognizes
    the bare wake → it never reaches here); it's the backstop for setups where STT garbles "Hey Aria". The
    brain's LLM is the second key on every high score, so the band only needs to be roughly right."""
    conf = max(0.0, min(1.0, score))
    if post_wake_ms is not None:
        span = max(1.0, post_ceil - post_floor)
        x = max(0.0, min(1.0, (post_ceil - post_wake_ms) / span))
        return round(x * conf, 3)
    if speech_dur_ms is not None:
        span = max(1.0, dur_ceil - dur_floor)
        x = max(0.0, min(1.0, (dur_ceil - speech_dur_ms) / span))
        return round(x * conf, 3)
    return round(0.5 * conf, 3)


def _tlog(message: str) -> None:
    """Write one line to the transcript log (loguru sink filtered on extra['transcript'])."""
    logger.bind(transcript=True).info(message)


class BrainLLMService(LLMService):
    """Streams an external brain's response into the voice pipeline."""

    def __init__(self, client: BrainClient, *, session_id: str | None = None, **kwargs):
        # The brain owns generation; we have no LLM knobs. Pass a fully-initialized (all-None)
        # settings so LLMService's completeness check doesn't log an error every startup.
        kwargs.setdefault(
            "settings",
            LLMSettings(
                model=None, system_instruction=None, temperature=None, max_tokens=None,
                top_p=None, top_k=None, frequency_penalty=None, presence_penalty=None,
                seed=None, filter_incomplete_user_turns=None, user_turn_completion_config=None,
            ),
        )
        super().__init__(**kwargs)
        self._client = client
        self._session_id = session_id or uuid.uuid4().hex
        self._pending_confirm: dict | None = None  # {"id":…, "method":…}
        self._reply_buf: list[str] = []  # accumulates this turn's spoken text (transcript log)
        # Speak a brain-down apology at most once per outage — re-armed by the first turn that completes
        # cleanly — so a fully-dead brain doesn't repeat "I lost my connection" on every subsequent turn.
        self._brain_error_spoken = False
        self._sleeping = False  # when True, ignore all input except a naming/wake-phrase transcript
        # Whether an acoustic wake-word gate sits upstream (set by main.py once the gate is built). Going to
        # sleep still forces that gate active (so ambient audio is mostly held out of STT), but it no longer
        # decides the wake: the wake-from-sleep transcript must name her / carry a wake phrase regardless,
        # because a phantom acoustic spike on un-AEC'd room audio can open the gate without a real wake
        # (2026-06-15). Kept for the main.py wiring and possible diagnostics; see the sleep branch below.
        self._acoustic_wake_gated = False
        # Bare-wake prompt: when the user says only the wake word, wait this long for a command before
        # asking "did you need something?" — a following command cancels the wait (see _arm/_cancel).
        self._bare_wake_delay = float(os.environ.get("BARE_WAKE_PROMPT_DELAY", "3.5"))
        self._bare_wake_task: asyncio.Task | None = None
        # The blind bare-wake timer used to fire even while the user was mid-command (it only cancelled on a
        # NEW transcript, which Whisper delivers seconds late) → "Yes? Did you need something?" talked OVER the
        # command and its bot-stop un-ducked the music mid-repeat (2026-06-18 22:29). Track in-flight user
        # speech off VADUserStartedSpeakingFrame (a SystemFrame that reaches this service unmuted at onset —
        # the same onset the upstream duck controller acts on) so the greeting can be cancelled/skipped the
        # instant the user starts answering, not only once the transcript lands.
        self._user_speaking = False
        # Refractory latch: once we've greeted (or are awaiting the command after a bare wake), don't RE-ARM
        # the timer off a second bare "Hey Aria" / TTS bleed — that re-arm is what double-greeted (22:29:40 +
        # 22:29:44). Cleared by the next real spoken turn (_process_context), so a genuine fresh bare wake
        # later in the session still greets normally.
        self._bare_wake_greeted = False
        # Listen-first (the maintainer, 2026-06-19): a bare wake gets NO spoken greeting by default — she opens the floor
        # and listens (the eye shows 'listening'), and speaks only in response to an actual command. A fixed
        # timer greeting can't help a user who pauses to formulate (4s pause → greeting fired right as he began
        # his command → barge-in cut her, 22:10:51 live). Default OFF; set BARE_WAKE_GREET=1 to restore the
        # old "Yes? Did you need something?" prompt (kept behind the flag, still speech-aware when on).
        self._bare_wake_greet = os.environ.get("BARE_WAKE_GREET", "0") not in ("0", "false", "False")
        # Re-sleep: when she's woken FROM SLEEP by a bare wake and no command follows, return to sleep silently
        # rather than lingering awake — "listen, wait, then go back to sleep if nothing comes" (the maintainer). A real
        # command cancels it. Only armed on a wake-from-sleep bare wake (an already-awake bare wake just lets
        # the gate's window idle-close). >= the wake window so a slow command still lands first.
        self._resleep_secs = float(os.environ.get("BARE_WAKE_RESLEEP_SECS", "15.0"))
        self._resleep_task: asyncio.Task | None = None
        # The in-flight reply turn, run as a cancellable task (NOT inline) so a mid-generation barge-in can
        # cancel it. Cancelling raises CancelledError inside `_consume` → POST /cancel reaches the brain AND
        # the token stream stops feeding TTS (she stays stopped, no resume). 2026-06-16: when this ran inline,
        # `broadcast_interruption()` flushed the current TTS buffer but never cancelled this service's own
        # task, so a long reply kept streaming and she resumed ~5s after the barge.
        self._turn_task: asyncio.Task | None = None
        # Item C — out-of-band acoustic wake signal. A fresh WakeEventFrame (from the upstream gate) stamps
        # _pending_wake; consumed by exactly the next turn. VAD edges give the turn's speech duration, which
        # fuses (duration-dominant) into a `bare_wake_likelihood` forwarded on /respond when the voice-side
        # wake-strip fails to find the name in a garbled transcript ("Hey Aria"→"how are you?"). Lets the
        # brain's listen-first guard trust the audio over the text. No emitting gate → _pending_wake stays
        # None → exact current behavior (back-compat).
        self._pending_wake: dict | None = None  # {"score": float, "ts": monotonic}
        self._vad_start_ts: float | None = None
        self._vad_stop_ts: float | None = None
        # Post-wake-dominant fusion band (cross-hardware default; see _fuse_bare_wake_likelihood). post = speech
        # after the wake fires: bare wake ~130ms, command ~1000ms+. dur_* is the fallback when post is
        # unmeasurable (name trailing a sentence).
        self._wake_post_floor_ms = float(os.environ.get("WAKE_BARE_POST_FLOOR_MS", "250"))
        self._wake_post_ceil_ms = float(os.environ.get("WAKE_BARE_POST_CEIL_MS", "1000"))
        self._wake_dur_floor_ms = float(os.environ.get("WAKE_BARE_DUR_FLOOR_MS", "400"))
        self._wake_dur_ceil_ms = float(os.environ.get("WAKE_BARE_DUR_CEIL_MS", "1500"))
        self._wake_signal_max_age = 30.0  # ignore a stale pending wake that produced no prompt turn
        # Lever A (is_addressed fast-pass): a turn that LED with the wake word (voice side stripped the
        # vocative the brain can't see) is addressed by definition → forward a `wake.fresh` marker so the
        # brain's is_addressed gate can skip its arya LLM classify. Gated OFF by default until the brain
        # consumer is live (flip WAKE_FRESH_MARKER=1 jointly); unset = today's behavior, no marker.
        self._wake_fresh_enabled = os.environ.get("WAKE_FRESH_MARKER", "0") not in ("0", "false", "False")
        self._this_turn_led_with_wake = False  # reset per turn in _process_context
        self._this_turn_wake_conf: float | None = None
        self._this_turn_residual_words = 0  # command words after the stripped wake (aside-safety hint)
        # Log the correlation key so the transcript can be lined up with the brain's own logs.
        _tlog(f"BRAIN | gabagent session_id={self._session_id}")

    def set_acoustic_wake_gated(self, gated: bool) -> None:
        """Tell the brain whether an upstream wake-word gate exists (called from main.py after the gate is
        built). Drives acoustic-only wake-from-sleep vs the text-phrase fallback."""
        self._acoustic_wake_gated = gated

    @property
    def brain_client(self):
        """The underlying BrainClient (used by config.start_brain/stop_brain)."""
        return self._client

    @property
    def session_id(self) -> str:
        """The brain session key — shared with MediaDuckController so duck calls correlate."""
        return self._session_id

    @property
    def is_sleeping(self) -> bool:
        """True while asleep (used to gate media ducking — nothing to make room for)."""
        return self._sleeping

    @property
    def is_resleep_pending(self) -> bool:
        """True in the provisional window after a wake-FROM-SLEEP bare wake, before any command lands (a
        re-sleep timer is armed; she's listening but will doze again if nothing comes). Gates the media
        ONSET duck only: a FAILED wake-from-sleep (she wakes on "Hey Aria", no command follows, re-sleeps)
        shouldn't dip the music for nothing (2026-06-20 — the user heard the bed flap on each failed wake). A real
        command cancels the re-sleep, and its confirmed-speech still ducks via should_duck (not gated here)."""
        t = self._resleep_task
        return t is not None and not t.done()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle LLMContextFrame by streaming the brain's reply; forward everything else."""
        await super().process_frame(frame, direction)

        # Observe user-speech VAD onset/stop so the bare-wake greeting is speech-aware (see _user_speaking).
        # These are SystemFrames — observe and push through (do NOT consume); the duck controller / aggregators
        # downstream still need them. Onset also cancels any pending greeting outright: the user is answering,
        # so a "did you need something?" must never land on top of the command.
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            # item C: capture the FIRST onset of the turn only — a multi-segment utterance (VAD flickers on
            # internal pauses) must measure the full span, not just the last fragment. (Bug 2026-06-20: resetting
            # on every onset made "Play some Led Zeppelin" measure 227ms = just "...Zeppelin".) Reset to None at
            # the turn boundary (_process_context finally), so the next turn captures its own first onset.
            if self._vad_start_ts is None:
                self._vad_start_ts = time.monotonic()
            self._cancel_bare_wake_prompt()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._vad_stop_ts = time.monotonic()  # last stop wins → span = first onset … last stop
            await self.push_frame(frame, direction)
            return

        from wake_word import StopWordFrame, WakeEventFrame

        if isinstance(frame, WakeEventFrame):
            # Item C: a FRESH acoustic wake fired upstream. Stamp it for the next turn (consume-and-clear in
            # _process_context). Carries the detector score; combined with this turn's VAD span it tells the
            # brain whether the (often garbled) transcript was really just the wake word. Push through — it's
            # a harmless SystemFrame to everything downstream.
            self._pending_wake = {"score": float(frame.score), "ts": time.monotonic()}
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, StopWordFrame):
            # The wake gate detected the interrupt/stop word mid bot-turn. Two actions, both needed:
            #   1. broadcast_interruption() — flushes the CURRENT TTS buffer downstream (output) so she stops
            #      speaking now. Issued from HERE, downstream of the half-duplex user-mute that DROPS
            #      InterruptionFrame at the user aggregator (`_maybe_mute_frame`) — the gate's own broadcast /
            #      a top-injected InterruptionFrame were both swallowed by that mute (2026-06-16, twice).
            #   2. cancel the in-flight `_turn_task` — broadcast_interruption does NOT route the interruption
            #      back into THIS service, so on a mid-GENERATION barge `_consume` keeps streaming tokens →
            #      TTS resumes ~5s later. Cancelling raises CancelledError in `_consume` → POST /cancel
            #      reaches the brain (aborts the live turn) AND stops further tokens → she stays stopped.
            # For a short reply already done generating, `_turn_task` is .done() → cancel no-ops (flush-only,
            # which is correct — there's nothing live to cancel). Consume the frame (it has done its job).
            _tlog("BARGE-IN| stop word — interrupting bot turn (flush TTS + cancel)")
            await self.broadcast_interruption()
            if self._turn_task is not None and not self._turn_task.done():
                self._turn_task.cancel()
            return

        if isinstance(frame, LLMContextFrame):
            # Guard against an overlapping turn: in normal flow the half-duplex mute (BotThinkingMute +
            # AlwaysUserMute) keeps the user muted across the whole bot turn, so no new context arrives
            # mid-turn. But after a barge opens the floor the mute lifts; await any still-unwinding prior
            # turn before starting a new one so two `_consume` loops never run (and so the prior /cancel has
            # landed before this turn's first /respond — closes the follow-up 409 race by sequencing).
            await self._await_prior_turn()
            self._turn_task = asyncio.create_task(self._run_turn(frame.context))
        else:
            await self.push_frame(frame, direction)

    async def _await_prior_turn(self) -> None:
        """Cancel + await any in-flight turn task so a new turn never overlaps the old one."""
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            try:
                await task  # _run_turn swallows its own CancelledError, so this returns cleanly
            except Exception:  # noqa: BLE001 - any turn error was already surfaced inside _run_turn
                pass
        self._turn_task = None

    async def _run_turn(self, context) -> None:
        """Run one reply turn (stream brain → frames). Runs as a cancellable task (see `_turn_task`); a
        mid-generation barge cancels it → `_consume` CancelledError → /cancel + stop streaming to TTS."""
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            # eye: request sent to the brain. But a nano false-fire on un-AEC'd room audio while asleep
            # opens a wake-candidate window → STT → an LLMContextFrame → this turn, even though
            # _process_context will ASLEEP-ignore it. Don't flash `thinking` (and below, settle to
            # `sleeping` not `idle`) for that phantom turn — honor resting_state() via is_resting().
            aria_state.set_state("thinking" if not aria_state.is_resting() else aria_state.resting_state())
            # Mute the user through the whole bot turn (not just while speaking): a SystemFrame
            # pushed UPSTREAM that BotThinkingMuteStrategy honors, closing the first-token latency
            # gap where an empty barge-in cancels the silent in-flight turn. Released in `finally`.
            await self._set_thinking(True)
            await self.start_processing_metrics()
            await self._process_context(context)
            self._brain_error_spoken = False  # a turn completed cleanly → re-arm the brain-down notice
        except asyncio.CancelledError:
            # Barge-in cancelled this turn. `_consume` already issued POST /cancel on its way out; just
            # let the finally restore state. Do NOT re-raise — this is a clean, expected stop, and the
            # task should end normally rather than propagate cancellation into the event loop.
            _tlog("BARGE-IN| turn cancelled (barge-in)")
        except Exception as e:  # noqa: BLE001 - surface as a pipeline error frame
            await self.push_error(error_msg=f"Brain error: {e}", exception=e)
            # Don't leave the user in dead air: the ErrorFrame is only logged. If nothing was spoken yet
            # this turn, speak a brief apology (TTS is local — works even with the brain down) so a brain
            # crash degrades to "try again" instead of silence. Dedup'd so a fully-down brain doesn't
            # repeat it every turn (re-armed by the next clean turn).
            if not self._reply_buf and not self._brain_error_spoken:
                self._brain_error_spoken = True
                await self._speak(_BRAIN_ERROR_REPLY, immediate=True)
                _tlog("BOT   | " + _BRAIN_ERROR_REPLY)
        finally:
            await self._set_thinking(False)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())
            # eye: an aside / suppressed turn (incl. an asleep-ignored phantom wake) produces no TTS, so
            # no BotStarted/StoppedSpeaking ever returns the eye to rest — do it here. Honor
            # resting_state() (`idle` awake, `sleeping` asleep) so a phantom turn while asleep settles
            # back to `sleeping`, not a hardcoded `idle`. A spoken reply (reply_buf populated) is owned by the TTS path
            # (speaking→rest on the bot-speaking frames); leave it.
            if not self._reply_buf:
                aria_state.set_state(aria_state.resting_state())

    async def cleanup(self) -> None:
        """Pipeline teardown: cancel any in-flight turn task so a shutdown mid-reply unwinds cleanly."""
        await self._await_prior_turn()
        await super().cleanup()

    async def _speak(self, text: str, *, immediate: bool = False) -> None:
        """Push text to TTS and record it for the transcript log.

        immediate=True pushes a TTSSpeakFrame (synthesized at once as a standalone utterance) instead of
        an LLMTextFrame. Use it for short status fillers: the TTS SENTENCE aggregator holds a lone phrase
        ending in "…" waiting for a following non-whitespace lookahead char that never arrives until the
        (often slow) result text — so a plain-LLMTextFrame filler stays silent for ~20s then speaks
        bundled with the result. TTSSpeakFrame bypasses that buffer so "Trying tidal…" is heard now.
        """
        if text:
            self._reply_buf.append(text)
            await self.push_frame(TTSSpeakFrame(text) if immediate else LLMTextFrame(text))

    def is_floor_free(self) -> bool:
        """The floor authority: True when bot-initiated speech may be voiced WITHOUT stepping on the
        conversation — not asleep, no reply turn generating, the user isn't speaking, Aria's own TTS isn't
        playing, no confirm is awaiting its spoken yes/no, and no bare-wake greeting is armed. Read by the
        DeferredAnnouncer (announce.py) to time out-of-band announcements; conservative by design (a deferred
        result is never latency-critical, so it waits for a clean gap rather than risk a talk-over).

        NOT yet considered: convo-hold (a held bed expecting a follow-up), tracked upstream in media_duck.
        Speaking in a convo-hold gap is acceptable and the producer (Brick B) doesn't exist yet, so wiring
        convo-hold awareness waits until the real channel lands."""
        if self._sleeping:
            return False
        if self._turn_task is not None and not self._turn_task.done():
            return False
        if self._user_speaking:
            return False
        if bot_speech.bot_speaking():
            return False
        if self._pending_confirm is not None:
            return False
        if self._bare_wake_task is not None and not self._bare_wake_task.done():
            return False
        return True

    def _arm_bare_wake_prompt(self) -> None:
        """User said only the wake word: wait `_bare_wake_delay`, then ask "did you need something?" — but
        a command arriving first cancels it (the natural pause before you answer a hanging name).

        Two guards stop the greeting from talking over a command or doubling (2026-06-18 22:29):
        - `_user_speaking`: a speech onset already arrived (the user is answering, but the command transcript
          lags behind the bare-wake transcript) — skip; don't greet over the command.
        - `_bare_wake_greeted`: we've already greeted / are awaiting the command this episode — skip the re-arm
          off a second bare "Hey Aria" / TTS bleed (the cause of the double greeting). Cleared by the next real
          spoken turn in _process_context.
        """
        if not self._bare_wake_greet:
            return  # listen-first: no spoken greeting on a bare wake (default)
        if self._user_speaking or self._bare_wake_greeted:
            return
        self._cancel_bare_wake_prompt()
        self._bare_wake_task = asyncio.create_task(self._bare_wake_prompt())

    def _cancel_bare_wake_prompt(self) -> None:
        if self._bare_wake_task is not None and not self._bare_wake_task.done():
            self._bare_wake_task.cancel()
        self._bare_wake_task = None

    def _arm_resleep(self) -> None:
        """Woken from sleep by a bare wake → if no command follows within `_resleep_secs`, return to sleep
        silently (listen-first: she waited, nothing came). A real command/confirm cancels it (_cancel_resleep
        at the command point); a fresh bare wake re-arms it. No spoken goodbye — going back to sleep is quiet."""
        self._cancel_resleep()
        self._resleep_task = asyncio.create_task(self._resleep())

    def _cancel_resleep(self) -> None:
        if self._resleep_task is not None and not self._resleep_task.done():
            self._resleep_task.cancel()
        self._resleep_task = None

    async def _resleep(self) -> None:
        try:
            await asyncio.sleep(self._resleep_secs)
        except asyncio.CancelledError:
            return
        if not self._sleeping:
            self._sleeping = True
            await self._set_sleep_gate(True)
            _tlog("WAKE  | no command after wake-from-sleep — returning to sleep (silent)")
        self._resleep_task = None

    async def _bare_wake_prompt(self) -> None:
        try:
            await asyncio.sleep(self._bare_wake_delay)
        except asyncio.CancelledError:
            return
        # The user (2026-06-20): NEVER greet while ANY media is playing — the auto-duck already signals she's
        # listening, and a spoken "Yes?" inevitably talks over the command. Greet only in a quiet room. Poll
        # is best-effort; None (no media capability / brain unreachable) → treat as not-playing and greet (the
        # "generally nice to hear something" default). Checked at FIRE time so a track that started during the
        # delay is honored.
        ms = await self._client.media_state(self._session_id)
        if ms and ms.get("playing"):
            _tlog("WAKE  | bare wake — media playing, duck is the ack (no greeting)")
            self._bare_wake_greeted = True
            self._bare_wake_task = None
            return
        # No command followed the bare wake — prompt once. immediate=True so the short standalone phrase
        # isn't held in the TTS sentence aggregator (see _speak). Logged as its own BOT line.
        _tlog("WAKE  | bare wake unanswered — prompting")
        self._reply_buf = []
        await self._speak(_BARE_WAKE_PROMPT, immediate=True)
        _tlog("BOT   | " + _BARE_WAKE_PROMPT)
        # Latch: greeted this episode — a second bare wake / TTS bleed must not re-arm a second greeting.
        # Cleared by the next real spoken turn (_process_context).
        self._bare_wake_greeted = True
        self._bare_wake_task = None

    async def _hold_wake(self, hold: bool) -> None:
        """Tell the wake gate (upstream, if present) to hold its command window open while a confirm is
        pending — so the user's yes/no answer needs no fresh wake — and release it after. No-op when
        there's no gate (the frame just flows to the transport)."""
        from wake_word import WakeHoldFrame

        await self.push_frame(WakeHoldFrame(hold=hold), FrameDirection.UPSTREAM)

    async def _keepalive_wake(self, ttl_secs: float) -> None:
        """Media keepalive: tell the wake gate to hold its command window open for `ttl_secs` after a media
        command (the gate clamps to its ceiling and self-releases on expiry). No-op when there's no gate."""
        from wake_word import WakeHoldFrame

        await self.push_frame(
            WakeHoldFrame(hold=True, ttl_secs=ttl_secs), FrameDirection.UPSTREAM
        )

    async def _set_thinking(self, active: bool) -> None:
        """Bracket the bot turn for the user-mute strategy: push BotThinkingFrame UPSTREAM so the user is
        muted from turn start (before any TTS) until it ends. No-op effect when the strategy isn't wired
        (full-duplex / MUTE_DURING_THINKING=0); the frame just flows to the transport."""
        from turn_mute import BotThinkingFrame

        await self.push_frame(BotThinkingFrame(active=active), FrameDirection.UPSTREAM)

    async def _set_sleep_gate(self, asleep: bool) -> None:
        """Tell the wake gate the sleep state changed: while asleep it forces acoustic gating (mic muted,
        STT starved on ambient TV). Pushed UPSTREAM; no-op when no gate is present (frame flows on)."""
        from wake_word import WakeSleepFrame

        await self.push_frame(WakeSleepFrame(asleep=asleep), FrameDirection.UPSTREAM)

    async def _release_duck_for_aside(self) -> None:
        """Tell the media-duck controller (upstream) to drop the VAD-onset pre-duck: this turn was an aside,
        so no reply is coming and there's nothing to make room for. Pushed UPSTREAM; no-op when no duck
        controller is present (frame flows on). See DuckReleaseFrame for the contract."""
        from brains.media_duck import DuckReleaseFrame

        await self.push_frame(DuckReleaseFrame(), FrameDirection.UPSTREAM)

    async def _release_convo_hold(self) -> None:
        """Tell the media-duck controller (upstream) that this reply is terminal — restore the bed at the
        turn's end instead of holding it open for a follow-up. Pushed UPSTREAM; no-op when no duck
        controller is present (frame flows on). See ConvoReleaseFrame for the contract + ordering."""
        from brains.media_duck import ConvoReleaseFrame

        await self.push_frame(ConvoReleaseFrame(), FrameDirection.UPSTREAM)

    async def _set_voice_volume(self, op: str | None, value: float | None) -> None:
        """Adjust Aria's OWN voice (TTS) level — push DOWNSTREAM to the TTS-gain processor (which lives in the
        TTS PCM path, below this service). Distinct from the media duck (upstream). No-op if no gain processor
        is downstream (frame flows on). See tts_gain.VoiceVolumeFrame."""
        from tts_gain import VoiceVolumeFrame

        await self.push_frame(VoiceVolumeFrame(op=op or "set", value=value), FrameDirection.DOWNSTREAM)

    def _consume_wake_signal(self) -> dict | None:
        """Pop a fresh acoustic wake (item C) and fuse it into a `wake` object for /respond. Consume-and-clear:
        a wake initiates exactly ONE turn. Returns None if there's no fresh wake or it's stale. Built only on
        the strip-failed path (the caller gates that), so a clean wake/command never carries a redundant signal."""
        pw = self._pending_wake
        self._pending_wake = None
        if pw is None:
            return None
        if time.monotonic() - pw["ts"] > self._wake_signal_max_age:
            return None  # the wake never produced a prompt turn → drop it
        # This turn's speech span (VAD onset→offset) and the speech AFTER the wake fired (raw features for the
        # brain to calibrate the threshold against receipts; the likelihood is duration-dominant).
        speech_dur_ms = post_wake_ms = None
        if (self._vad_start_ts is not None and self._vad_stop_ts is not None
                and self._vad_stop_ts >= self._vad_start_ts):
            speech_dur_ms = round((self._vad_stop_ts - self._vad_start_ts) * 1000)
        if self._vad_stop_ts is not None and self._vad_stop_ts >= pw["ts"]:
            post_wake_ms = round((self._vad_stop_ts - pw["ts"]) * 1000)
            if speech_dur_ms is not None:
                post_wake_ms = min(post_wake_ms, speech_dur_ms)  # post-wake is a subset of total — clamp frame jitter
        likelihood = _fuse_bare_wake_likelihood(
            pw["score"], post_wake_ms, speech_dur_ms,
            post_floor=self._wake_post_floor_ms, post_ceil=self._wake_post_ceil_ms,
            dur_floor=self._wake_dur_floor_ms, dur_ceil=self._wake_dur_ceil_ms)
        wake = {"bare_wake_likelihood": likelihood, "confidence": round(float(pw["score"]), 3)}
        if post_wake_ms is not None:
            wake["post_wake_voiced_ms"] = post_wake_ms
        if speech_dur_ms is not None:
            wake["speech_dur_ms"] = speech_dur_ms
        return wake

    def _fresh_wake_marker(self) -> dict:
        """Lever A: this turn led with the wake word → addressed by definition. Emit `{fresh: true}` (plus
        the acoustic confidence when available) so the brain's is_addressed gate fast-passes without an arya
        LLM classify. Only built when WAKE_FRESH_MARKER is on AND the turn led with the wake word."""
        marker: dict = {"fresh": True, "residual_words": self._this_turn_residual_words}
        if self._this_turn_wake_conf is not None:
            marker["confidence"] = self._this_turn_wake_conf
        return marker

    async def _process_context(self, context):
        user_text = self._latest_user_text(context)
        self._reply_buf = []
        self._this_turn_led_with_wake = False  # Lever A: set when this turn's text led with the wake word
        self._this_turn_wake_conf = None
        self._this_turn_residual_words = 0
        self._last_status: str | None = None  # suppress repeated identical status spam
        self._suppress_next_status = False  # skip the transitional fallback status after an error
        low = user_text.strip().lower()
        # Normalize STT punctuation before matching control phrases: Whisper sprinkles commas/periods
        # ("Shut down, voice mode.") that defeat a plain substring test against punctuation-free
        # phrases ("shut down voice mode") — which is exactly how a live shutdown failed. Strip
        # non-word chars to spaces and collapse whitespace, then match the gates against this.
        low_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", low)).strip()
        # Repair STT mishears of the keyword "voice" in the "… mode/agent" frame (boys→voice etc.)
        # so a misheard but unambiguous shutdown phrase still fires the gate.
        low_norm = _VOICE_SOUNDALIKE_RE.sub(r"voice \1", low_norm)
        # A question *about* the controls, not a command — don't fire the destructive gates.
        is_meta = any(m in low_norm for m in _META_COMMAND_MARKERS)
        # NARRATION guard for the destructive gates: a dictation self-label in the leading clause means
        # the user is narrating, not commanding (the brain's intent filter agrees, but it runs after these
        # gates). Checked against the first few words so a marker mid-sentence ("…transcribe it") doesn't
        # over-suppress a real command.
        lead = " ".join(low_norm.split()[:4])
        is_dictation = any(m in lead for m in _DICTATION_MARKERS)
        try:
            # A new spoken turn answers any pending "did you need something?" prompt, so cancel it (a bare
            # wake below re-arms the wait). A blank/blip turn isn't a real utterance — leave it pending.
            # NOTE: the greeted latch is NOT cleared here — a *second bare wake* (incl. greeting TTS bleeding
            # back as "Hey Aria" with no fresh VAD onset, the 22:29:40 / 11:44:00 double) is also a non-empty
            # turn, and clearing here would let it re-arm a second greeting. The latch is cleared only on a
            # real command / confirm turn below (the episode is genuinely over there).
            if user_text.strip():
                self._cancel_bare_wake_prompt()

            # --- shutdown: a clean voice exit of the whole agent. Works in any state
            # (even asleep). Speak a goodbye, then request graceful pipeline closure by
            # pushing EndTaskFrame UPSTREAM — the task flushes queued frames (so the
            # goodbye is spoken) then ends; main.py's finally tears the brain down.
            # Gated against narration (is_dictation) and against a phrase buried in a longer blob
            # (_command_is_standalone) so a 15s runaway-turn that merely ENDS with the phrase can't exit.
            shutdown_phrase = _matched_phrase(low_norm, _SHUTDOWN_PHRASES)
            if (not is_meta and not is_dictation and shutdown_phrase is not None
                    and _command_is_standalone(low_norm, shutdown_phrase)):
                _tlog(f"SHUTDOWN| {user_text!r}")
                await self._speak("Shutting down voice mode. Goodbye.")
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                return

            # --- sleep/wake gate. While asleep, LEAVE sleep only on a transcript that names her (a wake
            # vocative or soundalike) or carries a wake phrase, AND is not itself a sleep phrase. The old
            # rule woke on ANY transcript whenever an acoustic gate was upstream — but a phantom acoustic
            # spike on un-AEC'd room audio (a radio/TV the echo-canceller can't remove) opens the window and
            # lets the next line of ambient dialogue through, which then "confirmed" the wake (2026-06-15:
            # talk radio woke her repeatedly on non-wake lines; "Stop listening." — a sleep phrase arriving
            # on an in-flight turn — also wrongly woke her). Requiring the name/phrase rejects both: radio
            # chatter carries no "Aria", and a sleep phrase can never be a wake. Cost is deliberately
            # asymmetric — a real wake lost to STT noise just needs a repeat; a false wake while asleep is
            # the whole bug. (_acoustic_wake_gated no longer gates this decision; the text requirement
            # applies in both the gated and the degraded gate-less configs.) The soundalike (lenient) match
            # is used here too — an STT mis-spelling of the name should still wake her — but it carries its
            # own denylist (arid/arise/…) and a wake token/phrase is still REQUIRED, so nameless chatter is
            # rejected exactly as before. After waking we DON'T greet: the shared wake-word handling below
            # (awake or just-woken) forwards a riding command and only prompts on a truly bare wake.
            woke_from_sleep = False
            if self._sleeping:
                tokens = low_norm.split()
                has_wake = (any(_looks_like_vocative(v) for v in tokens)
                            or any(p in low_norm for p in _WAKE_PHRASES))
                is_sleep = any(p in low_norm for p in _SLEEP_PHRASES)
                if not has_wake or is_sleep:
                    _tlog(f"ASLEEP| ignoring: {user_text!r}")
                    return
                self._sleeping = False
                woke_from_sleep = True
                await self._set_sleep_gate(False)
                _tlog(f"WAKE  | {user_text!r}")
                # fall through to the shared wake-word handling below
            # Sleep gets only the narration guard (is_dictation), NOT the standalone/residual one:
            # sleep phrasings legitimately carry qualifiers ("mute voice mode unless I call your name" —
            # the qualifier is itself a sleep phrase), which a residual-length test would wrongly reject.
            # A stray sleep is self-correcting (wake up); the session-killing risk was shutdown-only.
            if not is_meta and not is_dictation and any(p in low_norm for p in _SLEEP_PHRASES):
                self._sleeping = True
                self._cancel_resleep()  # explicit sleep supersedes any pending wake-from-sleep re-sleep
                await self._set_sleep_gate(True)
                _tlog(f"SLEEP | {user_text!r}")
                await self._speak("Going to sleep. Say 'wake up' when you need me.")
                return

            # Empty/near-empty turn (e.g. a force-completed runaway turn that captured no words, or a
            # stray VAD blip) → don't spend a brain turn on it. Pending confirms still resolve below.
            if self._pending_confirm is None and not user_text.strip():
                _tlog("SKIP  | empty user turn ignored")
                return

            # --- shared wake-word handling (awake or just-woken). Natural conversation: when someone says
            # your name then a command, you answer the COMMAND, not the name; you only answer the name when
            # it's left hanging with nothing after — and even then you wait a beat to be sure (2026-06-15,
            # the maintainer). So strip a LEADING wake trigger ("Hey Aria, play a movie" → "play a movie") and forward
            # only the command; the brain never sees the wake word (it's consumed by the voice-side gate).
            # A truly bare wake ("Hey Aria") forwards nothing — instead it arms a short timer that a
            # following command cancels, and prompts ("Yes? Did you need something?") only if none comes.
            # Skipped during a pending confirm (the answer is handled below).
            if self._pending_confirm is None:
                had_wake, command, is_bare = _strip_leading_wake(user_text, low_norm)
                if is_bare:
                    # Listen-first: open the floor and LISTEN (the eye shows 'listening'); no spoken greeting
                    # unless BARE_WAKE_GREET=1 (then _arm_bare_wake_prompt speaks). If this was a wake FROM
                    # SLEEP, arm a silent re-sleep so she returns to sleep when no command follows.
                    self._pending_wake = None  # clean bare wake — STT got the name; handled voice-side
                    self._arm_bare_wake_prompt()  # no-op when listen-first (greeting disabled)
                    if woke_from_sleep:
                        self._arm_resleep()
                    _tlog(f"VOCATIVE| {user_text!r} (bare wake — listening"
                          f"{', re-sleep armed' if woke_from_sleep else ''})")
                    return
                if had_wake:
                    user_text = command  # forward only what followed the wake word
                    # Lever A: capture that this turn led with the wake word (addressed by definition) before
                    # clearing the pending acoustic wake — used to mark `wake.fresh` for the brain's
                    # is_addressed fast-pass. The brain can't see the stripped vocative otherwise.
                    self._this_turn_led_with_wake = True
                    self._this_turn_residual_words = len(command.split())  # aside-safety hint for the gate
                    if self._pending_wake is not None:
                        self._this_turn_wake_conf = round(float(self._pending_wake["score"]), 3)
                    self._pending_wake = None  # name survived in the text → no out-of-band signal needed

            # A real command / confirm answer is being processed → the bare-wake episode is genuinely over.
            # Clear the greeted latch here (not on every non-empty turn) so a fresh bare wake LATER greets
            # again, while a second bare wake WITHIN the episode stays latched (returns above before reaching
            # this point). A command also means the wake-from-sleep produced a real turn → cancel any re-sleep.
            self._bare_wake_greeted = False
            self._cancel_resleep()

            # --- normal turn / confirm resume ---
            if self._pending_confirm is not None:
                pending = self._pending_confirm
                self._pending_confirm = None
                await self._hold_wake(False)  # confirm answered → release the wake-gate hold
                if self._is_cancel(user_text):
                    # Explicit "wrong one / back out" — distinct from "no" (= new window). The brain reads
                    # this through the existing confirm passphrase field and aborts without playing.
                    _tlog(f"USER  | {user_text!r}  (confirm decision: cancelled)")
                    stream = self._client.confirm(
                        self._session_id, pending["id"], approved=False, passphrase="cancel"
                    )
                else:
                    approved = self._parse_yes_no(user_text)
                    _tlog(f"USER  | {user_text!r}  (confirm decision: approved={approved})")
                    stream = self._client.confirm(self._session_id, pending["id"], approved)
            else:
                # Item C: on the strip-failed path (the COMMON case over music — STT lost/garbled the name,
                # `had_wake=False`) forward the out-of-band acoustic wake signal so the brain's listen-first
                # guard can trust the audio over the text. Clean wake/command turns cleared _pending_wake
                # above, so this is None for them (back-compat).
                # Lever A: a wake-led turn (clean strip) forwards a `wake.fresh` marker so the brain's
                # is_addressed gate fast-passes (skips its arya classify). The strip-failed path still
                # forwards the bare_wake_likelihood backstop. Mutually exclusive per turn.
                if self._wake_fresh_enabled and self._this_turn_led_with_wake:
                    wake_obj = self._fresh_wake_marker()
                else:
                    wake_obj = None if had_wake else self._consume_wake_signal()
                if not wake_obj:
                    wake_detail = ""
                elif wake_obj.get("fresh"):
                    wake_detail = f"  (wake fresh — addressed conf={wake_obj.get('confidence')})"
                else:
                    wake_detail = (f"  (wake-signal bwl={wake_obj['bare_wake_likelihood']:.2f} "
                                   f"conf={wake_obj['confidence']:.2f} dur={wake_obj.get('speech_dur_ms')}ms "
                                   f"post={wake_obj.get('post_wake_voiced_ms')}ms)")
                _tlog(f"USER  | {user_text!r}{wake_detail}")
                stream = self._client.respond(self._session_id, user_text, wake=wake_obj)
            await self._consume(stream)
        finally:
            # Turn boundary: clear the VAD span so the NEXT turn captures its own first-onset … last-stop
            # (item C — the span must not bleed across turns; runs on every exit path, incl. early returns).
            self._vad_start_ts = None
            self._vad_stop_ts = None
            if self._reply_buf:
                _tlog("BOT   | " + " ".join(self._reply_buf))

    async def _consume(self, stream):
        """Push brain events as Pipecat frames. Cancelled on barge-in → close cleanly."""
        try:
            async for ev in stream:  # type: BrainEvent
                if ev.type == "token":
                    await self._speak(ev.text)
                elif ev.type == "error":
                    # Turn-level failure: non-terminal (a `done` follows). Speak the speakable
                    # `text`; log the structured `summary` cause (not spoken). Suppress the brain's
                    # transitional fallback status that follows, so the failure isn't said twice.
                    _tlog(f"ERROR | {ev.summary!r}")
                    await self._speak(ev.text or "Sorry, I hit a problem.")
                    self._suppress_next_status = True
                elif ev.type == "status":
                    # The brain emits a status per agent-loop step; a multi-tool turn can repeat
                    # the same one many times ("Looking into it." ×8). Speak each only once in a row.
                    if self._suppress_next_status:
                        self._suppress_next_status = False
                        _tlog(f"STATUS| (post-error fallback suppressed) {ev.text!r}")
                    elif ev.text and ev.text != self._last_status:
                        self._last_status = ev.text
                        # Log-only: the user found the spoken filler ("Trying Jellyfin…") annoying, especially
                        # on fast turns. We keep it in the transcript but never voice it; Aria stays quiet
                        # during a tool call rather than narrating each step.
                        _tlog(f"STATUS| {ev.text!r}")
                    elif ev.text:
                        _tlog(f"STATUS| (dupe suppressed) {ev.text!r}")
                elif ev.type == "confirm":
                    method = ev.method or "spoken_yesno"
                    summary = ev.summary or "perform that action"
                    _tlog(f"CONFIRM| tier={ev.tier} method={method} {summary!r}")
                    if method == "keyboard":
                        # Tier 3: out-of-band physical confirm via an on-screen dialog (kdialog),
                        # resolved in this same turn. Never read a raw tool-call summary aloud —
                        # the dialog body carries the full detail visually; speak only a short,
                        # human heads-up. Refer to the on-screen prompt (it's a mouse dialog —
                        # don't say "keyboard", which is inaccurate and confusing).
                        if self._looks_like_raw_tool(summary):
                            await self._speak("This one needs your approval — "
                                              "please confirm in the prompt on screen.")
                        else:
                            await self._speak(f"{self._confirm_prompt(summary, None, tail=False)} "
                                              "Please confirm in the prompt on screen.")
                        approved = await self._keyboard_confirm(summary, ev.reason)
                        _tlog(f"CONFIRM| keyboard decision: approved={approved}")
                        await self._consume(
                            self._client.confirm(self._session_id, ev.id, approved)
                        )
                        return
                    # spoken_yesno (Tier 2): two-turn — speak prompt, remember, end the turn.
                    self._pending_confirm = {"id": ev.id, "method": method}
                    await self._hold_wake(True)  # hold the wake gate open for the yes/no answer
                    if ev.prompt_is_complete:
                        # summary is the ENTIRE spoken line (carries its own yes/no choice):
                        # speak verbatim, append nothing.
                        await self._speak(summary)
                    else:
                        await self._speak(self._confirm_prompt(summary, ev.reason))
                    return
                elif ev.type == "blocked":
                    # NOT a stream boundary (per protocol): speak the reason, keep consuming;
                    # more frames and eventually `done` follow on the same stream.
                    _tlog(f"BLOCKED| action={ev.action!r} {ev.reason!r}")
                    await self._speak(ev.reason or "That action is blocked.")
                elif ev.type == "addressed":
                    # The brain's intent filter classified this turn as an aside (not addressed to Aria).
                    # The brain only ever emits this event on suppression, so its ARRIVAL — not the
                    # `addressed` bool, which a brain-side serializer could drop (False == 0) — is the
                    # signal. Release the VAD-onset pre-duck now; no reply is coming. NOT a stream
                    # boundary: keep consuming (a `done` still follows).
                    _tlog(f"ASIDE | not addressed (addressed={ev.addressed!r}) → release duck")
                    await self._release_duck_for_aside()
                elif ev.type == "wake_hold":
                    # Media keepalive: the brain ran a media command, so hold the wake gate's command window
                    # open for ttl_secs (a follow-up media command then needs no re-wake). The gate clamps
                    # the TTL to its ceiling and self-releases on expiry. NOT a stream boundary: keep
                    # consuming (a `done` still follows).
                    ttl = ev.ttl_secs
                    if ttl and ttl > 0:
                        _tlog(f"KEEPALIVE| media — hold wake window {ttl:.0f}s")
                        await self._keepalive_wake(ttl)
                elif ev.type == "convo_hold":
                    # Turn-terminality hint: the brain judged this reply terminal (one-shot Q&A / dismiss),
                    # so the media conversation-hold should restore the bed at this turn's end rather than
                    # holding it open for a follow-up. Arrival-keyed (like `addressed`/`wake_hold`): the
                    # brain only ever emits it to release. NOT a stream boundary: keep consuming (`done`
                    # still follows). No-op if no media controller is downstream (frame just flows on).
                    _tlog(f"CONVO | brain release hint (release={ev.release!r}) → no convo-hold this turn")
                    await self._release_convo_hold()
                elif ev.type == "voice_volume":
                    # The brain classified a my-voice-volume intent → adjust Aria's OWN TTS level (distinct
                    # from media volume). Push the change DOWNSTREAM to the TTS-gain processor (which lives in
                    # the TTS PCM path). NOT a stream boundary: keep consuming (`done` still follows). A no-op
                    # if no gain processor is downstream (frame just flows on).
                    _tlog(f"VOICE | own-volume hint op={ev.op!r} value={ev.value!r}")
                    await self._set_voice_volume(ev.op, ev.value)
                elif ev.type == "done":
                    return
        except asyncio.CancelledError:
            # Barge-in: tell the brain to abort the turn (NOT a full teardown — the
            # client/subprocess stay up for the next turn).
            _tlog(f"BARGE-IN| interrupted; said so far: {' '.join(self._reply_buf)!r}")
            try:
                await self._client.cancel(self._session_id)
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            # Always release this turn's stream (server already ended it after confirm/done).
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - best-effort
                    pass

    @staticmethod
    def _looks_like_raw_tool(summary: str) -> bool:
        """True if a summary looks like an un-speakable raw tool call (not for TTS)."""
        s = summary or ""
        return ("\n" in s) or ("(content=" in s) or ("command_id=" in s) or ("args=" in s)

    @staticmethod
    def _confirm_prompt(summary: str, reason: str | None = None, *, tail: bool = True) -> str:
        """Build the spoken Tier-2 confirm line, robust to either summary style.

        The brain may send an imperative fragment ("edit the README") or a fully-formed
        spoken question ("Play The Matrix in Jellyfin?"). Wrap the former with "I'll …"
        (lower-casing its leading verb so it reads as a sentence, not "I'll Open …"); speak
        the latter verbatim. With `tail`, append a single clean yes/no instruction — unless
        the summary already poses its own choice (a "yes…/no…" line), in which case we don't
        double it up. `tail=False` returns just the action clause (used by the keyboard path).
        """
        s = (summary or "perform that action").strip()
        if s.endswith((".", "?", "!")):
            prompt = s  # already a complete spoken line
        elif s.lower().startswith(("i ", "i'll", "i'd", "i will")):
            prompt = f"{s}."
        else:
            prompt = f"I'll {s[:1].lower()}{s[1:]}."  # "Open a URL" -> "I'll open a URL."
        if reason:
            prompt += f" {reason.strip()}"
        if not tail:
            return prompt
        # If the brain already spelled out the yes/no choice, don't append ours.
        low = prompt.lower()
        if "yes to" in low or "say yes" in low or ("yes or no" in low):
            return prompt
        return f"{prompt} Say yes to proceed, or no to cancel."

    @staticmethod
    def _latest_user_text(context) -> str:
        for msg in reversed(context.get_messages()):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "user":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            return part.get("text", "")
        return ""

    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        t = (text or "").strip().lower()
        return any(w in t for w in _YES_WORDS)

    @staticmethod
    def _is_cancel(text: str) -> bool:
        """True only for explicit back-out phrases. A plain 'no' is NOT cancel (it = new window)."""
        t = (text or "").strip().lower()
        return any(p in t for p in _CANCEL_PHRASES)

    async def _keyboard_confirm(self, summary: str, reason: str | None = None) -> bool:
        """Out-of-band physical confirm for Tier-3 actions (KDE kdialog, terminal fallback).

        Fail-closed: any error → not approved.
        """
        body = f"Voice agent wants to: {summary}"
        if reason:
            body += f"\n\n{reason}"
        body += "\n\nAllow this action?"
        kdialog = shutil.which("kdialog")
        try:
            if kdialog:
                proc = await asyncio.create_subprocess_exec(
                    kdialog, "--title", "Voice agent — confirm", "--yesno", body
                )
                rc = await proc.wait()
                return rc == 0  # kdialog: 0 = Yes
            loop = asyncio.get_event_loop()
            ans = await loop.run_in_executor(
                None, input, f"[CONFIRM] {summary} — type 'yes' to allow: "
            )
            return ans.strip().lower() in ("y", "yes")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self}: keyboard confirm failed ({e}); denying.")
            return False
