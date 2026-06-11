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
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from brains.brain_client import BrainClient, BrainEvent

_YES_WORDS = ("yes", "yeah", "yep", "yup", "confirm", "proceed", "go ahead", "do it", "affirmative", "sure")

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
        self._sleeping = False  # when True, ignore all input except a wake phrase
        # Whether an acoustic wake-word gate sits upstream (set by main.py once the gate is built). When
        # True, going to sleep forces that gate active so ambient TV never reaches STT and ANY transcript
        # arriving while asleep means "hey aria" already fired → wake. When False (no gate), fall back to
        # the text-phrase wake match (which a movie line can still trip — the degraded, gate-less mode).
        self._acoustic_wake_gated = False
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle LLMContextFrame by streaming the brain's reply; forward everything else."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                # Mute the user through the whole bot turn (not just while speaking): a SystemFrame
                # pushed UPSTREAM that BotThinkingMuteStrategy honors, closing the first-token latency
                # gap where an empty barge-in cancels the silent in-flight turn. Released in `finally`.
                await self._set_thinking(True)
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except Exception as e:  # noqa: BLE001 - surface as a pipeline error frame
                await self.push_error(error_msg=f"Brain error: {e}", exception=e)
            finally:
                await self._set_thinking(False)
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)

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

    async def _hold_wake(self, hold: bool) -> None:
        """Tell the wake gate (upstream, if present) to hold its command window open while a confirm is
        pending — so the user's yes/no answer needs no fresh wake — and release it after. No-op when
        there's no gate (the frame just flows to the transport)."""
        from wake_word import WakeHoldFrame

        await self.push_frame(WakeHoldFrame(hold=hold), FrameDirection.UPSTREAM)

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

    async def _process_context(self, context):
        user_text = self._latest_user_text(context)
        self._reply_buf = []
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

            # --- sleep/wake gate. When an acoustic wake gate is upstream (the normal config), going to
            # sleep FORCES it active, so ambient TV never reaches STT and a transcript can only arrive
            # here if "hey aria" already fired — so ANY transcript while asleep is the wake (no text
            # match against TV dialogue, which used to false-wake on lines like "we can wake up early").
            # Without a gate we degrade to the old text-phrase match. This is the real "stop listening".
            if self._sleeping:
                if self._acoustic_wake_gated or any(p in low_norm for p in _WAKE_PHRASES):
                    self._sleeping = False
                    await self._set_sleep_gate(False)
                    _tlog(f"WAKE  | {user_text!r}")
                    await self._speak("I'm awake. What do you need?")
                else:
                    _tlog(f"ASLEEP| ignoring: {user_text!r}")
                return
            # Sleep gets only the narration guard (is_dictation), NOT the standalone/residual one:
            # sleep phrasings legitimately carry qualifiers ("mute voice mode unless I call your name" —
            # the qualifier is itself a sleep phrase), which a residual-length test would wrongly reject.
            # A stray sleep is self-correcting (wake up); the session-killing risk was shutdown-only.
            if not is_meta and not is_dictation and any(p in low_norm for p in _SLEEP_PHRASES):
                self._sleeping = True
                await self._set_sleep_gate(True)
                _tlog(f"SLEEP | {user_text!r}")
                await self._speak("Going to sleep. Say 'wake up' when you need me.")
                return

            # Empty/near-empty turn (e.g. a force-completed runaway turn that captured no words, or a
            # stray VAD blip) → don't spend a brain turn on it. Pending confirms still resolve below.
            if self._pending_confirm is None and not user_text.strip():
                _tlog("SKIP  | empty user turn ignored")
                return

            # --- normal turn / confirm resume ---
            if self._pending_confirm is not None:
                pending = self._pending_confirm
                self._pending_confirm = None
                await self._hold_wake(False)  # confirm answered → release the wake-gate hold
                approved = self._parse_yes_no(user_text)
                _tlog(f"USER  | {user_text!r}  (confirm decision: approved={approved})")
                stream = self._client.confirm(self._session_id, pending["id"], approved)
            else:
                _tlog(f"USER  | {user_text!r}")
                stream = self._client.respond(self._session_id, user_text)
            await self._consume(stream)
        finally:
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
