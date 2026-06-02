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
import shutil
import uuid

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndTaskFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from brains.brain_client import BrainClient, BrainEvent

_YES_WORDS = ("yes", "yeah", "yep", "yup", "confirm", "proceed", "go ahead", "do it", "affirmative", "sure")

# Multi-word phrases only (single words like "sleep"/"wake" cause false hits, e.g. "otters sleep").
_SLEEP_PHRASES = ("go to sleep", "stop listening", "pause listening", "stop responding")
_WAKE_PHRASES = ("wake up", "i'm back", "are you awake", "you awake", "start listening")

# Clean exit of the whole voice agent. Deliberately specific to *this process* — never bare
# "shut down"/"power off", which are the brain's job (system control) and must pass through.
_SHUTDOWN_PHRASES = (
    "shut yourself down", "shut down voice mode", "shut down voice agent",
    "shut down the voice agent", "exit voice mode", "quit voice mode",
    "end voice mode", "turn yourself off", "stop the voice agent",
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
        # Media ducking: while the user speaks (and through Aria's reply), ask the brain to duck
        # music / pause video so playback doesn't drown out the mic (or get transcribed itself).
        self._media_ducked = False
        self._bot_spoke_since_duck = False
        self._restore_timer: asyncio.Task | None = None
        self._restore_grace = 2.0  # fallback restore if a ducked turn yields no bot speech
        # Log the correlation key so the transcript can be lined up with the brain's own logs.
        _tlog(f"BRAIN | gabagent session_id={self._session_id}")

    @property
    def brain_client(self):
        """The underlying BrainClient (used by config.start_brain/stop_brain)."""
        return self._client

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle LLMContextFrame by streaming the brain's reply; forward everything else."""
        await super().process_frame(frame, direction)

        self._maybe_duck(frame)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except Exception as e:  # noqa: BLE001 - surface as a pipeline error frame
                await self.push_error(error_msg=f"Brain error: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)

    def _maybe_duck(self, frame: Frame) -> None:
        """Drive media ducking off the VAD/bot speaking frames flowing through the pipeline.

        Duck on user-speech onset; restore when Aria finishes speaking (so playback stays low /
        paused through the whole exchange, not just while the user talks). A fallback timer
        restores if a ducked turn produces no bot speech. Skipped while asleep (we're ignoring
        the user, so there's nothing to make room for).
        """
        if isinstance(frame, UserStartedSpeakingFrame):
            if self._sleeping:
                return
            self._cancel_restore_timer()
            if not self._media_ducked:
                self._media_ducked = True
                self._bot_spoke_since_duck = False
                self._fire_duck(True)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_spoke_since_duck = True
            self._cancel_restore_timer()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._media_ducked:
                self._media_ducked = False
                self._fire_duck(False)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # If the bot never speaks this turn (e.g. an ignored/no-op turn), restore anyway.
            if self._media_ducked and not self._bot_spoke_since_duck:
                self._arm_restore_timer()

    def _fire_duck(self, on: bool) -> None:
        """Fire-and-forget the brain duck call — never block the audio pipeline."""
        duck = getattr(self._client, "duck", None)
        if duck is None:
            return

        async def _go():
            try:
                await duck(self._session_id, on)
            except Exception:  # noqa: BLE001 - best-effort; media control must never break audio
                pass

        asyncio.create_task(_go())

    def _arm_restore_timer(self) -> None:
        self._cancel_restore_timer()

        async def _later():
            try:
                await asyncio.sleep(self._restore_grace)
                if self._media_ducked and not self._bot_spoke_since_duck:
                    self._media_ducked = False
                    self._fire_duck(False)
            except asyncio.CancelledError:
                pass

        self._restore_timer = asyncio.create_task(_later())

    def _cancel_restore_timer(self) -> None:
        if self._restore_timer is not None and not self._restore_timer.done():
            self._restore_timer.cancel()
        self._restore_timer = None

    async def _speak(self, text: str) -> None:
        """Push text to TTS and record it for the transcript log."""
        if text:
            self._reply_buf.append(text)
            await self.push_frame(LLMTextFrame(text))

    async def _process_context(self, context):
        user_text = self._latest_user_text(context)
        self._reply_buf = []
        self._last_status: str | None = None  # suppress repeated identical status spam
        self._suppress_next_status = False  # skip the transitional fallback status after an error
        low = user_text.strip().lower()
        # A question *about* the controls, not a command — don't fire the destructive gates.
        is_meta = any(m in low for m in _META_COMMAND_MARKERS)
        try:
            # --- shutdown: a clean voice exit of the whole agent. Works in any state
            # (even asleep). Speak a goodbye, then request graceful pipeline closure by
            # pushing EndTaskFrame UPSTREAM — the task flushes queued frames (so the
            # goodbye is spoken) then ends; main.py's finally tears the brain down.
            if not is_meta and any(p in low for p in _SHUTDOWN_PHRASES):
                _tlog(f"SHUTDOWN| {user_text!r}")
                await self._speak("Shutting down voice mode. Goodbye.")
                await self.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
                return

            # --- sleep/wake gate (the mic stays on; while asleep we ignore all but a wake
            # phrase, and never call the brain). This is the real "stop listening" control.
            if self._sleeping:
                if any(p in low for p in _WAKE_PHRASES):
                    self._sleeping = False
                    _tlog(f"WAKE  | {user_text!r}")
                    await self._speak("I'm awake. What do you need?")
                else:
                    _tlog(f"ASLEEP| ignoring: {user_text!r}")
                return
            if not is_meta and any(p in low for p in _SLEEP_PHRASES):
                self._sleeping = True
                _tlog(f"SLEEP | {user_text!r}")
                await self._speak("Going to sleep. Say 'wake up' when you need me.")
                return

            # --- normal turn / confirm resume ---
            if self._pending_confirm is not None:
                pending = self._pending_confirm
                self._pending_confirm = None
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
                        _tlog(f"STATUS| {ev.text!r}")
                        await self._speak(ev.text)
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
