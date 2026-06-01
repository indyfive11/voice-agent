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
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from brains.brain_client import BrainClient, BrainEvent

_YES_WORDS = ("yes", "yeah", "yep", "yup", "confirm", "proceed", "go ahead", "do it", "affirmative", "sure")

# Multi-word phrases only (single words like "sleep"/"wake" cause false hits, e.g. "otters sleep").
_SLEEP_PHRASES = ("go to sleep", "stop listening", "pause listening", "stop responding")
_WAKE_PHRASES = ("wake up", "i'm back", "are you awake", "you awake", "start listening")


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
        # Log the correlation key so the transcript can be lined up with the brain's own logs.
        _tlog(f"BRAIN | gabagent session_id={self._session_id}")

    @property
    def brain_client(self):
        """The underlying BrainClient (used by config.start_brain/stop_brain)."""
        return self._client

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handle LLMContextFrame by streaming the brain's reply; forward everything else."""
        await super().process_frame(frame, direction)

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
        try:
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
            if any(p in low for p in _SLEEP_PHRASES):
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
                        # Tier 3: out-of-band physical confirm, resolved in this same turn.
                        await self._speak(f"{summary}. Please confirm at the keyboard.")
                        if ev.reason:
                            await self._speak(ev.reason)
                        approved = await self._keyboard_confirm(summary, ev.reason)
                        _tlog(f"CONFIRM| keyboard decision: approved={approved}")
                        await self._consume(
                            self._client.confirm(self._session_id, ev.id, approved)
                        )
                        return
                    # spoken_yesno (Tier 2): two-turn — speak prompt, remember, end the turn.
                    self._pending_confirm = {"id": ev.id, "method": method}
                    prompt = summary if summary.lower().startswith(("i ", "i'll")) else f"I'll {summary}"
                    if ev.reason:
                        prompt += f" {ev.reason}"
                    await self._speak(f"{prompt}. Say yes to proceed, or no to cancel.")
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
