"""Transport-agnostic interface to an external "brain" (Brain protocol).

A brain is any local process that takes a user utterance and streams back speakable
text, handling its own tools/state internally — plus a confirmation round-trip for gated
actions. voice-agent talks to it through this interface; the concrete transport
(HTTP/SSE vs stdio-JSONL) is a separate implementation chosen once the gabagent audit
locks the protocol. Only `FakeBrainClient` (for offline tests) ships today.

See ~/dev/gabagent/VOICE_MODE_HANDOFF.md for the protocol this mirrors.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass
class BrainEvent:
    """One event from a brain's response/confirm stream.

    type:
      - "token"     : a chunk of assistant text to speak (`text`)
      - "status"    : optional short narration to speak (`text`)
      - "confirm"   : a gated action awaiting a decision (`id`, `tier`, `method`, `summary`, `reason`)
      - "blocked"   : a refused action to announce (`reason`, `action`)
      - "error"     : a turn-level failure to speak (`text` speakable, `summary` structured cause)
      - "addressed" : the brain's "was this turn addressed to me?" verdict (`addressed`). Emitted by the
                      brain ONLY on suppression (an aside, `addressed:false`), the instant its intent
                      classifier returns — before any token/`done`. Lets voice-agent release a media duck
                      that the VAD-onset pre-duck engaged for an utterance that turns out not to be a
                      command (no reply → nothing to make room for). `addressed:true` is never emitted, so
                      the mere ARRIVAL of this event means "not addressed" — see brain_llm_service.
      - "wake_hold" : media keepalive (`hold:true`, `ttl_secs`). Emitted after any media-domain command,
                      before `done`, so the wake gate holds its command window open for `ttl_secs` and a
                      follow-up media command needs no re-wake. Release is by TTL (the brain never sends a
                      false) + the gate's own ceiling. Refreshed by each new media command.
      - "convo_hold": conversation-hold turn-terminality hint over playing media. Emitted before `done`
                      when the brain judges the reply TERMINAL (a one-shot Q&A, or an explicit dismiss
                      like "thanks, that's all") — so the media conversation-hold should NOT keep the bed
                      ducked for a follow-up but restore at this turn's end. Arrival-keyed (the brain
                      emits it only to release; `release` is carried for logging). A dropped event degrades
                      safely to the timed conversation-hold. (Extend — hold LONGER — is a future variant;
                      it must pair with `wake_hold` to also hold the gate window, see media_duck.)
      - "done"      : end of this turn
    """

    type: str
    text: str | None = None
    id: str | None = None
    tier: int | None = None
    method: str | None = None  # "spoken_yesno" | "keyboard" | "passphrase"
    summary: str | None = None
    reason: str | None = None
    action: str | None = None
    # confirm only: when True, `summary` is the ENTIRE spoken line (it carries its own
    # yes/no choice) — speak it verbatim and append no proceed/cancel tail. Omitted (None)
    # = normal Option A: `summary` is a bare action and voice-agent owns the yes/no tail.
    prompt_is_complete: bool | None = None
    # "addressed" event only: the verdict bool (always False on the wire — the brain emits the event
    # solely on suppression). Carried for logging; the handler keys on the event TYPE, not this value,
    # so a brain-side serializer that drops a literal `False` (Python `False == 0`) can't disarm it.
    addressed: bool | None = None
    # "wake_hold" event only: media keepalive. `hold` is always True on the wire (release is by TTL, never a
    # false); `ttl_secs` is how long the wake gate should hold its command window open from receipt.
    hold: bool | None = None
    ttl_secs: float | None = None
    # "convo_hold" event only: turn-terminality hint. `release` is True on the wire (the brain emits the
    # event only to release the conversation-hold); the handler keys on the event TYPE, not this value, so a
    # serializer that drops a literal True can't disarm it. (Carried for logging / future `extend` variant.)
    release: bool | None = None


@runtime_checkable
class BrainClient(Protocol):
    """What voice-agent needs from any brain. Implementations are async generators."""

    def respond(self, session_id: str, text: str) -> AsyncIterator[BrainEvent]:
        """Stream the brain's reply to a user utterance."""
        ...

    def confirm(
        self, session_id: str, confirm_id: str, approved: bool, passphrase: str | None = None
    ) -> AsyncIterator[BrainEvent]:
        """Resume a paused turn after a gated action's decision; stream the continuation."""
        ...

    async def cancel(self, session_id: str) -> None:
        """Abort the in-flight turn (barge-in) without tearing the client down."""
        ...

    async def duck(self, session_id: str, on: bool, mute: bool = False) -> None:
        """Best-effort: tell the brain to lower (on=True) or restore (on=False) the volume of its
        active media while the user is speaking, so playback doesn't drown out the mic. `mute=True`
        deepens the duck to a full mute (volume 0) for an open wake/command window, so a music vocal
        can't bleed into the transcription (per VOICE_PROTOCOL `/media/duck {on,mute?}`; defaults to a
        plain duck). No-op if the brain controls no media; must never raise."""
        ...

    async def media_state(self, session_id: str) -> dict | None:
        """Best-effort, provider-neutral playback snapshot: {"playing": bool, "state":
        "playing"|"paused"|"idle"} so the caller can skip ducking when nothing is playing. Brain-
        agnostic by contract — no per-provider keys. None if the brain doesn't expose it (older brain /
        404) or on any error; never raises."""
        ...

    async def aclose(self) -> None:
        """Full teardown — release transport + any spawned brain process (shutdown)."""
        ...


class FakeBrainClient:
    """Canned-event brain for offline tests. Satisfies BrainClient structurally.

    Optionally sleeps `delay` seconds between events so cancellation/barge-in can be
    exercised deterministically.
    """

    def __init__(
        self,
        respond_events: list[BrainEvent] | None = None,
        confirm_events: list[BrainEvent] | None = None,
        *,
        delay: float = 0.0,
    ):
        self._respond_events = respond_events or []
        self._confirm_events = confirm_events or []
        self._delay = delay
        self.closed = False
        self.respond_calls: list[tuple[str, str]] = []
        self.confirm_calls: list[tuple[str, str, bool, str | None]] = []
        self.cancel_calls: list[str] = []
        self.duck_calls: list[tuple[str, bool]] = []
        self.duck_mute_calls: list[tuple[str, bool, bool]] = []  # (session, on, mute)
        # Tests set this to drive media_state(); None → capability absent (callers assume "playing").
        self.media_state_value: dict | None = None
        self.media_state_calls = 0  # how many times media_state() was queried (debounce tests)

    async def respond(self, session_id: str, text: str) -> AsyncIterator[BrainEvent]:
        self.respond_calls.append((session_id, text))
        for ev in self._respond_events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield ev

    async def confirm(
        self, session_id: str, confirm_id: str, approved: bool, passphrase: str | None = None
    ) -> AsyncIterator[BrainEvent]:
        self.confirm_calls.append((session_id, confirm_id, approved, passphrase))
        for ev in self._confirm_events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield ev

    async def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)

    async def duck(self, session_id: str, on: bool, mute: bool = False) -> None:
        self.duck_calls.append((session_id, on))          # back-compat shape for existing tests
        self.duck_mute_calls.append((session_id, on, mute))  # full shape incl. mute

    async def media_state(self, session_id: str) -> dict | None:
        self.media_state_calls += 1
        return self.media_state_value

    async def aclose(self) -> None:
        self.closed = True
