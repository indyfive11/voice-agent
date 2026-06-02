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
      - "token"   : a chunk of assistant text to speak (`text`)
      - "status"  : optional short narration to speak (`text`)
      - "confirm" : a gated action awaiting a decision (`id`, `tier`, `method`, `summary`, `reason`)
      - "blocked" : a refused action to announce (`reason`, `action`)
      - "error"   : a turn-level failure to speak (`text` speakable, `summary` structured cause)
      - "done"    : end of this turn
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

    async def aclose(self) -> None:
        self.closed = True
