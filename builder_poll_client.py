"""BuilderPollClient — the steady, sleep-independent poll loop for proactive brain→voice announcements.

Brick B of the `send_to_builder` deferred-result seam (builder_spec.md §4.4); pairs with announce.py's
DeferredAnnouncer. On a fixed cadence — INDEPENDENT of wake/media/sleep (the C1 fix; the old `/media/state`
poll dies when asleep, `wake_word.py:504-507`) — this loop:

  1. Drains the announcer's just-spoken job_ids and piggybacks them as the `ack` (finalizes the brain's
     liveness-leased delivery: `delivering → delivered`).
  2. Polls `GET /builder/poll` for `[{job_id, text}]` items (builder results / timer rings).
  3. Hands each to `DeferredAnnouncer.announce(text, job_id=)`, which speaks it at the next FREE floor.

The split that matters (GA's note): the POLL/CLAIM runs even while `_sleeping` — so an item finishing while
the maintainer is away is claimed the moment it's enqueued and held by our continued polling (the brain reverts only if
we stop polling past its gone-timeout). The SPEAK is separately gated by `is_floor_free()` inside the
announcer, so a floor closed for hours never loses the item — we voice it the moment the floor frees. The
claim is liveness-keyed, never a speak-deadline.

Absent a brain that exposes `/builder/poll` this is harmless: the poll returns [] every tick (no-op).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

# Default cadence. ~1.5s keeps the brain's claim renewed well inside its ~8s gone-timeout (≈5 polls) while
# staying cheap (the steady case is an empty `{"deferred":[]}`). Override via the constructor / env at wiring.
DEFAULT_POLL_INTERVAL_SECS = 1.5


class BuilderPollClient:
    """Runs the fixed-cadence `/builder/poll` loop and feeds results to the DeferredAnnouncer."""

    def __init__(
        self,
        *,
        poll: Callable[[str, list[str]], Awaitable[list[dict]]],
        announce: Callable[..., Awaitable[None]],
        drain_delivered: Callable[[], list[str]],
        session_id: str,
        interval: float = DEFAULT_POLL_INTERVAL_SECS,
    ):
        self._poll = poll                      # async (session_id, ack) -> [{job_id, text}]
        self._announce = announce              # async (text, *, job_id) -> None  (DeferredAnnouncer.announce)
        self._drain_delivered = drain_delivered  # () -> [job_id]  (DeferredAnnouncer.drain_delivered)
        self._session_id = session_id
        self._interval = max(0.1, interval)
        self._task: asyncio.Task | None = None
        # Cheap dedup guard: job_ids we've already handed to the announcer but not yet ack'd. The brain
        # doesn't re-send a delivering item, so this only guards a same-item double-announce defensively.
        self._inflight: set[str] = set()

    async def start(self) -> None:
        """Begin polling. Idempotent — a second call while running is a no-op."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Builder poll: started (session={self._session_id}, interval={self._interval:.1f}s)")

    async def stop(self) -> None:
        """Cancel the loop and wait for it to unwind."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a tick must never kill the loop
                logger.debug(f"Builder poll: tick failed (ignored): {type(e).__name__}: {e}")
            await asyncio.sleep(self._interval)

    async def _tick(self) -> None:
        ack = self._drain_delivered()          # job_ids fully spoken since the last tick (piggybacked)
        for jid in ack:
            self._inflight.discard(jid)        # acked → no longer in flight on our side
        items = await self._poll(self._session_id, ack)
        if not items:
            return
        for item in items:
            job_id = item.get("job_id")
            text = item.get("text")
            if not text:
                continue                       # nothing to say — defensive (real items always carry text)
            if job_id is not None:
                if job_id in self._inflight:
                    continue                   # already handed to the announcer; don't double-speak
                self._inflight.add(job_id)
            await self._announce(text, job_id=job_id)
