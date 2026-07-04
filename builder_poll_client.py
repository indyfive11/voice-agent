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
        display: Callable[..., Awaitable[None]] | None = None,
        drain_displayed: Callable[[], list[str]] | None = None,
    ):
        self._poll = poll                      # async (session_id, ack) -> [{job_id, text[, display]}]
        self._announce = announce              # async (text, *, job_id) -> None  (DeferredAnnouncer.announce)
        self._drain_delivered = drain_delivered  # () -> [job_id]  (DeferredAnnouncer.drain_delivered)
        # Image display (roadmap ③). Optional — absent → display items are ignored (no-op, spoken path
        # unchanged), so a build without the sink wired behaves exactly as before.
        self._display = display                # async (descriptor, *, job_id) -> None  (ImageDisplaySink.show)
        self._drain_displayed = drain_displayed  # () -> [job_id]  (ImageDisplaySink.drain_displayed)
        self._session_id = session_id
        self._interval = max(0.1, interval)
        self._task: asyncio.Task | None = None
        # Display renders run as detached tasks so a long on-screen window (e.g. 30s, + a paused movie on the
        # Pi) NEVER blocks the steady poll loop — acks and other announcements keep flowing. Tracked so a task
        # ref isn't GC'd mid-flight and so stop() can cancel them.
        self._display_tasks: set[asyncio.Task] = set()
        # Cheap dedup guard: job_ids we've already handed off (announce OR display) but not yet ack'd. The
        # brain doesn't re-send a delivering item, so this only guards a same-item double-handoff defensively.
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
        # Cancel any in-flight display renders (a window still up at teardown).
        for t in list(self._display_tasks):
            t.cancel()
        self._display_tasks.clear()

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
        if self._drain_displayed is not None:
            ack = ack + self._drain_displayed()  # + job_ids of images fully shown (same ack channel)
        for jid in ack:
            self._inflight.discard(jid)        # acked → no longer in flight on our side
        items = await self._poll(self._session_id, ack)
        if not items:
            return
        for item in items:
            job_id = item.get("job_id")
            display = item.get("display")
            text = item.get("text")
            # A display-only item (image) carries `display` and empty text — render it, don't speak. An item
            # with neither is dropped (defensive; the brain always sends one or the other).
            if display and self._display is not None:
                if self._claim(job_id):
                    # Detach: rendering may hold the screen for the whole display window (+ a paused movie on
                    # the Pi); awaiting it here would freeze the poll loop. The sink records the job_id on
                    # completion, drained as the ack on a later tick.
                    t = asyncio.create_task(self._display(display, job_id=job_id))
                    self._display_tasks.add(t)
                    t.add_done_callback(self._display_tasks.discard)
                continue
            if not text:
                continue                       # nothing to say / can't display — skip
            if self._claim(job_id):
                await self._announce(text, job_id=job_id)

    def _claim(self, job_id) -> bool:
        """Record a job_id as in-flight; return False if it was already claimed (don't double-handle).
        A None job_id is always handled (no dedup key)."""
        if job_id is None:
            return True
        if job_id in self._inflight:
            return False
        self._inflight.add(job_id)
        return True
