"""DeferredAnnouncer — out-of-band, bot-initiated speech spoken at the next free conversational floor.

This is the speak side of the `send_to_builder` deferred-result seam (builder_spec.md §4.4), built as the
skeleton of the eventual floor/turn arbiter rather than as one feature bolted onto the LLM service:

  • The **floor authority** is whoever owns the conversational state — the BrainLLMService — which exposes a
    single `is_floor_free()` accessor. This processor depends only on that `Callable[[], bool]`, never on the
    service's internals, so the spine stays flat and the capability lives here.
  • This processor owns the **outbound queue** for bot-initiated speech and the etiquette: speak one item at
    a time, only at a free floor, draining on each free-floor edge (BotStopped). A barge between items stops
    the rest (the next item re-checks the floor).
  • Producers are trivial: call `announce(text, …)` (or push a SpeakDeferredFrame). Today the only producer is
    the builder-result poll client (Brick B, later); timers (G2) and proactive nudges are the same shape.

Deliberately NOT populated yet (no live consumer → would be designed/validated against nothing):
  • **Priority ordering** — `announce(priority=…)` is plumbed end-to-end and stored on the item, but `_drain`
    pops FIFO. It only means something with ≥2 producer classes (builder-result vs. confirm-question vs.
    timer), and the confirm/ask path is explicitly out of this queue until a later phase.
  • **Pre-emption / barge-aware non-ack + re-queue** — a barged item is dropped AND currently still recorded as
    delivered (so the brain ack'd it); not-acking a cut-off item so the brain re-offers it is coupled to
    pre-emption, deferred until that exists.
Both land additively (change `_drain`'s pop / the barge branch) when their consumers arrive — no caller churn.

The ACK SOURCE is built (the brain's two-phase delivery needs it now): each fully-spoken item's job_id is
recorded on its BotStopped edge and handed to the builder poll client (Brick B) via `drain_delivered()`, which
piggybacks it as `&ack=` on the next `GET /builder/poll`. The claim is held by poll-liveness, not a speak-
deadline (VAC's speak-delay is unbounded — the floor can stay closed while asleep), reverting only on the
brain's gone-timeout. The poll client itself is Brick B (built when GA's endpoint is frozen).

Absent any producer the queue stays empty → exact current behavior (no-op).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame, StartFrame, SystemFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Reserved priority band (higher = more urgent). Only DEFAULT is used today; the others name the future
# producers so the API is stable before the ordering logic exists.
PRIORITY_NUDGE = 0       # proactive, lowest — "by the way, …"
PRIORITY_DEFAULT = 10    # builder results and the like
PRIORITY_QUESTION = 20   # a deferred question that needs an answer (later phase)


@dataclass
class SpeakDeferredFrame(SystemFrame):
    """A piece of OUT-OF-BAND speech to announce at the next free floor — NOT a reply to the current turn.
    Either pushed into the pipeline (a future producer) or created via `DeferredAnnouncer.announce()`. `text`
    is the speaker's OWN ground-truth string (e.g. the builder's anti-confab summary, §7) — voiced verbatim,
    never recomposed. `job_id` lets the eventual two-phase ack key on the item; informational here (logged).
    `priority` is reserved for the future ordering (see module docstring) — carried, not yet acted on."""

    text: str = ""
    job_id: str | None = None
    priority: int = PRIORITY_DEFAULT


class DeferredAnnouncer(FrameProcessor):
    """Queues bot-initiated speech and voices it at a free conversational floor, one item at a time.

    Sits immediately after the LLM service in the pipeline: it injects TTSSpeakFrame DOWNSTREAM toward TTS,
    and observes BotStoppedSpeakingFrame on the upstream return path to know when the floor frees up. Frames
    it doesn't own are forwarded untouched (incl. BotStopped — media_duck / wake gate still need it)."""

    def __init__(self, floor_free: Callable[[], bool], **kwargs):
        super().__init__(**kwargs)
        self._floor_free = floor_free          # the floor authority (BrainLLMService.is_floor_free)
        self._queue: list[SpeakDeferredFrame] = []
        self._started = False                  # set on StartFrame; gates pushing until the pipeline is up
        self._in_flight = False                # latched while a popped item is being spoken (push→BotStopped)
        self._in_flight_job: str | None = None # job_id of the item being spoken (so BotStopped can ack it)
        self._delivered: list[str] = []        # spoken job_ids awaiting pickup by the poll client (the ack)

    async def announce(self, text: str, *, job_id: str | None = None, priority: int = PRIORITY_DEFAULT) -> None:
        """Producer entry point (the builder poll client / timers call this). Queue an out-of-band utterance
        and speak it at the next free floor. `text` is voiced verbatim — pass the ground-truth string."""
        await self._enqueue(SpeakDeferredFrame(text=text, job_id=job_id, priority=priority))

    async def enqueue_deferred(self, text: str, job_id: str | None = None) -> None:
        """Back-compat alias for the seam name GA's contract referenced; forwards to announce()."""
        await self.announce(text, job_id=job_id)

    def drain_delivered(self) -> list[str]:
        """Poll-client seam (Brick B): return the job_ids fully SPOKEN since the last call, and clear them.
        The builder poll client reads this each tick and piggybacks them as `&ack=<id>[,<id2>]` on the next
        `GET /builder/poll`, finalizing the brain's two-phase delivery (delivering → delivered; the claim is
        held by poll-liveness, never a speak-deadline). Drains so each spoken item is ack'd exactly once."""
        if not self._delivered:
            return []
        out = self._delivered
        self._delivered = []
        return out

    async def _enqueue(self, frame: SpeakDeferredFrame) -> None:
        if not frame.text:
            return  # nothing to say — a real result always carries its summary; drop empties defensively
        self._queue.append(frame)
        logger.bind(transcript=True).info(
            f"DEFER | queued (job={frame.job_id}, prio={frame.priority}, depth={len(self._queue)}): {frame.text!r}"
        )
        await self._drain()

    async def _drain(self) -> None:
        """Speak at most ONE queued item if the floor is free and we're not already mid-item. The next item
        waits for THIS one's BotStopped (which re-enters here), so a barge-in between items stops the rest and
        every item re-checks the floor. FIFO for now (priority ordering deferred — see module docstring)."""
        if not self._started or self._in_flight or not self._queue or not self._floor_free():
            return  # _started guard: never push before our StartFrame (push_frame would drop it and latch
            #         _in_flight forever — the pre-Start race that wedged the deferred path on a restart)
        frame = self._queue.pop(0)
        self._in_flight = True  # latched until this item's BotStopped, so no re-entrant double-speak
        self._in_flight_job = frame.job_id  # remember which item so its BotStopped can mark it delivered (ack)
        logger.bind(transcript=True).info(f"DEFER | speaking (job={frame.job_id}): {frame.text!r}")
        await self.push_frame(TTSSpeakFrame(frame.text), FrameDirection.DOWNSTREAM)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # A producer injected an out-of-band utterance into the pipeline → queue it (CONSUME: our control
        # frame, nothing downstream should see it). The method path (announce/enqueue_deferred) is equivalent.
        if isinstance(frame, SpeakDeferredFrame):
            await self._enqueue(frame)
            return

        # The pipeline is now started. Forward StartFrame downstream FIRST (so TTS/output start their tasks
        # before any TTSSpeakFrame reaches them — frames queue in order), then mark ready and flush anything a
        # producer queued before start. Without this, a poll/announce that fires during startup pushes a
        # TTSSpeakFrame pre-Start → push_frame drops it → _in_flight latches with no BotStopped → wedge.
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._started = True
            await self._drain()
            return

        # Any bot speech (a reply, or a just-spoken deferred item) finished → the floor may be free now.
        # Clear the latch, FORWARD the frame (media_duck / wake gate still need it), then drain the next item.
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._in_flight = False
            if self._in_flight_job is not None:
                # A deferred item we spoke just finished → record its job_id for the poll client to ack
                # (two-phase finalize: delivering → delivered). Items with no job_id aren't tracked (nothing
                # to key the ack on). NOTE: a user barge-in that cuts a deferred item also lands here and is
                # recorded as delivered — the item is consumed either way today; barge-aware non-ack + re-queue
                # rides with pre-emption (deferred, see module docstring).
                self._delivered.append(self._in_flight_job)
                self._in_flight_job = None
            await self.push_frame(frame, direction)
            await self._drain()
            return

        await self.push_frame(frame, direction)
