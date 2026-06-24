"""BuilderPollClient — the fixed-cadence /builder/poll loop (Brick B, builder_spec.md §4.4), offline.

The client depends only on three injected callables (poll / announce / drain_delivered), so these tests need
no HTTP, no brain, no pipeline — just fakes. The loop itself is driven by calling `_tick()` directly so the
assertions are deterministic (no sleeps / no real task scheduling).
"""

import asyncio

from builder_poll_client import BuilderPollClient


class _Harness:
    """Fakes the three seams the poll client talks to and records the interaction."""

    def __init__(self):
        self.poll_returns: list[list[dict]] = []   # queued responses, one per _tick (FIFO)
        self.poll_acks: list[list[str]] = []       # acks the client piggybacked on each poll
        self.delivered: list[str] = []             # what drain_delivered() will hand back next tick
        self.announced: list[tuple[str, str | None]] = []  # (text, job_id) handed to the announcer

    async def poll(self, session_id, ack):
        self.poll_acks.append(list(ack))
        return self.poll_returns.pop(0) if self.poll_returns else []

    async def announce(self, text, *, job_id=None):
        self.announced.append((text, job_id))

    def drain_delivered(self):
        out = self.delivered
        self.delivered = []
        return out

    def client(self, **kw):
        return BuilderPollClient(
            poll=self.poll, announce=self.announce, drain_delivered=self.drain_delivered,
            session_id="sess-1", **kw,
        )


def test_polls_and_announces_each_item():
    h = _Harness()
    h.poll_returns = [[{"job_id": "j1", "text": "Build done: 3 files."}]]
    asyncio.run(h.client()._tick())
    assert h.announced == [("Build done: 3 files.", "j1")]


def test_empty_response_announces_nothing():
    h = _Harness()
    h.poll_returns = [[]]
    asyncio.run(h.client()._tick())
    assert h.announced == []


def test_spoken_jobids_piggybacked_as_ack_next_tick():
    h = _Harness()
    c = h.client()
    # tick 1: nothing spoken yet → no ack
    asyncio.run(c._tick())
    assert h.poll_acks[-1] == []
    # the announcer reports j1 spoken → next tick must carry it as the ack
    h.delivered = ["j1"]
    asyncio.run(c._tick())
    assert h.poll_acks[-1] == ["j1"]


def test_dedup_does_not_reannounce_same_jobid_before_ack():
    h = _Harness()
    c = h.client()
    h.poll_returns = [[{"job_id": "j1", "text": "hi"}]]
    asyncio.run(c._tick())
    # brain shouldn't re-send, but defensively: same item again before it's acked → not re-announced
    h.poll_returns = [[{"job_id": "j1", "text": "hi"}]]
    asyncio.run(c._tick())
    assert h.announced == [("hi", "j1")]


def test_reannounce_allowed_after_ack_clears_inflight():
    h = _Harness()
    c = h.client()
    h.poll_returns = [[{"job_id": "j1", "text": "first"}]]
    asyncio.run(c._tick())
    h.delivered = ["j1"]            # spoken → ack clears it from in-flight this tick
    h.poll_returns = [[]]
    asyncio.run(c._tick())
    # a fresh item reusing the id (new build) would now be allowed through again
    h.poll_returns = [[{"job_id": "j1", "text": "second"}]]
    asyncio.run(c._tick())
    assert h.announced == [("first", "j1"), ("second", "j1")]


def test_item_without_text_skipped():
    h = _Harness()
    h.poll_returns = [[{"job_id": "j1", "text": ""}, {"job_id": "j2", "text": "ok"}]]
    asyncio.run(h.client()._tick())
    assert h.announced == [("ok", "j2")]


def test_item_without_job_id_still_announced():
    h = _Harness()
    h.poll_returns = [[{"job_id": None, "text": "anon ring"}]]
    asyncio.run(h.client()._tick())
    assert h.announced == [("anon ring", None)]


def test_loop_survives_poll_errors():
    h = _Harness()
    calls = {"n": 0}

    async def flaky(session_id, ack):
        calls["n"] += 1
        raise RuntimeError("brain down")

    c = h.client(interval=0.1)      # the production floor (max(0.1, …)); a tick fires every ~0.1s
    c._poll = flaky

    async def _run():
        await c.start()
        for _ in range(50):         # wait (≤2.5s) for at least 2 failing ticks, then assert still alive
            if calls["n"] >= 2:
                break
            await asyncio.sleep(0.05)
        assert not c._task.done()   # the loop swallowed every error and is still running
        await c.stop()

    asyncio.run(_run())
    assert calls["n"] >= 2          # it kept polling despite the failures


def test_start_is_idempotent_and_stop_cleans_up():
    h = _Harness()
    c = h.client(interval=0.01)

    async def _run():
        await c.start()
        first = c._task
        await c.start()           # second start is a no-op (same task)
        assert c._task is first
        await asyncio.sleep(0.03)  # let a couple of ticks run
        await c.stop()
        assert c._task is None

    asyncio.run(_run())
    assert h.poll_acks  # at least one poll happened while running
