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
        self.displayed_items: list[tuple[dict, str | None]] = []  # (descriptor, job_id) to the display sink
        self.to_display_ack: list[str] = []        # what drain_displayed() hands back next tick

    async def poll(self, session_id, ack):
        self.poll_acks.append(list(ack))
        return self.poll_returns.pop(0) if self.poll_returns else []

    async def announce(self, text, *, job_id=None):
        self.announced.append((text, job_id))

    def drain_delivered(self):
        out = self.delivered
        self.delivered = []
        return out

    async def display(self, descriptor, *, job_id=None):
        self.displayed_items.append((descriptor, job_id))

    def drain_displayed(self):
        out = self.to_display_ack
        self.to_display_ack = []
        return out

    def client(self, *, with_display=False, **kw):
        if with_display:
            kw.update(display=self.display, drain_displayed=self.drain_displayed)
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


# ---- roadmap ③: display (image) item routing ------------------------------

def test_display_item_routed_to_sink_not_spoken():
    h = _Harness()
    desc = {"path": "/x.png", "url": "https://cdn/x.png"}
    h.poll_returns = [[{"job_id": "img-1", "text": "", "display": desc}]]
    asyncio.run(h.client(with_display=True)._tick())
    assert h.displayed_items == [(desc, "img-1")]
    assert h.announced == []           # display-only → nothing spoken


def test_display_and_spoken_items_both_handled_same_tick():
    h = _Harness()
    desc = {"url": "https://cdn/x.png"}
    h.poll_returns = [[
        {"job_id": "img-1", "text": "", "display": desc},
        {"job_id": "j2", "text": "build done"},
    ]]
    asyncio.run(h.client(with_display=True)._tick())
    assert h.displayed_items == [(desc, "img-1")]
    assert h.announced == [("build done", "j2")]


def test_display_jobid_piggybacked_as_ack_next_tick():
    h = _Harness()
    c = h.client(with_display=True)
    asyncio.run(c._tick())
    assert h.poll_acks[-1] == []
    # the sink reports img-1 shown → next tick carries it as the ack (same channel as spoken rings)
    h.to_display_ack = ["img-1"]
    asyncio.run(c._tick())
    assert h.poll_acks[-1] == ["img-1"]


def test_display_dedup_before_ack():
    h = _Harness()
    c = h.client(with_display=True)
    desc = {"url": "https://cdn/x.png"}
    h.poll_returns = [[{"job_id": "img-1", "text": "", "display": desc}]]
    asyncio.run(c._tick())
    h.poll_returns = [[{"job_id": "img-1", "text": "", "display": desc}]]  # re-sent before ack
    asyncio.run(c._tick())
    assert h.displayed_items == [(desc, "img-1")]   # rendered once


def test_display_item_ignored_when_no_sink_wired():
    h = _Harness()
    # display present but the client has no sink → not spoken, not displayed (graceful no-op)
    h.poll_returns = [[{"job_id": "img-1", "text": "", "display": {"url": "u"}}]]
    asyncio.run(h.client(with_display=False)._tick())
    assert h.announced == []
    assert h.displayed_items == []


def test_item_with_neither_text_nor_display_skipped():
    h = _Harness()
    h.poll_returns = [[{"job_id": "j1", "text": ""}, {"job_id": "j2", "text": "ok"}]]
    asyncio.run(h.client(with_display=True)._tick())
    assert h.displayed_items == []
    assert h.announced == [("ok", "j2")]


def test_display_does_not_block_the_tick():
    """A slow render (long on-screen window) must not freeze the poll loop — the tick returns while the
    display runs detached; the item is recorded once the render finishes."""
    h = _Harness()
    rendered: list[str] = []

    async def slow_display(descriptor, *, job_id=None):
        await asyncio.sleep(0.2)                 # simulate a long on-screen window
        rendered.append(job_id)

    async def _run():
        c = h.client(with_display=True)
        c._display = slow_display
        h.poll_returns = [[{"job_id": "img-1", "text": "", "display": {"url": "u"}}]]
        # the tick itself must return promptly, NOT wait ~0.2s for the render
        t0 = asyncio.get_event_loop().time()
        await c._tick()
        assert asyncio.get_event_loop().time() - t0 < 0.1
        assert rendered == []                    # render still in flight
        await asyncio.sleep(0.3)                 # let the detached render finish
        assert rendered == ["img-1"]
        await c.stop()

    asyncio.run(_run())


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
