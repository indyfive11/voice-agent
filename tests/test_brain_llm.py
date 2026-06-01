"""Offline tests for the external-brain adapter — no gabagent, no audio needed.

Verifies BrainLLMService turns BrainEvents into the right Pipecat frames, the turn-based
confirmation flow, blocked actions, and barge-in (CancelledError) cleanup. Async bodies are
driven with asyncio.run so plain pytest (no pytest-asyncio) works.
"""

import asyncio

from pipecat.frames.frames import LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from brains.brain_client import BrainEvent, FakeBrainClient
from brains.brain_llm_service import BrainLLMService


def _service_with_recorder(client):
    svc = BrainLLMService(client, session_id="sess-test")
    pushed = []

    async def _rec(frame, direction=None):
        pushed.append(frame)

    svc.push_frame = _rec  # capture frames without a running pipeline
    return svc, pushed


def _ctx(user_text):
    return LLMContext(messages=[{"role": "user", "content": user_text}])


def _texts(frames):
    return [f.text for f in frames if isinstance(f, LLMTextFrame)]


def test_tokens_become_text_frames():
    client = FakeBrainClient(respond_events=[
        BrainEvent("token", text="Hi "),
        BrainEvent("token", text="there."),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("hello")))
    assert _texts(pushed) == ["Hi ", "there."]
    assert client.respond_calls == [("sess-test", "hello")]


def test_turn_based_confirmation_flow():
    client = FakeBrainClient(
        respond_events=[
            BrainEvent("token", text="Okay. "),
            BrainEvent("confirm", id="c1", tier=2, method="spoken_yesno",
                       summary="I'll edit the project README"),
        ],
        confirm_events=[BrainEvent("token", text="Done."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)

    # Turn 1: brain asks to confirm → service speaks prompt, ends turn, remembers pending id.
    asyncio.run(svc._process_context(_ctx("update the readme")))
    spoken = " ".join(_texts(pushed))
    assert "edit the project README" in spoken and "Say yes" in spoken
    assert svc._pending_confirm == {"id": "c1", "method": "spoken_yesno"}

    # Turn 2: user says "yes" → service calls confirm(approved=True) and streams continuation.
    pushed.clear()
    asyncio.run(svc._process_context(_ctx("yes please")))
    assert client.confirm_calls == [("sess-test", "c1", True, None)]
    assert _texts(pushed) == ["Done."]
    assert svc._pending_confirm is None


def test_confirmation_declined():
    # Tier-2 spoken_yesno, declined on the second turn.
    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="c2", tier=2, method="spoken_yesno",
                                   summary="overwrite that file")],
        confirm_events=[BrainEvent("token", text="Okay, cancelled."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("overwrite it")))  # turn 1: speaks prompt, pending set
    assert svc._pending_confirm == {"id": "c2", "method": "spoken_yesno"}
    pushed.clear()
    asyncio.run(svc._process_context(_ctx("no, stop")))      # turn 2: decision = no
    assert client.confirm_calls == [("sess-test", "c2", False, None)]
    assert "cancelled" in " ".join(_texts(pushed)).lower()


def test_error_event_spoken_once_fallback_status_suppressed():
    client = FakeBrainClient(respond_events=[
        BrainEvent("error", text="Sorry, I hit a problem.", summary="TimeoutError: jellyfin.play"),
        BrainEvent("status", text="Sorry, I hit a problem — ask me what went wrong for details."),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("open a jellyfin movie")))
    # error.text spoken once; the transitional fallback status is NOT spoken (no double-speak).
    assert _texts(pushed) == ["Sorry, I hit a problem."]


def test_repeated_status_spoken_once():
    client = FakeBrainClient(respond_events=[
        BrainEvent("status", text="Looking into it."),
        BrainEvent("status", text="Looking into it."),
        BrainEvent("status", text="Looking into it."),
        BrainEvent("token", text="Found it."),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("what was the error")))
    assert _texts(pushed) == ["Looking into it.", "Found it."]  # status spoken once, not ×3


def test_sleep_wake_gates_input():
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)

    # "go to sleep" → muted; never hits the brain; speaks a sleep ack.
    asyncio.run(svc._process_context(_ctx("ok, go to sleep now please")))
    assert svc._sleeping is True
    assert client.respond_calls == []
    assert "sleep" in " ".join(_texts(pushed)).lower()

    # while asleep, ordinary input is ignored (no brain call, no speech).
    pushed.clear()
    asyncio.run(svc._process_context(_ctx("tell me a joke")))
    assert client.respond_calls == []
    assert _texts(pushed) == []

    # "wake up" → un-mutes; speaks a wake ack.
    pushed.clear()
    asyncio.run(svc._process_context(_ctx("hey, wake up")))
    assert svc._sleeping is False
    assert "awake" in " ".join(_texts(pushed)).lower()

    # back to normal — input reaches the brain again.
    pushed.clear()
    asyncio.run(svc._process_context(_ctx("hello")))
    assert client.respond_calls == [("sess-test", "hello")]
    assert _texts(pushed) == ["Hi!"]


def test_blocked_is_spoken_then_turn_continues():
    # blocked is NOT a stream boundary — reason is spoken, then consuming continues to done.
    client = FakeBrainClient(respond_events=[
        BrainEvent("blocked", action="vpn", reason="That needs a passphrase, not set up yet."),
        BrainEvent("token", text="Anything else?"),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("turn off the vpn")))
    assert _texts(pushed) == ["That needs a passphrase, not set up yet.", "Anything else?"]


def test_barge_in_cancels_and_closes_stream():
    client = FakeBrainClient(
        respond_events=[BrainEvent("token", text=f"chunk{i} ") for i in range(20)],
        delay=0.02,
    )
    svc, _ = _service_with_recorder(client)

    async def run_and_cancel():
        task = asyncio.create_task(svc._process_context(_ctx("tell me a long story")))
        await asyncio.sleep(0.05)  # let a couple of chunks flow
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return True
        return False

    cancelled = asyncio.run(run_and_cancel())
    assert cancelled is True
    assert client.cancel_calls == ["sess-test"]  # /cancel sent on barge-in (not full teardown)
    assert client.closed is False  # client/subprocess stay up for the next turn


def test_keyboard_confirm_resolves_in_turn():
    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="k1", tier=3, method="keyboard",
                                   summary="delete the build directory")],
        confirm_events=[BrainEvent("token", text="Done, removed it."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)

    async def _fake_kbd(summary, reason=None):  # don't spawn kdialog in tests
        return True

    svc._keyboard_confirm = _fake_kbd
    asyncio.run(svc._process_context(_ctx("delete the build dir")))

    # Keyboard tier resolves within the same turn (no pending two-turn state).
    assert client.confirm_calls == [("sess-test", "k1", True, None)]
    assert "Done, removed it." in _texts(pushed)
    assert svc._pending_confirm is None
