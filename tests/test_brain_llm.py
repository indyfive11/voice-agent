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


def test_confirm_prompt_phrasing():
    cp = BrainLLMService._confirm_prompt
    # Imperative fragment → wrapped with "I'll …" + our yes/no tail.
    assert cp("edit the project README") == \
        "I'll edit the project README. Say yes to proceed, or no to cancel."
    # Leading verb is lower-cased so it reads as a sentence ("I'll open…", not "I'll Open…").
    assert cp("Open a URL in the default browser") == \
        "I'll open a URL in the default browser. Say yes to proceed, or no to cancel."
    # Already-formed question → spoken verbatim, no "I'll", no double "?.".
    assert cp("Play The Matrix in Jellyfin?") == \
        "Play The Matrix in Jellyfin? Say yes to proceed, or no to cancel."
    # Summary that already poses its own yes/no choice → we don't double it up.
    assert cp("Play on your open Chrome? Yes to use it, no to open a new window.") == \
        "Play on your open Chrome? Yes to use it, no to open a new window."
    # tail=False → just the action clause (keyboard path), no yes/no instruction.
    assert cp("Open a URL in the default browser", tail=False) == \
        "I'll open a URL in the default browser."
    # Never emits a broken "I'll …?" or a double period.
    for s in ("Play The Matrix in Jellyfin?", "Delete the file.", "Restart now!"):
        out = cp(s)
        assert "I'll Play" not in out and ".." not in out and "?." not in out


def test_keyboard_confirm_never_speaks_raw_tool_call():
    # Tier-3 keyboard confirm with a raw, multi-line tool-call summary → spoken heads-up
    # must NOT contain the raw payload; kdialog (stubbed) carries the detail.
    raw = ("memory_write (content=Voice Agent Feature Requests:\n1. Close current browser tab.\n"
           "2. Capture screenshot.)")
    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="k9", tier=3, method="keyboard", summary=raw)],
        confirm_events=[BrainEvent("token", text="Saved it."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)

    async def _fake_kbd(summary, reason=None):
        return True

    svc._keyboard_confirm = _fake_kbd
    asyncio.run(svc._process_context(_ctx("save those to memory")))
    spoken = " ".join(_texts(pushed))
    assert "memory_write" not in spoken and "content=" not in spoken and "\n" not in spoken
    # Heads-up points at the on-screen dialog (it's a mouse prompt) — never says "keyboard".
    assert ("screen" in spoken.lower() or "prompt" in spoken.lower())
    assert "keyboard" not in spoken.lower()
    assert "Saved it." in spoken


def test_shutdown_phrase_ends_pipeline_cleanly():
    from pipecat.frames.frames import EndTaskFrame

    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("okay, please shut down voice mode now")))
    assert client.respond_calls == []                          # never reaches the brain
    assert any(isinstance(f, EndTaskFrame) for f in pushed)    # graceful end requested
    assert "goodbye" in " ".join(_texts(pushed)).lower()


def test_shutdown_phrase_survives_stt_punctuation():
    # Live regression (session 621ac1b2…): Whisper emitted "Shut down, voice mode." with a comma,
    # which defeated the plain substring match → shutdown never fired, session only ended on Ctrl-C.
    # Normalization must strip the punctuation so the gate fires.
    from pipecat.frames.frames import EndTaskFrame

    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("Shut down, voice mode.")))
    assert client.respond_calls == []                          # never reaches the brain
    assert any(isinstance(f, EndTaskFrame) for f in pushed)    # graceful end requested
    assert "goodbye" in " ".join(_texts(pushed)).lower()


def test_shutdown_close_variant_fires():
    # Live regression (session 61a6228b): "Close down voice mode" missed the gate (no "close"
    # variant) → brain fielded it, couldn't exit. Close-variants must fire the clean exit.
    from pipecat.frames.frames import EndTaskFrame

    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("Close down voice mode.")))
    assert client.respond_calls == []
    assert any(isinstance(f, EndTaskFrame) for f in pushed)


def test_meta_question_about_shutdown_does_not_exit():
    # "Do you know how to turn yourself off yet?" is a QUESTION about the control — must not exit;
    # it should reach the brain so Aria can explain (regression from live session 3ba40222…).
    from pipecat.frames.frames import EndTaskFrame
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Just say 'shut down voice mode'."),
                                             BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("Do you know how to turn yourself off yet?")))
    assert not any(isinstance(f, EndTaskFrame) for f in pushed)
    assert client.respond_calls == [("sess-test", "Do you know how to turn yourself off yet?")]


def test_meta_question_about_sleep_does_not_sleep():
    # "What's the command to make you stop listening?" must not put it to sleep (live regression).
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Say 'go to sleep'."),
                                             BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("What's the command to make you stop listening to me?")))
    assert svc._sleeping is False
    assert client.respond_calls and "stop listening" in client.respond_calls[0][1].lower()


def test_bare_shutdown_passes_through_to_brain():
    # "shut down the computer" is system control (brain's job) — must NOT exit voice mode.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Okay."), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    from pipecat.frames.frames import EndTaskFrame
    asyncio.run(svc._process_context(_ctx("shut down the computer")))
    assert client.respond_calls == [("sess-test", "shut down the computer")]
    assert not any(isinstance(f, EndTaskFrame) for f in pushed)


def test_confirm_prompt_is_complete_spoken_verbatim():
    # prompt_is_complete → summary is the whole line (own yes/no), append nothing.
    line = "Play The Matrix on your open Chrome? Say yes to play there, or no to open a new window."
    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="c9", tier=2, method="spoken_yesno",
                                   summary=line, prompt_is_complete=True)],
        confirm_events=[BrainEvent("token", text="Playing."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("yes on chrome")))
    spoken = " ".join(_texts(pushed))
    assert spoken == line  # verbatim, no "I'll", no appended proceed/cancel tail
    assert "Say yes to proceed" not in spoken
    assert svc._pending_confirm == {"id": "c9", "method": "spoken_yesno"}


# --- media ducking (MediaDuckController; redesigned 2026-06-02) ---------------
from brains.media_duck import MediaDuckController  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)


def _duck(client, **kw):
    kw.setdefault("min_words", 2)
    kw.setdefault("restore_grace", 8.0)
    return MediaDuckController(client, "sess-test", **kw)


def _spoken(text):
    return TranscriptionFrame(text, "user", "2026-01-01T00:00:00Z")


def test_media_duck_on_confirmed_speech_restore_after_bot():
    client = FakeBrainClient()
    ctl = _duck(client)

    async def go():
        ctl._handle(_spoken("let's play a movie"))   # confirmed words → duck on
        await asyncio.sleep(0)
        ctl._handle(BotStartedSpeakingFrame())        # Aria replies
        ctl._handle(BotStoppedSpeakingFrame())        # Aria done → restore
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_duck_ignores_non_speech_vad_noise():
    # Sub-min_words transcription (movie effect / blip) must NOT duck — this is the whole point of
    # gating on confirmed speech instead of raw VAD.
    client = FakeBrainClient()
    ctl = _duck(client, min_words=2)

    async def go():
        ctl._handle(_spoken("hmm"))                   # 1 word < min_words → ignored
        ctl._handle(InterimTranscriptionFrame("", "user", "t"))  # empty → ignored
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == []


def test_media_duck_not_doubled_on_repeat_speech():
    client = FakeBrainClient()
    ctl = _duck(client)

    async def go():
        ctl._handle(_spoken("play some music"))
        ctl._handle(_spoken("play some music now"))   # already ducked → no second on
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]


def test_media_duck_fallback_restores_when_no_bot_speech():
    client = FakeBrainClient()
    ctl = _duck(client, restore_grace=0.01)  # short idle fallback for the test

    async def go():
        ctl._handle(_spoken("turn it down please"))
        await asyncio.sleep(0.05)                      # no bot speech → idle restore fires

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_duck_skipped_when_nothing_playing():
    client = FakeBrainClient()
    client.media_state_value = {"playing": False, "state": "idle"}  # neutral shape, nothing playing
    ctl = _duck(client)

    async def go():
        ctl._handle(_spoken("hello there aria"))
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == []  # media-state gate → no duck when nothing plays


def test_media_duck_fires_when_something_playing():
    client = FakeBrainClient()
    client.media_state_value = {"playing": True, "state": "playing"}  # neutral shape, playing
    ctl = _duck(client)

    async def go():
        ctl._handle(_spoken("hello there aria"))
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]


def test_media_duck_skipped_while_asleep():
    client = FakeBrainClient()
    ctl = _duck(client, should_duck=lambda: False)  # asleep → gate closed

    async def go():
        ctl._handle(_spoken("wake up please aria"))
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == []


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
