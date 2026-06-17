"""Offline tests for the external-brain adapter — no gabagent, no audio needed.

Verifies BrainLLMService turns BrainEvents into the right Pipecat frames, the turn-based
confirmation flow, blocked actions, and barge-in (CancelledError) cleanup. Async bodies are
driven with asyncio.run so plain pytest (no pytest-asyncio) works.
"""

import asyncio

from pipecat.frames.frames import LLMTextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from brains.brain_client import BrainEvent, FakeBrainClient
from brains.brain_llm_service import BrainLLMService, _BARE_WAKE_PROMPT


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


def _spoken_texts(frames):
    # Both the normal LLM token stream and immediate short utterances (the bare-wake prompt is a TTSSpeakFrame).
    return [f.text for f in frames if isinstance(f, (LLMTextFrame, TTSSpeakFrame))]


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


def test_wake_hold_event_pushes_keepalive_frame_upstream():
    # A media command emits wake_hold → the service pushes WakeHoldFrame(hold=True, ttl_secs=N) UPSTREAM so
    # the wake gate holds its window open for a follow-up command (no re-wake).
    from wake_word import WakeHoldFrame
    client = FakeBrainClient(respond_events=[
        BrainEvent("token", text="Playing."),
        BrainEvent("wake_hold", hold=True, ttl_secs=30),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("play some music")))
    holds = [f for f in pushed if isinstance(f, WakeHoldFrame)]
    assert len(holds) == 1
    assert holds[0].hold is True and holds[0].ttl_secs == 30


def test_wake_hold_zero_ttl_pushes_no_frame():
    from wake_word import WakeHoldFrame
    client = FakeBrainClient(respond_events=[
        BrainEvent("wake_hold", hold=True, ttl_secs=0),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("hello")))
    assert [f for f in pushed if isinstance(f, WakeHoldFrame)] == []


def test_status_filler_logged_but_not_spoken():
    # the user disliked the spoken filler ("Trying tidal…"). Status events are now LOG-ONLY: no TTSSpeakFrame
    # and no LLMTextFrame for the status; only the token speaks.
    from pipecat.frames.frames import TTSSpeakFrame

    client = FakeBrainClient(respond_events=[
        BrainEvent("status", text="Trying tidal…"),
        BrainEvent("token", text="Done."),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("play music")))

    assert not any(isinstance(f, TTSSpeakFrame) for f in pushed)  # filler never voiced
    assert _texts(pushed) == ["Done."]                           # only the token speaks


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


def test_confirm_pushes_wake_hold_then_release():
    # A Tier-2 spoken confirm must push WakeHoldFrame(True) so the gate holds its window open for the
    # yes/no answer, then WakeHoldFrame(False) once the answer resolves.
    from wake_word import WakeHoldFrame

    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="c9", tier=2, method="spoken_yesno",
                                   summary="do the thing")],
        confirm_events=[BrainEvent("token", text="Done."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)

    asyncio.run(svc._process_context(_ctx("do it")))          # turn 1 → hold ON
    holds = [f.hold for f in pushed if isinstance(f, WakeHoldFrame)]
    assert holds == [True]

    pushed.clear()
    asyncio.run(svc._process_context(_ctx("yes")))            # turn 2 → hold OFF
    holds = [f.hold for f in pushed if isinstance(f, WakeHoldFrame)]
    assert holds == [False]


def test_addressed_aside_event_pushes_duck_release():
    # The brain emits an "addressed" event ONLY on suppression (an aside). Its arrival must push a
    # DuckReleaseFrame upstream so the media-duck controller drops the VAD-onset pre-duck early, and it
    # must NOT be a stream boundary — the following token/done still flow.
    from brains.media_duck import DuckReleaseFrame

    client = FakeBrainClient(respond_events=[
        BrainEvent("addressed", addressed=False),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("she's in the back")))
    assert sum(isinstance(f, DuckReleaseFrame) for f in pushed) == 1


def test_addressed_event_keys_on_type_not_bool():
    # Robustness: the handler keys on the event TYPE, not the `addressed` value — a brain-side serializer
    # that drops a literal False (Python False == 0) leaves `addressed=None`, and the duck must STILL release.
    from brains.media_duck import DuckReleaseFrame

    client = FakeBrainClient(respond_events=[
        BrainEvent("addressed"),  # bool absent on the wire → parsed as None
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("parks and things every time")))
    assert sum(isinstance(f, DuckReleaseFrame) for f in pushed) == 1


def test_addressed_event_is_not_a_stream_boundary():
    # An aside that the brain answers anyway (addressed event THEN a token) keeps consuming: the token
    # after the addressed event must still be spoken.
    client = FakeBrainClient(respond_events=[
        BrainEvent("addressed", addressed=False),
        BrainEvent("token", text="Actually, here you go."),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("whatever")))
    assert _texts(pushed) == ["Actually, here you go."]


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


def test_confirm_cancel_sends_passphrase():
    # Controllable-client play confirm (yes-here / no-new-window / cancel). An explicit "wrong one"
    # must back out entirely: approved=False WITH passphrase="cancel" so the brain aborts the play.
    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="c3", tier=2, method="spoken_yesno",
                                   summary="play that movie on your open Chrome")],
        confirm_events=[BrainEvent("token", text="Okay, I won't play it."), BrainEvent("done")],
    )
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("play the movie")))
    assert svc._pending_confirm == {"id": "c3", "method": "spoken_yesno"}
    asyncio.run(svc._process_context(_ctx("no, wrong one")))
    assert client.confirm_calls == [("sess-test", "c3", False, "cancel")]
    assert svc._pending_confirm is None


def test_confirm_plain_no_is_not_cancel():
    # A bare "no" still means new window (approved=False, passphrase=None) — NOT cancel.
    client = FakeBrainClient(
        respond_events=[BrainEvent("confirm", id="c4", tier=2, method="spoken_yesno",
                                   summary="play that movie on your open Chrome")],
        confirm_events=[BrainEvent("token", text="Opening a new window."), BrainEvent("done")],
    )
    svc, _ = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("play the movie")))
    asyncio.run(svc._process_context(_ctx("no")))
    assert client.confirm_calls == [("sess-test", "c4", False, None)]


def test_is_cancel_phrase_matching():
    f = BrainLLMService._is_cancel
    assert f("cancel")
    assert f("never mind")
    assert f("forget it")
    assert f("that's the wrong movie")
    assert f("neither")
    assert f("don't play it")
    # plain yes/no decisions are not cancel
    assert not f("no")
    assert not f("yes")
    assert not f("no, open a new window")


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


def test_shutdown_survives_stt_voice_mishear():
    # Live regression (session 99462a1f): Whisper heard "shut down voice mode" as "shut down boys
    # mode" → missed the gate → brain falsely claimed to shut down while the process kept running.
    # The soundalike repair canonicalizes "<soundalike> mode/agent" → "voice mode/agent".
    from pipecat.frames.frames import EndTaskFrame

    for misheard in ("shut down boys mode", "Shut down boyce mode.", "close down voice mode",
                     "shut down boys agent"):
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
        svc, pushed = _service_with_recorder(client)
        asyncio.run(svc._process_context(_ctx(misheard)))
        assert client.respond_calls == [], f"reached brain for {misheard!r}"
        assert any(isinstance(f, EndTaskFrame) for f in pushed), f"no shutdown for {misheard!r}"


def test_bare_soundalike_does_not_shut_down():
    # The repair must only fire in the "<soundalike> mode/agent" frame — a bare soundalike elsewhere
    # ("the boys are loud") must reach the brain, never trigger a false shutdown.
    from pipecat.frames.frames import EndTaskFrame

    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Sure."), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("the boys are being loud, can you turn it down")))
    assert client.respond_calls == [("sess-test", "the boys are being loud, can you turn it down")]
    assert not any(isinstance(f, EndTaskFrame) for f in pushed)


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


def test_dictation_blob_does_not_shut_down():
    # Live leak (2026-06-10, session ffa503ef): the maintainer dictated "Dictating. The last time we told you...
    # shut down voice mode" — narration swept into one 15s runaway-turn blob whose tail hit the shutdown
    # gate and KILLED the session. The dictation-lead guard (and the standalone guard) must let it reach
    # the brain as an ordinary turn instead of exiting.
    from pipecat.frames.frames import EndTaskFrame

    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Noted."), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    text = "Dictating. The last time we told you to shut down voice mode."
    asyncio.run(svc._process_context(_ctx(text)))
    assert not any(isinstance(f, EndTaskFrame) for f in pushed)   # session survives
    assert client.respond_calls == [("sess-test", text)]          # narration reaches the brain


def test_buried_shutdown_phrase_does_not_fire_without_label():
    # Even WITHOUT a dictation self-label, a shutdown phrase buried at the end of a long blob (the
    # runaway-turn signature) must not exit — the standalone guard blocks it on residual length alone.
    from pipecat.frames.frames import EndTaskFrame

    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Sure."), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    text = "earlier we were chatting about whether the kids should shut down voice mode"
    asyncio.run(svc._process_context(_ctx(text)))
    assert not any(isinstance(f, EndTaskFrame) for f in pushed)
    assert client.respond_calls == [("sess-test", text)]


def test_dictation_blob_does_not_sleep():
    # Same narration guard protects the sleep gate: "Just dictating — go to sleep was what I said" must
    # not actually put her to sleep.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Noted."), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("Just dictating, go to sleep was the phrase we used earlier")))
    assert svc._sleeping is False
    assert client.respond_calls and "go to sleep" in client.respond_calls[0][1].lower()


def test_bare_vocative_not_forwarded_to_brain():
    # 2026-06-15: awake, a lone "Aria"/"Hey Aria" must NOT reach the brain — the acoustic gate already
    # opened the window, and a chit-chat reply ("Yeah?") would barge over the command spoken a beat later.
    for text in ("Aria", "Hey Aria", "Hey, Aria!", "Aria?", "okay Aria",
                 "Hey, Aria.  Hey, Ari.",   # STT clips "Aria"→"Ari" (2026-06-15 live leak)
                 "Hey Ariette", "Hey, Ariane."):  # STT spelling drift (2026-06-15 live) — fuzzy ari* match
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="Yeah?"), BrainEvent("done")])
        svc, pushed = _service_with_recorder(client)  # awake (not sleeping)
        asyncio.run(svc._process_context(_ctx(text)))
        assert client.respond_calls == [], f"forwarded bare vocative {text!r}"
        assert _texts(pushed) == [], f"spoke a reply to bare vocative {text!r}"


def test_vocative_plus_command_is_forwarded_with_wake_word_stripped():
    # name + command reaches the brain, and the wake word is STRIPPED — the brain hears only the command
    # (the wake word is consumed by the voice-side gate). 2026-06-15, the maintainer.
    cases = {
        "Aria, pause the movie": "pause the movie",
        "Hey Aria turn it up": "turn it up",
        "Hey Aria, play a movie from jellyfin": "play a movie from jellyfin",
    }
    for text, expected in cases.items():
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="ok"), BrainEvent("done")])
        svc, pushed = _service_with_recorder(client)
        asyncio.run(svc._process_context(_ctx(text)))
        assert client.respond_calls == [("sess-test", expected)], f"{text!r} → {client.respond_calls}"


def test_bare_wake_prompts_when_no_command_follows():
    # "Hey Aria" with nothing after → after a short wait with no command, ask once. Never forwarded.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    svc._bare_wake_delay = 0.01

    async def go():
        await svc._process_context(_ctx("Hey Aria"))
        await asyncio.sleep(0.05)  # let the prompt timer fire
    asyncio.run(go())
    assert client.respond_calls == []
    assert _spoken_texts(pushed) == [_BARE_WAKE_PROMPT]


def test_bare_wake_prompt_cancelled_by_following_command():
    # The natural pause: a command arriving after the bare wake answers it — the prompt never fires, and
    # only the command (no wake word) reaches the brain.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="ok"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    svc._bare_wake_delay = 0.05

    async def go():
        await svc._process_context(_ctx("Hey Aria"))   # bare → arms the prompt
        await asyncio.sleep(0.01)                       # command arrives before it fires
        await svc._process_context(_ctx("turn it up"))
        await asyncio.sleep(0.1)                        # well past the prompt delay
    asyncio.run(go())
    assert client.respond_calls == [("sess-test", "turn it up")]
    assert _BARE_WAKE_PROMPT not in _spoken_texts(pushed)


def test_non_vocative_short_turn_still_forwarded():
    # No vocative token / wake phrase → not a bare wake; a short utterance must still reach the brain.
    # "arid"/"arise" start with "ari" but are denylisted, so they must NOT read as the wake name.
    # ("are you awake" is now a wake phrase → covered by the bare-wake-prompt test, not here.)
    for text in ("okay", "what time is it", "arid", "arise and shine"):
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="ok"), BrainEvent("done")])
        svc, pushed = _service_with_recorder(client)
        asyncio.run(svc._process_context(_ctx(text)))
        assert client.respond_calls, f"dropped non-vocative turn {text!r}"


def test_fuzzy_vocative_wakes_from_sleep():
    # 2026-06-15 (the maintainer): an STT mis-spelling of the name ("Ariette") SHOULD wake her from sleep — the
    # soundalike match is now used by the asleep gate too. It carries its own denylist and still REQUIRES a
    # name-ish/wake token, so nameless ambient stays asleep (see test_asleep_ambient_dialogue_does_not_wake).
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _sleeping_service(client)
    svc._bare_wake_delay = 999  # don't let the bare-wake prompt fire mid-test
    asyncio.run(svc._process_context(_ctx("Ariette")))
    svc._cancel_bare_wake_prompt()
    assert svc._sleeping is False, "soundalike did not wake from sleep"
    assert client.respond_calls == []   # bare wake → nothing forwarded, no greeting


def _sleeping_service(client, *, gated=True):
    """A service already asleep, with the acoustic gate flag set as in the normal config."""
    svc, pushed = _service_with_recorder(client)
    svc.set_acoustic_wake_gated(gated)
    svc._sleeping = True
    return svc, pushed


def test_asleep_ambient_dialogue_does_not_wake():
    # 2026-06-15 regression: a phantom acoustic spike on un-AEC'd talk radio opened the mic and the next
    # line of radio dialogue ("Hike through the wilderness…") "confirmed" the wake. The transcript names no
    # wake word, so she must stay asleep and never reach the brain.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _sleeping_service(client)
    asyncio.run(svc._process_context(_ctx("Hike through the wilderness with packs on")))
    assert svc._sleeping is True
    assert client.respond_calls == []
    assert not any("awake" in t.lower() for t in _texts(pushed))


def test_asleep_sleep_phrase_does_not_wake():
    # "Stop listening." — a sleep phrase arriving on an in-flight turn right after sleep — used to short-
    # circuit to a wake whenever the acoustic gate was up. A sleep phrase can never be a wake.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _sleeping_service(client)
    asyncio.run(svc._process_context(_ctx("Stop listening.")))
    assert svc._sleeping is True
    assert client.respond_calls == []


def test_asleep_bare_vocative_wakes_silently():
    # Asleep, a bare wake leaves sleep but does NOT greet and does NOT forward — the duck is the ack, and a
    # bare wake only prompts (after a wait) if no command follows it. 2026-06-15, the maintainer.
    for text in ("Aria", "Hey Aria", "Aria, wake up", "Hey, Aria.  Hey, Ari."):
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
        svc, pushed = _sleeping_service(client)
        svc._bare_wake_delay = 999  # don't let the prompt fire during the test
        asyncio.run(svc._process_context(_ctx(text)))
        svc._cancel_bare_wake_prompt()
        assert svc._sleeping is False, f"did not wake on {text!r}"
        assert client.respond_calls == [], f"bare wake forwarded {text!r}"
        assert _spoken_texts(pushed) == [], f"bare wake spoke something for {text!r}"


def test_asleep_vocative_plus_command_forwards_stripped():
    # Asleep, "Hey Aria, <command>" wakes AND forwards only the command (wake word stripped), no greeting.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="ok"), BrainEvent("done")])
    svc, pushed = _sleeping_service(client)
    asyncio.run(svc._process_context(_ctx("Hey Aria, play a movie from jellyfin")))
    assert svc._sleeping is False
    assert client.respond_calls == [("sess-test", "play a movie from jellyfin")]
    assert not any("awake" in t.lower() for t in _spoken_texts(pushed))


def test_asleep_wake_phrase_wakes():
    # A wake phrase with no vocative still wakes (covers the gate-less degraded mode too).
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _sleeping_service(client)
    asyncio.run(svc._process_context(_ctx("okay, wake up")))
    assert svc._sleeping is False


def test_asleep_substring_vocative_does_not_wake():
    # Whole-word match: "malaria" embeds "aria" but must not wake.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _sleeping_service(client)
    asyncio.run(svc._process_context(_ctx("the malaria outbreak spread quickly")))
    assert svc._sleeping is True
    assert client.respond_calls == []


def test_deliberate_shutdown_still_fires_with_politeness():
    # The guards must NOT break a real, deliberate shutdown — a short vocative/polite command reduces to
    # ~nothing after fillers and still exits cleanly.
    from pipecat.frames.frames import EndTaskFrame

    for text in ("shut down voice mode", "Aria, shut down voice mode please",
                 "okay please shut down voice mode now"):
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
        svc, pushed = _service_with_recorder(client)
        asyncio.run(svc._process_context(_ctx(text)))
        assert client.respond_calls == [], f"reached brain for {text!r}"
        assert any(isinstance(f, EndTaskFrame) for f in pushed), f"no shutdown for {text!r}"


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
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)


def _duck(client, **kw):
    kw.setdefault("min_words", 2)
    kw.setdefault("restore_grace", 8.0)
    # The shared provider fails OPEN (unset/empty media_state → NOT playing → skip duck). Duck-*timing*
    # tests don't care about media state, so default it to "playing" here; gating tests set it explicitly.
    if getattr(client, "media_state_value", None) is None:
        client.media_state_value = {"playing": True, "state": "playing"}
    if "media_status" not in kw:
        # Wire the REAL shared media-state provider (the same one the wake gate uses) reading this
        # client — so the duck tests exercise the actual divergence-free path, not a stub.
        import types

        from config import build_media_state_provider
        kw["media_status"] = build_media_state_provider(
            types.SimpleNamespace(brain_client=client, session_id="sess-test")
        )
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


def _onset():
    return VADUserStartedSpeakingFrame()


def _offset():
    return VADUserStoppedSpeakingFrame()


def test_media_duck_on_vad_onset_immediately():
    # The latency fix: duck fires on speech *onset*, before any transcription.
    client = FakeBrainClient()
    ctl = _duck(client)

    async def go():
        ctl._handle(_onset())
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # ducked with no transcription yet


def test_media_duck_onset_confirmed_then_restores_on_bot():
    client = FakeBrainClient()
    ctl = _duck(client)

    async def go():
        ctl._handle(_onset())                          # duck on (fast)
        await asyncio.sleep(0)
        ctl._handle(_offset())                         # arm false-onset confirm timer
        ctl._handle(_spoken("play the next track"))    # real words → confirmed, cancels confirm timer
        await asyncio.sleep(0)
        ctl._handle(BotStartedSpeakingFrame())
        ctl._handle(BotStoppedSpeakingFrame())         # restore
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_duck_false_onset_no_flap_over_media_restores_via_idle_grace():
    # 2026-06-14 flap fix: while media is PLAYING, an onset with no qualifying transcription does NOT snap
    # back at confirm_grace — that flapped the bed down→up→re-duck and raced the confirmed `on` against a
    # stale `off`. It holds through confirm_grace and restores on the longer idle grace instead. (When
    # media is NOT playing the quick snap-back is preserved — see tests/test_media_duck.py.)
    client = FakeBrainClient()
    ctl = _duck(client, confirm_grace=0.01, restore_grace=0.08)

    async def go():
        ctl._handle(_onset())                          # duck on (media playing → real duck)
        await asyncio.sleep(0)
        ctl._handle(_offset())                         # speech stopped, no words → arm confirm timer
        await asyncio.sleep(0.04)                      # confirm grace elapsed → NO snap-back
        assert client.duck_calls == [("sess-test", True)]
        await asyncio.sleep(0.10)                      # idle grace elapses → restore

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_duck_continued_speech_cancels_false_onset_restore():
    # A pause mid-utterance (offset) then resumed speech (onset) must NOT flap-restore.
    client = FakeBrainClient()
    ctl = _duck(client, confirm_grace=0.02)

    async def go():
        ctl._handle(_onset())
        await asyncio.sleep(0)
        ctl._handle(_offset())                         # arm confirm timer
        ctl._handle(_onset())                          # resumed within grace → cancels it
        await asyncio.sleep(0.05)                      # would have fired if not cancelled

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]  # still ducked, no restore


def test_media_duck_no_restore_while_speech_in_flight():
    # A long monologue (onset → confirmed → STILL speaking) must NOT restore the bed mid-utterance
    # even after the idle grace elapses — the "music interjects during a long monologue" bug. The
    # idle-restore is guarded on an active VAD segment; it re-arms instead of yanking the bed.
    client = FakeBrainClient()
    ctl = _duck(client, restore_grace=0.01)

    async def go():
        ctl._handle(_onset())                              # duck on, speech in flight
        ctl._handle(_spoken("let me tell you a long story"))  # confirmed → arms idle restore
        await asyncio.sleep(0.05)                          # grace elapses while STILL speaking
        await asyncio.sleep(0.05)                          # second grace window — still no restore

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]      # never restored mid-monologue


def test_media_duck_restores_after_speech_stops():
    # Idle grace is measured from the actual speech *stop*: once the user stops, the bed restores.
    client = FakeBrainClient()
    ctl = _duck(client, restore_grace=0.01)

    async def go():
        ctl._handle(_onset())
        ctl._handle(_spoken("let me tell you a long story"))  # confirmed
        await asyncio.sleep(0.05)                          # in flight → no restore yet
        ctl._handle(_offset())                             # speech stops → idle grace starts here
        await asyncio.sleep(0.05)                          # idle grace elapses → restore

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_duck_aside_release_restores_immediately():
    # A1: the brain classifies the turn as an aside → DuckReleaseFrame → restore now, not after the idle
    # grace. Use a LONG grace so a passing test can only be the aside-release path, not the timer.
    from brains.media_duck import DuckReleaseFrame

    client = FakeBrainClient()
    ctl = _duck(client, restore_grace=100.0)

    async def go():
        ctl._handle(_onset())                              # pre-duck on
        ctl._handle(_spoken("she's in the back"))          # confirmed words (an aside, but we don't know yet)
        await asyncio.sleep(0)
        ctl._handle(_offset())                             # speech stops → idle grace armed (100s)
        ctl._handle(DuckReleaseFrame())                    # brain: not addressed → release now
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True), ("sess-test", False)]


def test_media_duck_aside_release_does_not_cut_live_tts():
    # Guard: a DuckReleaseFrame that arrives while Aria is already speaking (an aside-then-command race,
    # or a stale verdict) must NOT restore — BotStopped owns the restore in that case.
    from brains.media_duck import DuckReleaseFrame

    client = FakeBrainClient()
    ctl = _duck(client)

    async def go():
        ctl._handle(_onset())                              # pre-duck on
        ctl._handle(BotStartedSpeakingFrame())             # Aria is replying → _bot_spoke=True
        ctl._handle(DuckReleaseFrame())                    # stale aside verdict → must be ignored
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == [("sess-test", True)]      # still ducked under live TTS


def test_media_duck_aside_release_noop_when_not_ducked():
    # No duck engaged → a DuckReleaseFrame is a harmless no-op (no spurious /media/duck off).
    from brains.media_duck import DuckReleaseFrame

    client = FakeBrainClient()
    ctl = _duck(client)

    async def go():
        ctl._handle(DuckReleaseFrame())
        await asyncio.sleep(0)

    asyncio.run(go())
    assert client.duck_calls == []


def test_media_state_debounced_across_rapid_onsets():
    # With onset-triggering, a talkative paused-media stretch fires many duck attempts; the media_state
    # gate must be debounced so it doesn't re-query the brain on each one.
    client = FakeBrainClient()
    client.media_state_value = {"playing": False, "state": "idle"}  # nothing playing → all skip
    ctl = _duck(client)  # default 1.0s TTL

    async def go():
        for _ in range(6):
            ctl._handle(_onset())
        await asyncio.sleep(0.05)  # drain the fire-and-forget duck tasks

    asyncio.run(go())
    assert client.duck_calls == []           # all skipped (nothing playing)
    assert client.media_state_calls == 1     # but only one /media/state query (debounced + coalesced)


def test_status_never_spoken_only_token():
    client = FakeBrainClient(respond_events=[
        BrainEvent("status", text="Looking into it."),
        BrainEvent("status", text="Looking into it."),
        BrainEvent("status", text="Looking into it."),
        BrainEvent("token", text="Found it."),
        BrainEvent("done"),
    ])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("what was the error")))
    from pipecat.frames.frames import TTSSpeakFrame
    # status is log-only now — no status is ever voiced (TTSSpeakFrame or text); only the token speaks.
    assert not any(isinstance(f, TTSSpeakFrame) for f in pushed)
    assert _texts(pushed) == ["Found it."]


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

    # "wake up" → un-mutes; no spoken ack (the duck is the ack; a bare wake only prompts if nothing follows).
    pushed.clear()
    svc._bare_wake_delay = 999  # don't let the bare-wake prompt fire during the test
    asyncio.run(svc._process_context(_ctx("hey, wake up")))
    svc._cancel_bare_wake_prompt()
    assert svc._sleeping is False
    assert _spoken_texts(pushed) == []

    # back to normal — input reaches the brain again.
    pushed.clear()
    asyncio.run(svc._process_context(_ctx("hello")))
    assert client.respond_calls == [("sess-test", "hello")]
    assert _texts(pushed) == ["Hi!"]


def test_mute_phrases_trigger_sleep():
    # Live regression (session 064099d2): the user said "mute voice mode" 4x and got pointed at SHUTDOWN.
    # Muting / "stop until I call you" IS the sleep gate — these phrasings must put it to sleep.
    for phrase in ("Aria, mute voice mode unless I call your name.",
                   "Mute voice mode.", "mute yourself", "only respond when I call"):
        client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
        svc, pushed = _service_with_recorder(client)
        asyncio.run(svc._process_context(_ctx(phrase)))
        assert svc._sleeping is True, f"did not sleep for {phrase!r}"
        assert client.respond_calls == [], f"reached brain for {phrase!r}"


def test_empty_user_turn_skipped():
    # A force-completed runaway turn (or stray VAD blip) with no words must NOT reach the brain.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="x"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("   ")))
    assert client.respond_calls == []
    assert _texts(pushed) == []


def test_mute_the_music_does_not_sleep():
    # Guard: "mute the music" is a MEDIA command, not a request to stop listening — must reach the
    # brain, never trigger sleep (why bare "mute" is deliberately not a sleep phrase).
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="OK."), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("mute the music")))
    assert svc._sleeping is False
    assert client.respond_calls == [("sess-test", "mute the music")]


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


# --- P1: mute through the brain's "thinking" gap (empty-barge-in fix) -----------------
from pipecat.frames.frames import LLMContextFrame  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from turn_mute import BotThinkingFrame  # noqa: E402


def _thinking(frames):
    return [f.active for f in frames if isinstance(f, BotThinkingFrame)]


def test_thinking_frame_brackets_a_normal_turn():
    # The bot turn must be bracketed by BotThinkingFrame(True) … (False), with True pushed BEFORE any
    # spoken text (so the user is muted through the first-token gap) and False after — covering the
    # whole turn for BotThinkingMuteStrategy.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi "), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc.process_frame(LLMContextFrame(_ctx("hello")), FrameDirection.DOWNSTREAM))
    assert _thinking(pushed) == [True, False]
    # active=True is emitted before the first LLMTextFrame.
    kinds = [type(f).__name__ for f in pushed]
    assert kinds.index("BotThinkingFrame") < kinds.index("LLMTextFrame")


def test_thinking_frame_cleared_on_barge_in():
    # On a barge-in (CancelledError mid-stream) the finally must still push BotThinkingFrame(False), or
    # the user would stay muted forever after an interrupted turn.
    client = FakeBrainClient(
        respond_events=[BrainEvent("token", text=f"c{i} ") for i in range(20)], delay=0.02,
    )
    svc, pushed = _service_with_recorder(client)

    async def run_and_cancel():
        task = asyncio.create_task(
            svc.process_frame(LLMContextFrame(_ctx("tell me a story")), FrameDirection.DOWNSTREAM)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_and_cancel())
    assert _thinking(pushed)[-1] is False           # released despite the cancel
    assert _thinking(pushed).count(True) == _thinking(pushed).count(False)


def test_thinking_frame_cleared_when_turn_speaks_nothing():
    # An empty/sleep/skip turn that never produces a token must still release the mute (True then False).
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="x"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc.process_frame(LLMContextFrame(_ctx("   ")), FrameDirection.DOWNSTREAM))
    assert _thinking(pushed) == [True, False]
    assert _texts(pushed) == []                      # nothing spoken, but mute still cleared
    assert client.respond_calls == []                # empty turn never hit the brain


# --- P2: acoustic-wake-while-asleep (WakeSleepFrame + gate force) ----------------------
def _sleep_frames(frames):
    from wake_word import WakeSleepFrame
    return [f.asleep for f in frames if isinstance(f, WakeSleepFrame)]


def test_going_to_sleep_forces_the_gate():
    # Entering sleep must push WakeSleepFrame(asleep=True) UPSTREAM so the gate forces acoustic gating.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("go to sleep please")))
    assert svc._sleeping is True
    assert _sleep_frames(pushed) == [True]


def test_acoustic_gated_still_requires_a_wake_token():
    # Updated contract (2026-06-15): even with an upstream gate, a bare command arriving while asleep does
    # NOT wake — a phantom acoustic spike on un-AEC'd room audio can open the gate without a real wake, so
    # the transcript must name her / carry a wake phrase. The vocative does wake and releases the gate force.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    svc.set_acoustic_wake_gated(True)

    asyncio.run(svc._process_context(_ctx("go to sleep")))
    assert svc._sleeping is True

    pushed.clear()
    # A bare command (no wake token) must NOT wake — this is the radio/TV self-wake the fix closes.
    asyncio.run(svc._process_context(_ctx("what time is it")))
    assert svc._sleeping is True
    assert _texts(pushed) == []

    pushed.clear()
    # Naming her wakes and releases the gate force; a bare wake speaks nothing and doesn't hit the brain.
    svc._bare_wake_delay = 999  # don't let the bare-wake prompt fire during the test
    asyncio.run(svc._process_context(_ctx("Aria")))
    svc._cancel_bare_wake_prompt()
    assert svc._sleeping is False
    assert _sleep_frames(pushed) == [False]
    assert _spoken_texts(pushed) == []                     # no greeting — the duck is the ack
    assert client.respond_calls == []                # the wake turn itself doesn't hit the brain


def test_gateless_fallback_uses_text_match():
    # The text wake requirement applies in the gate-less config too (_acoustic_wake_gated stays False): a
    # non-wake transcript is ignored; a wake phrase wakes. Note the loose wake-phrase match means an ambient
    # "wake up" line can still trip it — which is why the vocative ("Aria") is the primary, tighter wake.
    client = FakeBrainClient(respond_events=[BrainEvent("token", text="Hi!"), BrainEvent("done")])
    svc, pushed = _service_with_recorder(client)
    asyncio.run(svc._process_context(_ctx("go to sleep")))

    pushed.clear()
    asyncio.run(svc._process_context(_ctx("what time is it")))   # no wake token → ignored
    assert svc._sleeping is True
    assert _texts(pushed) == []

    pushed.clear()
    asyncio.run(svc._process_context(_ctx("we can wake up very early")))  # wake phrase present → wakes
    assert svc._sleeping is False


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


def test_stop_word_frame_triggers_interruption():
    # V2 barge-in: the wake gate's StopWordFrame survives the half-duplex user-mute (which drops
    # InterruptionFrame), and BrainLLMService — sitting below that mute — converts it into the real
    # interruption (broadcast_interruption → flush TTS downstream + cancel this turn → POST /cancel).
    from pipecat.processors.frame_processor import FrameDirection

    from wake_word import StopWordFrame

    client = FakeBrainClient()
    svc, _pushed = _service_with_recorder(client)
    calls = []

    async def _spy():
        calls.append(True)

    svc.broadcast_interruption = _spy  # type: ignore[assignment]

    asyncio.run(svc.process_frame(StopWordFrame(), FrameDirection.DOWNSTREAM))
    assert calls == [True]  # interruption issued from below the mute


def test_stop_word_cancels_inflight_turn_and_posts_cancel():
    # V2 Test B fix (2026-06-16 live): a mid-GENERATION barge must CANCEL the in-flight turn task, not just
    # flush the current TTS buffer. When the turn ran inline, broadcast_interruption() flushed audio but never
    # cancelled this service's own _consume → the long reply kept streaming → she resumed ~5s later, and no
    # /cancel ever reached the brain. Now the turn runs as a cancellable task; StopWordFrame cancels it →
    # _consume CancelledError → POST /cancel + token stream stops.
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection

    from wake_word import StopWordFrame

    # A slow, long stream so the turn is genuinely still in-flight when the barge lands (the overlap case).
    client = FakeBrainClient(
        respond_events=[BrainEvent("token", text=f"w{i} ") for i in range(20)] + [BrainEvent("done")],
        delay=0.02,
    )
    svc, pushed = _service_with_recorder(client)

    async def _noop():  # broadcast_interruption needs a live pipeline; stub the flush, test the cancel
        pass

    svc.broadcast_interruption = _noop  # type: ignore[assignment]

    async def go():
        await svc.process_frame(LLMContextFrame(_ctx("tell me a long story")), FrameDirection.DOWNSTREAM)
        task = svc._turn_task
        assert task is not None and not task.done()      # turn runs as a task, not inline
        await asyncio.sleep(0.05)                          # let a few tokens stream
        await svc.process_frame(StopWordFrame(), FrameDirection.DOWNSTREAM)  # mid-gen barge → cancel
        await task                                         # unwind: _consume CancelledError → /cancel

    asyncio.run(go())
    assert client.cancel_calls == ["sess-test"]            # POST /cancel reached the brain (aborts live turn)
    spoken = _texts(pushed)
    assert 0 < len(spoken) < 20                            # stopped early — did NOT stream all 20 tokens


def test_inflight_turn_runs_as_cancellable_task_not_inline():
    # The LLMContextFrame handler must hand the turn to a task (so a SystemFrame barge can cancel it),
    # rather than awaiting it inline (which left nothing to cancel). process_frame returns while the turn
    # is still streaming.
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.frame_processor import FrameDirection

    client = FakeBrainClient(
        respond_events=[BrainEvent("token", text="slow "), BrainEvent("token", text="reply."),
                        BrainEvent("done")],
        delay=0.05,
    )
    svc, _pushed = _service_with_recorder(client)

    async def go():
        await svc.process_frame(LLMContextFrame(_ctx("hello")), FrameDirection.DOWNSTREAM)
        # process_frame returned but the turn is still in flight (delay keeps it streaming)
        assert svc._turn_task is not None and not svc._turn_task.done()
        await svc._turn_task                               # let it finish cleanly
        assert svc._turn_task.done()

    asyncio.run(go())


# --- brain-down resilience: a crashed brain must not leave the user in dead air -------------------
# Live (2026-06-17): gabagent died mid web-lookup ("peer closed connection ... incomplete chunked
# read", then "All connection attempts failed"). The turn's except only logged an ErrorFrame, so Aria
# went SILENT and the user waited on a reply that never came. _run_turn now speaks a brief apology
# (TTS is local — works even with the brain unreachable), dedup'd per outage and re-armed by a clean turn.
from brains.brain_llm_service import _BRAIN_ERROR_REPLY


async def _noop_metrics(*a, **k):  # start/stop_processing_metrics need a live pipeline; no-op offline
    pass


def _arm_offline(svc):
    svc.start_processing_metrics = _noop_metrics
    svc.stop_processing_metrics = _noop_metrics


class _CrashingClient(FakeBrainClient):
    """respond() dies before yielding — simulates the brain dropping the chunked stream mid-request."""

    def __init__(self):
        super().__init__()
        self.mode = "fail"  # "fail" → raise; "ok" → stream a normal reply

    async def respond(self, session_id, text):
        self.respond_calls.append((session_id, text))
        if self.mode == "fail":
            raise RuntimeError("peer closed connection without sending complete message body")
        for ev in (BrainEvent("token", text="Done."), BrainEvent("done")):
            yield ev


def test_brain_crash_speaks_apology_not_silence():
    svc, pushed = _service_with_recorder(_CrashingClient())
    _arm_offline(svc)
    asyncio.run(svc._run_turn(_ctx("find us a campsite")))
    assert _BRAIN_ERROR_REPLY in _spoken_texts(pushed)     # audible signal, not dead air


def test_brain_crash_apology_dedups_while_down():
    client = _CrashingClient()
    svc, pushed = _service_with_recorder(client)
    _arm_offline(svc)
    asyncio.run(svc._run_turn(_ctx("q1")))                 # 1st failure speaks
    assert _BRAIN_ERROR_REPLY in _spoken_texts(pushed)
    pushed.clear()
    asyncio.run(svc._run_turn(_ctx("q2")))                 # brain still down → no repeat
    assert _BRAIN_ERROR_REPLY not in _spoken_texts(pushed)


def test_brain_error_notice_rearms_after_clean_turn():
    client = _CrashingClient()
    svc, pushed = _service_with_recorder(client)
    _arm_offline(svc)
    asyncio.run(svc._run_turn(_ctx("q1")))                 # fail → speaks
    assert _BRAIN_ERROR_REPLY in _spoken_texts(pushed)
    client.mode = "ok"; pushed.clear()
    asyncio.run(svc._run_turn(_ctx("q2")))                 # clean turn → re-arms
    assert "Done." in _texts(pushed)
    client.mode = "fail"; pushed.clear()
    asyncio.run(svc._run_turn(_ctx("q3")))                 # fails again → speaks again
    assert _BRAIN_ERROR_REPLY in _spoken_texts(pushed)
