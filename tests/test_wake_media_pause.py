"""Tests for WakeMediaPauser (pause a playing video for the wake command window) + its config wiring."""

import asyncio

import pytest

from wake_media_pause import WakeMediaPauser


class FakeController:
    """Stand-in JellyfinRoomController: records pause/resume/forget calls."""

    def __init__(self, *, enabled=True, playing=True, raise_on=None):
        self._enabled = enabled
        self._playing = playing
        self._raise_on = raise_on or set()
        self.paused = False
        self.resumed = False
        self.forgot = False

    @property
    def enabled(self):
        return self._enabled

    async def pause_if_playing(self):
        if "pause" in self._raise_on:
            raise RuntimeError("pause boom")
        if self._playing:
            self.paused = True
            return True
        return False

    async def resume(self):
        if "resume" in self._raise_on:
            raise RuntimeError("resume boom")
        self.resumed = True

    def forget(self):
        self.forgot = True


def _run(coro):
    return asyncio.run(coro)


# ---- engage --------------------------------------------------------------

def test_engage_pauses_a_playing_video():
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c)
    _run(p.engage())
    assert c.paused is True
    assert p._engaged is True


def test_engage_is_idempotent_within_a_window():
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c)
    _run(p.engage())
    c.paused = False  # a second engage must NOT re-pause
    _run(p.engage())
    assert c.paused is False


def test_engage_noop_when_nothing_playing():
    c = FakeController(playing=False)
    p = WakeMediaPauser(controller=c)
    _run(p.engage())
    assert c.paused is False
    assert p._engaged is False


def test_engage_failsoft_swallows_controller_error():
    c = FakeController(raise_on={"pause"})
    p = WakeMediaPauser(controller=c)
    _run(p.engage())  # must not raise
    assert p._engaged is False


# ---- release -------------------------------------------------------------

def test_release_resumes_when_engaged_and_no_transport():
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c)
    _run(p.engage())
    _run(p.release())
    assert c.resumed is True
    assert c.forgot is False
    assert p._engaged is False


def test_release_suppressed_on_transport_intent():
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c, transport_intent=lambda: True)
    _run(p.engage())
    _run(p.release())
    # The turn was "pause it"/"stop it" → honor it: forget the marker, do NOT resume.
    assert c.resumed is False
    assert c.forgot is True
    assert p._engaged is False


def test_release_resumes_when_transport_intent_false():
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c, transport_intent=lambda: False)
    _run(p.engage())
    _run(p.release())
    assert c.resumed is True
    assert c.forgot is False


def test_release_noop_when_not_engaged():
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c)
    _run(p.release())  # never engaged → nothing to do
    assert c.resumed is False
    assert c.forgot is False


def test_release_failsoft_swallows_controller_error():
    c = FakeController(playing=True, raise_on={"resume"})
    p = WakeMediaPauser(controller=c)
    _run(p.engage())
    _run(p.release())  # must not raise
    assert p._engaged is False  # marker cleared even though resume threw


# ---- transport latch reset at fresh window -------------------------------

def test_engage_resets_transport_latch_at_fresh_window():
    # A fresh window clears any stale transport latch before listening, so a prior window's flag can't
    # suppress THIS window's resume.
    c = FakeController(playing=True)
    reset_calls = []
    p = WakeMediaPauser(controller=c, reset_transport_intent=lambda: reset_calls.append(1))
    _run(p.engage())
    assert reset_calls == [1]


def test_engage_resets_latch_even_when_nothing_playing():
    # The reset must fire on every fresh window entry — even a no-video window — so a lingering latch from a
    # "pause the music" turn (paused no video) is cleared before a later video window reads it.
    c = FakeController(playing=False)
    reset_calls = []
    p = WakeMediaPauser(controller=c, reset_transport_intent=lambda: reset_calls.append(1))
    _run(p.engage())
    assert reset_calls == [1]
    assert p._engaged is False


def test_engage_does_not_reset_latch_on_duplicate_within_window():
    # Idempotent engage: a duplicate wake in the same window must NOT clear a latch set earlier this window.
    c = FakeController(playing=True)
    reset_calls = []
    p = WakeMediaPauser(controller=c, reset_transport_intent=lambda: reset_calls.append(1))
    _run(p.engage())
    _run(p.engage())
    assert reset_calls == [1]  # only the first (fresh) engage reset


def test_engage_reset_failsoft_swallows_error():
    def boom():
        raise RuntimeError("reset boom")
    c = FakeController(playing=True)
    p = WakeMediaPauser(controller=c, reset_transport_intent=boom)
    _run(p.engage())  # must not raise
    assert c.paused is True  # reset failure did not abort the pause


def test_build_wires_reset_transport_intent_from_llm():
    import config

    class _LLM:
        last_transport_intent = False
        _reset = 0

        def reset_transport_intent(self):
            type(self)._reset += 1

    p = config.build_wake_media_pauser(_LLM(), controller=FakeController(enabled=True))
    assert p is not None and p._reset_transport_intent is not None
    p._reset_transport_intent()
    assert _LLM._reset == 1


# ---- enabled -------------------------------------------------------------

def test_enabled_reflects_controller():
    assert WakeMediaPauser(controller=FakeController(enabled=True)).enabled is True
    assert WakeMediaPauser(controller=FakeController(enabled=False)).enabled is False


# ---- config.build_wake_media_pauser --------------------------------------

def test_build_returns_none_without_controller():
    import config
    assert config.build_wake_media_pauser(object(), controller=None) is None


def test_build_returns_none_when_controller_disabled():
    import config
    assert config.build_wake_media_pauser(object(), controller=FakeController(enabled=False)) is None


def test_build_wires_transport_intent_from_llm():
    import config

    class _LLM:
        last_transport_intent = True

    p = config.build_wake_media_pauser(_LLM(), controller=FakeController(enabled=True))
    assert p is not None
    assert p._transport_intent is not None and p._transport_intent() is True


def test_build_transport_intent_none_when_llm_lacks_it():
    import config
    p = config.build_wake_media_pauser(object(), controller=FakeController(enabled=True))
    assert p is not None
    assert p._transport_intent is None


# ---- JellyfinRoomController.forget + NullRoomController.forget ------------

def test_jellyfin_controller_forget_drops_marker_without_unpausing():
    from image_display import JellyfinRoomController
    c = JellyfinRoomController(url="http://x", token="t", device="bedroom-jellyfin")
    c._paused_session = "sess-1"
    c.forget()
    assert c._paused_session is None


def test_null_controller_forget_is_noop():
    from image_display import NullRoomController
    NullRoomController().forget()  # must not raise
