"""ImageDisplaySink + JellyfinRoomController — roadmap ③ image display, offline.

Pure/injected design: the sink takes a fake runner + fake controller, the controller takes a fake httpx
client, so these tests need no mpv, no Jellyfin, no display. Driven by calling the async methods directly
via asyncio.run for deterministic assertions.
"""

import asyncio

from image_display import ImageDisplaySink, JellyfinRoomController, NullRoomController


# --------------------------------------------------------------------------- fakes
class _FakeRunner:
    def __init__(self, ok=True):
        self.calls: list[tuple[str, int]] = []
        self._ok = ok

    async def __call__(self, src, secs):
        self.calls.append((src, secs))
        return self._ok


class _FakeController:
    def __init__(self, playing):
        self._playing = playing
        self.paused = False
        self.resumed = False

    async def pause_if_playing(self):
        self.paused = self._playing
        return self._playing

    async def resume(self):
        self.resumed = True


def _sink(runner, controller=None, **kw):
    return ImageDisplaySink(runner=runner, controller=controller or NullRoomController(),
                            enabled=kw.pop("enabled", True), **kw)


# --------------------------------------------------------------------------- locality routing
def test_resolve_prefers_existing_local_path(tmp_path):
    f = tmp_path / "img.png"
    f.write_bytes(b"x")
    assert ImageDisplaySink._resolve_source({"path": str(f), "url": "https://cdn/x.png"}) == str(f)


def test_resolve_falls_back_to_url_when_path_absent():
    d = {"path": "/nonexistent/here.png", "url": "https://cdn/x.png"}
    assert ImageDisplaySink._resolve_source(d) == "https://cdn/x.png"


def test_resolve_none_when_neither_usable():
    assert ImageDisplaySink._resolve_source({"path": "/nope.png"}) is None
    assert ImageDisplaySink._resolve_source({}) is None


# --------------------------------------------------------------------------- show()
def test_show_renders_and_acks(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    r = _FakeRunner()
    s = _sink(r, display_secs=25)
    asyncio.run(s.show({"url": "https://cdn/x.png"}, job_id="img-1"))
    assert r.calls == [("https://cdn/x.png", 25)]
    assert s.drain_displayed() == ["img-1"]
    assert s.drain_displayed() == []            # drained → cleared


def test_show_skipped_when_disabled_but_still_acked(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    r = _FakeRunner()
    s = _sink(r, enabled=False)
    asyncio.run(s.show({"url": "u"}, job_id="img-1"))
    assert r.calls == []                        # not rendered
    assert s.drain_displayed() == ["img-1"]     # but acked (don't re-send an image we chose not to show)


def test_show_skipped_when_no_display_present(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    r = _FakeRunner()
    s = _sink(r)
    asyncio.run(s.show({"url": "u"}, job_id="img-1"))
    assert r.calls == []
    assert s.drain_displayed() == ["img-1"]


def test_show_acks_even_when_no_source(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    r = _FakeRunner()
    s = _sink(r)
    asyncio.run(s.show({}, job_id="img-1"))     # nothing renderable
    assert r.calls == []
    assert s.drain_displayed() == ["img-1"]


def test_show_acks_even_on_render_failure(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    r = _FakeRunner(ok=False)
    s = _sink(r)
    asyncio.run(s.show({"url": "u"}, job_id="img-1"))
    assert r.calls == [("u", 30)]               # attempted (default 30s window)
    assert s.drain_displayed() == ["img-1"]


# --------------------------------------------------------------------------- pause → show → resume
def test_pause_and_resume_when_media_playing(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    r = _FakeRunner()
    ctl = _FakeController(playing=True)
    s = _sink(r, controller=ctl)
    asyncio.run(s.show({"url": "u"}, job_id="img-1"))
    assert ctl.paused and ctl.resumed           # paused before, resumed after
    assert r.calls == [("u", 30)]


def test_no_resume_when_nothing_playing(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    r = _FakeRunner()
    ctl = _FakeController(playing=False)
    s = _sink(r, controller=ctl)
    asyncio.run(s.show({"url": "u"}, job_id="img-1"))
    assert not ctl.paused and not ctl.resumed   # nothing to pause → nothing to resume
    assert r.calls == [("u", 30)]


def test_resume_runs_even_if_render_raises(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    ctl = _FakeController(playing=True)

    async def boom(src, secs):
        raise RuntimeError("mpv exploded")

    s = _sink(boom, controller=ctl)
    asyncio.run(s.show({"url": "u"}, job_id="img-1"))
    assert ctl.resumed                          # finally-resume even on a render exception
    assert s.drain_displayed() == ["img-1"]


# --------------------------------------------------------------------------- JellyfinRoomController
class _FakeHttp:
    def __init__(self, sessions):
        self._sessions = sessions
        self.posts: list[str] = []

    async def get(self, url, params=None):
        return _Resp(200, self._sessions)

    async def post(self, url, params=None):
        self.posts.append(url)
        return _Resp(204)


class _Resp:
    def __init__(self, status, data=None):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


def _ctl(sessions):
    http = _FakeHttp(sessions)
    c = JellyfinRoomController(url="http://jf:8096", token="tok", device="bedroom-jellyfin", http=http)
    return c, http


def test_jf_disabled_when_unconfigured():
    c = JellyfinRoomController(url="", token="", device="")
    assert not c.enabled
    assert asyncio.run(c.pause_if_playing()) is False


def test_jf_pauses_playing_then_resumes():
    c, http = _ctl([{"DeviceName": "bedroom-jellyfin", "Id": "S1",
                     "NowPlayingItem": {"Name": "Movie"}, "PlayState": {"IsPaused": False}}])
    assert asyncio.run(c.pause_if_playing()) is True
    assert http.posts == ["http://jf:8096/Sessions/S1/Playing/Pause"]
    asyncio.run(c.resume())
    assert http.posts[-1] == "http://jf:8096/Sessions/S1/Playing/Unpause"


def test_jf_skips_when_nothing_playing():
    c, http = _ctl([{"DeviceName": "bedroom-jellyfin", "Id": "S1", "NowPlayingItem": None}])
    assert asyncio.run(c.pause_if_playing()) is False
    assert http.posts == []


def test_jf_skips_when_already_paused():
    c, http = _ctl([{"DeviceName": "bedroom-jellyfin", "Id": "S1",
                     "NowPlayingItem": {"Name": "M"}, "PlayState": {"IsPaused": True}}])
    assert asyncio.run(c.pause_if_playing()) is False
    assert http.posts == []


def test_jf_ignores_other_device():
    c, http = _ctl([{"DeviceName": "someone-else", "Id": "S9",
                     "NowPlayingItem": {"Name": "M"}, "PlayState": {"IsPaused": False}}])
    assert asyncio.run(c.pause_if_playing()) is False
    assert http.posts == []


def test_jf_resume_noop_when_nothing_paused():
    c, http = _ctl([])
    asyncio.run(c.resume())                     # never paused → no post, no error
    assert http.posts == []


# --------------------------------------------------------------------------- fail-soft factory
def test_make_sink_returns_none_when_module_absent():
    """Regression for the 2026-07-04 Pi crash: an untracked image_display.py shipped its tracked importers
    to the satellite but not itself → ModuleNotFoundError at startup took the whole voice agent down.
    make_image_display_sink() must degrade to None (feature disabled) instead of crashing — an OPTIONAL
    display feature can never nuke the voice loop over deploy skew."""
    import sys
    from unittest.mock import patch

    import config

    # sys.modules[name] = None makes `from image_display import ...` raise ImportError (import machinery
    # treats a None entry as "known-absent"), simulating the module never reaching the box.
    with patch.dict(sys.modules, {"image_display": None}):
        assert config.make_image_display_sink() is None
