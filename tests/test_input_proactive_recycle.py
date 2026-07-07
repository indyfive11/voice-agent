"""Tests for config.recycle_input_device + proactive_input_recycle_loop — the proactive capture recycle
(task #84) that reopens the mic stream at ~18:00 to dodge the XVF3800's deterministic 1200s stream-age
firmware wedge BEFORE it fires (vs #83's reactive ~66s recovery every ~21min).

Design + non-root ladder ratified with GA 2026-07-06 (stream-scoped clock proven by ten clean, never-
USB-reset 20:00 lifetimes). Guardrails under test:
  - LIGHT recycle reopens on the SHARED PyAudio client (output untouched — never terminated).
  - HEAVY (#2.5a) opens a FRESH dedicated PyAudio instance, swaps it in, and NEVER terminates the original
    shared instance (pipecat hands one instance to both input and output) — only a PRIOR dedicated one.
  - open() failure degrades to False + a nulled stream (the reactive watchdog is the backstop).
  - an explicit index pin is respected as a valid recycle target; an unset name falls back to the bound index.
  - the loop is OFF (returns immediately) unless INPUT_PROACTIVE_RECYCLE_SECS is set.
"""

import asyncio

import pytest

import config


# --- fakes (self-contained; add terminate()-tracking the reactive-test fakes don't have) -----------

class _FakeStream:
    def __init__(self, *, started=False):
        self.started = started
        self.stopped = False
        self.closed = False

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakeParams:
    def __init__(self, *, input_device_index, audio_in_channels=1, audio_in_sample_rate=0):
        self.input_device_index = input_device_index
        self.audio_in_channels = audio_in_channels
        self.audio_in_sample_rate = audio_in_sample_rate


class _FakePyAudio:
    def __init__(self, *, open_raises=False):
        self.open_raises = open_raises
        self.open_calls = []
        self.terminated = False

    def get_format_from_width(self, width):
        return f"fmt{width}"

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        if self.open_raises:
            raise OSError("[Errno -9997] Invalid sample rate")
        return _FakeStream(started=False)

    def terminate(self):
        self.terminated = True


class _FakeInput:
    def __init__(self, *, cur_idx, in_stream=None, py_audio=None):
        self._params = _FakeParams(input_device_index=cur_idx)
        self._py_audio = py_audio or _FakePyAudio()
        self._in_stream = in_stream if in_stream is not None else _FakeStream(started=True)
        self._sample_rate = 0

    def _audio_in_callback(self, *a):
        return None


def _base_env(monkeypatch, *, name="reSpeaker", index_pin=None):
    if name is not None:
        monkeypatch.setenv("AUDIO_INPUT_DEVICE_NAME", name)
    else:
        monkeypatch.delenv("AUDIO_INPUT_DEVICE_NAME", raising=False)
    monkeypatch.delenv("AUDIO_INPUT_DEVICE_INDEX", raising=False)
    if index_pin is not None:
        monkeypatch.setenv("AUDIO_INPUT_DEVICE_INDEX", str(index_pin))
    monkeypatch.setattr(config, "_supported_input_rate", lambda idx, **k: 48000)


# --- LIGHT recycle --------------------------------------------------------------------------------

def test_light_recycle_reopens_on_shared_client_output_untouched(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)  # same index (healthy device)
    pa = _FakePyAudio()
    inp = _FakeInput(cur_idx=11, py_audio=pa)
    old = inp._in_stream

    assert config.recycle_input_device(inp, heavy=False) is True
    assert old.stopped and old.closed                      # live stream torn down
    assert inp._in_stream is not old and inp._in_stream.started
    assert inp._py_audio is pa                              # SHARED client reused (output stays on it)...
    assert pa.terminated is False                          # ...and never terminated
    assert inp._params.input_device_index == 11
    assert inp._sample_rate == 48000 and inp._params.audio_in_sample_rate == 48000
    assert len(pa.open_calls) == 1
    kw = pa.open_calls[0]
    assert kw["input_device_index"] == 11 and kw["rate"] == 48000 and kw["input"] is True
    assert kw["frames_per_buffer"] == int(48000 / 100) * 2  # 20ms, mirrors pipecat start()


def test_light_recycle_open_failure_returns_false_and_nulls_stream(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)
    inp = _FakeInput(cur_idx=11, py_audio=_FakePyAudio(open_raises=True))

    assert config.recycle_input_device(inp, heavy=False) is False
    assert inp._in_stream is None                           # left closed → reactive watchdog recovers
    assert inp._params.input_device_index == 11             # pin unchanged on failure


def test_index_pin_is_a_valid_recycle_target(monkeypatch):
    # Unlike the reactive reopen (which respects a pin by NOT re-resolving), a proactive recycle still
    # recycles a pinned device — the pin is just the target index, resolve is skipped.
    _base_env(monkeypatch, index_pin=7)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 99)  # must NOT be consulted
    pa = _FakePyAudio()
    inp = _FakeInput(cur_idx=7, py_audio=pa)

    assert config.recycle_input_device(inp, heavy=False) is True
    assert pa.open_calls[0]["input_device_index"] == 7      # the pin, not the resolver's 99


def test_name_unset_falls_back_to_bound_index(monkeypatch):
    _base_env(monkeypatch, name=None)
    pa = _FakePyAudio()
    inp = _FakeInput(cur_idx=11, py_audio=pa)

    assert config.recycle_input_device(inp, heavy=False) is True
    assert pa.open_calls[0]["input_device_index"] == 11     # the currently bound index


def test_not_a_recognized_transport_returns_false(monkeypatch):
    _base_env(monkeypatch)

    class _Bare:
        _params = None

    assert config.recycle_input_device(_Bare(), heavy=False) is False


# --- HEAVY (#2.5a) recycle ------------------------------------------------------------------------

def test_heavy_recycle_uses_fresh_instance_and_never_terminates_shared(monkeypatch):
    pyaudio = pytest.importorskip("pyaudio")
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)

    fresh = _FakePyAudio()
    monkeypatch.setattr(pyaudio, "PyAudio", lambda: fresh)   # the fresh dedicated input client

    shared = _FakePyAudio()                                  # the original instance (also backs output)
    inp = _FakeInput(cur_idx=11, py_audio=shared)
    old = inp._in_stream

    assert config.recycle_input_device(inp, heavy=True) is True
    assert old.stopped and old.closed                       # old input stream torn down
    assert inp._py_audio is fresh                            # swapped to the fresh dedicated client
    assert inp._in_stream.started
    assert shared.terminated is False                       # ORIGINAL shared instance NEVER terminated
    assert getattr(fresh, "_va_input_dedicated", False) is True  # marked for a future heavy recycle
    assert fresh.open_calls[0]["input_device_index"] == 11


def test_repeated_heavy_recycle_terminates_prior_dedicated_not_shared(monkeypatch):
    pyaudio = pytest.importorskip("pyaudio")
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)

    first = _FakePyAudio()
    second = _FakePyAudio()
    seq = iter([first, second])
    monkeypatch.setattr(pyaudio, "PyAudio", lambda: next(seq))

    shared = _FakePyAudio()
    inp = _FakeInput(cur_idx=11, py_audio=shared)

    assert config.recycle_input_device(inp, heavy=True) is True   # shared -> first (dedicated)
    assert inp._py_audio is first and shared.terminated is False

    assert config.recycle_input_device(inp, heavy=True) is True   # first -> second; first is dedicated → freed
    assert inp._py_audio is second
    assert first.terminated is True                          # prior DEDICATED instance terminated
    assert shared.terminated is False                        # shared instance still never touched


def test_heavy_open_failure_terminates_fresh_instance_no_leak(monkeypatch):
    # A (GA review 2026-07-06): a heavy open that raises AFTER pyaudio.PyAudio() stood up the client must
    # terminate that fresh client — else repeated heavy failures on a flaky device leak one client each.
    pyaudio = pytest.importorskip("pyaudio")
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)

    fresh = _FakePyAudio(open_raises=True)                   # PyAudio() succeeds, then open() throws
    monkeypatch.setattr(pyaudio, "PyAudio", lambda: fresh)

    shared = _FakePyAudio()
    inp = _FakeInput(cur_idx=11, py_audio=shared)

    assert config.recycle_input_device(inp, heavy=True) is False
    assert fresh.terminated is True                          # fresh client reaped, not orphaned
    assert shared.terminated is False                        # shared/output client never touched
    assert inp._in_stream is None                            # left closed → reactive watchdog recovers


# --- B: mutual exclusion with the reactive watchdog -----------------------------------------------

def test_recycle_skips_when_recovery_lock_held(monkeypatch):
    # The reactive path holds _input_recovery_lock while it recovers; a proactive recycle must NOT race it —
    # it skips (returns False) and mutates nothing. And a normal recycle must not leave the lock held.
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)
    pa = _FakePyAudio()
    inp = _FakeInput(cur_idx=11, py_audio=pa)
    old = inp._in_stream

    assert config._input_recovery_lock.acquire(blocking=False) is True   # stand in for the reactive path
    try:
        assert config.recycle_input_device(inp, heavy=False) is False    # proactive backs off
        assert inp._in_stream is old and not old.stopped and not old.closed
        assert pa.open_calls == []                                       # never touched the device
    finally:
        config._input_recovery_lock.release()

    # lock free again → a normal recycle proceeds AND releases the lock (finally: fires on the success path)
    assert config.recycle_input_device(inp, heavy=False) is True
    assert config._input_recovery_lock.acquire(blocking=False) is True
    config._input_recovery_lock.release()


# --- the loop -------------------------------------------------------------------------------------

def test_loop_disabled_returns_immediately(monkeypatch):
    monkeypatch.delenv("INPUT_PROACTIVE_RECYCLE_SECS", raising=False)
    called = []
    monkeypatch.setattr(config, "recycle_input_device", lambda *a, **k: called.append(1) or True)
    # returns without ever sleeping or recycling
    asyncio.run(config.proactive_input_recycle_loop(_FakeInput(cur_idx=11)))
    assert called == []


def test_loop_recycles_after_interval_and_honors_mode(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("INPUT_PROACTIVE_RECYCLE_SECS", "1080")
    monkeypatch.setenv("INPUT_PROACTIVE_RECYCLE_MODE", "heavy")
    calls = []
    monkeypatch.setattr(config, "recycle_input_device", lambda inp, *, heavy: calls.append(heavy) or True)

    # Drive exactly one iteration: first sleep returns, then the second sleep aborts the loop.
    sleeps = {"n": 0}

    async def _fake_sleep(_secs):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(config.proactive_input_recycle_loop(_FakeInput(cur_idx=11), is_idle=lambda: True))
    assert calls == [True]  # one recycle, heavy mode honored


def test_loop_defers_until_idle(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("INPUT_PROACTIVE_RECYCLE_SECS", "1080")
    monkeypatch.setenv("INPUT_PROACTIVE_RECYCLE_MAX_DEFER_SECS", "90")
    recycled = []
    monkeypatch.setattr(config, "recycle_input_device", lambda inp, *, heavy: recycled.append(heavy) or True)

    idle_states = iter([False, False, True])  # busy for two polls, then the floor frees
    probed = []

    def _is_idle():
        v = next(idle_states)
        probed.append(v)
        return v

    sleeps = {"n": 0}

    async def _fake_sleep(_secs):
        sleeps["n"] += 1
        # iter1: interval(1) + two idle-polls(2,3) → recycle; iter2: next interval sleep(4) aborts
        # BEFORE a second recycle, so we assert exactly one deferred recycle.
        if sleeps["n"] >= 4:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(config.proactive_input_recycle_loop(_FakeInput(cur_idx=11), is_idle=_is_idle))
    assert recycled == [False]       # recycled once (light), only AFTER idle returned True
    assert probed[-1] is True        # deferred through the busy polls until the floor was free
