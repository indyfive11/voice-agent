"""Tests for config.maybe_reopen_input_device — the part-3 self-heal (task #78) that reopens capture on
a DIFFERENT device index when a mic stall coincides with the pinned name re-resolving elsewhere (replug).

Design ratified with GA 2026-07-05. Guardrails under test: reopen only on an actual index change (else
kick), only when name-pinned (respect an explicit index pin), kill-switch, a failed open() degrades to
the escalate ladder (returns "failed" → caller returns False → os._exit), and E1 (a retry after a nulled
stream genuinely re-attempts open()). Plus a drift test that the reopen's open() kwargs mirror pipecat's
LocalAudioInputTransport.start() so a pipecat upgrade that changes the signature fails loudly.
"""

import config


# --- fakes mirroring pipecat LocalAudioInputTransport internals ------------------------------------

class _FakeStream:
    def __init__(self, *, started=False):
        self.started = started
        self.stopped = False
        self.closed = False

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.stopped = True

    def stop(self):  # unused alias
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

    def get_format_from_width(self, width):
        return f"fmt{width}"

    def open(self, **kwargs):
        self.open_calls.append(kwargs)
        if self.open_raises:
            raise OSError("[Errno -9997] Invalid sample rate")
        return _FakeStream(started=False)


class _FakeInput:
    def __init__(self, *, cur_idx, in_stream=None, py_audio=None):
        self._params = _FakeParams(input_device_index=cur_idx)
        self._py_audio = py_audio or _FakePyAudio()
        self._in_stream = in_stream if in_stream is not None else _FakeStream(started=True)
        self._sample_rate = 0

    def _audio_in_callback(self, *a):  # identity referenced by the reopen
        return None


def _base_env(monkeypatch, *, name="reSpeaker", index_pin=None, killswitch=None):
    monkeypatch.setenv("AUDIO_INPUT_DEVICE_NAME", name) if name is not None else \
        monkeypatch.delenv("AUDIO_INPUT_DEVICE_NAME", raising=False)
    monkeypatch.delenv("AUDIO_INPUT_DEVICE_INDEX", raising=False)
    if index_pin is not None:
        monkeypatch.setenv("AUDIO_INPUT_DEVICE_INDEX", str(index_pin))
    monkeypatch.delenv("INPUT_STALL_RERESOLVE", raising=False)
    if killswitch is not None:
        monkeypatch.setenv("INPUT_STALL_RERESOLVE", killswitch)


# --- happy path: index changed → reopen -----------------------------------------------------------

def test_reopen_on_index_change(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)   # new idx
    monkeypatch.setattr(config, "_supported_input_rate", lambda idx, **k: 48000)
    inp = _FakeInput(cur_idx=11)
    old = inp._in_stream

    outcome = config.maybe_reopen_input_device(inp)

    assert outcome == "reopened"
    assert old.stopped and old.closed                 # old stream torn down
    assert inp._in_stream is not old and inp._in_stream.started  # new stream started
    assert inp._params.input_device_index == 16       # pin advanced only on success
    assert inp._sample_rate == 48000                  # rate re-probed + retagged
    assert inp._params.audio_in_sample_rate == 48000


def test_reopen_open_kwargs_mirror_pipecat_start_args(monkeypatch):
    # Drift guard: pipecat LocalAudioInputTransport.start() opens with EXACTLY these kwargs. If a pipecat
    # upgrade changes the signature, this fails so we notice (the reopen hand-replicates that open()).
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)
    monkeypatch.setattr(config, "_supported_input_rate", lambda idx, **k: 16000)
    pa = _FakePyAudio()
    inp = _FakeInput(cur_idx=11, py_audio=pa)

    config.maybe_reopen_input_device(inp)

    assert len(pa.open_calls) == 1
    kw = pa.open_calls[0]
    assert set(kw) == {"format", "channels", "rate", "frames_per_buffer",
                       "stream_callback", "input", "input_device_index"}
    assert kw["format"] == "fmt2"                      # get_format_from_width(2)
    assert kw["channels"] == 1                         # from _params.audio_in_channels (NOT re-probed)
    assert kw["rate"] == 16000
    assert kw["frames_per_buffer"] == int(16000 / 100) * 2   # 20ms, mirrors start()
    assert kw["stream_callback"] == inp._audio_in_callback
    assert kw["input"] is True
    assert kw["input_device_index"] == 16


# --- no-reopen cases → None (caller does the in-place kick) ----------------------------------------

def test_same_index_returns_none(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 11)   # unchanged
    inp = _FakeInput(cur_idx=11)
    assert config.maybe_reopen_input_device(inp) is None
    assert inp._py_audio.open_calls == []             # never touched the stream


def test_no_name_match_returns_none(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: None)
    inp = _FakeInput(cur_idx=11)
    assert config.maybe_reopen_input_device(inp) is None


def test_name_unset_returns_none(monkeypatch):
    _base_env(monkeypatch, name=None)
    # should short-circuit before resolving
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)
    inp = _FakeInput(cur_idx=11)
    assert config.maybe_reopen_input_device(inp) is None


def test_index_pinned_returns_none(monkeypatch):
    _base_env(monkeypatch, index_pin=11)   # explicit index pin wins — never re-resolve by name
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)
    inp = _FakeInput(cur_idx=11)
    assert config.maybe_reopen_input_device(inp) is None


def test_kill_switch_off_returns_none(monkeypatch):
    _base_env(monkeypatch, killswitch="0")
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)
    inp = _FakeInput(cur_idx=11)
    assert config.maybe_reopen_input_device(inp) is None


# --- failure path + E1 retry ----------------------------------------------------------------------

def test_open_failure_returns_failed_and_nulls_stream(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)
    monkeypatch.setattr(config, "_supported_input_rate", lambda idx, **k: 48000)
    inp = _FakeInput(cur_idx=11, py_audio=_FakePyAudio(open_raises=True))

    outcome = config.maybe_reopen_input_device(inp)

    assert outcome == "failed"                         # caller returns False → escalate ladder
    assert inp._in_stream is None                      # left closed, never a deaf latch w/o escalation
    assert inp._params.input_device_index == 11        # pin NOT advanced on failure


def test_e1_retry_after_nulled_stream_reattempts_open(monkeypatch):
    # E1 (GA): guardrail #2 nulls _in_stream on failure. A second stall attempt must genuinely RE-ATTEMPT
    # open() (a transient may have cleared), not crash on old.stop_stream() with old=None.
    _base_env(monkeypatch)
    monkeypatch.setattr(config, "_resolve_device_index", lambda *a, **k: 16)
    monkeypatch.setattr(config, "_supported_input_rate", lambda idx, **k: 48000)

    pa = _FakePyAudio(open_raises=True)
    inp = _FakeInput(cur_idx=11, py_audio=pa)
    assert config.maybe_reopen_input_device(inp) == "failed"
    assert inp._in_stream is None

    pa.open_raises = False                             # the transient clears
    outcome = config.maybe_reopen_input_device(inp)    # must NOT raise on old=None; must re-attempt
    assert outcome == "reopened"
    assert len(pa.open_calls) == 2                     # both calls genuinely reached open()
    assert inp._in_stream is not None and inp._in_stream.started
