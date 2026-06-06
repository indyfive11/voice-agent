"""Tests for the boot-time AEC-mic readiness wait (config.wait_for_input_device / wait_for_aec_mic).

The wait polls PyAudio with a FRESH instance each iteration (PortAudio snapshots the device list at
init), so we fake `pyaudio` via sys.modules and let the fake's device list change across PyAudio() inits
to simulate the AEC node appearing late.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

import config


class _FakePyAudioModule:
    """Minimal pyaudio stand-in. `devices` is read live on each PyAudio() init, so a test can mutate it
    (or pass a callable) to model a device that appears after some polls."""

    def __init__(self, devices):
        self._devices = devices  # list[dict] or zero-arg callable returning list[dict]

    def PyAudio(self):  # noqa: N802 - mirror pyaudio's API
        devs = self._devices() if callable(self._devices) else self._devices
        return _FakePyAudio(list(devs))


class _FakePyAudio:
    def __init__(self, devices):
        self._devices = devices

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, i):
        return self._devices[i]

    def terminate(self):
        pass


def _dev(name, in_ch=1):
    return {"name": name, "maxInputChannels": in_ch, "maxOutputChannels": 0}


@pytest.fixture
def fake_pyaudio(monkeypatch):
    def _install(devices):
        monkeypatch.setitem(sys.modules, "pyaudio", _FakePyAudioModule(devices))
    return _install


def test_unset_name_returns_immediately():
    # No name to wait for → no-op True even with no pyaudio installed.
    assert asyncio.run(config.wait_for_input_device(None, timeout=5)) is True


def test_zero_timeout_disables_wait(fake_pyaudio):
    fake_pyaudio([])  # device absent, but timeout<=0 means "disabled"
    assert asyncio.run(config.wait_for_input_device("AEC Mic", timeout=0)) is True


def test_present_immediately(fake_pyaudio):
    fake_pyaudio([_dev("Built-in Mic"), _dev("AEC Mic")])
    assert asyncio.run(config.wait_for_input_device("aec mic", timeout=5, poll=0.01)) is True


def test_appears_after_a_few_polls(fake_pyaudio):
    calls = {"n": 0}

    def devices():
        calls["n"] += 1
        # Absent for the first two enumerations, then the AEC node shows up.
        return [_dev("Built-in Mic")] if calls["n"] < 3 else [_dev("Built-in Mic"), _dev("AEC Mic")]

    fake_pyaudio(devices)
    assert asyncio.run(config.wait_for_input_device("AEC Mic", timeout=5, poll=0.01)) is True
    assert calls["n"] >= 3


def test_never_appears_times_out(fake_pyaudio):
    fake_pyaudio([_dev("Built-in Mic")])  # AEC node never shows up
    assert asyncio.run(config.wait_for_input_device("AEC Mic", timeout=0.1, poll=0.02)) is False


def test_output_only_device_does_not_match(fake_pyaudio):
    # A sink named "AEC ..." with zero input channels must not satisfy an input wait.
    fake_pyaudio([{"name": "AEC Mic Monitor", "maxInputChannels": 0, "maxOutputChannels": 2}])
    assert asyncio.run(config.wait_for_input_device("AEC Mic", timeout=0.1, poll=0.02)) is False


def test_pyaudio_unavailable_does_not_block(monkeypatch):
    # Importing pyaudio fails → don't block startup; return True.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyaudio":
            raise ImportError("no pyaudio")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert asyncio.run(config.wait_for_input_device("AEC Mic", timeout=5)) is True


def test_aec_mic_entrypoint_skips_when_pinned_by_index(monkeypatch, fake_pyaudio):
    # AUDIO_INPUT_DEVICE_INDEX set → no name wait, returns True without touching pyaudio.
    monkeypatch.setenv("AUDIO_INPUT_DEVICE_INDEX", "3")
    fake_pyaudio([])  # would time out if it polled
    assert asyncio.run(config.wait_for_aec_mic()) is True


def test_aec_mic_entrypoint_uses_env_name_and_timeout(monkeypatch, fake_pyaudio):
    monkeypatch.delenv("AUDIO_INPUT_DEVICE_INDEX", raising=False)
    monkeypatch.setenv("AUDIO_INPUT_DEVICE_NAME", "AEC Mic")
    monkeypatch.setenv("AUDIO_INPUT_WAIT_SECS", "0.1")
    fake_pyaudio([_dev("Built-in Mic")])  # AEC absent → should time out (False), not hang
    assert asyncio.run(config.wait_for_aec_mic()) is False
