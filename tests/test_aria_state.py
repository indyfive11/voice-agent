"""Tests for aria_state — the eye-indicator state-file writer (the seam to the Conky panel)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import aria_state


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """Point the writer at a temp file and reset its module-level dedup/disable state per test."""
    p = tmp_path / "aria" / "state"
    monkeypatch.setattr(aria_state, "_path", p)
    monkeypatch.setattr(aria_state, "_disabled", False)
    monkeypatch.setattr(aria_state, "_last_state", None)
    monkeypatch.setattr(aria_state, "_last_level", -1.0)
    monkeypatch.setattr(aria_state, "_level_gain", 4.0)
    return p


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def test_writes_valid_json_with_required_fields(state_file):
    aria_state.set_state("thinking")
    obj = _read(state_file)
    assert obj["state"] == "thinking"
    assert obj["level"] == 0.0
    assert isinstance(obj["ts"], (int, float)) and obj["ts"] > 0


def test_creates_parent_dir(state_file):
    assert not state_file.parent.exists()
    aria_state.set_state("idle")
    assert state_file.exists()


def test_level_is_clamped(state_file):
    aria_state.set_state("speaking", 5.0)
    assert _read(state_file)["level"] == 1.0
    aria_state._last_state = None  # force a fresh write past the dedup guard
    aria_state._last_level = -1.0
    aria_state.set_state("speaking", -3.0)
    assert _read(state_file)["level"] == 0.0


def test_dedup_skips_identical_write(state_file):
    aria_state.set_state("idle")
    first_ts = _read(state_file)["ts"]
    aria_state.set_state("idle")  # identical → must NOT rewrite
    assert _read(state_file)["ts"] == first_ts


def test_dedup_allows_meaningful_level_change_while_speaking(state_file):
    aria_state.set_state("speaking", 0.10)
    aria_state.set_state("speaking", 0.80)  # >0.02 delta → rewrites
    assert _read(state_file)["level"] == 0.8


def test_state_change_always_writes(state_file):
    aria_state.set_state("idle")
    aria_state.set_state("listening")
    assert _read(state_file)["state"] == "listening"


def test_current_tracks_last_written(state_file):
    assert aria_state.current() is None
    aria_state.set_state("listening")
    assert aria_state.current() == "listening"


def test_disabled_writes_nothing(state_file, monkeypatch):
    monkeypatch.setattr(aria_state, "_disabled", True)
    aria_state.set_state("speaking", 0.5)
    assert not state_file.exists()


def test_errors_are_swallowed(state_file, monkeypatch):
    # An unwritable target must not raise into the agent.
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(aria_state.os, "replace", boom)
    aria_state.set_state("thinking")  # must not raise


def test_atomic_no_tmp_left_behind(state_file):
    aria_state.set_state("speaking", 0.3)
    leftovers = list(state_file.parent.glob(".state.*"))
    assert leftovers == []


def test_speaking_level_empty_is_zero():
    assert aria_state.speaking_level_from_pcm(b"") == 0.0


def test_speaking_level_silence_is_zero(state_file):
    silence = (b"\x00\x00") * 1600  # int16 zeros
    assert aria_state.speaking_level_from_pcm(silence) == 0.0


def test_speaking_level_loud_pcm_is_positive_and_clamped(state_file, monkeypatch):
    import numpy as np

    monkeypatch.setattr(aria_state, "_level_gain", 4.0)
    full = (np.ones(1600, dtype=np.int16) * 32767).tobytes()  # ~full-scale → RMS ~1.0 * gain → clamp
    lvl = aria_state.speaking_level_from_pcm(full)
    assert lvl == 1.0


def test_speaking_level_midlevel_scales_with_gain(monkeypatch):
    import numpy as np

    monkeypatch.setattr(aria_state, "_level_gain", 1.0)  # no boost → raw normalized RMS
    half = (np.ones(1600, dtype=np.int16) * 16384).tobytes()  # ~0.5 of full-scale
    lvl = aria_state.speaking_level_from_pcm(half)
    assert 0.45 < lvl < 0.55
