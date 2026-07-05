"""Tests for name→device resolution ranking (config._rank_name_matches / _collect_name_matches /
_resolve_device_index).

Regression cover for EM 2026-07-05: a bare `AUDIO_INPUT_DEVICE_NAME=reSpeaker` substring-matched BOTH
the PipeWire/pulse node ("… Analog Stereo", in×4) AND the raw ALSA alias ("… : USB Audio (hw:4,0)",
in×2) of the same card. The old first-match resolver grabbed whichever enumerated first; on a replug it
landed the raw `hw:` node, which stalls (PipeWire owns the card → direct grab starves) → watchdog thrash.
The ranked matcher must deterministically prefer the routable pulse node while never filtering the set to
empty (a bare-ALSA box with only a `hw:` match must still get that node, not fall through to OS default).
"""

import sys
import types

import config


# --- _rank_name_matches (pure ranking) ------------------------------------------------------------

def test_prefers_pulse_node_over_raw_hw_alias_even_when_hw_enumerates_first():
    # The reSpeaker case: raw hw: (in×2) at index 0, pulse node (in×4) at index 1. hw: enumerates FIRST
    # (that was the bug) but must NOT win.
    matches = [
        (0, "reSpeaker XVF3800 4-Mic Array: USB Audio (hw:4,0)", 2),
        (1, "reSpeaker XVF3800 4-Mic Array Analog Stereo", 4),
    ]
    ranked = config._rank_name_matches(matches)
    assert ranked[0][0] == 1  # the pulse node wins despite the raw alias sorting first in enumeration


def test_plughw_alias_also_deprioritized():
    matches = [
        (0, "reSpeaker Analog Stereo", 2),           # routable pulse node, fewer channels
        (1, "reSpeaker plughw:4,0", 8),              # raw ALSA plugin, MORE channels — must still lose
    ]
    ranked = config._rank_name_matches(matches)
    assert ranked[0][0] == 0  # routable beats raw even with fewer channels (card-ownership dominates)


def test_most_channels_breaks_tie_among_routable_nodes():
    matches = [
        (0, "reSpeaker Analog Mono", 1),
        (1, "reSpeaker Analog Stereo", 4),
    ]
    ranked = config._rank_name_matches(matches)
    assert ranked[0][0] == 1  # both routable → more channels wins


def test_lowest_index_breaks_final_tie_deterministically():
    matches = [
        (5, "reSpeaker Analog Stereo", 4),
        (2, "reSpeaker Analog Stereo", 4),
    ]
    ranked = config._rank_name_matches(matches)
    assert ranked[0][0] == 2  # identical routable+channels → lower index (stable, deterministic)


def test_sole_raw_hw_match_is_never_dropped():
    # A bare-ALSA box: the ONLY match is a raw hw: node. Ranking must still return it (best-available),
    # NOT empty the set — else the caller would fall through to the OS default and surprise the user.
    matches = [(3, "reSpeaker XVF3800: USB Audio (hw:0,0)", 2)]
    ranked = config._rank_name_matches(matches)
    assert ranked and ranked[0][0] == 3


def test_empty_stays_empty():
    assert config._rank_name_matches([]) == []


# --- _collect_name_matches (substring + channel-direction filter) ----------------------------------

class _FakePyAudio:
    def __init__(self, devices):
        self._devices = devices

    def get_device_count(self):
        return len(self._devices)

    def get_device_info_by_index(self, i):
        return self._devices[i]

    def terminate(self):
        pass


_RESPEAKER_DEVICES = [
    {"name": "reSpeaker XVF3800 4-Mic Array: USB Audio (hw:4,0)", "maxInputChannels": 2, "maxOutputChannels": 2},
    {"name": "reSpeaker XVF3800 4-Mic Array Analog Stereo", "maxInputChannels": 4, "maxOutputChannels": 2},
    {"name": "pulse", "maxInputChannels": 32, "maxOutputChannels": 32},
    {"name": "Webcam C110 Mono", "maxInputChannels": 1, "maxOutputChannels": 0},
]


def test_collect_matches_filters_by_substring_and_input_direction():
    pa = _FakePyAudio(_RESPEAKER_DEVICES)
    matches = config._collect_name_matches(pa, "reSpeaker", want_output=False)
    idxs = {m[0] for m in matches}
    assert idxs == {0, 1}  # both reSpeaker interfaces; "pulse"/"Webcam" don't match the name


def test_collect_matches_skips_zero_channel_devices_for_direction():
    # A capture-only device (out=0) must not match an OUTPUT query.
    pa = _FakePyAudio([{"name": "reSpeaker capture-only", "maxInputChannels": 2, "maxOutputChannels": 0}])
    assert config._collect_name_matches(pa, "reSpeaker", want_output=True) == []
    assert len(config._collect_name_matches(pa, "reSpeaker", want_output=False)) == 1


# --- _resolve_device_index (end-to-end with a fake pyaudio module) --------------------------------

def _install_fake_pyaudio(monkeypatch, devices):
    fake = types.ModuleType("pyaudio")
    fake.PyAudio = lambda: _FakePyAudio(devices)
    monkeypatch.setitem(sys.modules, "pyaudio", fake)


def test_resolve_picks_ranked_best_not_first_match(monkeypatch):
    _install_fake_pyaudio(monkeypatch, _RESPEAKER_DEVICES)
    idx = config._resolve_device_index("reSpeaker", want_output=False)
    assert idx == 1  # the pulse node, not the raw hw: alias at index 0


def test_resolve_returns_none_when_no_match(monkeypatch):
    _install_fake_pyaudio(monkeypatch, _RESPEAKER_DEVICES)
    assert config._resolve_device_index("nonexistent-mic", want_output=False) is None


def test_resolve_none_name_is_noop(monkeypatch):
    _install_fake_pyaudio(monkeypatch, _RESPEAKER_DEVICES)
    assert config._resolve_device_index(None, want_output=False) is None
