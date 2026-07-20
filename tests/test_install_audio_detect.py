"""Tests for voice_agent_install.audio_detect — the chipmunk-guard's install-time detect step.

Fixtures are the REAL /proc/asound layout captured from the Pi (EMEET) + EM 2026-07-20, written into a
fake proc tree so the parser is validated against actual hardware output, not a guess."""
from __future__ import annotations

import pytest

from voice_agent_install import audio_detect as a


# Real Pi capture: `cat /proc/asound/cards`
PI_CARDS = """\
 0 [vc4hdmi0       ]: vc4-hdmi - vc4-hdmi-0
                      vc4-hdmi-0
 1 [vc4hdmi1       ]: vc4-hdmi - vc4-hdmi-1
                      vc4-hdmi-1
 2 [Headphones     ]: bcm2835_headpho - bcm2835 Headphones
                      bcm2835 Headphones
 3 [Plus           ]: USB-Audio - EMEET OfficeCore M0 Plus
                      EMEET OfficeCore M0 Plus EMEET OfficeCore M0 Plus at usb-0000:01:00.0-1.2, full
"""

# Real Pi capture: `cat /proc/asound/card3/stream0` — Playback 48000 fixed, Capture 16000 (must NOT leak)
EMEET_STREAM0 = """\
EMEET OfficeCore M0 Plus EMEET OfficeCore M0 Plus at usb-0000:01:00.0-1.2, full : USB Audio

Playback:
  Status: Stop
  Interface 2
    Altset 1
    Format: S16_LE
    Channels: 2
    Endpoint: 0x01 (1 OUT) (ASYNC)
    Rates: 48000
    Bits: 16
    Channel map: FL FR

Capture:
  Status: Running
  Interface 1
    Altset 1
    Format: S16_LE
    Channels: 1
    Endpoint: 0x81 (1 IN) (ASYNC)
    Rates: 16000
    Bits: 16
    Channel map: MONO
"""


def _write_proc(tmp_path, cards=PI_CARDS, stream0_by_card=None):
    """Build a fake /proc/asound tree; returns its path string."""
    root = tmp_path / "asound"
    root.mkdir()
    (root / "cards").write_text(cards)
    for card_idx, stream0 in (stream0_by_card or {}).items():
        d = root / f"card{card_idx}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stream0").write_text(stream0)
    return str(root)


# --------------------------------------------------------------- classify_output_device
@pytest.mark.parametrize(
    "name,kind",
    [
        ("pulse", "server"), ("pipewire", "server"), ("default", "server"),
        ("sysdefault", "server"), ("jack", "server"), ("PULSE", "server"),
        ("", "server"), (None, "server"), ("   ", "server"),
        ("EMEET", "hardware"), ("hw:3,0", "hardware"), ("reSpeaker", "hardware"),
    ],
)
def test_classify_output_device(name, kind):
    assert a.classify_output_device(name) == kind


# --------------------------------------------------------------- list_alsa_cards
def test_list_alsa_cards_parses_index_id_desc(tmp_path):
    proc = _write_proc(tmp_path)
    cards = a.list_alsa_cards(proc)
    assert (3, "Plus", "USB-Audio - EMEET OfficeCore M0 Plus EMEET OfficeCore M0 Plus "
            "EMEET OfficeCore M0 Plus at usb-0000:01:00.0-1.2, full") == cards[3]
    assert [c[0] for c in cards] == [0, 1, 2, 3]


def test_list_alsa_cards_missing_file_is_empty(tmp_path):
    assert a.list_alsa_cards(str(tmp_path / "nope")) == []


# --------------------------------------------------------------- alsa_card_index_for
def test_card_index_matches_emeet_by_substring(tmp_path):
    proc = _write_proc(tmp_path)
    assert a.alsa_card_index_for("EMEET", proc_asound=proc) == 3
    assert a.alsa_card_index_for("emeet", proc_asound=proc) == 3  # case-insensitive
    assert a.alsa_card_index_for("Plus", proc_asound=proc) == 3   # matches the id too


def test_card_index_none_when_no_match(tmp_path):
    proc = _write_proc(tmp_path)
    assert a.alsa_card_index_for("NoSuchDevice", proc_asound=proc) is None


def test_card_index_none_when_ambiguous(tmp_path):
    # "vc4-hdmi" substring matches BOTH card 0 and card 1 → ambiguous → None (never pin the wrong card)
    proc = _write_proc(tmp_path)
    assert a.alsa_card_index_for("vc4-hdmi", proc_asound=proc) is None


# --------------------------------------------------------------- alsa_playback_rates
def test_playback_rates_reads_fixed_48000_not_capture_16000(tmp_path):
    proc = _write_proc(tmp_path, stream0_by_card={3: EMEET_STREAM0})
    # the crux: Playback is 48000, Capture is 16000 — only the Playback rate may come back
    assert a.alsa_playback_rates(3, proc_asound=proc) == [48000]


def test_playback_rates_multiple_values(tmp_path):
    s0 = "Playback:\n    Rates: 44100, 48000, 96000\nCapture:\n    Rates: 16000\n"
    proc = _write_proc(tmp_path, stream0_by_card={5: s0})
    assert a.alsa_playback_rates(5, proc_asound=proc) == [44100, 48000, 96000]


def test_playback_rates_continuous_range_is_none(tmp_path):
    s0 = "Playback:\n    Rates: 8000 - 48000\n"
    proc = _write_proc(tmp_path, stream0_by_card={6: s0})
    assert a.alsa_playback_rates(6, proc_asound=proc) is None


def test_playback_rates_missing_stream0_is_none(tmp_path):
    proc = _write_proc(tmp_path)  # no stream0 written for card 3
    assert a.alsa_playback_rates(3, proc_asound=proc) is None


# --------------------------------------------------------------- detect_output_sample_rate (top-level)
def test_detect_server_endpoint_stays_unset(tmp_path):
    # EM: AUDIO_OUTPUT_DEVICE_NAME=pulse → UNSET
    proc = _write_proc(tmp_path, stream0_by_card={3: EMEET_STREAM0})
    d = a.detect_output_sample_rate("pulse", proc_asound=proc)
    assert d.sample_rate is None and d.device_kind == "server"


def test_detect_unset_name_stays_unset(tmp_path):
    d = a.detect_output_sample_rate(None, proc_asound=_write_proc(tmp_path))
    assert d.sample_rate is None and d.device_kind == "server"


def test_detect_fixed_hardware_pins_native_rate(tmp_path):
    # Pi: AUDIO_OUTPUT_DEVICE_NAME=EMEET → PIN 48000
    proc = _write_proc(tmp_path, stream0_by_card={3: EMEET_STREAM0})
    d = a.detect_output_sample_rate("EMEET", proc_asound=proc)
    assert d.sample_rate == 48000 and d.device_kind == "hardware"


def test_detect_hardware_no_match_stays_unset(tmp_path):
    proc = _write_proc(tmp_path, stream0_by_card={3: EMEET_STREAM0})
    d = a.detect_output_sample_rate("Bogus", proc_asound=proc)
    assert d.sample_rate is None and d.device_kind == "unknown"


def test_detect_hardware_multirate_stays_unset(tmp_path):
    s0 = "Playback:\n    Rates: 44100, 48000\n"
    proc = _write_proc(tmp_path, cards=" 5 [Multi          ]: USB-Audio - Multi Device\n",
                       stream0_by_card={5: s0})
    d = a.detect_output_sample_rate("Multi", proc_asound=proc)
    assert d.sample_rate is None and d.device_kind == "hardware"
