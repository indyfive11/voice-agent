"""Offline tests for usb_reset.locate — parse uhubctl's listing to find a device's hub+port by VID:PID.

No hardware: the real `uhubctl` stdout (captured from the EM MEGA4 setup) is fed in via a fake
subprocess.run, so the parser is verified against the exact format it must handle in the field.
"""

import subprocess
import types

import usb_reset

# Real `uhubctl` output from EM (2026-07-07): reSpeaker (2886:001a) on the MEGA4 (2109:2817) port 2.
UHUBCTL_OUT = """Current status for hub 11-1.4 [0bda:5411 Generic 4-Port USB 2.0 Hub, USB 2.10, 4 ports, ppps]
  Port 1: 0100 power
  Port 2: 0103 power enable connect [046d:c52b Logitech USB Receiver]
  Port 3: 0103 power enable connect [062a:4102 MOSART Semi. 2.4G Wireless Mouse]
  Port 4: 0103 power enable connect [1ea7:0169 2.4G Mouse]
Current status for hub 11-1.2 [2109:2817 VIA Labs, Inc. USB2.0 Hub, USB 2.10, 4 ports, ppps]
  Port 1: 0100 power
  Port 2: 0503 power highspeed enable connect [2886:001a Seeed Studio reSpeaker XVF3800 4-Mic Array 101991441261600230]
  Port 3: 0100 power
  Port 4: 0100 power
Current status for hub 9-2 [0bda:5411 Generic USB2.1 Hub, USB 2.10, 4 ports, ppps]
  Port 4: 0507 power highspeed suspend enable connect [046d:0829 Webcam C110]
"""


def _fake_run(out="", rc=0):
    def run(cmd, capture_output=True, text=True, timeout=None):
        return types.SimpleNamespace(stdout=out, stderr="", returncode=rc)
    return run


def test_locate_finds_hub_and_port(monkeypatch):
    monkeypatch.setattr(usb_reset.subprocess, "run", _fake_run(UHUBCTL_OUT))
    assert usb_reset.locate("2886:001a") == ("11-1.2", 2)


def test_locate_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(usb_reset.subprocess, "run", _fake_run(UHUBCTL_OUT))
    assert usb_reset.locate("2886:001A") == ("11-1.2", 2)


def test_locate_matches_the_right_device_among_many(monkeypatch):
    # The Logitech receiver on a DIFFERENT hub must resolve to its own hub/port, not bleed across hubs.
    monkeypatch.setattr(usb_reset.subprocess, "run", _fake_run(UHUBCTL_OUT))
    assert usb_reset.locate("046d:c52b") == ("11-1.4", 2)


def test_locate_returns_none_when_absent(monkeypatch):
    monkeypatch.setattr(usb_reset.subprocess, "run", _fake_run(UHUBCTL_OUT))
    assert usb_reset.locate("dead:beef") is None


def test_locate_returns_none_when_uhubctl_raises(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("uhubctl not installed")
    monkeypatch.setattr(usb_reset.subprocess, "run", boom)
    assert usb_reset.locate("2886:001a") is None


def test_cycle_disabled_on_empty_vidpid(monkeypatch):
    # Safe universal default: empty VID:PID = feature off, never shells out.
    called = []
    monkeypatch.setattr(usb_reset.subprocess, "run", lambda *a, **k: called.append(1))
    assert usb_reset.cycle("") is False
    assert not called


def test_cycle_returns_false_when_device_not_found(monkeypatch):
    monkeypatch.setattr(usb_reset.subprocess, "run", _fake_run(UHUBCTL_OUT))
    assert usb_reset.cycle("dead:beef") is False
