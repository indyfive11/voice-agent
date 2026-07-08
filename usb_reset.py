"""USB port power-cycle — the hardware rung of the input-stall recovery ladder.

Why this exists: the reSpeaker XVF3800 has a HARD wedge class (task #79/#83) where the device stays
enumerated on the bus but its capture stream's C-layer `open()` blocks forever. No in-process reopen —
light OR heavy — clears it; only removing VBUS power to the port forces the device firmware to cold
re-enumerate. With the reSpeaker on a PPPS-capable hub (the UUGear MEGA4, VIA Labs 2109:2817), `uhubctl`
can cut that one port's power. This module locates the device's hub+port by USB VID:PID — so it survives
bus renumbering / a moved port with no hardcoded topology path — cycles it, and waits for the device to
re-appear before returning, so the caller's in-process reopen re-resolves the fresh enumeration by name.

Deliberately OUTSIDE `config._input_recovery_lock` (GA flag 2026-07-07): a hung in-process proactive
recycle holding that lock must not be able to starve the one recovery step that actually works. `uhubctl`
is an external subprocess touching OS-level port power, independent of the in-process stream mutations the
lock serializes.

Disabled unless `INPUT_USB_RESET_VIDPID` is set (default '' = historical no-op), per the hardware-
portability SOP: an unconfigured install behaves exactly as before; the env names the device, the hub is
detected. Runtime needs non-root access to the hub (a udev rule granting MODE on the MEGA4's node) or the
cycle simply fails-soft to False and the ladder falls through to the process-exit rung as before.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time

from loguru import logger


def _tlog(msg: str) -> None:
    logger.bind(transcript=True).info(msg)


_STATUS_HUB = re.compile(r"status for hub (\S+)")
# a connected-port line, e.g.:
#   "  Port 2: 0503 power highspeed enable connect [2886:001a Seeed Studio reSpeaker ...]"
# (unconnected ports like "Port 1: 0100 power" have no [vid:pid] and are skipped)
_PORT_LINE = re.compile(r"Port (\d+):\s+[0-9a-fA-F]{4}\b.*\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})")


def _run(bin_: str, *args: str, timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run([bin_, *args], capture_output=True, text=True, timeout=timeout)


def locate(vidpid: str, *, uhubctl_bin: str = "uhubctl"):
    """Return (hub_location, port) for the PPPS hub port whose connected device matches `vidpid`, else None.

    Parses `uhubctl`'s listing (the authoritative OS view of switchable hubs). `vidpid` like '2886:001a'
    (case-insensitive)."""
    vp = vidpid.lower()
    try:
        r = _run(uhubctl_bin)
    except Exception as e:  # noqa: BLE001 - probing must never raise into the caller
        logger.debug(f"usb_reset.locate: uhubctl failed: {type(e).__name__}: {e}")
        return None
    hub = None
    for line in r.stdout.splitlines():
        m = _STATUS_HUB.search(line)
        if m:
            hub = m.group(1)
            continue
        p = _PORT_LINE.search(line)
        if p and hub and p.group(2).lower() == vp:
            return hub, int(p.group(1))
    return None


def cycle(vidpid: str, *, uhubctl_bin: str = "uhubctl", off_delay: float = 2.0,
          settle_timeout: float = 12.0) -> bool:
    """Power-cycle the hub port hosting `vidpid`, then wait until the device re-appears. Returns success.

    Bounded and best-effort: every failure path returns False so the caller's escalation ladder continues
    to its next rung. Does NOT touch `config._input_recovery_lock` (see module docstring)."""
    if not vidpid:
        return False
    bin_ = shutil.which(uhubctl_bin) or uhubctl_bin
    loc = locate(vidpid, uhubctl_bin=bin_)
    if loc is None:
        _tlog(f"INPUT STALL | usb-reset: device {vidpid} not found on any PPPS hub — cannot power-cycle")
        return False
    hub, port = loc
    _tlog(f"INPUT STALL | usb-reset: power-cycling {vidpid} at hub {hub} port {port} (off {off_delay:.0f}s)")
    try:
        r = _run(bin_, "-l", hub, "-p", str(port), "-a", "cycle", "-d", str(off_delay),
                 timeout=off_delay + 15.0)
    except Exception as e:  # noqa: BLE001 - recovery must never raise into the watch loop
        _tlog(f"INPUT STALL | usb-reset: uhubctl cycle errored: {type(e).__name__}: {e}")
        return False
    if r.returncode != 0:
        _tlog(f"INPUT STALL | usb-reset: uhubctl exit {r.returncode}: {(r.stderr or '').strip()[:200]}")
        return False
    # Wait for re-enumeration before returning, so the caller's in-process reopen re-resolves the fresh
    # device by name instead of racing an empty port mid-cold-boot.
    deadline = time.monotonic() + settle_timeout
    while time.monotonic() < deadline:
        if locate(vidpid, uhubctl_bin=bin_) is not None:
            _tlog(f"INPUT STALL | usb-reset: {vidpid} re-enumerated — handing back to in-process recovery")
            return True
        time.sleep(0.5)
    _tlog(f"INPUT STALL | usb-reset: {vidpid} did NOT re-appear within {settle_timeout:.0f}s")
    return False
