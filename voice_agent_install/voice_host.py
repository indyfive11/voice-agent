"""The voice-host role provisioner — the caller that turns the brain's mDNS advertiser ON.

INSTALL-TIME ONLY. Same rule as ``satellite.py``: nothing in the running application may import this
package (the satellite sync manifest is ``git ls-files '*.py'`` + ``run.sh``, so a runtime importer
drags every module here onto that manifest and any untracked module becomes a guaranteed satellite
crash). Run as ``python -m voice_agent_install.voice_host`` — what ``bootstrap.sh --role voice-host``
execs after it has provisioned ``uv`` and the venv.

WHY THIS ROLE EXISTS. The satellite role's discovery/§10d remedy text points the operator at "enable
``voice_advertise`` on the voice-host role" — but until this module existed there was no such role, so
that instruction named a caller that did not exist (``profile.py`` still raises for every role but
``satellite``). This is that caller. It runs ON THE BRAIN BOX and its one job is to make the brain
broadcast ``_voice-brain._tcp`` so a satellite can discover it without a hand-typed IP.

WHY IT SHELLS OUT INSTEAD OF WRITING THE FLAG. ``voice_advertise`` is a gabagent ``settings.json``
field, and voice-agent cannot ``import gabagent`` (separate venv). So Layer B DETECTS the box's LAN IP
and invokes the Layer-C seam cross-process:

    gabagent-install --enable-voice-host --host <LAN_IP> [--room-id <room>]

THE GRAFT (maintainer-ratified 2026-07-27, a §10d/§6 amendment). ``voice_advertise:true`` is INERT on a
loopback bind — the runtime advertiser skips loopback and ``voice_host`` defaults to ``127.0.0.1`` — so
the flag alone broadcasts nothing AND makes the satellite-side remedy text lie. The division of labor:
**Layer B detects the LAN IP (primary-route interface) and passes it via ``--host``; Layer C validates
it is non-loopback and writes ``voice_host`` + ``voice_advertise`` together, else REFUSES (exit 2).**
We therefore never enable a silent no-op.

DISCOVERY IS AN ENHANCEMENT, NOT A HARD DEPENDENCY (CR-4). Every failure below degrades to the honest
floor: a satellite can still be pointed at the brain by typing its IP. So the summary always says so,
and the exit is non-zero only because this role's *own* job (turning the advertiser on) was not done —
which is what keeps ``$?`` trustworthy (an installer whose only signal is ``$?`` cannot report a no-op
as success), never because the overall system is broken.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Callable, Optional, Sequence

from voice_agent_install.satellite import Console, StepResult

__all__ = [
    "detect_lan_ip", "resolve_install_bin", "enable_advertise", "provision", "main",
    "EXIT_OK", "EXIT_USAGE", "EXIT_NO_SEAM", "EXIT_ENABLE_FAILED",
]

# Exit contract (mirrors satellite.py: every non-success is non-zero and the summary names the step).
EXIT_OK = 0            # advertiser enabled (or already enabled)
EXIT_USAGE = 2         # no non-loopback LAN IP to advertise on, and none supplied via --host
EXIT_NO_SEAM = 3       # the gabagent-install console script could not be found or run
EXIT_ENABLE_FAILED = 4 # the seam ran but did not enable (loopback refuse, or any other non-zero)

# The default-route destination used purely to make the kernel SELECT a source address. No packet is
# ever sent: `ip route get` is a routing-table lookup, and the UDP fallback only sets a connect()
# destination. 1.1.1.1 is a stable public address; the box's answer is its own LAN source IP, not 1.1.1.1.
_ROUTE_PROBE_TARGET = "1.1.1.1"


def _is_loopbackish(ip: str) -> bool:
    ip = (ip or "").strip()
    return not ip or ip == "0.0.0.0" or ip.startswith("127.")


def _lan_ip_via_iproute(target: str, run: Callable[..., subprocess.CompletedProcess]) -> Optional[str]:
    """Parse ``ip -4 route get <target>`` for the ``src`` address — the authoritative source the kernel
    would use toward the default route. Read-only; sends nothing."""
    try:
        cp = run(["ip", "-4", "route", "get", target], capture_output=True, text=True)
    except OSError:
        return None
    if cp.returncode != 0 or not cp.stdout:
        return None
    tokens = cp.stdout.split()
    if "src" in tokens:
        i = tokens.index("src")
        if i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def _lan_ip_via_socket(target: str) -> Optional[str]:
    """Fallback: a connected UDP socket's local address is the kernel's chosen source toward `target`.
    connect() on a datagram socket sets the destination only — no packet leaves the box."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target, 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def detect_lan_ip(*, target: str = _ROUTE_PROBE_TARGET,
                  route_run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Optional[str]:
    """The box's LAN IPv4 as the kernel would source it toward the default route — the authoritative
    primary-route address per the no-hardcodes portability SOP. Returns None if only loopback resolves;
    the caller then refuses or asks for ``--host`` rather than guessing a value it cannot justify."""
    ip = _lan_ip_via_iproute(target, route_run) or _lan_ip_via_socket(target)
    return ip if ip and not _is_loopbackish(ip) else None


def resolve_install_bin(*, which: Callable[[str], Optional[str]] = shutil.which,
                        env: "os._Environ[str] | dict[str, str]" = os.environ) -> Optional[str]:
    """Locate the ``gabagent-install`` console script WITHOUT importing gabagent (cross-venv → the same
    ModuleNotFoundError that made us vendor installkit): explicit override → PATH. A real brain box has
    it on PATH (pip / pipx `~/.local/bin` / system `/usr/bin`), and ``GABAGENT_INSTALL_BIN`` names it for
    anything off PATH — including an editable checkout. No install-specific path is hardcoded here (the
    portability SOP: an installation-specific value must never be a bare constant in code)."""
    override = env.get("GABAGENT_INSTALL_BIN")
    if override:
        return override
    return which("gabagent-install")


def enable_advertise(install_bin: str, lan_ip: str, room_id: Optional[str] = None, *,
                     run: Callable[..., subprocess.CompletedProcess] = subprocess.run
                     ) -> subprocess.CompletedProcess:
    """Invoke the Layer-C seam. Argv exactly as the contract locked with GA:
    ``gabagent-install --enable-voice-host --host <ip> [--room-id <room>]``."""
    argv = [install_bin, "--enable-voice-host", "--host", lan_ip]
    if room_id:
        argv += ["--room-id", room_id]
    return run(argv, capture_output=True, text=True)


def provision(*, host: Optional[str] = None, room_id: Optional[str] = None,
              detect: Optional[Callable[..., Optional[str]]] = None,
              resolve: Optional[Callable[..., Optional[str]]] = None,
              invoke: Optional[Callable[..., subprocess.CompletedProcess]] = None
              ) -> "tuple[int, list[StepResult]]":
    """Detect the LAN IP → resolve the seam → invoke it → interpret the exit. Pure of side effects
    apart from the injected subprocess calls; the seams are injectable so tests never touch a socket or
    a subprocess. Returns ``(exit_code, steps)`` in satellite.py's shape.

    The seams default to the module-level functions but are resolved by name at call time, so ``main``
    stays hermetically testable (monkeypatch the module attribute) without threading injection through."""
    detect = detect or detect_lan_ip
    resolve = resolve or resolve_install_bin
    invoke = invoke or enable_advertise
    steps: list[StepResult] = []

    # 1. The host to advertise on: an explicit --host wins, else detect the primary-route address.
    supplied = (host or "").strip()
    lan_ip = supplied or (detect() or "")
    if not lan_ip:
        steps.append(StepResult("detect-lan-ip", False,
                                "no non-loopback LAN address found on this box; pass --host <LAN_IP>"))
        return EXIT_USAGE, steps
    if _is_loopbackish(lan_ip):
        steps.append(StepResult("detect-lan-ip", False,
                                f"{lan_ip!r} is loopback — the advertiser would broadcast nothing; "
                                "pass the brain's LAN address via --host"))
        return EXIT_USAGE, steps
    steps.append(StepResult("detect-lan-ip", True,
                            f"{lan_ip} ({'from --host' if supplied else 'primary route'})"))

    # 2. The Layer-C seam (a separate venv's console script — never an import).
    install_bin = resolve()
    if not install_bin:
        steps.append(StepResult("resolve-seam", False,
                                "gabagent-install not found — is gabagent installed on this box? "
                                "set GABAGENT_INSTALL_BIN to its path"))
        return EXIT_NO_SEAM, steps
    steps.append(StepResult("resolve-seam", True, install_bin))

    # 3. Flip the flag cross-process; the seam refuses on loopback (exit 2) and writes on success.
    try:
        cp = invoke(install_bin, lan_ip, room_id)
    except OSError as e:
        steps.append(StepResult("enable-advertise", False, f"could not run {install_bin}: {e}"))
        return EXIT_NO_SEAM, steps

    if cp.returncode == EXIT_OK:
        detail = f"brain now advertises _voice-brain._tcp on {lan_ip}"
        if room_id:
            detail += f" (room {room_id})"
        steps.append(StepResult("enable-advertise", True, detail))
        return EXIT_OK, steps

    reason = (cp.stderr or "").strip() or (cp.stdout or "").strip() or f"exit {cp.returncode}"
    steps.append(StepResult("enable-advertise", False,
                            f"the brain-side seam did not enable advertising (exit {cp.returncode}): {reason}"))
    return EXIT_ENABLE_FAILED, steps


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent_install.voice_host",
        description="Provision this box as the voice-host: detect its LAN IP and enable the brain's "
                    "mDNS advertiser so satellites can discover it without a hand-typed address.")
    parser.add_argument("--host", default=None,
                        help="the brain's LAN address to advertise on (default: auto-detect the "
                             "primary-route interface). Pass this if detection picks the wrong NIC.")
    parser.add_argument("--room-id", dest="room_id", default=None,
                        help="this brain's room identity (written as voice_room_id)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    console = Console(interactive=False)
    code, steps = provision(host=args.host, room_id=args.room_id)

    console.out("\n--- voice-host summary ---")
    for step in steps:
        console.out(f"  {'OK  ' if step.ok else 'FAIL'} {step.name}: {step.detail}")
    if code == EXIT_OK:
        console.out("\nAdvertising enabled. A satellite on this LAN can now auto-discover this brain.")
    else:
        console.out("\nAdvertising NOT enabled (see FAIL above). This is not fatal: satellites still "
                    "work — point them at this brain by typing its IP at install time. Fix the step "
                    "above and re-run to turn on auto-discovery.")
    return code


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m`
    raise SystemExit(main())
