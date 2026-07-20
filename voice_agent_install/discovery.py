"""LAN-brain discovery — the ordered provider seam.

maintainer-locked design (2026-07-13): the satellite finds its brain via an ORDERED
PROVIDER SEAM — each provider returns a :class:`BrainEndpoint` or ``None``; they
are tried in order, first hit wins, and the **manual host is always the last,
always-available floor** (a flaky-multicast Wi-Fi satellite is never bricked).
UDP-broadcast was rejected as a speculative second path (it fails *correlated*
with mDNS — same link-local/subnet/Wi-Fi-multicast envelope — so it rescues
almost nothing mDNS-only misses; the manual host is the only *independent*
fallback). The seam is deliberately the cheapest possible form: a plain ordered
tuple of callables, NOT a registry/entry-points/decorator (that would re-import
the very asymmetry the three-layer installer split exists to prevent).

GA↔VAC consensus (2026-07-20):
- **Manual floor is 3a's hard floor** — it needs no brain-side code. mDNS is the
  enhancement that lights up once the brain ships its zeroconf advertiser; until
  then :func:`mdns_discover` returns ``None`` and the seam degrades to manual.
- **Write the resolved LAN IP, never a ``.local`` name** — a ``.local`` in
  ``GAB_HOST`` makes every httpx connect eat a synchronous ``getaddrinfo``/nss-mdns
  lookup (~5s when Avahi is down), landing inside the boot-safety connect budget.
  IP keeps that path stall-free and drops ``libnss-mdns`` from the Pi's deps.
- **Positive-overwrite invariant** — a discovery miss (or a loopback result) NEVER
  overwrites a good written host. See :func:`decide_host`.

Boot-safety (Tier-0 hard rule): the at-boot re-resolve is a LIBRARY CALL inside
the app's async startup, never a systemd ``ExecStartPre``/oneshot with a remote
dependency. The reachability probe is ONE bounded ``GET /health``; the mDNS browse
is a SINGLE bounded ``Event.wait`` — never a ``while not found: rebrowse`` loop.
Worst case = one probe + one browse, kept well under the 10s boot ceiling, and
always non-fatal (a miss falls through to the written host and lets the app's
existing in-process connect-retry take over).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

# Net-new mDNS service type for the brain advert. NEUTRAL-NAMED on purpose
# (protocol de-branding is a ship-blocker pillar — a net-new artifact starts clean,
# it does not inherit the ``gab`` wire-name). The brain's advertiser MUST publish
# this exact type; the name is a GA↔VAC coordination point carried in the A4/A5 spec.
BRAIN_MDNS_TYPE = "_voice-brain._tcp.local."

# Matches config.py's GAB_PORT default (the brain's per-room HTTP/SSE port).
DEFAULT_BRAIN_PORT = 8765

# Boot-safety budgets (seconds). probe + browse worst case = 8s < the 10s ceiling.
HEALTH_PROBE_TIMEOUT = 3.0
MDNS_BROWSE_TIMEOUT = 5.0


@dataclass(frozen=True)
class BrainEndpoint:
    """A resolved brain location. ``host`` is always a routable address/IP, never
    loopback when it came from discovery. ``source`` records how it was found
    (``mdns`` | ``manual`` | ``written``) for logging/provenance."""

    host: str
    port: int
    source: str


def is_loopback(host: str) -> bool:
    """True for loopback / unspecified addresses that must never be *discovered* into
    a satellite's ``GAB_HOST`` (a satellite pointed at itself is broken)."""
    if not host:
        return True
    h = host.strip().lower()
    return (
        h in ("localhost", "::1", "0.0.0.0", "::")
        or h.startswith("127.")
    )


def _room_matches(advertised: Optional[bytes], want_room: Optional[bytes]) -> bool:
    """Room-filter policy for an mDNS advert (contract with the brain advertiser).

    - the satellite asks for no particular room (``want_room is None``) → match anything.
    - the advert carries an ABSENT or EMPTY ``room_id`` → it's the default/only brain and ALWAYS
      matches (a single-room brain advertises ``room_id=""`` and must be found even when the satellite
      asks for its hostname-defaulted room).
    - only a NON-EMPTY advertised room that DIFFERS from the requested one is a genuinely other room's
      brain → no match.
    """
    if want_room is None:
        return True
    if not advertised:  # None or b"" → default/only brain
        return True
    return advertised == want_room


def decide_host(
    written_host: str,
    reachable: bool,
    discovered_host: Optional[str],
) -> tuple[str, str]:
    """Boot re-resolve policy (pure). Returns ``(host, reason)``.

    - written host reachable  → keep it (no discovery needed).
    - unreachable + a POSITIVELY-resolved routable discovery → adopt the new host.
    - unreachable + miss/loopback → **keep the written host** (never overwrite a good
      value with nothing, never write loopback). The app's in-process connect-retry
      then handles a transiently-down brain.
    """
    if reachable:
        return written_host, "written-reachable"
    if discovered_host and not is_loopback(discovered_host):
        return discovered_host, "rediscovered"
    return written_host, "written-kept"


# --------------------------------------------------------------------------- probes
def http_health_ok(
    host: str,
    port: int,
    *,
    timeout: float = HEALTH_PROBE_TIMEOUT,
    path: str = "/health",
) -> bool:
    """One bounded ``GET /health`` — stdlib only (usable at install time and, wrapped
    off the event loop, in-app). Never raises; any failure → ``False`` (non-fatal)."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed scheme)
            status = getattr(resp, "status", None) or resp.getcode()
            return 200 <= int(status) < 300
    except Exception:
        return False


def mdns_discover(
    *,
    timeout: float = MDNS_BROWSE_TIMEOUT,
    service_type: str = BRAIN_MDNS_TYPE,
    room_id: Optional[str] = None,
) -> Optional[BrainEndpoint]:
    """Single bounded mDNS browse for the brain advert. Fail-soft by construction:

    - ``zeroconf`` not installed → return ``None`` (→ manual floor). This is why the
      Pi needs no mDNS dependency to ship 3a.
    - no advert within ``timeout`` → return ``None``.
    - a matched advert with a routable address → :class:`BrainEndpoint`.

    NEVER loops: one :class:`~zeroconf.ServiceBrowser` + one ``Event.wait(timeout)``,
    with the ``Zeroconf`` instance always closed in ``finally`` so no browser thread
    or socket outlives the call (GA reinforcement — discovery never leaks a thread
    out of startup, just as it never raises out of it).
    """
    try:
        from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
    except Exception:
        return None

    import threading

    found: dict[str, object] = {}
    done = threading.Event()
    want_room = room_id.encode() if room_id is not None else None
    info_timeout_ms = max(1, int(timeout * 1000))

    class _Listener(ServiceListener):  # type: ignore[misc]
        def _consider(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=info_timeout_ms)
            if info is None:
                return
            if not _room_matches((info.properties or {}).get(b"room_id"), want_room):
                return  # a different room's brain — keep browsing
            for addr in info.parsed_addresses():
                if not is_loopback(addr):
                    found["host"] = addr
                    found["port"] = int(info.port or DEFAULT_BRAIN_PORT)
                    done.set()
                    return

        def add_service(self, zc, type_, name):
            self._consider(zc, type_, name)

        def update_service(self, zc, type_, name):
            self._consider(zc, type_, name)

        def remove_service(self, zc, type_, name):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, service_type, _Listener())
        done.wait(timeout=timeout)  # single bounded wait — NEVER a retry loop
    finally:
        zc.close()

    if "host" in found:
        return BrainEndpoint(str(found["host"]), int(found["port"]), "mdns")  # type: ignore[arg-type]
    return None


def manual_prompt(
    *,
    default_port: int = DEFAULT_BRAIN_PORT,
    input_fn: Callable[[str], str] = input,
) -> Optional[BrainEndpoint]:
    """The always-available floor: ask the operator for the brain's LAN IP (+ port).

    Returns ``None`` on empty input so a caller can loop/re-prompt. Rejects a
    loopback answer (a satellite must point at another box), re-prompting via the
    caller's loop rather than silently accepting a self-pointing host.
    """
    host = input_fn("LAN brain host (IP address): ").strip()
    if not host:
        return None
    if is_loopback(host):
        return None
    port_s = input_fn(f"Brain port [{default_port}]: ").strip()
    try:
        port = int(port_s) if port_s else default_port
    except ValueError:
        port = default_port
    return BrainEndpoint(host, port, "manual")


# --------------------------------------------------------------------------- seam
def discover_brain(
    providers: Iterable[Callable[[], Optional[BrainEndpoint]]],
) -> Optional[BrainEndpoint]:
    """Run the ordered provider seam: first provider returning an endpoint wins.

    ``providers`` is a plain ordered iterable of zero-arg callables — the manual
    provider is expected LAST so it is the floor. Returns ``None`` only if every
    provider (including manual) yields nothing.
    """
    for provider in providers:
        endpoint = provider()
        if endpoint is not None:
            return endpoint
    return None


def default_providers(
    *,
    room_id: Optional[str] = None,
    input_fn: Callable[[str], str] = input,
) -> tuple[Callable[[], Optional[BrainEndpoint]], ...]:
    """The maintainer-locked ordered seam: ``(mdns, manual)``. Add a future provider (e.g. a
    broadcast fallback, only if live evidence ever warrants it) as another entry on
    this tuple — no call-site change."""
    return (
        lambda: mdns_discover(room_id=room_id),
        lambda: manual_prompt(input_fn=input_fn),
    )


# --------------------------------------------------------------- in-app boot re-resolve
async def reresolve_brain_host_async(
    written_host: str,
    port: int,
    *,
    room_id: Optional[str] = None,
    health_timeout: float = HEALTH_PROBE_TIMEOUT,
    browse_timeout: float = MDNS_BROWSE_TIMEOUT,
    probe_fn: Optional[Callable[[str, int], bool]] = None,
    discover_fn: Optional[Callable[[], Optional[BrainEndpoint]]] = None,
) -> tuple[str, str]:
    """In-app boot re-resolve of the brain host — the async runtime counterpart to the
    install-time seam. Returns ``(host, reason)`` (same shape as :func:`decide_host`).

    This is the SANCTIONED form of "re-find the brain at boot" (Tier-0 boot-safety): a
    plain library call inside the app's async startup, NEVER a systemd ``ExecStartPre``/
    oneshot with a remote dependency. Bounded and non-fatal by construction:

    - a loopback ``written_host`` is a local brain → skip (never re-resolve to another box).
    - ONE bounded ``GET /health`` on the written host; reachable → keep it, no browse.
    - only if unreachable, ONE bounded mDNS browse; then :func:`decide_host` decides —
      adopt a routable rediscovery, else KEEP the written host (a miss/loopback never
      overwrites a good value). The app's existing in-process connect-retry then covers a
      transiently-down brain.

    Worst case = one probe + one browse, each run OFF the event loop via ``asyncio.to_thread``
    so a slow lookup never stalls the loop. ANY internal error → ``(written_host, "error-kept")``
    so app startup never dies on discovery. The port is left as written (per-room brains use a
    stable port; :func:`decide_host` policy is host-only) — a moved port is the operator's
    manual re-provision, not an auto-adopt.

    ``probe_fn``/``discover_fn`` are injection seams for tests (sync stubs); unset ⇒ the real
    stdlib health probe and mDNS browse.
    """
    import asyncio

    if is_loopback(written_host):
        return written_host, "loopback-skip"

    async def _probe() -> bool:
        if probe_fn is not None:
            return probe_fn(written_host, port)
        return await asyncio.to_thread(http_health_ok, written_host, port, timeout=health_timeout)

    async def _discover() -> Optional[BrainEndpoint]:
        if discover_fn is not None:
            return discover_fn()
        return await asyncio.to_thread(mdns_discover, timeout=browse_timeout, room_id=room_id)

    try:
        reachable = await _probe()
        discovered_host: Optional[str] = None
        if not reachable:
            endpoint = await _discover()
            discovered_host = endpoint.host if endpoint is not None else None
        return decide_host(written_host, reachable, discovered_host)
    except Exception:
        return written_host, "error-kept"
