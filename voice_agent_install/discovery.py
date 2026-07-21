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

# Boot-safety budgets (seconds). Re-budgeted 2026-07-20 after a 4-way adversarial pass found the old
# 3+5=8-against-a-10s-ceiling had NO headroom and was not actually enforced anywhere:
#   - `urlopen(timeout=)` is applied PER RESOLVED ADDRESS and excludes `getaddrinfo` entirely (measured:
#     timeout=1.0 against 3 blackhole addresses returned in 3.01s), so a hostname host is unbounded →
#     `reresolve_brain_host_async` now refuses to probe a non-IP-literal host at all (see is_ip_literal).
#   - the per-advert `get_service_info` resolve used to be handed the WHOLE browse budget, so one slow
#     advert on zeroconf's callback thread could eat the entire browse and starve the right brain's
#     callback — a miss DESPITE the brain advertising. It gets its own small slice now.
# Worst case is probe + confirm-probe + browse = 8.0s, and the CALLER still wraps the whole thing in a hard
# outer deadline (config.reresolve_brain_host) because these are the callee's promises, not a guarantee.
HEALTH_PROBE_TIMEOUT = 2.0
MDNS_BROWSE_TIMEOUT = 4.0
# Per-advert resolve slice — deliberately a small fraction of the browse budget, never equal to it.
MDNS_INFO_TIMEOUT = 1.0


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


def is_ip_literal(host: str) -> bool:
    """True if ``host`` is a bare IPv4/IPv6 address rather than a name.

    Load-bearing for boot-safety, not cosmetics: ``urllib``'s ``timeout=`` is applied per-resolved-address
    and covers no part of ``getaddrinfo``, so probing a NAME has an unbounded DNS phase that nothing in
    this module can bound (an nss-mdns lookup with Avahi down is the ~5s stall this file's own header
    warns about). ``GAB_HOST`` is a free-form hand-edited field, so the guard has to live here.
    """
    import ipaddress

    try:
        ipaddress.ip_address((host or "").strip())
        return True
    except ValueError:
        return False


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
    info_timeout: float = MDNS_INFO_TIMEOUT,
    stats: Optional[dict] = None,
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

    ``info_timeout`` bounds EACH advert's resolve and is deliberately a small fraction of ``timeout``:
    handing a per-advert resolve the whole browse budget lets one slow advert on zeroconf's callback
    thread starve the right brain's callback, producing a miss even though the brain was advertising.

    ``stats`` (optional) is filled in place with ``{"seen": n, "filtered": m}`` so a caller can tell
    "nothing was advertising" apart from "a brain advertised and I filtered it out as another room's" —
    the second is an operator misconfiguration and is otherwise invisible.
    """
    if stats is not None:
        stats.setdefault("seen", 0)
        stats.setdefault("filtered", 0)
    try:
        from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
    except Exception:
        return None

    import threading

    found: dict[str, object] = {}
    done = threading.Event()
    want_room = room_id.encode() if room_id is not None else None
    info_timeout_ms = max(1, int(info_timeout * 1000))

    class _Listener(ServiceListener):  # type: ignore[misc]
        def _consider(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=info_timeout_ms)
            if info is None:
                return
            if stats is not None:
                stats["seen"] += 1
            if not _room_matches((info.properties or {}).get(b"room_id"), want_room):
                if stats is not None:
                    stats["filtered"] += 1
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
    stats: Optional[dict] = None,
) -> tuple[str, str]:
    """In-app boot re-resolve of the brain host — the async runtime counterpart to the
    install-time seam. Returns ``(host, reason)`` (same shape as :func:`decide_host`).

    This is the SANCTIONED form of "re-find the brain at boot" (Tier-0 boot-safety): a
    plain library call inside the app's async startup, NEVER a systemd ``ExecStartPre``/
    oneshot with a remote dependency. Bounded and non-fatal by construction:

    - a loopback ``written_host`` is a local brain → skip (never re-resolve to another box).
    - a NON-IP-LITERAL ``written_host`` → skip: its DNS phase is unbounded and nothing here can
      bound it (see :func:`is_ip_literal`). Boot-safety beats convenience; the manual host still works.
    - ONE bounded ``GET /health`` on the written host; reachable → keep it, no browse.
    - only if unreachable, ONE bounded mDNS browse, then **one bounded confirm-probe of the discovered
      host before adopting it**. zeroconf answers from its cache, so an advert on its own is NOT
      evidence the brain is up: adopting on the advert alone would swap a merely-unreachable host for a
      definitely-unreachable one while logging "rediscovered" — a success claim never observed. Only a
      200 earns adoption; otherwise :func:`decide_host` KEEPS the written host.

    Worst case = probe + browse + confirm-probe, each run OFF the event loop via ``asyncio.to_thread``
    so a slow lookup never stalls the loop. ANY internal error → ``(written_host, "error-kept")`` so app
    startup never dies on discovery. These are the callee's promises, not a guarantee — ``to_thread`` is
    not cancellable, so the CALLER must still impose a hard outer deadline.

    The port is left as written (per-room brains use a stable port; :func:`decide_host` policy is
    host-only) — a moved port is the operator's manual re-provision, not an auto-adopt. But a divergence
    is reported in the reason rather than swallowed: silently pointing at a live host on a dead port is
    the confusing failure this used to produce.

    The ``reason`` is deliberately diagnosable — ``written-kept(no-adverts)`` (nothing was advertising)
    reads differently from ``written-kept(filtered=2)`` (brains advertised; the room filter rejected
    them), which is an operator misconfiguration that is otherwise invisible.

    ``probe_fn``/``discover_fn`` are injection seams for tests (sync stubs); unset ⇒ the real
    stdlib health probe and mDNS browse. ``probe_fn`` is called as ``probe_fn(host, port)`` for BOTH the
    written host and the confirm-probe of a discovered one.
    """
    import asyncio

    if is_loopback(written_host):
        return written_host, "loopback-skip"
    if not is_ip_literal(written_host):
        return written_host, "not-ip-skip"

    stats = {} if stats is None else stats

    async def _probe(host: str) -> bool:
        if probe_fn is not None:
            return probe_fn(host, port)
        return await asyncio.to_thread(http_health_ok, host, port, timeout=health_timeout)

    async def _discover() -> Optional[BrainEndpoint]:
        if discover_fn is not None:
            return discover_fn()
        return await asyncio.to_thread(
            mdns_discover, timeout=browse_timeout, room_id=room_id, stats=stats
        )

    try:
        if await _probe(written_host):
            return decide_host(written_host, True, None)

        endpoint = await _discover()
        if endpoint is None:
            filtered = stats.get("filtered", 0)
            detail = f"filtered={filtered}" if filtered else "no-adverts"
            return written_host, f"written-kept({detail})"
        if is_loopback(endpoint.host):
            return written_host, "written-kept(loopback-advert)"

        port_note = f",advert-port={endpoint.port}" if endpoint.port != port else ""
        if not await _probe(endpoint.host):
            return written_host, f"written-kept(unconfirmed{port_note})"

        host, reason = decide_host(written_host, False, endpoint.host)
        if port_note:
            reason = f"{reason}({port_note.lstrip(',')}!={port} — re-provision GAB_PORT)"
        return host, reason
    except Exception:
        return written_host, "error-kept"
