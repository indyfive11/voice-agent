"""Tests for voice_agent_install.discovery — the ordered provider seam + the
boot re-resolve policy. Pure/mocked only; no real zeroconf or network I/O."""

import builtins

import pytest

from voice_agent_install import discovery as d
from voice_agent_install.discovery import BrainEndpoint


# --------------------------------------------------------------------------- is_loopback
@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("127.1.2.3", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", True),
        ("", True),
        ("192.168.1.10", False),
        ("10.0.0.5", False),
        ("brain.local", False),  # a name is not loopback (though we standardize on IP)
    ],
)
def test_is_loopback(host, expected):
    assert d.is_loopback(host) is expected


# --------------------------------------------------------------------------- decide_host (CR-5 invariant)
def test_decide_host_reachable_keeps_written():
    host, reason = d.decide_host("192.168.1.10", reachable=True, discovered_host="192.168.1.99")
    assert host == "192.168.1.10"  # reachable → never re-discovers, even if a new host was found
    assert reason == "written-reachable"


def test_decide_host_unreachable_adopts_routable_discovery():
    host, reason = d.decide_host("192.168.1.10", reachable=False, discovered_host="192.168.1.42")
    assert host == "192.168.1.42"
    assert reason == "rediscovered"


def test_decide_host_unreachable_miss_keeps_written():
    # The core invariant: a discovery MISS must never overwrite a good written host.
    host, reason = d.decide_host("192.168.1.10", reachable=False, discovered_host=None)
    assert host == "192.168.1.10"
    assert reason == "written-kept"


def test_decide_host_never_overwrites_with_loopback():
    # A loopback discovery result would point the satellite at itself — must be rejected.
    host, reason = d.decide_host("192.168.1.10", reachable=False, discovered_host="127.0.0.1")
    assert host == "192.168.1.10"
    assert reason == "written-kept"


# --------------------------------------------------------------------------- discover_brain seam
def test_discover_brain_first_hit_wins():
    ep = BrainEndpoint("10.0.0.1", 8765, "mdns")
    providers = [lambda: ep, lambda: pytest.fail("second provider must not run")]
    assert d.discover_brain(providers) is ep


def test_discover_brain_falls_through_to_floor():
    floor = BrainEndpoint("10.0.0.9", 8765, "manual")
    providers = [lambda: None, lambda: floor]
    assert d.discover_brain(providers) is floor


def test_discover_brain_all_miss_returns_none():
    assert d.discover_brain([lambda: None, lambda: None]) is None


def test_discover_brain_manual_is_last_in_default_seam(monkeypatch):
    # The floor must be ordered LAST so mDNS is tried first. Patch both underlying
    # calls and record the call order when the seam runs.
    calls = []
    monkeypatch.setattr(d, "mdns_discover", lambda **k: calls.append("mdns") or None)
    floor = BrainEndpoint("10.0.0.9", 8765, "manual")
    monkeypatch.setattr(d, "manual_prompt", lambda **k: calls.append("manual") or floor)
    seam = d.default_providers()
    assert len(seam) == 2
    assert d.discover_brain(seam) is floor  # mDNS misses → falls to the manual floor
    assert calls == ["mdns", "manual"]  # mDNS tried FIRST, manual LAST


# --------------------------------------------------------------------------- manual_prompt (the floor)
def test_manual_prompt_reads_host_and_port():
    answers = iter(["192.168.1.50", "9001"])
    ep = d.manual_prompt(input_fn=lambda _prompt: next(answers))
    assert ep == BrainEndpoint("192.168.1.50", 9001, "manual")


def test_manual_prompt_default_port_on_blank():
    answers = iter(["192.168.1.50", "  "])
    ep = d.manual_prompt(input_fn=lambda _prompt: next(answers))
    assert ep == BrainEndpoint("192.168.1.50", d.DEFAULT_BRAIN_PORT, "manual")


def test_manual_prompt_empty_host_returns_none():
    assert d.manual_prompt(input_fn=lambda _prompt: "") is None


def test_manual_prompt_rejects_loopback():
    # A satellite must point at another box; a loopback answer returns None so the
    # caller re-prompts rather than provisioning a self-pointing host.
    assert d.manual_prompt(input_fn=lambda _prompt: "127.0.0.1") is None


def test_manual_prompt_bad_port_falls_back_to_default():
    answers = iter(["192.168.1.50", "not-a-port"])
    ep = d.manual_prompt(input_fn=lambda _prompt: next(answers))
    assert ep.port == d.DEFAULT_BRAIN_PORT


# --------------------------------------------------------------------------- mdns fail-soft
def test_mdns_discover_returns_none_when_zeroconf_absent(monkeypatch):
    # Simulate zeroconf not installed → the import inside mdns_discover raises → None
    # (→ the seam degrades to the manual floor). This is why 3a needs no mDNS dep.
    real_import = builtins.__import__

    def _no_zeroconf(name, *args, **kwargs):
        if name == "zeroconf" or name.startswith("zeroconf."):
            raise ImportError("no zeroconf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_zeroconf)
    assert d.mdns_discover(timeout=0.01) is None


# --------------------------------------------------------------------------- health probe
def test_http_health_ok_true_on_2xx(monkeypatch):
    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # http_health_ok imports urllib.request lazily; patch at the source module.
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert d.http_health_ok("192.168.1.10", 8765, timeout=0.1) is True


def test_http_health_ok_false_on_error(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert d.http_health_ok("192.168.1.10", 8765, timeout=0.1) is False


# --------------------------------------------------------- reresolve_brain_host_async
# Sync tests drive the coroutine via asyncio.run (no pytest-asyncio dependency).
import asyncio


def _reresolve(**kw):
    return asyncio.run(d.reresolve_brain_host_async(**kw))


def test_reresolve_skips_loopback_written_host():
    # a local brain is never re-resolved onto another box; probe must not even run
    def _probe(_h, _p):
        raise AssertionError("probe must not be called for a loopback host")
    host, reason = _reresolve(written_host="127.0.0.1", port=8765, probe_fn=_probe)
    assert (host, reason) == ("127.0.0.1", "loopback-skip")


def test_reresolve_keeps_reachable_and_never_browses():
    def _discover():
        raise AssertionError("must not browse when the written host is reachable")
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: True, discover_fn=_discover,
    )
    assert (host, reason) == ("10.0.0.5", "written-reachable")


def test_reresolve_adopts_routable_rediscovery_when_unreachable():
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False,
        discover_fn=lambda: BrainEndpoint("10.0.0.9", 8765, "mdns"),
    )
    assert (host, reason) == ("10.0.0.9", "rediscovered")


def test_reresolve_keeps_written_on_discovery_miss():
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False,
        discover_fn=lambda: None,
    )
    assert (host, reason) == ("10.0.0.5", "written-kept")


def test_reresolve_rejects_loopback_rediscovery():
    # a discovery that somehow yields loopback must never overwrite a good written host
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False,
        discover_fn=lambda: BrainEndpoint("127.0.0.1", 8765, "mdns"),
    )
    assert (host, reason) == ("10.0.0.5", "written-kept")


def test_reresolve_never_raises_returns_error_kept():
    def _boom(_h, _p):
        raise RuntimeError("probe blew up")
    host, reason = _reresolve(written_host="10.0.0.5", port=8765, probe_fn=_boom)
    assert (host, reason) == ("10.0.0.5", "error-kept")


# --------------------------------------------------------------------- _room_matches
@pytest.mark.parametrize(
    "advertised,want_room,expected",
    [
        (None, None, True),                 # satellite asks nothing → match anything
        (b"bedroom", None, True),             # satellite asks nothing → match a named brain too
        (None, b"bedroom", True),             # advert has no room_id → default brain, matches
        (b"", b"bedroom", True),              # advert empty room_id → single-room default, matches (the PoC path)
        (b"bedroom", b"bedroom", True),         # exact room match
        (b"kitchen", b"bedroom", False),      # a genuinely different room's brain → skip
        (b"", None, True),                  # empty advert + no ask → match
    ],
)
def test_room_matches(advertised, want_room, expected):
    assert d._room_matches(advertised, want_room) is expected
