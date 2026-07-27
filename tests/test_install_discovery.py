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


def test_reresolve_adopts_rediscovery_only_after_confirm_probe():
    # written host down, discovered host answers /health → adopt. The confirm-probe is what earns
    # the word "rediscovered": zeroconf answers from cache, so the advert alone proves nothing.
    probed: list[str] = []

    def _probe(h, _p):
        probed.append(h)
        return h == "10.0.0.9"

    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=_probe,
        discover_fn=lambda: BrainEndpoint("10.0.0.9", 8765, "mdns"),
    )
    assert (host, reason) == ("10.0.0.9", "rediscovered")
    assert probed == ["10.0.0.5", "10.0.0.9"]  # written first, then the confirm


def test_reresolve_keeps_written_when_advert_fails_confirm_probe():
    # THE cache-stale case: a brain that died inside its mDNS TTL still resolves. Adopting it would
    # swap a merely-unreachable host for a definitely-unreachable one and log a success never observed.
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False,  # nothing is reachable, including the advert
        discover_fn=lambda: BrainEndpoint("10.0.0.9", 8765, "mdns"),
    )
    assert (host, reason) == ("10.0.0.5", "written-kept(unconfirmed)")


def test_reresolve_reports_port_divergence_instead_of_silently_pointing_at_a_dead_port():
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: h == "10.0.0.9",
        discover_fn=lambda: BrainEndpoint("10.0.0.9", 8766, "mdns"),
    )
    assert host == "10.0.0.9"
    assert reason.startswith("rediscovered(")
    assert "advert-port=8766" in reason and "8765" in reason


def test_reresolve_keeps_written_on_discovery_miss():
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False,
        discover_fn=lambda: None,
    )
    assert (host, reason) == ("10.0.0.5", "written-kept(no-adverts)")


def test_reresolve_rejects_loopback_rediscovery():
    # a discovery that somehow yields loopback must never overwrite a good written host
    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False,
        discover_fn=lambda: BrainEndpoint("127.0.0.1", 8765, "mdns"),
    )
    assert (host, reason) == ("10.0.0.5", "written-kept(loopback-advert)")


def test_reresolve_miss_distinguishes_filtered_from_no_adverts():
    # "nothing was advertising" and "brains advertised but the room filter rejected them" are the same
    # outcome but completely different diagnoses — the second is an operator misconfiguration.
    shared: dict = {}

    def _discover():
        shared.update({"seen": 2, "filtered": 2})
        return None

    host, reason = _reresolve(
        written_host="10.0.0.5", port=8765,
        probe_fn=lambda h, p: False, discover_fn=_discover, stats=shared,
    )
    assert (host, reason) == ("10.0.0.5", "written-kept(filtered=2)")


def test_mdns_discover_seeds_stats_even_when_zeroconf_absent(monkeypatch):
    real_import = builtins.__import__

    def _no_zeroconf(name, *a, **kw):
        if name == "zeroconf":
            raise ImportError("no zeroconf")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_zeroconf)
    stats: dict = {}
    assert d.mdns_discover(stats=stats) is None
    assert stats == {"seen": 0, "filtered": 0}


def test_reresolve_skips_a_non_ip_written_host():
    # a hostname has an unbounded getaddrinfo phase that urlopen's timeout does not cover, so we
    # refuse to probe it at all rather than pretend it is bounded. The probe must not run.
    def _probe(_h, _p):
        raise AssertionError("probe must not be called for a non-IP host")

    host, reason = _reresolve(written_host="brain.local", port=8765, probe_fn=_probe)
    assert (host, reason) == ("brain.local", "not-ip-skip")


@pytest.mark.parametrize(
    "host,expected",
    [
        ("192.168.1.100", True), ("10.0.0.5", True), ("::1", True), ("fe80::1", True),
        ("brain.local", False), ("em", False), ("", False), ("192.168.1.100:8765", False),
    ],
)
def test_is_ip_literal(host, expected):
    assert d.is_ip_literal(host) is expected


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


# ============================================================ §10d firewall-aware discovery diagnosis
# All injected — no sockets. `brain_browse_fn` returns (endpoint, stats) or raises the internal
# unavailable/broken signals; `broad_browse_fn` returns the OTHER-service count; `probe_fn` confirms.
def _diagnose(**kw):
    return d.diagnose_brain_discovery(**kw)


def test_diagnose_found_when_routable_and_probe_answers():
    ep = BrainEndpoint("192.168.1.42", 8765, "mdns")
    diag = _diagnose(brain_browse_fn=lambda: (ep, {}), probe_fn=lambda h, p: True)
    assert diag.ok is True
    assert diag.reason == "found"
    assert diag.endpoint is ep
    assert diag.remedy == ""            # no remedy needed on success


def test_diagnose_loopback_advert():
    ep = BrainEndpoint("127.0.0.1", 8765, "mdns")
    diag = _diagnose(brain_browse_fn=lambda: (ep, {}),
                     probe_fn=lambda h, p: pytest.fail("must not probe a loopback advert"))
    assert diag.ok is False and diag.reason == "loopback-advert"
    assert "127.0.0.1" in diag.detail and "localhost" in diag.remedy


def test_diagnose_unconfirmed_when_advert_fails_probe():
    ep = BrainEndpoint("192.168.1.42", 8765, "mdns")
    diag = _diagnose(brain_browse_fn=lambda: (ep, {}), probe_fn=lambda h, p: False)
    assert diag.ok is False and diag.reason == "unconfirmed"
    assert diag.endpoint is ep and "8765" in diag.remedy


def test_diagnose_room_filtered_when_adverts_seen_but_room_mismatch():
    # brain adverts WERE seen (mDNS works) but none matched → a room-id mismatch, NOT a firewall gap.
    diag = _diagnose(room_id="bedroom",
                     brain_browse_fn=lambda: (None, {"seen": 2, "filtered": 2}),
                     broad_browse_fn=lambda: pytest.fail("room-filter is decided before the broad browse"))
    assert diag.reason == "room-filtered"
    assert "different room" in diag.remedy.lower() and "firewall" in diag.remedy.lower()


def test_diagnose_filtered_when_other_services_visible_but_no_brain():
    # THE firewall discriminator: multicast reaches this box (other services seen) but the brain does
    # not → the gap is at the brain (advertiser off, or its host firewall drops 5353/udp).
    diag = _diagnose(brain_browse_fn=lambda: (None, {}), broad_browse_fn=lambda: 4)
    assert diag.reason == "filtered"
    assert diag.detail == "4" and diag.label == "filtered=4"
    assert d.MDNS_MULTICAST_HINT in diag.remedy and "voice_advertise" in diag.remedy


def test_diagnose_no_adverts_when_nothing_at_all():
    diag = _diagnose(brain_browse_fn=lambda: (None, {}), broad_browse_fn=lambda: 0)
    assert diag.reason == "no-adverts"
    assert d.MDNS_MULTICAST_HINT in diag.remedy   # firewall-aware


def test_diagnose_no_mdns_when_zeroconf_absent_is_not_a_fault():
    def _unavailable():
        raise d._ZeroconfUnavailable("no zeroconf")
    diag = _diagnose(brain_browse_fn=_unavailable)
    assert diag.reason == "no-mdns"
    assert "not an error" in diag.remedy.lower()


def test_diagnose_broken_zeroconf_is_not_masked_as_no_adverts():
    # THE corollary: an installed-but-broken zeroconf must render as its own reason, never as
    # "no-adverts" — otherwise a local mDNS fault points the operator at the brain/firewall.
    def _broken():
        raise d._ZeroconfBroken("multicast socket denied")
    diag = _diagnose(brain_browse_fn=_broken)
    assert diag.reason == "zeroconf-error"
    assert diag.reason != "no-adverts"
    assert "multicast socket denied" in diag.remedy and "broken" in diag.remedy.lower()


def test_diagnose_never_runs_broad_browse_when_brain_is_found():
    ep = BrainEndpoint("192.168.1.42", 8765, "mdns")
    _diagnose(brain_browse_fn=lambda: (ep, {}), probe_fn=lambda h, p: True,
              broad_browse_fn=lambda: pytest.fail("broad browse must not run once the brain is found"))


# ----------------------------------------------- default_providers report wiring (detect-and-report)
def test_default_providers_report_fires_on_miss_then_falls_to_manual(monkeypatch):
    # With a `report` callback the mDNS leg runs the diagnosis; on a miss it reports and returns None,
    # and the seam falls through to the manual floor — NON-FATAL by construction.
    reported: list = []
    miss = d.DiscoveryDiagnosis(False, "no-adverts", "none", None, "remedy text")
    monkeypatch.setattr(d, "diagnose_brain_discovery", lambda **k: miss)
    floor = BrainEndpoint("10.0.0.9", 8765, "manual")
    monkeypatch.setattr(d, "manual_prompt", lambda **k: floor)

    seam = d.default_providers(report=reported.append)
    assert d.discover_brain(seam) is floor
    assert reported == [miss]           # the operator was told WHY before being asked to type a host


def test_default_providers_report_hit_returns_endpoint_without_manual(monkeypatch):
    ep = BrainEndpoint("192.168.1.42", 8765, "mdns")
    hit = d.DiscoveryDiagnosis(True, "found", "192.168.1.42:8765", ep, "")
    monkeypatch.setattr(d, "diagnose_brain_discovery", lambda **k: hit)
    monkeypatch.setattr(d, "manual_prompt", lambda **k: pytest.fail("manual floor must not run on a hit"))

    seam = d.default_providers(report=lambda _diag: pytest.fail("report must not fire on a hit"))
    assert d.discover_brain(seam) is ep


def test_default_providers_without_report_is_unchanged(monkeypatch):
    # Backward-compat: no report → the mDNS leg is a plain mdns_discover, no diagnosis.
    calls = []
    monkeypatch.setattr(d, "mdns_discover", lambda **k: calls.append("mdns") or None)
    monkeypatch.setattr(d, "diagnose_brain_discovery",
                        lambda **k: pytest.fail("diagnosis must not run without a report callback"))
    monkeypatch.setattr(d, "manual_prompt", lambda **k: BrainEndpoint("10.0.0.9", 8765, "manual"))
    seam = d.default_providers()
    assert d.discover_brain(seam).source == "manual"
    assert calls == ["mdns"]
