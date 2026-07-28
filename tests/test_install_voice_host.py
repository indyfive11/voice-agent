"""Voice-host role provisioner (§10d caller) — hermetic: no real socket, subprocess, or gabagent."""
from __future__ import annotations

import subprocess

import pytest

from voice_agent_install import voice_host as vh


def _cp(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# --- LAN-IP detection -------------------------------------------------------------------------------

def test_is_loopbackish():
    assert vh._is_loopbackish("127.0.0.1") and vh._is_loopbackish("") and vh._is_loopbackish("0.0.0.0")
    assert not vh._is_loopbackish("192.168.1.100")


def test_detect_lan_ip_parses_iproute_src():
    out = "1.1.1.1 via 192.168.1.1 dev eth0 src 192.168.1.100 uid 1000 \n    cache"
    got = vh.detect_lan_ip(route_run=lambda *a, **k: _cp(stdout=out))
    assert got == "192.168.1.100"


def test_detect_lan_ip_rejects_loopback_src_and_does_not_fall_to_socket(monkeypatch):
    # iproute yields loopback → filtered to None; socket fallback stubbed so the test stays hermetic.
    monkeypatch.setattr(vh, "_lan_ip_via_socket", lambda target: None)
    got = vh.detect_lan_ip(route_run=lambda *a, **k: _cp(stdout="1.1.1.1 dev lo src 127.0.0.1"))
    assert got is None


def test_detect_lan_ip_iproute_failure_falls_back_to_socket(monkeypatch):
    monkeypatch.setattr(vh, "_lan_ip_via_socket", lambda target: "10.0.0.7")
    got = vh.detect_lan_ip(route_run=lambda *a, **k: _cp(returncode=1))
    assert got == "10.0.0.7"


def test_detect_lan_ip_iproute_missing_binary_falls_back(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ip binary")
    monkeypatch.setattr(vh, "_lan_ip_via_socket", lambda target: "10.0.0.9")
    assert vh.detect_lan_ip(route_run=boom) == "10.0.0.9"


# --- seam resolution --------------------------------------------------------------------------------

def test_resolve_install_bin_prefers_env_override():
    got = vh.resolve_install_bin(which=lambda n: "/on/path", env={"GABAGENT_INSTALL_BIN": "/override/bin"})
    assert got == "/override/bin"


def test_resolve_install_bin_uses_path_when_no_override():
    got = vh.resolve_install_bin(which=lambda n: "/usr/bin/gabagent-install", env={})
    assert got == "/usr/bin/gabagent-install"


def test_resolve_install_bin_none_when_no_override_and_not_on_path():
    # No hardcoded dev-path fallback: off PATH with no override → None (portability SOP).
    assert vh.resolve_install_bin(which=lambda n: None, env={}) is None


# --- seam invocation argv ---------------------------------------------------------------------------

def test_enable_advertise_argv_with_room():
    seen = {}
    def run(argv, **k):
        seen["argv"] = argv
        return _cp()
    vh.enable_advertise("/x/gabagent-install", "192.168.1.5", "living-room", run=run)
    assert seen["argv"] == ["/x/gabagent-install", "--enable-voice-host",
                            "--host", "192.168.1.5", "--room-id", "living-room"]


def test_enable_advertise_argv_without_room():
    seen = {}
    vh.enable_advertise("/x/gabagent-install", "192.168.1.5",
                        run=lambda argv, **k: seen.setdefault("argv", argv) or _cp())
    assert "--room-id" not in seen["argv"]


# --- provision orchestration ------------------------------------------------------------------------

def test_provision_success_detected_ip():
    code, steps = vh.provision(detect=lambda: "192.168.1.50",
                               resolve=lambda: "/x/gabagent-install",
                               invoke=lambda b, ip, rid: _cp(returncode=0, stdout="wrote"))
    assert code == vh.EXIT_OK
    assert steps[-1].ok and "advertises" in steps[-1].detail and "192.168.1.50" in steps[-1].detail


def test_provision_host_override_labelled_and_passed():
    seen = {}
    def invoke(b, ip, rid):
        seen["ip"], seen["rid"] = ip, rid
        return _cp()
    code, steps = vh.provision(host="10.0.0.5", room_id="den",
                               detect=lambda: pytest.fail("detect must not run when --host given"),
                               resolve=lambda: "/x/bin", invoke=invoke)
    assert code == vh.EXIT_OK and seen == {"ip": "10.0.0.5", "rid": "den"}
    assert "from --host" in steps[0].detail


def test_provision_no_host_detected_is_usage_error():
    code, steps = vh.provision(detect=lambda: None, resolve=lambda: "/x/bin",
                               invoke=lambda *a: _cp())
    assert code == vh.EXIT_USAGE and not steps[-1].ok and "--host" in steps[-1].detail


def test_provision_loopback_override_refused_before_seam():
    code, steps = vh.provision(host="127.0.0.1",
                               resolve=lambda: pytest.fail("must not resolve seam on loopback"),
                               invoke=lambda *a: pytest.fail("must not invoke seam on loopback"))
    assert code == vh.EXIT_USAGE and "loopback" in steps[-1].detail


def test_provision_no_seam_found():
    code, steps = vh.provision(detect=lambda: "192.168.1.50", resolve=lambda: None,
                               invoke=lambda *a: pytest.fail("no seam to invoke"))
    assert code == vh.EXIT_NO_SEAM and "gabagent-install not found" in steps[-1].detail


def test_provision_seam_oserror_is_no_seam():
    def invoke(*a):
        raise OSError("permission denied")
    code, steps = vh.provision(detect=lambda: "192.168.1.50", resolve=lambda: "/x/bin", invoke=invoke)
    assert code == vh.EXIT_NO_SEAM


def test_provision_seam_loopback_refuse_surfaces_reason():
    code, steps = vh.provision(detect=lambda: "192.168.1.50", resolve=lambda: "/x/bin",
                               invoke=lambda *a: _cp(returncode=2, stderr="effective voice_host is loopback"))
    assert code == vh.EXIT_ENABLE_FAILED
    assert "exit 2" in steps[-1].detail and "loopback" in steps[-1].detail


def test_provision_seam_other_failure():
    code, steps = vh.provision(detect=lambda: "192.168.1.50", resolve=lambda: "/x/bin",
                               invoke=lambda *a: _cp(returncode=5, stderr="config write failed"))
    assert code == vh.EXIT_ENABLE_FAILED and "exit 5" in steps[-1].detail


# --- main() (hermetic via module-attr monkeypatch) --------------------------------------------------

def test_main_success(monkeypatch):
    monkeypatch.setattr(vh, "detect_lan_ip", lambda **k: "192.168.1.50")
    monkeypatch.setattr(vh, "resolve_install_bin", lambda **k: "/x/gabagent-install")
    monkeypatch.setattr(vh, "enable_advertise", lambda *a, **k: _cp(returncode=0))
    assert vh.main([]) == vh.EXIT_OK


def test_main_host_arg_and_failure_exit(monkeypatch):
    monkeypatch.setattr(vh, "resolve_install_bin", lambda **k: "/x/gabagent-install")
    monkeypatch.setattr(vh, "enable_advertise", lambda *a, **k: _cp(returncode=2, stderr="loopback"))
    # --host given → detect must not be consulted; failure propagates as a non-zero exit.
    monkeypatch.setattr(vh, "detect_lan_ip", lambda **k: pytest.fail("detect must not run"))
    assert vh.main(["--host", "192.168.1.9", "--room-id", "kitchen"]) == vh.EXIT_ENABLE_FAILED
