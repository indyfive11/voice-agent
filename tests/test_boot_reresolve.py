"""Tests for the boot re-resolve CALL SITE (config.reresolve_brain_host).

The primitive's own policy is covered in test_install_discovery.py. What is tested HERE is the thing
that file cannot see: that the call site is off by default, fail-soft on every path, hard-bounded, and
that its result actually reaches the CONSUMER. A test asserting only "os.environ was set" is vacuous —
it passes even if the write is dropped when the value happens to be unchanged, and it says nothing
about whether anything downstream reads it. So the load-bearing test asserts the brain client's
base_url, i.e. the write->read handoff itself.

Every test sets BRAIN/GAB_* explicitly: config.py calls load_dotenv(override=True) at IMPORT, so a
developer's .env would otherwise leak in and silently flip the early-return branches.
"""
from __future__ import annotations

import asyncio

import pytest

import config
from voice_agent_install import discovery as d


@pytest.fixture(autouse=True)
def _clean_brain_env(monkeypatch):
    """A known-good remote-satellite shape; individual tests narrow it."""
    monkeypatch.setenv("BRAIN", "gabagent")
    monkeypatch.setenv("BRAIN_REDISCOVER", "1")
    monkeypatch.setenv("GAB_HOST", "10.0.0.5")
    monkeypatch.setenv("GAB_PORT", "8765")
    for k in ("STT_REMOTE_URL", "TTS_REMOTE_URL", "BRAIN_RERESOLVE_BUDGET_SECS"):
        monkeypatch.delenv(k, raising=False)


def _stub_reresolve(monkeypatch, result):
    """Replace the primitive; returns a list that records the (host, port) it was called with."""
    calls: list[tuple[str, int]] = []

    async def _fake(written_host, port, **kw):
        calls.append((written_host, port))
        return result

    monkeypatch.setattr(d, "reresolve_brain_host_async", _fake)
    return calls


# --------------------------------------------------------------------------- the off-switch
def test_off_by_default_is_a_true_no_op(monkeypatch):
    # The portability SOP's "safe universal default = the historical no-op": an install that never
    # asked for discovery must not even probe. BRAIN_REDISCOVER unset == off.
    monkeypatch.delenv("BRAIN_REDISCOVER", raising=False)

    async def _must_not_run(*a, **kw):
        raise AssertionError("re-resolve must not run when BRAIN_REDISCOVER is unset")

    monkeypatch.setattr(d, "reresolve_brain_host_async", _must_not_run)
    asyncio.run(config.reresolve_brain_host())
    assert config._env("GAB_HOST") == "10.0.0.5"


@pytest.mark.parametrize("off", ["0", "false", "False", "no"])
def test_explicit_off_values(monkeypatch, off):
    monkeypatch.setenv("BRAIN_REDISCOVER", off)

    async def _must_not_run(*a, **kw):
        raise AssertionError("re-resolve must not run when explicitly disabled")

    monkeypatch.setattr(d, "reresolve_brain_host_async", _must_not_run)
    asyncio.run(config.reresolve_brain_host())


def test_local_brain_never_re_resolves(monkeypatch):
    monkeypatch.setenv("BRAIN", "local")

    async def _must_not_run(*a, **kw):
        raise AssertionError("a local LLM brain has no host to find")

    monkeypatch.setattr(d, "reresolve_brain_host_async", _must_not_run)
    asyncio.run(config.reresolve_brain_host())


# --------------------------------------------------------------------------- fail-soft
def test_import_error_is_survivable(monkeypatch):
    """THE crash-loop guard. main() catches only KeyboardInterrupt and the unit is Restart=on-failure
    with StartLimitBurst=5, so an escaping ImportError on a partially-synced satellite would leave
    voice-agent permanently dead. It must degrade to "keep the written host"."""
    import builtins

    real_import = builtins.__import__

    def _no_discovery(name, *a, **kw):
        if name == "voice_agent_install.discovery":
            raise ImportError("module not synced")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_discovery)
    asyncio.run(config.reresolve_brain_host())  # must not raise
    assert config._env("GAB_HOST") == "10.0.0.5"


def test_primitive_exception_is_survivable(monkeypatch):
    async def _boom(*a, **kw):
        raise RuntimeError("discovery exploded")

    monkeypatch.setattr(d, "reresolve_brain_host_async", _boom)
    asyncio.run(config.reresolve_brain_host())
    assert config._env("GAB_HOST") == "10.0.0.5"


def test_overrunning_primitive_is_bounded_and_keeps_written_host(monkeypatch):
    """The outer deadline is the only HARD bound: the callee's internal timeouts are promises
    (urlopen's timeout excludes DNS; to_thread cannot be cancelled)."""
    monkeypatch.setenv("BRAIN_RERESOLVE_BUDGET_SECS", "0.05")

    async def _hangs(*a, **kw):
        await asyncio.sleep(30)
        return ("10.9.9.9", "rediscovered")

    monkeypatch.setattr(d, "reresolve_brain_host_async", _hangs)
    asyncio.run(config.reresolve_brain_host())
    assert config._env("GAB_HOST") == "10.0.0.5"  # never adopted a value we didn't get in time


def test_non_integer_port_disables_quietly(monkeypatch):
    monkeypatch.setenv("GAB_PORT", "not-a-port")

    async def _must_not_run(*a, **kw):
        raise AssertionError("must not run with an unparseable port")

    monkeypatch.setattr(d, "reresolve_brain_host_async", _must_not_run)
    asyncio.run(config.reresolve_brain_host())


# --------------------------------------------------------------------------- the write->read handoff
def test_adopted_host_reaches_the_brain_client_base_url(monkeypatch):
    """The test that actually bites. Asserting os.environ is vacuous — this asserts the CONSUMER sees
    it, so it fails if the write is dropped OR if a second GAB_HOST reader is ever added upstream of
    the call site in main.run()."""
    _stub_reresolve(monkeypatch, ("10.9.9.9", "rediscovered"))
    asyncio.run(config.reresolve_brain_host())

    llm = config.build_llm()
    assert llm.brain_client._base_url == "http://10.9.9.9:8765"


def test_kept_host_leaves_the_brain_client_pointed_at_the_written_host(monkeypatch):
    _stub_reresolve(monkeypatch, ("10.0.0.5", "written-kept(no-adverts)"))
    asyncio.run(config.reresolve_brain_host())

    llm = config.build_llm()
    assert llm.brain_client._base_url == "http://10.0.0.5:8765"


def test_primitive_is_called_with_the_written_host_and_port(monkeypatch):
    calls = _stub_reresolve(monkeypatch, ("10.0.0.5", "written-reachable"))
    asyncio.run(config.reresolve_brain_host())
    assert calls == [("10.0.0.5", 8765)]


# --------------------------------------------------------------------------- STT/TTS follow-the-brain
def test_remote_stt_tts_follow_the_brain_to_its_new_host(monkeypatch):
    """Without this the flagship Pi re-finds its brain and stays deaf and mute: STT/TTS live on the
    same box and their URLs are absolute + pinned."""
    monkeypatch.setenv("STT_REMOTE_URL", "http://10.0.0.5:8770")
    monkeypatch.setenv("TTS_REMOTE_URL", "http://10.0.0.5:8771")
    _stub_reresolve(monkeypatch, ("10.9.9.9", "rediscovered"))

    asyncio.run(config.reresolve_brain_host())

    assert config._env("STT_REMOTE_URL") == "http://10.9.9.9:8770"
    assert config._env("TTS_REMOTE_URL") == "http://10.9.9.9:8771"


def test_services_on_a_third_box_are_never_blind_repointed(monkeypatch):
    # STT lives elsewhere than the brain — following the brain would break a working service.
    monkeypatch.setenv("STT_REMOTE_URL", "http://10.0.0.77:8770")
    monkeypatch.setenv("TTS_REMOTE_URL", "http://10.0.0.5:8771")
    _stub_reresolve(monkeypatch, ("10.9.9.9", "rediscovered"))

    asyncio.run(config.reresolve_brain_host())

    assert config._env("STT_REMOTE_URL") == "http://10.0.0.77:8770"  # untouched
    assert config._env("TTS_REMOTE_URL") == "http://10.9.9.9:8771"   # followed


def test_no_repoint_when_the_host_is_kept(monkeypatch):
    monkeypatch.setenv("STT_REMOTE_URL", "http://10.0.0.5:8770")
    _stub_reresolve(monkeypatch, ("10.0.0.5", "written-kept(no-adverts)"))

    asyncio.run(config.reresolve_brain_host())

    assert config._env("STT_REMOTE_URL") == "http://10.0.0.5:8770"


def test_malformed_service_url_does_not_break_startup(monkeypatch):
    monkeypatch.setenv("STT_REMOTE_URL", "::: not a url :::")
    _stub_reresolve(monkeypatch, ("10.9.9.9", "rediscovered"))

    asyncio.run(config.reresolve_brain_host())  # must not raise

    assert config._env("GAB_HOST") == "10.9.9.9"


# --------------------------------------------------------------------------- URL rewriting
@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://10.0.0.5:8770", "http://10.0.0.5x:8770"),
        ("http://10.0.0.5", "http://10.0.0.5x"),
        ("http://10.0.0.5:8770/v1/audio", "http://10.0.0.5x:8770/v1/audio"),
        ("https://10.0.0.5:8443/x?y=1", "https://10.0.0.5x:8443/x?y=1"),
    ],
)
def test_swap_url_host_preserves_everything_else(url, expected):
    assert config._swap_url_host(url, "10.0.0.5x") == expected
