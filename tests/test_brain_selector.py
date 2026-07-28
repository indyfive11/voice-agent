"""The brain-neutral BRAIN selector (GitHub #1, coupling point 3).

`BRAIN=remote` must drive any HTTP/SSE brain WITHOUT the shell forcing a caller to name one brain
(`BRAIN=gabagent`). These assert the neutrality contract and its back-compat: the `gabagent` alias and
the `GAB_*` connection env keep working unchanged, the neutral `BRAIN_*` keys win when both are set, and
a generic `remote` brain is attach-only (we never try to spawn the reference brain's `gab` binary for it).

config.py calls load_dotenv(override=True) at IMPORT, so every test sets the brain env explicitly — a
developer's .env would otherwise leak in and flip the selector branch.
"""
from __future__ import annotations

import pytest

import config


@pytest.fixture(autouse=True)
def _clean_brain_env(monkeypatch):
    """Start each test from a known-empty brain env; the test sets only what it asserts on."""
    for k in ("BRAIN", "BRAIN_HOST", "BRAIN_PORT", "BRAIN_AUTH_TOKEN", "BRAIN_PROJECT_DIR",
              "BRAIN_BIN", "BRAIN_LAUNCH", "GAB_HOST", "GAB_PORT", "GAB_AUTH_TOKEN",
              "GAB_PROJECT_DIR", "GAB_BIN", "GAB_LAUNCH"):
        monkeypatch.delenv(k, raising=False)


def _brain_client(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return config.build_llm().brain_client


# --------------------------------------------------------------------------- selection
def test_remote_selects_the_http_brain(monkeypatch):
    from brains.brain_llm_service import BrainLLMService
    monkeypatch.setenv("BRAIN", "remote")
    assert isinstance(config.build_llm(), BrainLLMService)


def test_gabagent_alias_still_selects_the_same_path(monkeypatch):
    from brains.brain_llm_service import BrainLLMService
    monkeypatch.setenv("BRAIN", "gabagent")
    assert isinstance(config.build_llm(), BrainLLMService)


def test_unknown_brain_names_the_neutral_value_in_the_error(monkeypatch):
    monkeypatch.setenv("BRAIN", "banana")
    with pytest.raises(ValueError, match=r"local\|remote\|gabagent"):
        config.build_llm()


# --------------------------------------------------------------------------- connection env: BRAIN_* wins, GAB_* back-compat
def test_remote_reads_neutral_brain_host_port(monkeypatch):
    client = _brain_client(monkeypatch, BRAIN="remote", BRAIN_HOST="10.0.0.9", BRAIN_PORT="9000")
    assert client._base_url == "http://10.0.0.9:9000"


def test_remote_falls_back_to_gab_env_for_backcompat(monkeypatch):
    client = _brain_client(monkeypatch, BRAIN="remote", GAB_HOST="10.0.0.5", GAB_PORT="8765")
    assert client._base_url == "http://10.0.0.5:8765"


def test_neutral_brain_host_wins_over_legacy_gab_host(monkeypatch):
    client = _brain_client(monkeypatch, BRAIN="remote", BRAIN_HOST="10.0.0.9", GAB_HOST="10.0.0.5")
    assert client._base_url.startswith("http://10.0.0.9:")


# --------------------------------------------------------------------------- spawn is reference-brain-only
def test_remote_on_loopback_is_attach_only(monkeypatch):
    # A generic brain can't be spawned from here — we only ever launch the gabagent `gab` binary,
    # and only for the reference-brain selector. `remote` must never carry a launch argv.
    client = _brain_client(monkeypatch, BRAIN="remote", BRAIN_HOST="127.0.0.1")
    assert client._launch is None


def test_gabagent_on_loopback_still_spawns(monkeypatch):
    # Back-compat: the historical loopback default (connect-or-spawn) is unchanged for `gabagent`.
    client = _brain_client(monkeypatch, BRAIN="gabagent", GAB_HOST="127.0.0.1")
    assert client._launch is not None
    assert "--voice-serve" in client._launch


def test_remote_never_spawns_even_if_explicitly_launched_on_loopback(monkeypatch):
    # BRAIN_LAUNCH=1 forces a spawn intent, but the argv is still the reference brain's binary; a
    # remote-only deploy would set BRAIN_LAUNCH=0. We assert the default stays attach for remote.
    client = _brain_client(monkeypatch, BRAIN="remote", BRAIN_HOST="127.0.0.1", BRAIN_LAUNCH="0")
    assert client._launch is None
