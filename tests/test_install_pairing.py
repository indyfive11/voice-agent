"""Tests for the satellite pairing client (3c) and its wiring into the provisioner.

Hermetic by construction: every side effect is injected. `post_fn` scripts the brain's responses, a fake
clock advances only when the loop sleeps, and `PairState` writes to a tmp dir. No socket, no wall-clock
wait, no live brain — the live end-to-end (`gab pairvoiceagent` → satellite pair) is a separate check that
needs the brain lane and is not what these prove. What they DO prove: the CLAIM state machine, both
dropped-response recoveries, the capability gate, the bounded-deadline abort, and that a pairing failure
writes/enables nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent_install import pairing
from voice_agent_install import satellite as sat
from voice_agent_install.pairing import HTTPResponse, PairState, pair_for_token


class FakeTime:
    """Time advances only when the loop sleeps — so `deadline_s` deterministically bounds poll count."""

    def __init__(self) -> None:
        self.t = 0.0

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def scripted(*responses: HTTPResponse):
    """A post_fn that returns each response in turn, then repeats the last forever. Records posted bodies."""
    bodies: list[dict] = []
    seq = list(responses)

    def post(url: str, body: dict) -> HTTPResponse:
        bodies.append(dict(body))
        return seq[len(bodies) - 1] if len(bodies) <= len(seq) else seq[-1]

    post.bodies = bodies  # type: ignore[attr-defined]
    return post


def _pair(tmp_path, post, *, deadline_s=30.0, **kw):
    ft = FakeTime()
    return pair_for_token("10.0.0.9", 8765, room_id="den", label="pi-den",
                          state=PairState.for_root(tmp_path), post_fn=post,
                          clock=ft.clock, sleep=ft.sleep, deadline_s=deadline_s, **kw), ft


# --- PairState: the durable idempotency key -------------------------------------------------------- #

def test_client_id_is_128_bit_random_and_persisted(tmp_path):
    st = PairState.for_root(tmp_path)
    cid = st.client_id()
    assert len(cid) == 32 and int(cid, 16) >= 0        # 16 bytes hex == 128-bit
    assert (tmp_path / ".pairing.json").exists()
    # A fresh instance (a re-run / crash-resume) reads back the SAME id — never mints a second candidate.
    assert PairState.for_root(tmp_path).client_id() == cid


def test_client_id_not_derived_from_anything_guessable(tmp_path):
    # Two boxes with the same room/host must still get different ids — it's random, not derived.
    a = PairState.for_root(tmp_path / "a").client_id()
    b = PairState.for_root(tmp_path / "b").client_id()
    assert a != b


def test_state_file_is_0600(tmp_path):
    PairState.for_root(tmp_path).client_id()
    assert oct((tmp_path / ".pairing.json").stat().st_mode & 0o777) == "0o600"


def test_claim_secret_persists_and_clear_removes_the_file(tmp_path):
    st = PairState.for_root(tmp_path)
    st.client_id()
    st.save_claim_secret("s3cr3t")
    assert PairState.for_root(tmp_path).claim_secret() == "s3cr3t"
    st.clear()
    assert not (tmp_path / ".pairing.json").exists()


# --- happy path + the two dropped-response recoveries --------------------------------------------- #

def test_happy_path_409_then_202_then_200(tmp_path):
    post = scripted(
        HTTPResponse(409, {"error": "no_pairing_window_open"}),          # operator hasn't opened yet
        HTTPResponse(202, {"status": "pending", "claim_secret": "K"}),   # window open, candidate registered
        HTTPResponse(202, {"status": "pending"}),                        # awaiting accept
        HTTPResponse(200, {"auth_token": "T0K", "token_scheme": "bearer"}),
    )
    (outcome, _ft) = _pair(tmp_path, post)
    assert outcome.ok and outcome.token == "T0K" and outcome.reason == "paired"
    # secretless until the secret is known, then always carried:
    assert "claim_secret" not in post.bodies[0] and "claim_secret" not in post.bodies[1]
    assert post.bodies[2]["claim_secret"] == "K" and post.bodies[3]["claim_secret"] == "K"
    # state cleared on success — no short-lived secret left on disk:
    assert not (tmp_path / ".pairing.json").exists()


def test_dropped_first_202_recovers_via_same_secret(tmp_path):
    # The 202 carrying claim_secret is dropped (network error, status 0). A secretless re-POST re-returns
    # the SAME secret (server idempotent on {client_id+peer-IP}), then accept issues the token.
    post = scripted(
        HTTPResponse(0, {}),                                             # first 202 dropped on the wire
        HTTPResponse(202, {"status": "pending", "claim_secret": "K"}),   # re-POST re-returns same secret
        HTTPResponse(200, {"auth_token": "T0K", "token_scheme": "bearer"}),
    )
    (outcome, _ft) = _pair(tmp_path, post)
    assert outcome.ok and outcome.token == "T0K"
    assert "claim_secret" not in post.bodies[0] and "claim_secret" not in post.bodies[1]


def test_dropped_200_recovers_within_ttl(tmp_path):
    # We hold the secret; the 200 is dropped; the retry (still carrying secret) re-returns the same token.
    post = scripted(
        HTTPResponse(202, {"status": "pending", "claim_secret": "K"}),
        HTTPResponse(0, {}),                                             # 200 dropped
        HTTPResponse(200, {"auth_token": "T0K", "token_scheme": "bearer"}),
    )
    (outcome, _ft) = _pair(tmp_path, post)
    assert outcome.ok and outcome.token == "T0K"
    assert post.bodies[1]["claim_secret"] == "K" and post.bodies[2]["claim_secret"] == "K"


def test_crash_resume_reuses_persisted_client_id_and_secret(tmp_path):
    # First run gets the secret then "crashes" (we stop). A second run must reuse both the id and secret.
    st1 = PairState.for_root(tmp_path)
    cid = st1.client_id()
    st1.save_claim_secret("K")
    post = scripted(HTTPResponse(200, {"auth_token": "T0K", "token_scheme": "bearer"}))
    ft = FakeTime()
    outcome = pair_for_token("10.0.0.9", 8765, state=PairState.for_root(tmp_path), post_fn=post,
                             clock=ft.clock, sleep=ft.sleep)
    assert outcome.ok
    assert post.bodies[0]["client_id"] == cid and post.bodies[0]["claim_secret"] == "K"


# --- capability gate + hard-stop cases ------------------------------------------------------------ #

def test_501_is_pairing_unsupported_and_does_not_retry(tmp_path):
    post = scripted(HTTPResponse(501, {"error": "pairing_unsupported"}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert not outcome.ok and outcome.reason == "pairing_unsupported"
    assert len(post.bodies) == 1        # hard stop — no polling


def test_404_is_pairing_unavailable_distinct_from_409(tmp_path):
    post = scripted(HTTPResponse(404, {}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert not outcome.ok and outcome.reason == "pairing_unavailable"
    assert len(post.bodies) == 1


def test_400_bad_client_id_is_terminal_not_retried(tmp_path):
    # The brain rejects the id below its entropy floor; a retry sends the same id → same 400. Fail fast.
    post = scripted(HTTPResponse(400, {"error": "bad_client_id"}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert not outcome.ok and outcome.reason == "bad_client_id"
    assert len(post.bodies) == 1


def test_403_bad_claim_secret_is_terminal_not_retried(tmp_path):
    post = scripted(HTTPResponse(403, {"error": "bad_claim_secret"}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert not outcome.ok and outcome.reason == "bad_claim_secret"
    assert len(post.bodies) == 1


def test_unknown_token_scheme_is_refused_not_guessed(tmp_path):
    post = scripted(HTTPResponse(200, {"auth_token": "T0K", "token_scheme": "macaroon"}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert not outcome.ok and outcome.reason == "unknown_token_scheme"


def test_empty_200_token_is_never_stored(tmp_path):
    post = scripted(HTTPResponse(200, {"auth_token": "", "token_scheme": "bearer"}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert not outcome.ok and outcome.reason == "empty_token"


def test_missing_token_scheme_defaults_to_bearer(tmp_path):
    # token_scheme absent (older-but-present brain) is treated as bearer, not an unknown-scheme refusal.
    post = scripted(HTTPResponse(200, {"auth_token": "T0K"}))
    (outcome, _ft) = _pair(tmp_path, post)
    assert outcome.ok and outcome.token == "T0K"


def test_timeout_is_bounded_and_aborts(tmp_path):
    # The window never opens: 409 forever. The deadline must stop it (no infinite headless poll).
    post = scripted(HTTPResponse(409, {"error": "no_pairing_window_open"}))
    (outcome, ft) = _pair(tmp_path, post, deadline_s=5.0)
    assert not outcome.ok and outcome.reason == "timeout"
    assert ft.t >= 5.0 and len(post.bodies) <= 6      # ~5 polls at 1s, then the deadline check fails


def test_429_busy_is_retried_then_succeeds(tmp_path):
    post = scripted(
        HTTPResponse(429, {"error": "pairing_busy"}),
        HTTPResponse(202, {"status": "pending", "claim_secret": "K"}),
        HTTPResponse(200, {"auth_token": "T0K", "token_scheme": "bearer"}),
    )
    (outcome, _ft) = _pair(tmp_path, post)
    assert outcome.ok and outcome.token == "T0K"


# --- provisioner wiring: brain token from pairing, and abort-before-apply on failure -------------- #

def _endpoint(host="10.0.0.9", port=8765):
    from voice_agent_install import discovery
    return lambda: discovery.BrainEndpoint(host, port, "test")


class _PairConsole(sat.Console):
    """Scripted console for a --pair run: no brain-token secret is prompted (pairing supplies it), so the
    secret list carries only the STT + TTS tokens."""

    def __init__(self, answers, secrets=("stok", "ttok")):
        self._answers = list(answers)
        self._secrets = list(secrets)
        self.lines: list[str] = []
        super().__init__(input_fn=lambda p="": self._answers.pop(0) if self._answers else "",
                         secret_fn=lambda p: self._secrets.pop(0) if self._secrets else "",
                         out=self.lines.append, interactive=True)


PAIR_ANSWERS = ["den", "mic-name", "spk-name", "aria", "y"]   # room, in, out, wake, confirm


def _probes_ok(monkeypatch):
    for leg in ("brain", "stt", "tts"):
        monkeypatch.setattr(sat.verify, f"probe_{leg}",
                            lambda *a, **k: sat.verify.LegResult("x", True, "ok"))


def test_provision_pair_writes_brain_token_from_handshake(tmp_path, monkeypatch):
    _probes_ok(monkeypatch)
    ok = lambda *a, **k: pairing.PairOutcome(True, "paired", "x", token="WIRE-TOKEN")
    console = _PairConsole(PAIR_ANSWERS)
    code, steps = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),),
                                pair=True, pair_fn=ok, run=lambda a: __import__("subprocess").CompletedProcess(a, 0),
                                unit_dir=tmp_path / "systemd", proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_OK, [str(s) for s in steps]
    env = (tmp_path / ".env").read_text()
    assert "GAB_AUTH_TOKEN=WIRE-TOKEN" in env


def test_provision_pair_failure_aborts_before_apply(tmp_path, monkeypatch):
    # A pairing failure must write and enable NOTHING — the headless-safe honest floor.
    fail = lambda *a, **k: pairing.PairOutcome(False, "timeout", "no token", remedy="open a window")
    calls = []
    console = _PairConsole(PAIR_ANSWERS)
    code, steps = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),),
                                pair=True, pair_fn=fail,
                                run=lambda a: calls.append(a) or __import__("subprocess").CompletedProcess(a, 0),
                                unit_dir=tmp_path / "systemd", proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_PROVISION_FAILED
    assert not (tmp_path / ".env").exists()                 # nothing written
    assert not (tmp_path / "systemd").exists()              # no unit
    assert calls == []                                      # nothing enabled/lingered
    assert any("pairing failed" in s.detail for s in steps)
