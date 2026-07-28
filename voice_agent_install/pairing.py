"""The satellite-side pairing client — obtain the brain's auth token over the wire (3c).

INSTALL-TIME ONLY, like the rest of ``voice_agent_install``. Nothing in the running application imports
this; it runs once, inside :func:`voice_agent_install.satellite.gather`, to replace the hand-typed brain
token with one the brain hands out during a short, human-authorized window.

WHAT THIS IS (and is NOT). This is the CLIENT half of the agnostic ``POST /pair`` contract that the brain
documents (ownership of the wire spec is brain-side; this file conforms to it, it does not define it). The
satellite is **fully headless** by design — every human action happens on the BRAIN console
(``/pairvoiceagent`` opens a window and accepts a specific candidate). This client only ever POSTs and
polls; it never prompts, needs a TTY, or reads a keystroke.

THE STATE MACHINE (locked GA↔VAC, 2026-07-27). Two lifecycles, deliberately split:

- **WINDOW** (operator-facing, brain-side): opened by ``/pairvoiceagent``, closed on operator-accept or
  timeout. Not visible to this client except as response codes.
- **CLAIM** (satellite-facing, this file): the accepted candidate becomes a retrievable *claim* with its
  own short TTL, replay-gated on ``{bound peer-IP + claim_secret}``.

Wire flow this client drives:

1. ``POST /pair {client_id}`` (no secret yet) → ``202 {status:"pending", claim_secret}``. The brain
   binds the candidate to this box's peer-IP and mints ``claim_secret``.
2. The operator accepts *this* candidate on the brain console.
3. ``POST /pair {client_id, claim_secret}`` from the same peer-IP → ``200 {auth_token}``.

TWO DROPPED-RESPONSE HAZARDS, both handled by idempotency (cleartext HTTP loses responses, and this runs
mid-install on a satellite):

- A dropped **first ``202``** (carrying ``claim_secret``): a secretless re-POST ``{client_id}`` re-returns
  the SAME ``claim_secret`` — the candidate is idempotent on ``{client_id + peer-IP}`` and the secret is
  never rotated. So we simply keep polling until we hold a secret, then keep polling with it.
- A dropped **``200``** (carrying the token): retrieval is replay-until-TTL_claim gated on
  ``{peer-IP + claim_secret}`` — a retry within the claim's TTL re-returns the same token.

So the client's loop is uniform: **send ``client_id`` always; include ``claim_secret`` as soon as we have
one; a dropped response is just another poll.** A network error is status ``0`` → retry within the deadline.

WHY ``client_id`` IS 128-BIT RANDOM AND PERSISTED. It is the idempotency key; a guessable value (derived
from ``room_id``/hostname/MAC) would let an on-LAN party who knows the string collect the token during a
legitimate window, and ``room_id`` is already broadcast in mDNS. So it is ``secrets.token_hex(16)`` and
persisted to disk **before the first POST**, so a crash/re-invoke mid-install resumes the same candidate
rather than stranding an already-accepted one behind a fresh id. (``claim_secret`` is the server-generated
retrieval capability that guarantees entropy regardless — but the random ``client_id`` is the belt.)

CAPABILITY GATE — POST-and-interpret, not an mDNS flag. ``discovery`` drops all mDNS TXT but ``room_id``
and mDNS is often off, so the ``pairproto`` TXT hint can't be the gate. The POST status *is* the gate:
``404`` = brain predates pairing; ``501`` = pairing-capable brain with no auth configured (misconfigured —
a provisioned voice-host always holds a token); ``409``/``429``/``202`` = keep polling; ``200`` = done.
"""

from __future__ import annotations

import json
import secrets as _secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

__all__ = ["PairOutcome", "PairState", "pair_for_token", "pair_url",
           "DEFAULT_DEADLINE_S", "DEFAULT_POLL_INTERVAL_S", "HTTPResponse"]

# The whole ceremony's patience. Long enough for an operator to walk to the brain, run
# ``/pairvoiceagent``, and accept — but bounded, because a satellite that polls forever is a headless box
# wedged with no one watching. On expiry the client aborts and the caller writes/enables NOTHING.
DEFAULT_DEADLINE_S = 180.0
# Matches the brain's "re-POST after ~1s" guidance. The post-accept ``200`` lands on the first poll after
# the operator accepts, so this is the felt latency of the happy path, not a throttle.
DEFAULT_POLL_INTERVAL_S = 1.0
# Per-request network timeout. Cleartext HTTP on a LAN; a slow/lost response is expected and simply
# becomes another poll (status 0 → retry). Kept short so a black-holed request doesn't eat the deadline.
DEFAULT_REQUEST_TIMEOUT_S = 5.0


def pair_url(host: str, port: int) -> str:
    """The agnostic pairing endpoint. Always a routable host/IP we were handed by discovery — never a
    ``.local`` name (same rule as :func:`voice_agent_install.satellite.service_url`: a ``.local`` makes
    every connect eat a synchronous nss-mdns lookup)."""
    return f"http://{host}:{port}/pair"


@dataclass(frozen=True)
class HTTPResponse:
    """A POST result reduced to what the state machine reads: a status code and a parsed JSON body. The
    default transport never raises for a 4xx/5xx (it reads the status), and a transport error is
    ``status=0`` so the loop treats a dropped/failed request as a retryable poll rather than a crash."""

    status: int
    body: dict


@dataclass(frozen=True)
class PairOutcome:
    """The result of a pairing attempt. ``ok`` ⇒ ``token`` is set. Otherwise ``reason`` is a stable
    vocabulary token, ``detail`` is the specific instance, and ``remedy`` is operator-facing guidance the
    caller prints before it aborts (nothing is written/enabled on a non-ok outcome)."""

    ok: bool
    reason: str
    detail: str
    token: Optional[str] = None
    remedy: str = ""


# --------------------------------------------------------------------------- client_id / secret state
@dataclass
class PairState:
    """Durable pairing state for ONE install, persisted to a 0600 JSON file.

    Holds the 128-bit random ``client_id`` (the idempotency key, generated once and reused across a
    crash/re-run so an already-accepted candidate is never stranded behind a fresh id) and, transiently,
    the ``claim_secret`` (so a mid-install crash resumes the claim rather than re-registering). Stored in
    the install root beside ``.env`` — per-box by construction, and untracked so it is never rsynced onto
    another satellite's tree.
    """

    path: Path
    _cache: Optional[dict] = field(default=None, repr=False)

    @classmethod
    def for_root(cls, root: Path | str) -> "PairState":
        return cls(Path(root) / ".pairing.json")

    def _load(self) -> dict:
        if self._cache is None:
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self._cache, dict):
                    self._cache = {}
            except (FileNotFoundError, ValueError):
                self._cache = {}
        return self._cache

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache or {}), encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass  # a filesystem that can't chmod (e.g. a test tmpfs quirk) is not a provisioning failure

    def client_id(self) -> str:
        """The stable idempotency key — generated (128-bit random) and persisted the first time it is
        asked for, then returned unchanged for the life of this install state."""
        data = self._load()
        cid = data.get("client_id")
        if not cid:
            cid = _secrets.token_hex(16)  # 128-bit, NOT derived from room_id/hostname/MAC (the guess trap)
            data["client_id"] = cid
            self._save()
        return cid

    def claim_secret(self) -> Optional[str]:
        return (self._load().get("claim_secret")) or None

    def save_claim_secret(self, secret: str) -> None:
        data = self._load()
        if secret and data.get("claim_secret") != secret:
            data["claim_secret"] = secret
            self._save()

    def clear(self) -> None:
        """Called on success: the token is now in ``.env``; the ``client_id``/``claim_secret`` are spent,
        and ``claim_secret`` is a (short-lived) secret we do not leave on disk."""
        self._cache = {}
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- transport (default seam)
def _urllib_post(url: str, body: dict, *, timeout: float = DEFAULT_REQUEST_TIMEOUT_S) -> HTTPResponse:
    """Default transport: a stdlib JSON POST that NEVER raises for a 4xx/5xx (it reads the status off the
    error) and maps any transport failure to ``status=0`` (→ retryable). stdlib only, so this is usable at
    install time with no extra dependency, exactly like ``discovery.http_health_ok``."""
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})

    def _read(resp) -> dict:
        try:
            raw = resp.read().decode("utf-8") or "{}"
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed http scheme)
            status = getattr(resp, "status", None) or resp.getcode()
            return HTTPResponse(int(status), _read(resp))
    except urllib.error.HTTPError as e:
        # A 4xx/5xx is a real, meaningful response here (409/429/501/404), not a transport failure.
        return HTTPResponse(int(e.code), _read(e))
    except Exception:
        # Connection refused / reset / timeout / DNS — a dropped or failed request. Status 0 ⇒ the loop
        # retries within the deadline (this is the dropped-202/dropped-200 resilience on the client side).
        return HTTPResponse(0, {})


# --------------------------------------------------------------------------- the loop
def _remedy(reason: str, host: str, port: int) -> str:
    manual = ("Or omit pairing and provide the brain's auth token directly on an interactive run.")
    if reason == "pairing_unsupported":
        return (f"The brain at {host}:{port} supports pairing but has NO auth token configured, so it has "
                "nothing to hand out. On the BRAIN, provision its auth token (it should be minted at "
                "install and present before the brain starts), then re-run. " + manual)
    if reason == "pairing_unavailable":
        return (f"The brain at {host}:{port} has no /pair endpoint — it predates the pairing protocol. "
                "Upgrade the brain to a version that serves /pair, or type the token instead. " + manual)
    if reason == "timeout":
        return (f"No token was issued within the window. On the BRAIN, run `gab pairvoiceagent` to open a "
                f"pairing window and ACCEPT this box, then re-run — this client polls {host}:{port}/pair "
                "and needs a human to open+accept on the brain side. " + manual)
    if reason == "unknown_token_scheme":
        return ("The brain offered a token scheme this client does not understand; refusing to guess. "
                "Upgrade this satellite, or type the token. " + manual)
    if reason == "empty_token":
        return ("The brain returned success but no token — treat as a brain bug, do not store an empty "
                "token. " + manual)
    if reason == "bad_client_id":
        return (f"The brain at {host}:{port} rejected this box's pairing id as too weak. This should not "
                "happen with a fresh install; delete this box's .pairing.json and re-run to mint a new id. "
                + manual)
    if reason == "bad_claim_secret":
        return (f"The brain at {host}:{port} rejected the pairing claim — the window may have expired or "
                "this box's address changed mid-pairing. On the BRAIN, open a fresh window "
                "(`gab pairvoiceagent`) and re-run this install to re-pair. " + manual)
    return manual


def pair_for_token(
    host: str,
    port: int,
    *,
    room_id: Optional[str] = None,
    label: Optional[str] = None,
    state: PairState,
    deadline_s: float = DEFAULT_DEADLINE_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    post_fn: Callable[[str, dict], HTTPResponse] = _urllib_post,
    clock: Optional[Callable[[], float]] = None,
    sleep: Optional[Callable[[float], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
) -> PairOutcome:
    """Drive the CLAIM state machine to obtain the brain's auth token, or return a non-ok outcome.

    Bounded by ``deadline_s`` — on expiry it returns ``reason="timeout"`` and the caller aborts BEFORE
    writing or enabling anything (a headless box has no prompt floor, so the honest floor is "fail the
    install," not "wedge polling"). Pure of side effects apart from the injected ``post_fn``/``sleep`` and
    the ``state`` file; ``clock``/``sleep`` are injectable so tests run without wall-clock waits.

    Capability gate (POST-and-interpret): ``404`` → ``pairing_unavailable`` (brain too old, hard stop);
    ``501`` → ``pairing_unsupported`` (pairing-capable but unprovisioned brain — misconfigured, hard stop).
    ``409``/``429`` are retryable (window not open yet / brain busy); ``202`` captures/keeps the secret and
    polls; ``200`` returns the token. ``400`` (client_id rejected) and ``403`` (claim rejected) are the
    brain's TERMINAL validation errors — a retry sends the same value and gets the same code, so they hard
    stop rather than poll to the deadline. An unknown ``token_scheme`` or an empty ``200`` token is a hard
    stop (never guess a scheme; never store an empty token). Any other status / a network error (``0``) is
    a retryable poll — the server is idempotent, so a re-POST re-returns the same state.
    """
    import time as _time

    now = clock or _time.monotonic
    nap = sleep or _time.sleep
    say = on_status or (lambda _msg: None)

    client_id = state.client_id()          # generated + persisted BEFORE the first POST (crash-resume)
    claim_secret = state.claim_secret()    # may be set already if a prior run got this far then died

    url = pair_url(host, port)
    start = now()
    announced_wait = False

    while now() - start < deadline_s:
        body: dict = {"client_id": client_id}
        if label:
            body["label"] = label
        if room_id is not None:
            body["room_id"] = room_id
        if claim_secret:
            body["claim_secret"] = claim_secret

        resp = post_fn(url, body)
        st = resp.status

        if st == 200:
            scheme = resp.body.get("token_scheme")
            if scheme not in (None, "bearer"):     # closed enum — reject unknown rather than guess
                return PairOutcome(False, "unknown_token_scheme", str(scheme),
                                   remedy=_remedy("unknown_token_scheme", host, port))
            token = (resp.body.get("auth_token") or "").strip()
            if not token:
                return PairOutcome(False, "empty_token", "200 with no auth_token",
                                   remedy=_remedy("empty_token", host, port))
            state.clear()
            say("accepted — received the brain auth token")
            return PairOutcome(True, "paired", f"{host}:{port}", token=token)

        if st == 202:
            secret = resp.body.get("claim_secret")
            if secret and not claim_secret:
                claim_secret = str(secret)
                state.save_claim_secret(claim_secret)  # persist so a crash resumes the claim
                say("registered — waiting for the operator to accept on the brain console")
                announced_wait = True   # past "open a window" now; don't regress the message

            elif not announced_wait:
                say("waiting for the operator to open a pairing window on the brain (`gab pairvoiceagent`)")
                announced_wait = True

        elif st == 501:   # brain has no auth configured → nothing to hand out. Misconfig. Hard stop.
            return PairOutcome(False, "pairing_unsupported", f"{host}:{port} returned 501",
                               remedy=_remedy("pairing_unsupported", host, port))

        elif st == 404:   # no /pair route → brain predates pairing. Distinct from 409. Hard stop.
            return PairOutcome(False, "pairing_unavailable", f"{host}:{port} has no /pair route",
                               remedy=_remedy("pairing_unavailable", host, port))

        elif st == 400:   # brain rejected the client_id (below its entropy floor). Terminal — a retry
            # sends the same id and gets the same 400. Cannot happen with our 128-bit id, but fail fast.
            return PairOutcome(False, "bad_client_id", f"{host}:{port} rejected the client_id (400)",
                               remedy=_remedy("bad_client_id", host, port))

        elif st == 403:   # claim_secret rejected (wrong secret, or this box's IP no longer matches the
            # candidate — e.g. a DHCP renew mid-pair). Terminal per the contract; the remedy is to re-pair.
            return PairOutcome(False, "bad_claim_secret", f"{host}:{port} rejected the claim (403)",
                               remedy=_remedy("bad_claim_secret", host, port))

        elif st == 409:   # supported, no window open yet → retryable; the operator hasn't opened one.
            if not announced_wait:
                say("waiting for the operator to open a pairing window on the brain (`gab pairvoiceagent`)")
                announced_wait = True

        elif st == 429:   # another candidate mid-accept → back off and retry within the deadline.
            pass

        # st == 0 (network error / dropped response) and any other status: fall through → retry. The
        # server is idempotent, so a re-POST re-returns the same pending/secret/token state.

        nap(poll_interval_s)

    return PairOutcome(False, "timeout", f"no token issued within {int(deadline_s)}s",
                       remedy=_remedy("timeout", host, port))
