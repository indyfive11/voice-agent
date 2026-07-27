"""Secret/token engine (A.5) — SECRETS layer. Pure stdlib.

Mints and reconciles the shared secrets a provisioned box needs. Two functions, TWO AUDIENCES — and the
split is the whole point (audit hole H4):

  - ensure_tokens()  — the AUTHORITY side. Reuse a present value, else MINT one. Correct only where THIS
                       box is the source of truth for the secret (the brain minting GAB_AUTH_TOKEN; an
                       STT/TTS server minting its own token).
  - require_tokens() — the CONSUMER / satellite side. Reuse a present value, else FAIL LOUD — never mint.
                       A satellite's tokens are shared secrets it must RECEIVE from the server it talks to;
                       a freshly-minted one is a token the server has never heard of → a silent 401 on
                       exactly the satellite topology. Fail at provision time, not at first request.

Idempotent by construction: a token that already works is never rotated (rotating it silently breaks the
paired peer that still holds the old one). Secrets are RETURNED as data; the caller writes them (via the
A.4 0600 `.env` writer) — this module never touches the filesystem and never templates a secret.
"""
from __future__ import annotations

import secrets as _secrets

__all__ = ["generate_token", "ensure_tokens", "require_tokens", "MissingTokens"]

# 128 bits is the floor for a real bearer secret; below it the "token" is guessable, not a secret.
_MIN_NBYTES = 16


class MissingTokens(Exception):
    """Raised by require_tokens when a consumer-side token is absent — fail-loud beats a silently
    invented secret the paired server never heard of (the satellite auth-failure class, hole H4)."""

    def __init__(self, names: list[str]) -> None:
        self.names = list(names)
        super().__init__("missing required token(s): " + ", ".join(self.names))


def _nonempty(v: object) -> bool:
    """A usable existing secret is a non-blank string. None / "" / whitespace ⇒ treat as absent."""
    return isinstance(v, str) and v.strip() != ""


def generate_token(nbytes: int = 32) -> str:
    """A fresh URL-safe secret (secrets.token_urlsafe). URL-safe [A-Za-z0-9_-] so it drops straight into
    an .env value AND an HTTP bearer header with zero escaping. nbytes is the entropy source width, not
    the string length (the result is ~1.3× longer, base64url)."""
    if nbytes < _MIN_NBYTES:
        raise ValueError(f"nbytes must be >= {_MIN_NBYTES} (128 bits) — shorter is not a real secret")
    return _secrets.token_urlsafe(nbytes)


def ensure_tokens(existing: dict[str, str | None], names: list[str], *, nbytes: int = 32) -> dict[str, str]:
    """AUTHORITY side. For each name: reuse a present non-empty value, else MINT one. Returns the full
    name→value set. IDEMPOTENT — re-running never rotates a token that already works.

    Mint-when-absent is correct ONLY where this box owns the secret (the server/authority). On a
    CONSUMER / satellite, use require_tokens — minting a shared secret the server never heard of is
    broken-on-arrival (hole H4)."""
    return {name: (existing.get(name) if _nonempty(existing.get(name)) else generate_token(nbytes))  # type: ignore[misc]
            for name in names}


def require_tokens(existing: dict[str, str | None], names: list[str]) -> dict[str, str]:
    """CONSUMER / satellite side. Return name→value for `names`, or raise MissingTokens if ANY is
    absent/empty — NEVER mints. A satellite's tokens (GAB_AUTH_TOKEN↔brain, STT/TTS_REMOTE_TOKEN↔the
    remote STT/TTS) are shared secrets it must RECEIVE; inventing one yields a token the server rejects,
    a silent auth failure on exactly this topology. Fail loud at provision time instead (hole H4)."""
    missing = [n for n in names if not _nonempty(existing.get(n))]
    if missing:
        raise MissingTokens(missing)
    return {n: existing[n] for n in names}  # type: ignore[misc]  # _nonempty above guarantees str
