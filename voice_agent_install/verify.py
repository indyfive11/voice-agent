"""Install-time verification of the three service legs — on AUTHENTICATED routes.

WHY NOT ``/health``. All three services expose an unauthenticated ``/health``:

    brain  gabagent/voice/server.py:57   short-circuits the bearer guard for path == "/health"
    stt    stt_service/server.py:224     web.get("/health", …) registered outside the guard
    tts    tts_service/server.py:292     same

So a ``/health`` probe answers ``200 {"status": "ok"}`` for a service whose token we got *wrong*.
The install then reports success and the failure surfaces at the first spoken utterance as a 401
inside the pipeline — which the operator experiences as "she just never answers." That is the
green-while-broken shape this whole increment exists to eliminate, so every leg is probed on a route
that actually requires the credential.

SECOND TRAP, and it is subtler. Both services treat an EMPTY token as "run open":

    stt_service/server.py:183-184   if not self._token: return True   # loopback / no-token mode: open
    tts_service/server.py:164-165   (identical)

which means an *absent* token in our config is indistinguishable from "that service is deliberately
open" — an authenticated probe passes either way. So the caller must decide the two cases apart
BEFORE probing (see :func:`classify_credential`), and a token that is missing when the service is
in fact guarded has to be an error, not a shrug.

Everything here is stdlib-only and bounded: this runs at install time on a box that may have no
project dependencies installed yet, and a hung probe during provisioning is indistinguishable to the
operator from a hung installer.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Optional

__all__ = ["LegResult", "classify_credential", "probe_brain", "probe_stt", "probe_tts", "silence_wav"]

PROBE_TIMEOUT = 5.0


@dataclass(frozen=True)
class LegResult:
    """Outcome of one leg's probe. ``ok`` is the only thing a caller should gate on; ``detail`` is
    written for a human staring at a failed install, so it names the remedy, not just the symptom."""

    leg: str
    ok: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{'OK  ' if self.ok else 'FAIL'} {self.leg}: {self.detail}"


def classify_credential(token: Optional[str], *, present_in_config: bool) -> str:
    """Tell "deliberately open" apart from "operator forgot the token".

    Returns ``open`` | ``guarded`` | ``missing``:

    - ``present_in_config`` False → ``missing``: the key was never supplied. We cannot prove the
      service is open, and assuming so is how a satellite ends up talking to a service it has no
      credential for. This is an ERROR the caller must surface, not a default.
    - key present and explicitly empty → ``open``: the operator asserted an unauthenticated service.
      Empty-but-present is a real, distinct statement in this file format (see ``envfile.parse``).
    - otherwise → ``guarded``.
    """
    if not present_in_config:
        return "missing"
    if (token or "").strip() == "":
        return "open"
    return "guarded"


def _request(url: str, *, method: str = "GET", token: Optional[str] = None,
             body: Optional[bytes] = None, content_type: Optional[str] = None,
             timeout: float = PROBE_TIMEOUT) -> tuple[int, bytes, dict]:
    """One bounded HTTP request. Returns ``(status, body, headers)``; raises on transport failure.

    A 401 is a RESULT, not an exception — it is the single most informative answer this module can
    get, and swallowing it into a generic failure would lose the distinction between "wrong token"
    and "service unreachable", which have completely different remedies.
    """
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method=method, data=body)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed http scheme)
            return int(getattr(resp, "status", 0) or resp.getcode()), resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return int(e.code), e.read() if hasattr(e, "read") else b"", dict(getattr(e, "headers", {}) or {})


def silence_wav(*, seconds: float = 0.2, rate: int = 16000) -> bytes:
    """A minimal mono 16-bit PCM WAV of silence, built by hand.

    Deliberately not numpy/soundfile: this runs before project dependencies exist. It is the smallest
    payload that exercises the STT service's real decode path rather than just its auth layer — a
    probe that only checks auth would miss a service that is authenticated but has no model loaded.
    """
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(data))
    return hdr + data


def probe_brain(host: str, port: int, token: Optional[str], *, room_id: str = "install-check",
                timeout: float = PROBE_TIMEOUT) -> LegResult:
    """Probe the brain on ``POST /attach`` — authenticated, and the same call the app makes at start.

    A purpose-built authenticated ping was considered and rejected (GA's point): it would be
    ``/attach`` with extra steps, and a probe that exercises a DIFFERENT code path than the app is a
    probe that can pass while the app fails.
    """
    url = f"http://{host}:{port}/attach"
    payload = json.dumps({"room_id": room_id, "capabilities": {}}).encode()
    try:
        status, body, _ = _request(url, method="POST", token=token, body=payload,
                                   content_type="application/json", timeout=timeout)
    except Exception as e:  # transport-level: unreachable, refused, DNS, timeout
        return LegResult("brain", False, f"unreachable at {host}:{port} ({type(e).__name__}) — "
                                         "is the brain running, and is this the right host?")
    if status == 401:
        return LegResult("brain", False, "401 unauthorized — GAB_AUTH_TOKEN does not match the "
                                         "brain's GABAI_VOICE_AUTH_TOKEN")
    if 200 <= status < 300:
        return LegResult("brain", True, f"authenticated at {host}:{port}")
    return LegResult("brain", False, f"HTTP {status} from /attach at {host}:{port}")


def probe_stt(url: str, token: Optional[str], *, timeout: float = PROBE_TIMEOUT) -> LegResult:
    """Probe STT on ``POST /stt`` with 0.2s of silence — authenticated AND decode-exercising."""
    endpoint = url.rstrip("/") + "/stt"
    try:
        status, body, _ = _request(endpoint, method="POST", token=token, body=silence_wav(),
                                   content_type="audio/wav", timeout=timeout)
    except Exception as e:
        return LegResult("stt", False, f"unreachable at {url} ({type(e).__name__}) — is aria-stt "
                                       "running on the brain host?")
    if status == 401:
        return LegResult("stt", False, "401 unauthorized — STT_REMOTE_TOKEN does not match the "
                                       "service's STT_SERVICE_AUTH_TOKEN")
    if 200 <= status < 300:
        return LegResult("stt", True, f"authenticated and decoding at {url}")
    return LegResult("stt", False, f"HTTP {status} from /stt at {url}")


def probe_tts(url: str, token: Optional[str], *, timeout: float = PROBE_TIMEOUT) -> LegResult:
    """Probe TTS on ``POST /tts/stream`` with one word.

    Checks for the ``X-Sample-Rate`` response header, because that is what ``remote_tts.py`` reads to
    drive playback: a service that answers 200 without it would pass a naive probe and then produce
    the wrong playback rate — the chipmunk failure wearing a different hat.
    """
    endpoint = url.rstrip("/") + "/tts/stream"
    payload = json.dumps({"text": "ready"}).encode()
    try:
        status, body, headers = _request(endpoint, method="POST", token=token, body=payload,
                                         content_type="application/json", timeout=timeout)
    except Exception as e:
        return LegResult("tts", False, f"unreachable at {url} ({type(e).__name__}) — is aria-tts "
                                       "running on the brain host?")
    if status == 401:
        return LegResult("tts", False, "401 unauthorized — TTS_REMOTE_TOKEN does not match the "
                                       "service's TTS_SERVICE_AUTH_TOKEN")
    if not (200 <= status < 300):
        return LegResult("tts", False, f"HTTP {status} from /tts/stream at {url}")
    rate = headers.get("X-Sample-Rate") or headers.get("x-sample-rate")
    if not rate:
        return LegResult("tts", False, f"200 from {url} but no X-Sample-Rate header — playback would "
                                       "run at the wrong rate; is this really the voice-agent TTS service?")
    return LegResult("tts", True, f"authenticated at {url}, emitting {rate} Hz")
