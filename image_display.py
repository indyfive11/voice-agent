"""ImageDisplaySink — render a brain `display` descriptor (a generated image) on the room's screen.

Consumer half of roadmap item ③ (voice-driven image generation). The brain (gabagent) runs the
`generate_image` tool, downloads the PNG, and enqueues a **display-only** poll item carrying an R1
descriptor::

    {"job_id": "img-<id>", "text": "",
     "display": {"path": "/abs/local.png", "url": "https://cdn.gab.ai/…/<id>.png",
                 "w": 1024, "h": 1024, "ttl_secs": 86400, "credits_used": 5, …}}

`BuilderPollClient` hands the `display` payload here; the turn's own spoken reply narrates it, so there is
nothing to say — this only puts the picture on screen.

Three things this owns (the VAC display lane; the wire is GA's):

  * **Locality routing (`:L`/`:R`), by ground truth, not topology config.** `path` is a file on the BRAIN
    host. When the brain is co-located with this room (EM / laptop loopback) that file exists locally and we
    render it directly (fast, no network). When the brain is remote (Pi satellite → EM brain) the path is an
    EM filesystem path that does NOT exist on the Pi, so we fall back to the **public CDN `url`** (mpv fetches
    it). The test is simply ``os.path.exists(path)`` — self-configuring on every host, zero per-room wiring.

  * **A BOUNDED display duration** — ``IMAGE_DISPLAY_SECS`` (default 30), NOT the descriptor's ``ttl_secs``
    (that is the brain's *local-file GC* retention, e.g. 86400s; showing an image for a day would be absurd,
    and on the Pi TV it must be time-boxed so the paused movie resumes).

  * **Media pause→show→resume**, via an INJECTED controller so the sink stays pure/testable and the
    Jellyfin-specific Pi logic lives in `JellyfinRoomController`. On a room whose media is actively playing,
    ``pause_if_playing()`` pauses it; we always ``resume()`` in a ``finally`` (walk-away safe).

Best-effort throughout: a headless box (no display), a missing mpv, an unreachable CDN, or a pause failure is
a logged no-op that still ACKs the item — an undisplayable image must never wedge the poll loop or the turn.
"""

from __future__ import annotations

import asyncio
import os
import shutil

from loguru import logger


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


class NullRoomController:
    """No media to pause (desktop rooms, or any room without a known player). Always a no-op."""

    async def pause_if_playing(self) -> bool:
        return False

    async def resume(self) -> None:
        return None


class JellyfinRoomController:
    """Pause/resume the room's Jellyfin playback via the Session API (roadmap ③, Pi living-room TV).

    The Pi's jellyfin-mpv-shim runs EMBEDDED libmpv (``mpv_ext: false``) — there is NO external mpv IPC
    socket to drive, so we pause at the Jellyfin layer instead: the shim registers as a controllable session
    while playing, and a ``POST /Sessions/{id}/Playing/Pause`` (Unpause) is relayed to its mpv. Jellyfin
    tracks the position, so resume is exact. This touches NOTHING in the working #62 movie path (no shim
    config change).

    All three knobs default empty → the controller is DISABLED (``NullRoomController`` semantics: never
    pauses), so any install without a configured Jellyfin room is byte-identical to before. Enable per-device
    (the Pi ``.env``) with ``JELLYFIN_URL`` / ``JELLYFIN_TOKEN`` / ``JELLYFIN_DEVICE`` (the shim's device
    name, e.g. ``bedroom-jellyfin``). Best-effort: any HTTP/parse failure → no pause (image still shows over
    whatever's on screen), never an error.
    """

    def __init__(self, *, url: str = "", token: str = "", device: str = "", http=None):
        self._url = (url or os.environ.get("JELLYFIN_URL", "")).rstrip("/")
        self._token = token or os.environ.get("JELLYFIN_TOKEN", "")
        self._device = device or os.environ.get("JELLYFIN_DEVICE", "")
        self._http = http  # injected httpx.AsyncClient-like for tests; lazy-made in prod
        self._paused_session: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._token and self._device)

    async def pause_if_playing(self) -> bool:
        """If this room's session is actively playing (present, has NowPlaying, not already paused), pause it.
        Returns True iff we issued a pause (so the caller resumes only what we paused)."""
        if not self.enabled:
            return False
        sess = await self._find_playing_session()
        if not sess:
            return False
        if await self._post(f"/Sessions/{sess}/Playing/Pause"):
            self._paused_session = sess
            return True
        return False

    async def resume(self) -> None:
        """Unpause the session we paused (if any). Idempotent; clears the pause marker regardless."""
        sess, self._paused_session = self._paused_session, None
        if sess:
            await self._post(f"/Sessions/{sess}/Playing/Unpause")

    async def _find_playing_session(self) -> str | None:
        """Session id for our device that is actively playing (NowPlayingItem present, not paused)."""
        client = self._client()
        try:
            r = await client.get(f"{self._url}/Sessions", params={"api_key": self._token})
            if r.status_code != 200:
                return None
            for s in r.json():
                if s.get("DeviceName") != self._device:
                    continue
                if not s.get("NowPlayingItem"):
                    return None  # our device is connected but nothing is playing → nothing to pause
                if (s.get("PlayState") or {}).get("IsPaused"):
                    return None  # already paused (e.g. user paused) → don't touch, don't resume later
                return s.get("Id")
        except Exception as e:  # noqa: BLE001 - a control probe must never break display
            logger.debug(f"IMAGE | Jellyfin /Sessions probe failed (ignored): {type(e).__name__}: {e}")
        return None

    async def _post(self, path: str) -> bool:
        client = self._client()
        try:
            r = await client.post(f"{self._url}{path}", params={"api_key": self._token})
            return r.status_code in (200, 204)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"IMAGE | Jellyfin POST {path} failed (ignored): {type(e).__name__}: {e}")
            return False

    def _client(self):
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        return self._http


class ImageDisplaySink:
    """Render display descriptors on the local room screen via mpv; ack the ones shown.

    Injectables (all defaulted for production; overridden in tests):
      * ``runner``     — async ``(src, secs) -> bool``; shows ``src`` for ``secs``, True on clean render.
      * ``controller`` — object with async ``pause_if_playing() -> bool`` and ``resume() -> None``.
      * ``display_secs`` / ``enabled`` — read from env when not passed.
    """

    def __init__(
        self,
        *,
        runner=None,
        controller=None,
        display_secs: int | None = None,
        enabled: bool | None = None,
    ):
        self._runner = runner or self._run_mpv
        self._controller = controller or NullRoomController()
        self._display_secs = (
            int(os.environ.get("IMAGE_DISPLAY_SECS", "30")) if display_secs is None else display_secs
        )
        # Default ENABLED — but a box with no display auto-skips (see _has_display). The env override is an
        # explicit off-switch; unset behaves as "show if this room has a screen".
        if enabled is None:
            enabled = os.environ.get("IMAGE_DISPLAY_ENABLED", "1") not in ("0", "false", "False", "")
        self._enabled = enabled
        # job_ids rendered (or terminally skipped) since the last drain — piggybacked as the poll ack so the
        # brain finalizes its liveness-leased delivery exactly like a spoken ring.
        self._displayed: list[str] = []

    # ---- poll-loop surface -------------------------------------------------

    def drain_displayed(self) -> list[str]:
        """job_ids fully handled since the last call (ack these). Clears the buffer."""
        out, self._displayed = self._displayed, []
        return out

    async def show(self, display: dict, *, job_id: str | None = None) -> None:
        """Render one descriptor. Never raises; always marks the job_id handled (even on a no-op path) so an
        undisplayable item is acked, not re-sent forever."""
        try:
            await self._show(display)
        except Exception as e:  # noqa: BLE001 - display must never break the poll loop or the turn
            _tlog(f"IMAGE | display FAILED (ignored): {type(e).__name__}: {e}")
        finally:
            if job_id is not None:
                self._displayed.append(job_id)

    async def _show(self, display: dict) -> None:
        if not self._enabled:
            _tlog("IMAGE | display disabled (IMAGE_DISPLAY_ENABLED=0) — skipped")
            return
        if not self._has_display():
            _tlog("IMAGE | no display on this host (no WAYLAND_DISPLAY/DISPLAY) — skipped")
            return
        src = self._resolve_source(display)
        if not src:
            _tlog(f"IMAGE | no renderable source (path absent + no url) — skipped: {display!r}")
            return

        paused = False
        try:
            paused = await self._controller.pause_if_playing()
            if paused:
                _tlog("IMAGE | room media paused for image display")
            ok = await self._runner(src, self._display_secs)
            _tlog(
                f"IMAGE | displayed {'ok' if ok else 'FAILED'} — {src} "
                f"({self._display_secs}s{', paused media' if paused else ''})"
            )
        finally:
            if paused:
                await self._controller.resume()
                _tlog("IMAGE | room media resumed")

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _has_display() -> bool:
        """Ground-truth: does this host have a display we could render on?"""
        return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))

    @staticmethod
    def _resolve_source(display: dict) -> str | None:
        """Locality routing: local file if it EXISTS here (:L, co-located brain), else the public CDN url
        (:R, remote brain). Returns None when neither is usable."""
        path = display.get("path")
        if path and os.path.exists(path):
            return path
        url = display.get("url")
        if url:
            return url
        return None

    async def _run_mpv(self, src: str, secs: int) -> bool:
        """Show a still image fullscreen for ``secs`` via mpv (proven on the Pi labwc TV). Returns True on a
        clean exit. mpv fetches ``src`` whether it is a local path or an https URL."""
        mpv = os.environ.get("IMAGE_DISPLAY_MPV", "mpv")
        if not shutil.which(mpv):
            _tlog(f"IMAGE | mpv binary '{mpv}' not found — cannot display")
            return False
        fullscreen = os.environ.get("IMAGE_DISPLAY_FULLSCREEN", "1") not in ("0", "false", "False", "")
        args = [
            mpv, "--no-config", "--vo=gpu", "--no-osc", "--no-input-default-bindings",
            "--force-window=immediate", "--keep-open=no", "--really-quiet", "--no-terminal",
            f"--image-display-duration={secs}",
        ]
        if fullscreen:
            args.append("--fullscreen")
        args.append(src)
        # Optional display-env overrides (a per-device .env sets these when the launch env lacks them — e.g.
        # the Pi aria-voice unit). Default: inherit the process env unchanged.
        env = os.environ.copy()
        for k in ("WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "DISPLAY"):
            v = os.environ.get(f"IMAGE_DISPLAY_{k}")
            if v:
                env[k] = v
        try:
            p = await asyncio.create_subprocess_exec(
                *args, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            # Bound the wait so a wedged mpv can't hang the sink; grace over the display duration.
            await asyncio.wait_for(p.wait(), timeout=secs + 10)
            return p.returncode == 0
        except asyncio.TimeoutError:
            try:
                p.kill()
            except ProcessLookupError:
                pass
            _tlog("IMAGE | mpv exceeded display window — killed")
            return False
        except Exception as e:  # noqa: BLE001
            _tlog(f"IMAGE | mpv spawn failed: {type(e).__name__}: {e}")
            return False
