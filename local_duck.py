"""Pi-side media-duck belt — attenuate the LOCAL media sink-input directly via PipeWire/pactl.

The brain ducks media over Mopidy's RPC software mixer, but on the Pi satellite that mixer can fail to
attenuate the real HDMI output (2026-06-23 drive: a duck fired, music stayed loud). The brain's own
`pactl` sink path is HOST-LOCAL and no-ops for a remote Pi room (the #62 design), and the media plays on
the Pi's OWN sink — so only the satellite can lower the real output node. This ducks the media PipeWire
sink-input directly on duck-on and restores it on duck-off.

Why this is robust regardless of the "decoupled vs phantom-zero mixer" question: with Mopidy
``mixer = software`` the volume is applied INSIDE the GStreamer pipeline (the samples are scaled) and the
stream still hits PipeWire at node-volume 100%. Lowering that **node** volume attenuates the real output
downstream of Mopidy's internal mixer — so it works whether or not the software mixer is reliable. Mirror
of the EM brain's ``_duck_tidal_sink``, run Pi-local.

Identifies the media stream by ``application.name`` / ``node.name`` match (default ``mopidy``); the
specific match keeps it off the assistant's own TTS output (a different app/node). Best-effort: any pactl
failure or a missing stream is a logged no-op, never an error that could break audio.

Env (all default to the historical no-op — DISABLED — per the no-hardcodes SOP, so EM / any other install
is byte-identical until the knob is set; enabled per-device, e.g. the Pi ``.env``):
  ``MEDIA_DUCK_LOCAL``         ``1`` to enable (default off).
  ``MEDIA_DUCK_LOCAL_PCT``     duck-to percentage (default 18, mirrors the brain's ``_DUCK_VOLUME``).
  ``MEDIA_DUCK_LOCAL_MATCH``   comma-separated case-insensitive substrings matched vs application.name/node.name
                               (default ``mopidy``; e.g. ``mopidy,mpv`` to also duck a Jellyfin/mpv video stream).
  ``MEDIA_DUCK_LOCAL_RESTORE`` baseline % to restore to when the read prior is a stranded duck artifact (default 100).

Two stranding bugs this guards against (2026-06-23 drive: "stuck on mute after switching songs"):
  * **Value stranding.** ``pactl set-sink-input-volume`` is persisted per-app by PipeWire's
    ``module-stream-restore``; on a song switch Mopidy rebuilds the stream and stream-restore REPLAYS the
    last (ducked, 18%) level. If the belt then reads 18 as the "prior", it restores to 18 → music stuck
    quiet. Guard: never save a prior <= duck_pct — fall back to the baseline so the belt re-asserts a real
    level and self-heals (its restore-to-100 is then what stream-restore remembers).
  * **Index change.** A song switch can spawn a NEW sink-input index, so restore must RE-FIND the current
    stream by name, not target the cached duck-time index (which is dead).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil

from loguru import logger


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0") not in ("0", "false", "False", "")


class LocalSinkDucker:
    """Duck/restore the LOCAL media sink-input's PipeWire node volume (Pi-side belt for the duck window)."""

    def __init__(self, *, enabled: bool | None = None, duck_pct: int | None = None,
                 match: str | None = None, restore_level: int | None = None, runner=None):
        self._enabled = _env_on("MEDIA_DUCK_LOCAL") if enabled is None else enabled
        self._duck_pct = int(os.environ.get("MEDIA_DUCK_LOCAL_PCT", "18")) if duck_pct is None else duck_pct
        # Comma-separated list of app/node-name substrings (so ONE belt ducks the music player AND a video
        # player, e.g. "mopidy,mpv" for #62 Jellyfin-on-Pi). A single token = the historical behavior; an
        # empty value matches NOTHING (safe — never duck every stream, incl. our own TTS).
        _raw = match if match is not None else os.environ.get("MEDIA_DUCK_LOCAL_MATCH", "mopidy")
        self._matches = [t.strip().lower() for t in _raw.split(",") if t.strip()]
        self._restore_level = (
            int(os.environ.get("MEDIA_DUCK_LOCAL_RESTORE", "100")) if restore_level is None else restore_level
        )
        self._runner = runner or self._run_pactl
        # A single GUARDED pre-duck level (> duck_pct), NOT a per-index map: a song switch can change the
        # sink-input index, so restore re-finds the CURRENT stream by name and lifts it to this level. None
        # ⟺ not currently ducked (also the idempotency guard — don't re-read/re-save while already ducked).
        self._prior_level: int | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def duck(self, on: bool) -> None:
        """Duck (on=True) or restore (on=False) the local media sink-input. Never raises."""
        if not self._enabled:
            return
        try:
            await (self._duck_on() if on else self._restore())
        except Exception as e:  # noqa: BLE001 - media control must never break audio
            _tlog(f"DUCK  | local belt {'on' if on else 'off'} FAILED: {type(e).__name__}: {e}")

    async def _duck_on(self) -> None:
        if self._prior_level is not None:  # already ducked — keep the saved prior (idempotent)
            return
        found = await self._find()
        if not found:
            _tlog("DUCK  | local belt: no media sink-input found (nothing to duck locally)")
            return
        read = next((v for _, v in found if v is not None), None)
        # Stranded-prior guard: a read at/below the duck level is a leftover duck (module-stream-restore
        # replays it across a song switch), not a real level — restoring to it would strand the music quiet.
        prior = read if (read is not None and read > self._duck_pct) else self._restore_level
        self._prior_level = prior
        for idx, _ in found:
            await self._runner("set-sink-input-volume", idx, f"{self._duck_pct}%")
        _tlog(
            f"DUCK  | local belt ON — sink-input(s) {[i for i, _ in found]} {read}→{self._duck_pct}% "
            f"(restore→{prior}%)"
        )

    async def _restore(self) -> None:
        if self._prior_level is None:
            return
        prior, self._prior_level = self._prior_level, None
        # RE-FIND by name (not the cached duck-time index): a song switch may have spawned a new sink-input,
        # and the old index is dead. Lift whatever Mopidy stream is live now back to the guarded prior level.
        found = await self._find()
        if not found:
            return
        for idx, _ in found:
            await self._runner("set-sink-input-volume", idx, f"{prior}%")
        _tlog(f"DUCK  | local belt OFF — sink-input(s) {[i for i, _ in found]} → {prior}%")

    async def media_playing(self) -> bool:
        """True iff a MATCHED local media sink-input is actively PLAYING (present AND not corked). A paused/
        stopped stream still has a sink-input (corked) — so presence alone isn't 'playing'. Used by the wake
        gate to suppress the local wake-ack only when audible media is already giving duck-dip feedback (the maintainer:
        never intrude on playing music). Best-effort: any pactl failure / no match → False (treated as 'no
        audible media' → the ack speaks). Reads live pactl, NOT the brain's media_state (which goes stale)."""
        rc, out = await self._runner("list", "sink-inputs")
        if rc != 0 or not out:
            return False
        return self._any_match_playing(out)

    def _any_match_playing(self, out: str) -> bool:
        """Pure parse: any name-matched sink-input block with `Corked: no` (= actively playing)."""
        for block in out.split("Sink Input #")[1:]:
            names = re.findall(r'(?:application|node)\.name = "([^"]*)"', block)
            if not any(tok in n.lower() for n in names for tok in self._matches):
                continue
            m = re.search(r"Corked:\s*(yes|no)", block)
            if m and m.group(1) == "no":
                return True
        return False

    async def _find(self) -> list[tuple[str, int | None]]:
        rc, out = await self._runner("list", "sink-inputs")
        if rc != 0 or not out:
            return []
        return self._parse_media_sink_inputs(out)

    def _parse_media_sink_inputs(self, out: str) -> list[tuple[str, int | None]]:
        """Return [(sink-input index, current volume %)] for streams whose application/node name matches.

        Mirrors the EM brain's `_parse_mopidy_sink_input`, generalized to a configurable substring match so
        the same belt works for any local player (mopidy/mpd/…). Volume is the first `NN%` in the block."""
        res: list[tuple[str, int | None]] = []
        for block in out.split("Sink Input #")[1:]:
            names = re.findall(r'(?:application|node)\.name = "([^"]*)"', block)
            if not any(tok in n.lower() for n in names for tok in self._matches):
                continue
            idx = block.splitlines()[0].strip()
            if not idx.isdigit():
                continue
            m = re.search(r"Volume:.*?(\d+)%", block)
            res.append((idx, int(m.group(1)) if m else None))
        return res

    async def _run_pactl(self, *args, timeout: float = 2.0) -> tuple[int, str]:
        if not shutil.which("pactl"):
            return 1, ""
        try:
            p = await asyncio.create_subprocess_exec(
                "pactl", *args,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await asyncio.wait_for(p.communicate(), timeout=timeout)
            return p.returncode or 0, out.decode(errors="replace")
        except Exception:  # noqa: BLE001 - treat any spawn/timeout failure as "no pactl result"
            return 1, ""
