"""WakeMediaPauser — pause the room's VIDEO for the wake command window, so a command over a movie lands
on a clean mic.

The problem (diagnosed from the 2026-07-04 Pi drive): co-located room media plays into the satellite mic
with NO acoustic-echo reference (it is not Aria's own TTS, and a Jellyfin movie bypasses the agent's output
path entirely). A movie's *dialogue* then floods STT — the mic transcribes the film as phantom turns and
masks the user's actual command. On that drive, 12 of 32 wake windows opened and then captured no command
("no speech after wake"); the maintainer's workaround was to PAUSE the movie by hand — the only thing that reliably
cleared the mic. This automates exactly that: on a fresh wake over a playing video, pause it for the command
window and resume when the window closes.

Scope — VIDEO only, deliberately:
  * The felt failure was movies. The same drive shows the MUSIC turns ("play some music", "turn up that
    music") succeeded — the media-duck belt attenuating music to 18% is sufficient, because non-dialogue
    audio doesn't defeat STT. So music is left to the existing belt; only intelligible video dialogue needs
    the hard clear of a pause.
  * Pause (0%), not a deep duck (~5%): a movie paused is a guaranteed-clean mic and matches the maintainer's manual
    workaround; a 5% dip risks STT still catching faint dialogue.

Resume-suppression: if the command IS itself a transport action ("pause it" / "stop it"), the user/brain now
owns the transport state, so we must NOT auto-resume and override them. Two signals gate the resume:
  * ``transport_intent()`` — the brain flags a turn that issued a media-transport action (rides the /respond
    response; wired optional so this degrades safely to "resume unless the session is gone" until it lands).
  * A stopped movie CLOSES its Jellyfin session, so a blind resume of a closed session is a harmless no-op
    (Unpause 404) — the stop case is safe even before the flag exists.

Fail-soft throughout: any failure is a logged no-op; nothing here may block the wake path or the voice loop.
Disabled (pure no-op) unless a controller that ``enabled`` is wired — so any install without a configured
Jellyfin room is byte-identical to before. Shares ONE ``JellyfinRoomController`` instance with the image-
display sink (roadmap ③) so the two never race a pause/resume on the same session.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from loguru import logger


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


class WakeMediaPauser:
    """Pause a playing video on fresh wake; resume on window-close unless the turn owned the transport.

    Injectables:
      * ``controller``       — object with async ``pause_if_playing() -> bool`` / ``resume() -> None`` and a
                               sync ``forget() -> None`` + ``enabled`` prop (a ``JellyfinRoomController``).
      * ``transport_intent`` — optional ``() -> bool``; True iff a turn in THIS command window issued a
                               media-transport command (then suppress the resume). None ⇒ never suppress
                               (safe default). The brain LATCHES this over the window, so a transport
                               command anywhere in the window — not only the last turn — suppresses resume
                               (GA's ``resume ⟺ … NOT saw(transport_intent)`` rule).
      * ``reset_transport_intent`` — optional ``() -> None``; clears the brain's window latch. Called at the
                               START of each fresh window so a transport flag from a prior window (e.g. a
                               "pause the music" turn that paused no *video*) can't leak into this one.
    """

    def __init__(
        self,
        *,
        controller,
        transport_intent: Callable[[], bool] | None = None,
        reset_transport_intent: Callable[[], None] | None = None,
    ):
        self._controller = controller
        self._transport_intent = transport_intent
        self._reset_transport_intent = reset_transport_intent
        self._engaged = False  # True iff WE paused the video for the current command window

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._controller, "enabled", False))

    async def engage(self) -> None:
        """Fresh wake opened a command window → pause a playing video. Idempotent within one window (a re-arm
        or a duplicate wake won't double-pause). No-op if nothing is playing / already paused."""
        if self._engaged:
            return
        # Fresh window: clear any transport latch left over from a prior window before we listen for this
        # window's command, so a stale flag can't suppress THIS window's resume.
        if self._reset_transport_intent is not None:
            try:
                self._reset_transport_intent()
            except Exception:  # noqa: BLE001 - latch reset must never break the wake path
                pass
        try:
            if await self._controller.pause_if_playing():
                self._engaged = True
                _tlog("WAKE  | video paused for command window (clean mic over movie)")
        except Exception as e:  # noqa: BLE001 - media control must never break the wake path
            _tlog(f"WAKE  | video pause FAILED (ignored): {type(e).__name__}: {e}")

    async def release(self) -> None:
        """Command window closed → resume the video we paused, UNLESS the turn issued a transport command
        (the user/brain owns the state now). A stopped movie's session is gone, so resume is a harmless
        no-op there even without the flag."""
        if not self._engaged:
            return
        self._engaged = False
        try:
            if self._transport_intent is not None and self._transport_intent():
                # The command was "pause it" / "stop it" — honor it; drop our marker WITHOUT unpausing.
                self._controller.forget()
                _tlog("WAKE  | video resume SUPPRESSED — turn issued a transport command")
                return
            await self._controller.resume()
            _tlog("WAKE  | video resumed after command window")
        except Exception as e:  # noqa: BLE001 - media control must never break the voice loop
            _tlog(f"WAKE  | video resume FAILED (ignored): {type(e).__name__}: {e}")
