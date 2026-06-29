"""Media-duck controller — a pass-through FrameProcessor that owns `/media/duck` *timing*.

Sits right after STT in the pipeline because that's the only place that sees transcription frames:
the user aggregator consumes `TranscriptionFrame`/`InterimTranscriptionFrame` and does **not**
forward them downstream, so this logic can't live in `BrainLLMService` (which is downstream of the
aggregator). The brain owns the duck *action* (music → volume via Mopidy; Jellyfin video →
`<video>.volume`); we only decide *when* to signal on/off.

Design (the 2026-06-02 low-latency revision — supersedes the duck-on-confirmed-transcription rev):
- **Duck on VAD speech *onset*** (`VADUserStartedSpeakingFrame`, ~0.2s after you start) rather than
  waiting for a finished, transcribed phrase. The old "duck on ≥min_words transcription" rule made the
  duck engage 1–3s late (it had to wait for the segment to end *and* for Whisper) — the thing the user
  heard as lag. Onset-triggering is safe now that the PipeWire echo-cancel source removes the
  speaker output from the mic, so playback no longer self-trips the VAD.
- **False-onset guard:** an onset that yields no real words (a cough, a stray VAD blip) restores
  quickly — on `VADUserStoppedSpeaking` we arm a short `confirm_grace` timer that restores unless a
  ≥`min_words` transcription confirms the onset was speech (which cancels it). Continued speech
  (another onset) also cancels it, so a multi-clause utterance never flaps. **Exception (2026-06-14):
  while media is actually *playing*, the confirm timer does NOT snap the bed back — over a movie the mic
  is gated behind the wake word, so an onset is almost always a real command whose transcript merely lags
  `confirm_grace` (Whisper latency). Snapping back then flapped the bed down→up→re-duck and raced the
  confirmed `on` against a stale `off` (full-volume mid-command). It falls back to the longer idle grace,
  which a slow real command confirms within (the confirmed path then owns restore) and a genuine
  non-speech onset still hits eventually.** The quick snap-back stands when nothing is playing.
- **Confirmed transcription** (≥`min_words`) marks the turn real, cancels the confirm timer, and arms
  the slow idle-restore — and still *triggers* the duck itself if the onset frame never arrived
  (graceful fallback to the old behavior; no regression).
- **Restore when Aria finishes** (`BotStoppedSpeakingFrame`), or after a generous quiet grace if a
  ducked turn never produces bot speech.
- **Idempotent on/off**, **skip when nothing is playing** (best-effort `GET /media/state`), and
  **skip while asleep**.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    SystemFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


@dataclass
class DuckReleaseFrame(SystemFrame):
    """Pushed UPSTREAM by the brain (BrainLLMService) when it classifies a turn as NOT addressed to Aria
    (an aside). The VAD-onset pre-duck has already engaged for the utterance — but an aside yields no reply,
    so there's nothing to make room for. This releases that duck immediately instead of waiting out the
    idle-restore grace (~8s). A SystemFrame so it reaches this controller through the user aggregator the
    same way WakeSleepFrame/BotThinkingFrame reach the wake gate."""


@dataclass
class ConvoReleaseFrame(SystemFrame):
    """Pushed UPSTREAM by the brain (BrainLLMService) mid-turn (before `done`) when it judges the reply
    TERMINAL — the exchange is over (a one-shot Q&A, or an explicit dismiss like "thanks, that's all").
    The conversation-hold (Phase 1) otherwise keeps the bed ducked for `DUCK_CONVO_HOLD_SECS` after the
    reply so a follow-up needs no re-wake; a terminal turn doesn't want that wait, so this asks the duck
    to restore the bed at this turn's end instead of holding.

    Ordering matters: the brain emits this DURING the response stream, which is BEFORE the convo-hold is
    armed (the hold arms at BotStopped, after TTS finishes). So this can't 'cancel + restore' — there's
    nothing armed yet. It sets a ONE-SHOT 'don't hold this turn' flag consumed at the next BotStopped,
    which then restores immediately instead of arming the hold. Arrival-keyed like DuckReleaseFrame (the
    brain only ever emits it to act). A dropped frame degrades safely to Phase-1 (the idle timer still
    restores after the hold). RELEASE only — 'extend the hold longer than the wake window' is a separate
    future step: it would break the Phase-1 `_convo_secs ≤ window_secs` clamp (the gate's window idle-close
    is the backstop restore) and so must pair with a `wake_hold` to co-extend the gate window."""


def _tlog(message: str) -> None:
    """One line to the transcript log (greppable alongside USER/BOT/DUCK)."""
    logger.bind(transcript=True).info(message)


class MediaDuckController(FrameProcessor):
    """Drive media ducking off confirmed-speech + bot-speaking frames flowing through the pipeline."""

    def __init__(
        self,
        client,
        session_id: str,
        *,
        min_words: int = 2,
        restore_grace: float = 8.0,
        confirm_grace: float = 2.5,
        sustained_secs: float = 4.0,
        max_hold_secs: float = 120.0,
        convo_hold_secs: float = 0.0,
        window_secs: float = 15.0,
        should_duck: Callable[[], bool] | None = None,
        should_duck_onset: Callable[[], bool] | None = None,
        media_status: Callable[[], "dict|None"] | None = None,
        time_source: Callable[[], float] | None = None,
        local_duck=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._client = client
        self._session_id = session_id
        # Pi-side local sink belt (local_duck.LocalSinkDucker): attenuates the LOCAL media sink-input's
        # PipeWire node volume alongside the brain's RPC duck. On a satellite where the brain's mixer-RPC
        # can't reliably attenuate the room's output (the Pi Mopidy software mixer — 2026-06-23 drive), this
        # does the real duck. None (default) or a disabled ducker = no-op (the brain duck is the only path,
        # exactly as before). The brain skips its now-redundant mixer-RPC for a `duck_local` room (GA side).
        self._local_duck = local_duck
        self._min_words = max(1, min_words)
        self._restore_grace = restore_grace
        # An aside (DuckReleaseFrame) normally releases the duck immediately — but if the user is in a
        # SUSTAINED utterance (still speaking, or the duck has been on this long), snapping the music back to
        # full mid-sentence is the "it kept playing while I talked" bug. Past this duration we HOLD the duck on
        # an aside and let the idle-grace restore own it (fires ~restore_grace after speech truly stops); the
        # immediate aside-release then applies only to SHORT asides (the ambient-blip case it was built for).
        self._sustained_secs = sustained_secs
        # Hold the bed ducked across the WHOLE brain think (BotThinkingFrame active=True → active=False),
        # not just the `restore_grace` after speech stops. Media/TIDAL lookups run 15–60s (worst seen 64s,
        # 2026-06-19 audit §1) — far longer than the 8s idle grace — so the grace fired mid-think and the
        # music popped to full volume, then re-ducked when she finally replied (the "bob-up" the maintainer reported).
        # `max_hold_secs` is a pure safety ceiling: BotThinkingFrame bracketing is bounded by the brain only
        # by convention (per-tool timeouts, no single turn-level cap — gabagent audit 2026-06-19 Q1), so a
        # pathological unbounded-but-finite turn could otherwise hold the bed ducked indefinitely. The ceiling
        # restores the bed if a think runs absurdly long; it never fires on a real turn (120s ≫ 64s worst).
        self._max_hold_secs = max_hold_secs
        self._bot_thinking = False
        self._hold_task: asyncio.Task | None = None
        # Conversation-hold (2026-06-19): after an ADDRESSED reply over playing media, keep the bed ducked +
        # the floor open across follow-up turns so the user can continue without re-waking and the bed doesn't
        # bob down→up→down between turns. Owned here (the duck owner) with its OWN timer — the idle-grace can't
        # serve: after a reply `_bot_spoke` is True, which gates `_arm_restore` off. Clamped to the wake
        # `window_secs`: the gate already re-arms its command window for ~window_secs on every reply and its
        # idle-close fires its own `_fire_duck(False)`, so keeping the hold ≤ window keeps the two in lockstep
        # (they end together) AND gives a free second restore guarantee — no wake_word change, no stuck-window.
        # 0 disables (today's immediate bot-stopped restore). Ended by: idle timer, sleep, or shutdown — every
        # path calls `_restore` (a held duck with no pending restore is the one regression this guards against).
        self._convo_secs = max(0.0, min(convo_hold_secs, window_secs)) if convo_hold_secs > 0 else 0.0
        self._convo_task: asyncio.Task | None = None
        # Phase-2 brain release hint (ConvoReleaseFrame): a one-shot "this turn was terminal — don't hold,
        # restore at BotStopped" flag. Set mid-turn (before the hold is armed), consumed at the next
        # BotStopped, reset there. See ConvoReleaseFrame for the ordering rationale.
        self._convo_release_pending = False
        self._time = time_source or time.monotonic
        self._duck_started_at = 0.0
        # How long after speech *stops* to wait for a confirming transcription before treating the
        # onset as a false trigger and restoring. Must comfortably exceed Whisper's per-segment
        # latency so real speech confirms before it fires (else a slow transcribe would flap).
        self._confirm_grace = confirm_grace
        # Gate (e.g. "not asleep"); default always-allow.
        self._should_duck = should_duck or (lambda: True)
        # Stricter gate for the RAW VAD-ONSET duck (the immediate, pre-transcription dip). Over playing media
        # the held-open keepalive/idle window would otherwise let the media's own VAD onsets dip the bed with
        # no wake (2026-06-16); this gate suppresses the onset duck there while the confirmed-speech path
        # (gated by should_duck) still ducks a real follow-up command. Defaults to should_duck (no change).
        self._should_duck_onset = should_duck_onset or self._should_duck
        self._ducked = False
        self._bot_spoke = False  # did the bot speak during the current duck?
        self._confirmed = False  # got ≥min_words this duck episode (so it's not a false onset)
        # Is the user currently inside an active VAD speech segment (onset seen, no stop yet)? The
        # idle-restore must never fire while this is True — yanking the bed out mid-sentence is the
        # "music interjects during a long monologue" bug. Idle is measured from speech *stop*, not
        # from the last transcription chunk (which can lag / fall under min_words mid-utterance).
        self._speech_in_flight = False
        self._restore_task: asyncio.Task | None = None
        self._confirm_task: asyncio.Task | None = None
        # Media-state — the SHARED provider (config.build_media_state_provider), the SAME callable the
        # wake gate uses, so the gate and the duck can never disagree (they used to keep separate caches
        # with opposite fail-modes). Returns {"playing": bool, "kind": …}; the duck only needs `playing`.
        # It owns the 1s cache + coalescing. Default → always-duck if unwired (the brain's duck is a
        # harmless no-op when nothing plays).
        self._media_status = media_status or (lambda: {"playing": True})

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self._handle(frame)
        await self.push_frame(frame, direction)

    # --- frame handling ----------------------------------------------------
    def _handle(self, frame: Frame) -> None:
        # BotThinkingFrame brackets the whole brain turn (pushed UPSTREAM by BrainLLMService._set_thinking at
        # turn start/finally). Imported lazily to avoid an import cycle at module load (turn_mute imports only
        # dataclasses + pipecat, so this is safe). Checked first so the think-hold owns restore timing through
        # a slow lookup. NOT keyed before the VAD branches because a BotThinkingFrame is never also a VAD frame.
        from turn_mute import BotThinkingFrame

        if isinstance(frame, BotThinkingFrame):
            if frame.active:
                # The brain accepted a real turn and is working; a reply (BotStopped) or an aside
                # (DuckReleaseFrame) is coming however long it takes. Hold the duck: kill the false-onset
                # confirm AND the idle-restore so neither pops the bed mid-think. Arm the safety ceiling.
                self._bot_thinking = True
                self._cancel_confirm()
                self._cancel_restore()
                if self._ducked:
                    self._arm_hold_ceiling()
            else:
                # Think ended. A spoken reply already restored via BotStopped (not ducked here → no-op). A
                # silent turn (empty model turn, or a turn that produced no bot speech and no aside) leaves the
                # bed ducked — re-arm the idle grace as the safety net so it still restores.
                self._bot_thinking = False
                self._cancel_hold_ceiling()
                if self._ducked and not self._bot_spoke:
                    self._arm_restore()
            return
        # Sleep / shutdown: a held conversation-duck (or any in-flight duck) must not survive either. Both clear
        # the hold and restore. WakeSleepFrame is pushed UPSTREAM by the brain and flows through here; lazy
        # import avoids an import cycle (wake_word never imports this module). EndFrame is the pipeline-teardown
        # frame — best-effort restore (fire-and-forget may not land before teardown; the brain owns the final
        # volume restore), and it closes a pre-existing "duck left on at shutdown" gap the hold would worsen.
        from wake_word import WakeSleepFrame

        if isinstance(frame, WakeSleepFrame):
            if frame.asleep and self._ducked:
                self._cancel_convo()
                self._restore("sleep")
            return
        if isinstance(frame, EndFrame):
            if self._ducked:
                self._cancel_convo()
                self._restore("shutdown")
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # Speech onset — duck immediately. Cancel any pending false-onset restore (still talking).
            # Also cancel a pending conversation-hold restore: the user is following up, so the bed must
            # stay ducked through this turn (its eventual BotStopped re-arms the hold).
            self._speech_in_flight = True
            self._cancel_confirm()
            self._cancel_convo()
            # Clear any stale brain release intent at the start of every user turn. The flag is normally
            # consumed at BotStopped, but a turn that ended another way (barge-in mid-reply, sleep, shutdown)
            # never reaches that consume point — resetting on each onset stops it leaking into the next turn
            # and wrongly suppressing that turn's conversation-hold.
            self._convo_release_pending = False
            self._duck_on("speech onset", allow=self._should_duck_onset)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Speech segment ended. If this onset hasn't been confirmed by real words yet, start the
            # short countdown that restores it as a false trigger (cancelled if a transcription
            # confirms, or if the user resumes speaking).
            self._speech_in_flight = False
            if self._ducked and not self._confirmed:
                self._arm_confirm()
            elif self._ducked and self._confirmed:
                # Confirmed turn just paused — (re)start the idle grace from the actual stop point so
                # the bed only restores after a genuine silence, never mid-monologue.
                self._arm_restore()
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            text = getattr(frame, "text", "") or ""
            if len(text.split()) >= self._min_words:
                self._on_confirmed_speech()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_spoke = True
            self._cancel_confirm()
            self._cancel_restore()
            self._cancel_hold_ceiling()  # she's replying — the think-hold ceiling is no longer needed
        elif isinstance(frame, BotStoppedSpeakingFrame):
            _tlog(f"DUCK  | bot-stopped (ducked={self._ducked})")
            if self._ducked:
                # Conversation-hold: an ADDRESSED reply (bot actually spoke) over still-ducked media → hold the
                # bed ducked + floor open for a follow-up instead of restoring now. Asides (`_bot_spoke=False`),
                # the disabled case (`_convo_secs==0`), AND a brain release hint (`_convo_release_pending` — the
                # turn was terminal) fall through to the normal immediate restore.
                if self._convo_secs > 0 and self._bot_spoke and not self._convo_release_pending:
                    # Reset the per-turn flags so the next follow-up is a clean duck episode (the duck stays
                    # physically on; `_duck_on` will no-op on it). The hold's own timer owns restore from here.
                    self._bot_spoke = False
                    self._confirmed = False
                    self._arm_convo()
                else:
                    self._restore("convo release" if self._convo_release_pending else "bot-stopped")
            # One-shot: consumed at this turn's end regardless of whether a hold would have engaged.
            self._convo_release_pending = False
        elif isinstance(frame, DuckReleaseFrame):
            # Brain says this turn was an aside (not addressed) → no reply is coming. Release the
            # pre-duck now rather than holding it for the idle grace. Keyed on the frame TYPE, which the
            # brain only ever sends on suppression (no bool to be dropped by a serializer). Guarded by
            # `not self._bot_spoke` so a stale aside-verdict can never cut the bed out from under live TTS
            # (e.g. an aside-then-command race where Aria is already replying).
            if self._ducked and not self._bot_spoke:
                sustained = (
                    self._speech_in_flight
                    or (self._time() - self._duck_started_at) >= self._sustained_secs
                )
                if sustained:
                    # Sustained utterance (dictation / long aside) — don't snap the bed back mid-sentence.
                    # Hold; the idle-grace restore (re-armed per transcription, gated on _speech_in_flight)
                    # owns it and fires only after speech truly stops. This is the automatic duck-policy.
                    _tlog("DUCK  | aside during sustained speech — holding (idle-grace owns restore)")
                else:
                    self._restore("aside")
        elif isinstance(frame, ConvoReleaseFrame):
            # Brain says this reply is TERMINAL → don't hold the bed for a follow-up; restore at this
            # turn's BotStopped. Arrives mid-turn (before the hold is armed), so set a one-shot flag the
            # BotStopped branch consumes — never restore here (the bot may still be mid-reply; cutting the
            # bed now would un-duck under live TTS). No-op when convo-hold is disabled. See ConvoReleaseFrame.
            if self._convo_secs > 0:
                _tlog("DUCK  | brain release hint — terminal turn, no convo-hold")
                self._convo_release_pending = True

    def _duck_on(self, reason: str, allow: Callable[[], bool] | None = None) -> None:
        # `allow` lets the raw-onset path use the stricter onset gate while the confirmed-speech path uses
        # the broad should_duck (default). Both still require something to be playing (checked in _fire).
        if not (allow or self._should_duck)() or self._ducked:
            return
        self._ducked = True
        self._bot_spoke = False
        self._confirmed = False
        self._duck_started_at = self._time()
        self._fire(True, reason)

    def _on_confirmed_speech(self) -> None:
        if not self._should_duck():
            return
        # Real words → this duck episode is confirmed; drop the false-onset guard.
        self._confirmed = True
        self._cancel_confirm()
        # Fallback: if the onset frame never reached us, the transcription still triggers the duck
        # (old behavior — no regression if VAD frames don't flow for some transport).
        self._duck_on("confirmed speech")
        # (Re)arm the idle fallback: a ducked turn that never yields bot speech restores after the
        # grace. Re-arming on each transcription means continuous speech keeps media ducked; the
        # grace only elapses once speech truly stops with no reply, so a normal thinking gap (bot
        # replies within the grace → BotStarted cancels it) never restores prematurely.
        self._arm_restore()

    # --- duck firing -------------------------------------------------------
    def _fire(self, on: bool, reason: str = "") -> None:
        """Fire-and-forget the brain duck call — never block the audio pipeline."""

        async def _go():
            try:
                # Arm if EITHER the brain's media_state reports media OR a matched LOCAL sink is live.
                # The belt ducks a physically-local sink, so a matched local sink playing is ground-truth
                # on its own — independent of the brain's media_state, which judges a CAST session by its
                # RemoteEndPoint and can flag a movie playing on THIS laptop's screen as 'remote' (→ not
                # 'playing') when the room has no room_media profile. Self-detecting via the belt avoids
                # gating a local-sink duck on the brain's session-locality call (the 2026-06-26 movie gap).
                if on and not (await self._media_playing() or await self.media_playing_local()):
                    # Nothing actually playing — don't bother the brain; undo the optimistic flag.
                    # DEBUG-only: in open-mic conversation (no media) every VAD onset hits this path, so
                    # logging it to the transcript drowns USER/BOT in "on SKIPPED" noise (~40% of the
                    # transcript in the 2026-06-17 open-mic drive). The skip is a no-op; keep it in
                    # session.log for media debugging, off the transcript.
                    self._ducked = False
                    logger.debug("DUCK  | on SKIPPED (nothing playing: brain media_state idle + no local sink)")
                    return
                await self._client.duck(self._session_id, on)
                # Pi-side belt: attenuate the LOCAL media sink-input too (no-op when unset/disabled). On a
                # satellite whose brain mixer-RPC can't reliably duck the room's output, this is the real
                # duck; elsewhere it's a no-op and the brain duck above is the only path. Best-effort itself.
                if self._local_duck is not None:
                    await self._local_duck.duck(on)
                _tlog(f"DUCK  | on ({reason}) sent" if on else "DUCK  | /media/duck on=False sent")
            except Exception as e:  # noqa: BLE001 - media control must never break audio
                _tlog(f"DUCK  | /media/duck on={on} FAILED: {type(e).__name__}: {e}")

        asyncio.create_task(_go())

    async def _media_playing(self) -> bool:
        """Delegate to the SHARED media-state provider (same callable the wake gate uses → no divergence).
        Accepts a sync OR async callable returning {"playing": …}; the provider owns the cache/fail-mode."""
        st = self._media_status()
        if asyncio.iscoroutine(st) or asyncio.isfuture(st):
            st = await st
        return bool(st and st.get("playing"))

    async def media_playing_local(self) -> bool:
        """Is local media ACTIVELY playing, per the live local sink belt (NOT the brain's media_state, which
        goes stale-frozen for a sleeping room)? Wired to the wake gate's wake-ack gate so the ack stays quiet
        only over genuinely-audible media. No local belt configured → False (the ack speaks)."""
        if self._local_duck is None or not getattr(self._local_duck, "enabled", False):
            return False
        return await self._local_duck.media_playing()

    # --- restore timer -----------------------------------------------------
    def _arm_restore(self) -> None:
        self._cancel_restore()

        async def _later():
            try:
                await asyncio.sleep(self._restore_grace)
                if self._ducked and not self._bot_spoke:
                    if self._speech_in_flight:
                        # User is still mid-utterance — don't restore the bed under them. Re-arm and
                        # let the next stop (VADUserStoppedSpeaking) measure the idle window cleanly.
                        self._arm_restore()
                        return
                    self._restore("idle-grace")
            except asyncio.CancelledError:
                pass

        self._restore_task = asyncio.create_task(_later())

    def _cancel_restore(self) -> None:
        if self._restore_task is not None and not self._restore_task.done():
            self._restore_task.cancel()
        self._restore_task = None

    # --- false-onset confirm timer ----------------------------------------
    def _arm_confirm(self) -> None:
        self._cancel_confirm()

        async def _later():
            try:
                await asyncio.sleep(self._confirm_grace)
                # Speech stopped and no qualifying transcription arrived within confirm_grace.
                if self._ducked and not self._confirmed and not self._bot_spoke:
                    # While media is actually playing, do NOT snap the bed back now. Over a movie the mic
                    # is gated behind the wake word, so an onset here is almost always a real command whose
                    # transcript is merely slower than confirm_grace (Whisper latency over the AEC mic). The
                    # immediate restore causes an audible flap — duck → up → re-duck — and a send-order race
                    # where its `off` can land *after* the imminent confirmed-speech `on`, leaving the movie
                    # at full volume mid-command (2026-06-14 log 19:36:34). Fall back to the longer idle
                    # grace instead: a slow real command confirms within it (the confirmed path then owns
                    # restore), while a genuine non-speech onset still restores once the idle window elapses.
                    if await self._media_playing():
                        self._arm_restore()
                        return
                    self._restore("unconfirmed onset")
            except asyncio.CancelledError:
                pass

        self._confirm_task = asyncio.create_task(_later())

    def _cancel_confirm(self) -> None:
        if self._confirm_task is not None and not self._confirm_task.done():
            self._confirm_task.cancel()
        self._confirm_task = None

    # --- think-hold safety ceiling ----------------------------------------
    def _arm_hold_ceiling(self) -> None:
        """Backstop the think-hold: if the brain stays "thinking" pathologically long (no reply, no aside, no
        error — the unbounded-but-finite turn the gabagent audit flagged), restore the bed rather than hold it
        ducked forever. Never fires on a real turn (max_hold_secs ≫ worst observed think)."""
        self._cancel_hold_ceiling()

        async def _later():
            try:
                await asyncio.sleep(self._max_hold_secs)
                if self._ducked and not self._bot_spoke:
                    _tlog(f"DUCK  | think-hold ceiling ({self._max_hold_secs:.0f}s) — restoring")
                    self._restore("think-hold ceiling")
            except asyncio.CancelledError:
                pass

        self._hold_task = asyncio.create_task(_later())

    def _cancel_hold_ceiling(self) -> None:
        if self._hold_task is not None and not self._hold_task.done():
            self._hold_task.cancel()
        self._hold_task = None

    # --- conversation-hold timer -------------------------------------------
    def _arm_convo(self) -> None:
        """Hold the bed ducked + floor open after an addressed reply, so a follow-up needs no re-wake and the
        bed doesn't bob between turns. Refreshed (cancelled + re-armed) per turn: a follow-up's onset cancels
        it, that turn's BotStopped re-arms it. On expiry (the conversation went quiet) restore — the primary
        of the design's restore guarantees (the gate's own idle-close ~window_secs later is the backstop)."""
        self._cancel_convo()
        _tlog(f"DUCK  | convo-hold ({self._convo_secs:.0f}s) — bed stays ducked for a follow-up")

        async def _later():
            try:
                await asyncio.sleep(self._convo_secs)
                if self._ducked:
                    self._restore("convo idle")
            except asyncio.CancelledError:
                pass

        self._convo_task = asyncio.create_task(_later())

    def _cancel_convo(self) -> None:
        if self._convo_task is not None and not self._convo_task.done():
            self._convo_task.cancel()
        self._convo_task = None

    def _restore(self, reason: str = "?") -> None:
        self._cancel_restore()
        self._cancel_confirm()
        self._cancel_hold_ceiling()
        self._cancel_convo()
        if self._ducked:
            self._ducked = False
            self._confirmed = False
            _tlog(f"DUCK  | off / restore (via {reason})")
            self._fire(False)
