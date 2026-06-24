"""SmartTurn-honoring user-turn stop strategy — don't end a turn the model judged INCOMPLETE.

The stock `TurnAnalyzerUserTurnStopStrategy` ends a turn when `_turn_complete` AND a transcript exist. But it
sets `_turn_complete = True` in a **transcript fallback** ("assume the turn is complete when a transcript
arrives with no VAD-stop recorded yet", `turn_analyzer_user_turn_stop_strategy.py`), then arms a short
STT-p99 timeout (`p99 - vad_stop_secs`, ~0.8s with the default 1.0s p99) that fires the turn stop. Over our
always-on Silero VAD, a Whisper transcript routinely lands ~1ms around the VAD-stop frame, so that fallback
trips and the STT-timeout ends the turn **even when SmartTurn just predicted INCOMPLETE** — cutting the user
off on a natural mid-sentence pause (2026-06-20 live: "Tell me a little story about…<pause>" → answered with a
generic story before the user named the topic; SmartTurn had scored it INCOMPLETE 0.64% 0.8s earlier).

Fix: gate the single trigger funnel (`trigger_user_turn_stopped`, which every stop path in the base routes
through) on SmartTurn's own state. `BaseSmartTurn.speech_triggered` is True exactly while the model considers
the turn unfinished — it's cleared ONLY on a real COMPLETE verdict or the `stop_secs` silence timeout
(`_clear` runs on COMPLETE, never on INCOMPLETE). So: while `speech_triggered` is True, suppress the stop —
let only a genuine COMPLETE verdict or the `stop_secs` (4s) silence end it. A user who resumes ("…a dog and a
cat") fires a fresh VAD onset that resets the turn; a user who truly stopped ends via SmartTurn's 4s fallback.
The parallel `MaxTurnDurationUserTurnStopStrategy` (15s) remains the runaway backstop, so a turn can never hang.

We do NOT exempt "finalized" transcripts: Whisper (a batch STT) marks **every** segment `finalized=True`
(pipecat stt_service.py: "every transcription is inherently finalized"), so a finalized check never fires and
left this guard a no-op (2026-06-20 live: turn ended on INCOMPLETE, no hold). SmartTurn — not STT
finalization — is the turn-end authority in this pipeline, so honor it regardless of `finalized`.

Env `TURN_HONOR_INCOMPLETE=0` reverts to the stock behavior (A/B / kill-switch).
"""

from __future__ import annotations

import asyncio
import re
import time

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


def _tlog(message: str) -> None:
    logger.bind(transcript=True).info(message)


# Trailing function words that signal an UNFINISHED utterance (the always-on long-form heuristic, below).
# If a turn's last word is one of these, the speaker almost certainly isn't done ("the task is", "and then",
# "create a file named hello.txt containing the"). Deliberately conservative — pronouns/common verb objects
# ("it", "this", "that" as objects) are EXCLUDED so normal complete commands ("turn it up", "do it", "what
# is it") stay snappy and forward immediately. A clause that ends on a plain noun ("…containing the text")
# does NOT look unfinished and won't be held by the heuristic alone — that case is why the explicit trigger
# phrase + the silence backstop exist as the reliable net.
_UNFINISHED_TAIL_WORDS = frozenset(
    "and or but then to of for with is are was were am the a an my your our his her its their "
    "into onto at in on by as so because if when while named called containing about from than "
    "create make write add set put give tell show".split()
)


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding sentence punctuation — for phrase matching."""
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .,!?;:")


class LongHoldState:
    """Shared one-bit flag: True while a long-form/dictation turn is being held open by the stop strategy.

    The max-turn-duration cap (turn_cap.py) runs as a PARALLEL stop strategy and would otherwise guillotine
    a legitimate long dictation at its normal wall-clock cap. Both strategies hold a reference to one instance
    of this: the honoring strategy sets `.active` while holding, and the cap reads it to extend its ceiling.
    """

    __slots__ = ("active",)

    def __init__(self):
        self.active = False


class SmartTurnHonoringStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """`TurnAnalyzerUserTurnStopStrategy` that never ends a turn while SmartTurn still judges it INCOMPLETE.

    Also emits a per-turn **finalization-latency** line (the #60 endpointing instrumentation): the felt lag
    between the user going quiet and the turn closing (so the brain can answer). The line attributes the
    latency to its cause so a live ear-test produces a tunable number instead of a guess:
      * ``prompt COMPLETE, no hold`` — SmartTurn fired COMPLETE right at the VAD-stop; the ``Nms from last
        speech`` figure is then mostly the SmartTurn **inference** cost (the suspect on weak HW like a Pi-4,
        where neural inference is slow — if this is large, the lever is offloading/lightening endpointing,
        NOT ``stop_secs``).
      * ``held …ms on INCOMPLETE → stop_secs silence fallback`` — SmartTurn kept the turn open and only the
        ``stop_secs`` (default 4s) silence timer ended it: this IS the ``SMART_TURN_STOP_SECS`` latency tax,
        the knob to lower (trading against mid-thought-pause fragmentation).
      * ``held …ms on INCOMPLETE → COMPLETE verdict`` — a real COMPLETE arrived during the pause well before
        ``stop_secs``: honor-incomplete did its job (don't lower ``stop_secs`` on account of these).
    """

    def __init__(
        self,
        *,
        turn_analyzer,
        stop_secs: float = 4.0,
        continuation_grace_secs: float = 0.0,
        continuation_max_words: int = 3,
        dictation_triggers: tuple[str, ...] = (),
        dictation_enders: tuple[str, ...] = (),
        dictation_silence_secs: float = 7.0,
        longform_heuristic: bool = False,
        long_hold: LongHoldState | None = None,
        **kwargs,
    ):
        super().__init__(turn_analyzer=turn_analyzer, **kwargs)
        self._stop_secs = stop_secs
        self._suppressed_logged = False
        self._hold_started: float | None = None  # monotonic ts of the first INCOMPLETE suppression this turn
        self._last_vad_stop: float | None = None  # monotonic ts of the most recent VAD stop this turn
        # Long-form / dictation hold (the "let me finish a long command" path). Two ways to ENTER a hold:
        #   1. Explicit trigger phrase ("start a builder task") anywhere in the turn → unconditional hold.
        #   2. Always-on heuristic: the turn's tail looks unfinished (_UNFINISHED_TAIL_WORDS) → hold.
        # While held, SmartTurn COMPLETE verdicts are SUPPRESSED (clause-boundary pauses don't cut the turn);
        # it ends only on an explicit stop phrase ("done", "that's all"), the silence backstop, or the cap.
        self._dictation_triggers = tuple(_norm(t) for t in dictation_triggers if _norm(t))
        self._dictation_enders = tuple(_norm(e) for e in dictation_enders if _norm(e))
        self._dictation_silence_secs = dictation_silence_secs
        self._longform_heuristic = longform_heuristic
        self._long_hold = long_hold if long_hold is not None else LongHoldState()
        self._held = False                          # True once this turn entered a long-form hold
        self._backstop_task: asyncio.Task | None = None  # silence timer that ends a held turn
        # Continuation grace (#62 drive fix): SmartTurn can mis-judge a SHORT garbled fragment as COMPLETE
        # and close the turn on it — the EMEET/remote-STT path split "…up <pause> tell me the joke" into two
        # VAD segments; the turn closed on "up", which arya then auto-ran as volume_up, and the real request
        # was dropped. When SmartTurn fires COMPLETE on a turn with <= continuation_max_words, hold the stop
        # for continuation_grace_secs; a fresh VAD onset (the user resuming) cancels the stop so the rest of
        # the utterance lands in the SAME turn. Only short turns pay the grace; full sentences forward
        # immediately. 0.0 = OFF = byte-identical to today (the safe-universal default per the portability SOP).
        self._continuation_grace_secs = continuation_grace_secs
        self._continuation_max_words = continuation_max_words
        self._text = ""  # accumulated transcript this turn (for the short-turn continuation heuristic)
        self._pending_stop_task: asyncio.Task | None = None

    async def reset(self):
        await super().reset()
        self._suppressed_logged = False
        self._hold_started = None
        self._last_vad_stop = None
        self._text = ""
        self._held = False
        self._long_hold.active = False
        await self._cancel_pending_stop()
        await self._cancel_backstop()

    async def cleanup(self):
        await super().cleanup()
        await self._cancel_pending_stop()
        await self._cancel_backstop()

    async def _cancel_pending_stop(self):
        if self._pending_stop_task:
            await self.task_manager.cancel_task(self._pending_stop_task)
            self._pending_stop_task = None

    async def _cancel_backstop(self):
        if self._backstop_task:
            await self.task_manager.cancel_task(self._backstop_task)
            self._backstop_task = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        # Timestamp the latest VAD-stop BEFORE delegating: the base handler can synchronously trigger the
        # turn-stop on a prompt COMPLETE verdict within this very call, so recording after super() would
        # leave _last_vad_stop=None for no-hold turns (the "?ms" gap). Set it first so the finalization
        # line reports the real end-to-end dead air (the latency a person feels).
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._last_vad_stop = time.monotonic()
        elif isinstance(frame, TranscriptionFrame):
            if frame.text:  # accumulate the turn's text for the short-turn continuation heuristic
                self._text = f"{self._text} {frame.text}".strip() if self._text else frame.text
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # The user resumed during the continuation grace → cancel the pending stop and keep the turn
            # open so the continuation joins this turn (instead of becoming a dropped next-turn fragment).
            if self._pending_stop_task is not None:
                await self._cancel_pending_stop()
                _tlog("TURN  | continuation — user resumed within grace, holding turn open")
            # Resumed mid long-form hold → cancel the silence backstop; it re-arms on the next pause's
            # COMPLETE. (Only real speech cancels it, so a held turn ends only after true trailing silence.)
            if self._held and self._backstop_task is not None:
                await self._cancel_backstop()
        return await super().process_frame(frame)

    async def trigger_user_turn_stopped(self):
        # The single funnel for every stop path in the base. Suppress it while SmartTurn is mid-turn (last
        # verdict INCOMPLETE → analyzer.speech_triggered). A real COMPLETE verdict or the stop_secs silence
        # both clear speech_triggered, so legitimate turn-ends pass straight through. We deliberately do NOT
        # exempt finalized transcripts: Whisper finalizes every segment, so that exemption made this a no-op.
        if getattr(self._turn_analyzer, "speech_triggered", False):
            if self._hold_started is None:
                self._hold_started = time.monotonic()
            if not self._suppressed_logged:
                self._suppressed_logged = True
                _tlog("TURN  | holding — SmartTurn INCOMPLETE (waiting for end-of-turn, not cutting mid-pause)")
            return

        # Long-form / dictation hold: SmartTurn fired a CONFIDENT COMPLETE, but this turn is (or should be)
        # held open so a multi-clause command isn't cut on a clause-boundary pause. Two entry paths, one exit.
        text_l = _norm(self._text)
        if self._held or self._should_enter_hold(text_l):
            if self._dictation_enders and self._matches_end(text_l):
                _tlog("TURN  | long-form END (stop phrase) — dispatching the full command")
                await self._end_hold_and_stop()
                return
            if not self._held:
                self._held = True
                self._long_hold.active = True  # let the parallel max-turn cap extend its ceiling
                reason = "trigger phrase" if self._matches_trigger(text_l) else "heuristic (tail looks unfinished)"
                _tlog(
                    f"TURN  | long-form HOLD on — {reason}; pauses won't cut. Ends on a stop phrase or "
                    f"~{self._dictation_silence_secs:.0f}s silence."
                )
            self._arm_backstop()  # (re)arm only if not already counting down
            return

        # Continuation grace: a COMPLETE verdict on a SHORT turn is the fragment-mis-close signature. Defer
        # the stop briefly so a resume can fold in; a longer turn (a real sentence) forwards immediately.
        if self._pending_stop_task is not None:
            return  # grace already running for this turn — don't double-schedule
        words = len(self._text.split())
        if self._continuation_grace_secs > 0 and 1 <= words <= self._continuation_max_words:
            _tlog(
                f"TURN  | short turn ({words}w \"{self._text}\") — {self._continuation_grace_secs:.2f}s "
                "continuation grace before forwarding (fragment guard)"
            )
            self._pending_stop_task = self.task_manager.create_task(
                self._grace_then_stop(), f"{self}::_grace_then_stop"
            )
            await asyncio.sleep(0)  # ensure it's scheduled
            return
        await self._do_stop()

    async def _grace_then_stop(self):
        try:
            await asyncio.sleep(self._continuation_grace_secs)
        except asyncio.CancelledError:
            return
        finally:
            self._pending_stop_task = None
        # If SmartTurn re-opened the turn during the grace (user resumed without a fresh onset frame we
        # caught), don't force the stop — let the turn continue and re-evaluate on the next COMPLETE.
        if getattr(self._turn_analyzer, "speech_triggered", False):
            return
        await self._do_stop()

    def _should_enter_hold(self, text_l: str) -> bool:
        """Decide whether a confident COMPLETE should be HELD instead of ending the turn. Two entry paths:
        an explicit trigger phrase (unconditional), or the always-on heuristic (the tail looks unfinished)."""
        if not text_l:
            return False
        if self._matches_trigger(text_l):
            return True
        if self._longform_heuristic and text_l.split()[-1] in _UNFINISHED_TAIL_WORDS:
            return True
        return False

    def _matches_trigger(self, text_l: str) -> bool:
        # Trigger phrases appear anywhere in the turn (usually at the start: "start a builder task …").
        return any(t in text_l for t in self._dictation_triggers)

    def _matches_end(self, text_l: str) -> bool:
        # Stop phrases end the turn — matched at the TAIL of the utterance (the natural place to say "…done")
        # so a phrase appearing mid-sentence ("are you done loading?") doesn't prematurely close a hold.
        return any(text_l == e or text_l.endswith(" " + e) for e in self._dictation_enders)

    def _arm_backstop(self) -> None:
        """Start the trailing-silence timer that ends a held turn. Armed only if not already counting down,
        so SmartTurn's own repeated silence-fallback COMPLETEs during one quiet stretch don't keep pushing
        it out — a real resume (VADUserStartedSpeaking) cancels it, and the next pause's COMPLETE re-arms."""
        if self._backstop_task is not None:
            return
        self._backstop_task = self.task_manager.create_task(self._backstop(), f"{self}::_backstop")

    async def _backstop(self) -> None:
        try:
            await asyncio.sleep(self._dictation_silence_secs)
        except asyncio.CancelledError:
            return
        finally:
            self._backstop_task = None
        # If SmartTurn re-opened the turn (the user resumed) during the wait, don't force the end.
        if getattr(self._turn_analyzer, "speech_triggered", False):
            return
        _tlog(f"TURN  | long-form END ({self._dictation_silence_secs:.0f}s silence) — dispatching the full command")
        await self._end_hold_and_stop()

    async def _end_hold_and_stop(self) -> None:
        await self._cancel_backstop()
        self._held = False
        self._long_hold.active = False
        await self._do_stop()

    async def _do_stop(self):
        self._log_finalization()
        await super().trigger_user_turn_stopped()

    def _log_finalization(self) -> None:
        """Attribute this turn's finalization latency to its cause (see the class docstring)."""
        now = time.monotonic()
        lead = f"{(now - self._last_vad_stop) * 1000:.0f}ms" if self._last_vad_stop is not None else "?ms"
        if self._hold_started is not None:
            held_ms = (now - self._hold_started) * 1000
            # Held ~stop_secs ⇒ the silence fallback ended it (the tax); a clearly shorter hold ⇒ a real
            # COMPLETE verdict released the pause early (honor-incomplete working as intended).
            via = "stop_secs silence fallback" if held_ms >= self._stop_secs * 1000 * 0.9 else "COMPLETE verdict"
            tail = f" (held {held_ms:.0f}ms on INCOMPLETE → {via})"
        else:
            tail = " (prompt COMPLETE, no hold)"
        _tlog(f"TURN  | finalized — {lead} from last speech{tail}")
