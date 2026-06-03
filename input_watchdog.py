"""Input-stall watchdog — turn a silent mic-capture freeze into a logged, self-healing event.

Why this exists: live (2026-06-03, session f1fbc3a5) the AEC echo-cancel mic stopped delivering usable
audio after a media window closed, and Aria went dead for **89 seconds** until the user hit Ctrl-C. Nothing
logged it — pipecat's `base_input._audio_task_handler` loops on its 0.5s queue-timeout *silently forever*
once it has seen one frame, and the PyAudio callback ignores PortAudio status flags. The brain stayed
alive (its `/media/state` kept answering), so the freeze was isolated to the voice audio path.

Crucially, GC's audit showed the capture did **not** stop — the gate kept polling `/media/state` every
1s (which only happens while frames flow), yet zero wake-detection happened. So the source kept *emitting
frames that were dead/silent*. A naive "no frames for N seconds" detector would miss that. So we watch
**both**:
  - **no frames** for `stall_secs` (the classic stop), and
  - **frames arriving but persistently silent** for `silent_secs` — `max(|int16|) < silence_eps`, with a
    tiny epsilon so a real quiet room's noise floor/dither does NOT trip it (only a dead source's near-zeros).

On a detected stall we log loudly (`INPUT STALL`) and call an injected best-effort `restart` (main.py
kicks the PortAudio stream), bounded by `max_restarts` so a truly-dead source can't loop. On good frames
returning we log `INPUT RESUMED`. **Honest caveat:** if the echo-cancel *module* itself died (not just the
stream), the in-process kick may not revive it — pinning the output sink (prevention) is the primary lever;
this watchdog guarantees the failure is *visible and actionable* instead of 89s of silence.

Sits FIRST in the pipeline (right after `transport.input()`, before the resampler) so it times the rawest
mic frames and is blind to downstream wake-gate swallowing (it measures capture health, not gating).
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
from loguru import logger

from pipecat.frames.frames import Frame, InputAudioRawFrame, StartFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _tlog(message: str) -> None:
    """One line to the transcript log (greppable alongside USER/BOT/DUCK/WAKE/GATE)."""
    logger.bind(transcript=True).info(message)


class InputStallDetector(FrameProcessor):
    """Detect a no-frames OR frames-but-silent mic-capture stall; log it and best-effort restart capture."""

    def __init__(
        self,
        *,
        stall_secs: float = 5.0,
        silent_secs: float = 8.0,
        silence_eps: int = 4,
        restart=None,  # async callable(start_frame) -> bool; None = log-only
        max_restarts: int = 3,
        heartbeat_secs: float = 10.0,  # periodic liveness line; 0 = off
        gate_state=None,  # optional callable → {"gated","open","wake_peak"} for the heartbeat
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stall_secs = stall_secs
        self._silent_secs = silent_secs
        self._silence_eps = silence_eps
        self._restart = restart
        self._max_restarts = max_restarts
        self._hb_secs = heartbeat_secs
        self._gate_state = gate_state

        self._last = 0.0           # monotonic of the last frame of ANY kind
        self._last_nonsilent = 0.0  # monotonic of the last frame with real audio
        self._armed = False         # seen ≥1 frame (don't warn before capture starts)
        self._stalled = False       # currently in a stall episode
        self._restarts = 0          # restart attempts this episode
        self._start_frame: StartFrame | None = None  # captured for the restart
        self._task: asyncio.Task | None = None
        # Heartbeat counters (deltas since the last beat).
        self._frames = 0
        self._silent = 0
        self._hb_at = 0.0
        self._hb_frames = 0
        self._hb_silent = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._start_frame = frame
        elif isinstance(frame, InputAudioRawFrame):
            now = time.monotonic()
            self._last = now
            self._frames += 1
            if self._is_silent(frame.audio):
                self._silent += 1
            else:
                self._last_nonsilent = now
            if not self._armed:
                self._armed = True
                self._last_nonsilent = now  # arm both clocks on first frame
                self._hb_at = now
                self._task = self.task_manager.create_task(self._watch(), f"{self}::watch")
        await self.push_frame(frame, direction)

    def _is_silent(self, audio: bytes) -> bool:
        if not audio:
            return True
        samples = np.frombuffer(audio, dtype=np.int16)
        return samples.size == 0 or int(np.abs(samples).max()) < self._silence_eps

    async def _watch(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                await self._tick()
        except asyncio.CancelledError:
            return

    async def _tick(self) -> None:
        """One watch step (extracted so it's unit-testable without driving the sleep loop)."""
        if not self._armed:
            return
        now = time.monotonic()
        no_frames = now - self._last
        silent = now - self._last_nonsilent
        if no_frames > self._stall_secs:
            await self._on_stall("no mic frames", no_frames)
        elif silent > self._silent_secs:
            await self._on_stall("frames but silent", silent)
        elif self._stalled:
            # good frames are flowing again
            self._stalled = False
            self._restarts = 0
            _tlog(f"INPUT RESUMED | mic audio back after {silent:.1f}s")
        self._maybe_heartbeat(now)

    def _maybe_heartbeat(self, now: float) -> None:
        """Periodic liveness line. Its ABSENCE means the loop froze; its CONTENT tells freeze vs lockout:
        frames=+0 → capture stopped; frames>0 with all-silent → dead/silent source; gated=True open=False
        with a low wake_peak → masked-out wake word (lockout), not a freeze."""
        if self._hb_secs <= 0 or now - self._hb_at < self._hb_secs:
            return
        df = self._frames - self._hb_frames
        ds = self._silent - self._hb_silent
        self._hb_frames, self._hb_silent, self._hb_at = self._frames, self._silent, now
        line = f"INPUT | hb frames=+{df} silent=+{ds} last_voice={now - self._last_nonsilent:.0f}s"
        if self._gate_state is not None:
            try:
                g = self._gate_state()
                line += f" | gate gated={g['gated']} open={g['open']} wake_peak={g['wake_peak']:.2f}"
            except Exception as e:  # noqa: BLE001 - diagnostics must never break the pipeline
                line += f" | gate <err {type(e).__name__}>"
        _tlog(line)

    async def _on_stall(self, mode: str, secs: float) -> None:
        if not self._stalled:
            self._stalled = True  # log once per episode
            _tlog(f"INPUT STALL | {mode} for {secs:.1f}s — capture may have stalled")
        if self._restart is None:
            return
        if self._restarts >= self._max_restarts:
            if self._restarts == self._max_restarts:
                self._restarts += 1  # log the give-up exactly once
                _tlog(
                    f"INPUT STALL | still dead after {self._max_restarts} restart attempts — "
                    "manual restart likely needed (echo-cancel source may be gone)"
                )
            return
        self._restarts += 1
        try:
            ok = await self._restart(self._start_frame)
            _tlog(f"INPUT STALL | restart attempt {self._restarts}/{self._max_restarts} "
                  f"({'kicked capture' if ok else 'no stream to kick'})")
        except Exception as e:  # noqa: BLE001 - recovery must never crash the pipeline
            _tlog(f"INPUT STALL | restart attempt {self._restarts} FAILED: {type(e).__name__}: {e}")

    async def cleanup(self):
        await super().cleanup()
        if self._task:
            await self.task_manager.cancel_task(self._task)
            self._task = None
