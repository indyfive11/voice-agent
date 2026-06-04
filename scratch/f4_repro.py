"""Minimal repro: `RuntimeWarning: coroutine 'TurnAnalyzerUserTurnStopStrategy._timeout_handler'
was never awaited` (pipecat 1.3.0).

Root: TaskManager.create_task wraps the passed coroutine in an inner `run_coroutine()` that only
awaits it once run_coroutine itself runs. When the task is cancelled BEFORE run_coroutine gets to
its `await coroutine` (exactly what TurnAnalyzerUserTurnStopStrategy.reset()/cleanup() do to the
just-created `_timeout_task`), the wrapped coroutine is dropped un-awaited.

Real-world trigger: a SECOND UserTurnStopStrategy (e.g. a max-turn-duration cap) force-completes the
turn while `_handle_vad_user_stopped_speaking` has just created the timeout task -> reset() cancels it
before it runs -> this warning. Cosmetic (the turn completes correctly).

Run:  uv run python scratch/f4_repro.py   ->  prints "reproduced: True"
"""
import asyncio
import gc
import warnings

from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


class _StubTurnAnalyzer:
    """Only needs to exist - the strategy stores it; this repro never calls its methods."""


async def main():
    tm = TaskManager()
    tm.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

    strat = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=_StubTurnAnalyzer())
    await strat.setup(tm)

    # _handle_vad_user_stopped_speaking creates the timeout task (line ~195)...
    task = tm.create_task(strat._timeout_handler(0.1), f"{strat}::_timeout_handler")
    # ...then the cap's force-stop runs reset() -> cancels it BEFORE run_coroutine reaches `await`.
    task.cancel()  # synchronous: no loop iteration has run run_coroutine yet
    try:
        await task
    except asyncio.CancelledError:
        pass


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    asyncio.run(main())
    gc.collect()

hits = [w for w in caught if "_timeout_handler" in str(w.message) and "never awaited" in str(w.message)]
print(f"reproduced: {bool(hits)}")
for w in hits:
    print(f"  {w.category.__name__}: {w.message}")
