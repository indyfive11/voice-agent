"""Voice Agent — real-time spoken assistant (Pipecat 1.3.x, local-first).

Phase 2: the core mic → STT → LLM → TTS → speaker loop with VAD + SmartTurn v3
turn-taking and automatic barge-in. Tools + the 3-tier safety model land in Phase 3
(see the marked hook below). Provider selection lives in config.py — this file is
provider-agnostic.

Run:  ./run.sh   (or:  uv run python main.py)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

# This machine mounts /tmp `noexec` (CachyOS hardening). Kokoro's phonemizer copies
# libespeak-ng.so into a tempdir and dlopen()s it; mapping an executable segment fails
# on a noexec mount ("failed to map segment from shared object"). Redirect TMPDIR to an
# exec-friendly dir in $HOME BEFORE anything touches tempfile (phonemizer uses
# tempfile.mkdtemp(), which honors TMPDIR).
_EXEC_TMP = os.path.expanduser("~/.cache/voice-agent/tmp")
os.makedirs(_EXEC_TMP, exist_ok=True)
os.environ["TMPDIR"] = _EXEC_TMP
tempfile.tempdir = None  # drop any cached value so TMPDIR is re-read

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.workers.runner import WorkerRunner

import config

load_dotenv(override=True)


def _setup_logging() -> None:
    """Mirror logs to files for offline/real-time analysis (in addition to the console).

    - logs/session.log    : everything at DEBUG (rolling) — for deep dives.
    - logs/transcript.log : only transcript-tagged lines — USER / BOT / STATUS / CONFIRM /
      BLOCKED / BARGE-IN (rolling) — a clean, greppable record of the conversation.
    Both roll by size and keep a few old copies, so they survive many short test runs.
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "session.log"),
        level="DEBUG", rotation="10 MB", retention=3, enqueue=True,
    )
    logger.add(
        os.path.join(log_dir, "transcript.log"),
        level="INFO",
        filter=lambda r: r["extra"].get("transcript"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="5 MB", retention=10, enqueue=True,
    )
    logger.bind(transcript=True).info(
        f"==== session start | BRAIN={os.environ.get('BRAIN', 'local')} "
        f"LLM_PROVIDER={os.environ.get('LLM_PROVIDER', '-')} "
        f"GAB_MODEL={os.environ.get('GAB_MODEL', '-')} ===="
    )


def _check_runtime_env() -> None:
    """Fail fast with a clear error if the selected LLM provider isn't usable."""
    # With an external brain (BRAIN=gabagent), the local LLM_PROVIDER is unused — its
    # readiness is the brain's concern (checked via /health when we start it).
    if os.environ.get("BRAIN", "local").lower() != "local":
        return

    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error(
            "ANTHROPIC_API_KEY is not set. Add it to ~/dev/voice-agent/.env "
            "(or switch LLM_PROVIDER=ollama for a fully local brain)."
        )
        raise SystemExit(1)

    if provider == "ollama":
        # Reach the daemon's /api/version (strip the OpenAI-compat /v1 suffix).
        import urllib.request

        root = (os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3].rstrip("/")
        try:
            with urllib.request.urlopen(f"{root}/api/version", timeout=2) as r:
                r.read()
        except Exception:
            logger.error(
                f"Ollama daemon not reachable at {root}. Start it with `ollama serve`, "
                "then re-run. (Or set LLM_BASE_URL if it's on another host/port.)"
            )
            raise SystemExit(1)


async def run() -> None:
    _setup_logging()
    _check_runtime_env()

    # Enumerate audio devices so the chosen mic/speaker indices are visible in the log.
    config.list_audio_devices()

    transport = config.build_transport()
    stt = config.build_stt()
    tts = config.build_tts()
    llm = config.build_llm()
    # Start an external brain (BRAIN=gabagent) — spawn `gab --voice-serve` + wait for /health.
    # No-op for local LLM brains.
    await config.start_brain(llm)

    # --- Phase 3 hook: register tools + safety here, then pass tools=ToolsSchema(...) below.
    # from tools import register_tools, build_tools_schema
    # register_tools(llm)
    # tools = build_tools_schema()
    tools = None  # talk-only for Phase 2

    # Universal context: system prompt as the first system-role message. Pipecat's
    # adapters translate it per provider (Anthropic extracts it into the API system param;
    # OpenAI/Ollama use it natively), so this stays provider-agnostic.
    context = LLMContext(
        messages=[{"role": "system", "content": config.SYSTEM_PROMPT}],
        **({"tools": tools} if tools is not None else {}),
    )

    # VAD goes on the user-aggregator params (1.3.x). SmartTurn v3 is the DEFAULT user-turn
    # stop strategy — we get end-of-turn detection for free without constructing it here.
    #
    # Half-duplex muting: without acoustic echo cancellation, open speakers feed the bot's
    # own TTS back into the mic, so it transcribes itself and self-interrupts in a loop.
    # AlwaysUserMuteStrategy mutes the mic while the bot is speaking, breaking the loop — at
    # the cost of barge-in. Default ON (works on speakers); set HALF_DUPLEX=0 with headphones
    # or a PipeWire echo-cancel source to restore full-duplex barge-in.
    half_duplex = os.environ.get("HALF_DUPLEX", "1") not in ("0", "false", "False")
    logger.info(
        "Turn-taking: "
        + ("half-duplex — mic muted while speaking, no barge-in (HALF_DUPLEX=0 for barge-in)"
           if half_duplex else "full-duplex — barge-in on (needs headphones or echo-cancel)")
    )
    # Turn-detection tuning: the VAD's `stop_secs` is how long a silence must last before a
    # turn is considered (possibly) over. The default (0.2s) is aggressive and fragments
    # natural multi-clause speech ("…start music… draft an email… can you do that?") into
    # separate turns. A more forgiving default (0.6s) absorbs short mid-thought pauses;
    # SmartTurn v3 still decides true end-of-turn. Tune via VAD_STOP_SECS.
    vad_stop_secs = float(os.environ.get("VAD_STOP_SECS", "0.6"))
    logger.info(f"VAD stop_secs={vad_stop_secs}s (pause tolerance before end-of-turn)")
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs)),
            user_mute_strategies=[AlwaysUserMuteStrategy()] if half_duplex else [],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),     # mic (PipeWire → PyAudio)
            stt,                   # Whisper (local)
            user_aggregator,       # VAD + SmartTurn v3 + user-side context
            llm,                   # Claude / OpenAI-compatible / local Ollama
            tts,                   # Kokoro (local)
            transport.output(),    # speaker
            assistant_aggregator,  # assistant-side context (after output, so it logs spoken text)
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    # Optional spoken greeting on start (confirms the TTS path end-to-end). GREET_ON_START=0
    # to disable. Uses a transient developer message — only meaningful for a raw-LLM brain;
    # an external brain (BRAIN=gabagent) consumes the latest *user* utterance, so skip it there
    # and just wait for the user to speak.
    greet = os.environ.get("GREET_ON_START", "1") not in ("0", "false", "False")
    if greet and os.environ.get("BRAIN", "local").lower() == "local":
        context.add_message(
            {"role": "developer", "content": "Greet the user in one short, friendly sentence."}
        )
        await worker.queue_frames([LLMRunFrame()])

    logger.info("Voice agent ready — start speaking. (Ctrl-C to quit.)")
    runner = WorkerRunner(handle_sigint=(sys.platform != "win32"))
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        # Tear down an external brain (stop `gab --voice-serve`); no-op for local brains.
        await config.stop_brain(llm)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
