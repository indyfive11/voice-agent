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

# Stamp a stable, collision-proof property on every PipeWire stream this process opens (mic + TTS output)
# so the gabagent brain's universal local-media duck can EXCLUDE Aria's own TTS output exactly (vs a
# node-name heuristic) and never mute her. PipeWire reads PIPEWIRE_PROPS at stream-connect, so — like
# TMPDIR above — it must be set BEFORE any audio stream opens. The brain skips any sink-input whose
# `pactl list sink-inputs` Properties block contains gabagent.duck_exclude = "1". NB: the ALSA→PipeWire
# path (our PyAudio output, node alsa_playback.python3.12) honours PIPEWIRE_PROPS, NOT PULSE_PROP —
# verified on this box 2026-06-04. The mic source-output also gets it (harmless: the duck only scans
# sink-inputs). setdefault so an explicit override wins.
os.environ.setdefault("PIPEWIRE_PROPS", "{ gabagent.duck_exclude = 1 }")

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

from turn_cap import MaxTurnDurationUserTurnStopStrategy
from turn_mute import BotThinkingMuteStrategy
from response_latency import ResponseLatencyObserver
from tts_gain import TTSGainProcessor
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    default_user_turn_start_strategies,
)
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

    # On a cold boot/login WirePlumber brings the AEC echo-cancel mic up asynchronously, so it may
    # not be in the PipeWire graph yet. Wait for it here (owns the device contract; lets the autostart
    # wrapper stay "wait for PipeWire, exec") before enumerating — on timeout we fall over to the default
    # mic, which build_transport's name-resolve does for us. Runtime AEC death is input_watchdog's job.
    await config.wait_for_aec_mic()

    # Enumerate audio devices so the chosen mic/speaker indices are visible in the log.
    config.list_audio_devices()
    # Heads-up if the default output sink is muted at startup (a stale device mute silences Aria with no
    # obvious cause — 2026-06-15). Diagnostic only; the app never force-unmutes the user's master volume.
    await config.warn_if_output_muted()

    transport = config.build_transport()
    stt = config.build_stt()
    tts = config.build_tts()
    # Attenuate Aria's TTS so her (duck-excluded) voice isn't louder than the ducked music. TTS_GAIN
    # (0..1, default 0.55; 1.0 = no change) — tune live to taste. Keeps duck-exclude; just lowers her level.
    # 0.55: the user judged 0.6 "about right, a teeny bit down" on the 2026-06-04 live tune.
    tts_gain = float(os.environ.get("TTS_GAIN", "0.55"))
    tts_gain_proc = TTSGainProcessor(gain=tts_gain)
    if tts_gain < 1.0:
        logger.info(f"TTS gain: attenuating Aria's output to {tts_gain:.2f} (TTS_GAIN; keeps duck-exclude).")
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
    # AlwaysUserMuteStrategy only mutes while the bot is *speaking* — it leaves the brain's "thinking"
    # gap (turn opened, no audio yet) unmuted, so a stray VAD onset there cancels the silent in-flight
    # turn (the 91% empty `said so far: ''` barge-ins on the higher-latency Claude backend). Add a second
    # strategy that also mutes through that gap (the aggregator OR-combines them). Half-duplex only — it's
    # the "no barge-in" mode anyway; default on, MUTE_DURING_THINKING=0 reverts to speaking-only muting.
    mute_during_thinking = half_duplex and os.environ.get(
        "MUTE_DURING_THINKING", "1"
    ) not in ("0", "false", "False")
    logger.info(
        "Turn-taking: "
        + ("half-duplex — mic muted while speaking, no barge-in (HALF_DUPLEX=0 for barge-in)"
           if half_duplex else "full-duplex — barge-in on (needs headphones or echo-cancel)")
        + (" + muting through the brain's thinking gap (MUTE_DURING_THINKING=0 to disable)"
           if mute_during_thinking else "")
    )
    # Turn-detection tuning. TWO distinct `stop_secs` knobs — don't conflate them:
    #   * VAD `stop_secs` — silence before the VAD says "user stopped", which *triggers*
    #     SmartTurn inference. SmartTurn v3 is trained against 0.2s segmentation and then
    #     dynamically decides the *real* end-of-turn itself, so 0.2 is correct here. (A higher
    #     value fights SmartTurn and trips a startup warning.)
    #   * SmartTurn `stop_secs` — SmartTurn's hard max-silence fallback (default 3s): when a
    #     mid-sentence pause hits it, the turn is force-completed regardless of the model verdict.
    #     That 3s default is what fragmented the user's multi-clause speech ("…start music… Tidal…
    #     recommendations…") into separate turns. Raise it (~4s) so a thinking pause doesn't split
    #     the turn; SmartTurn still ends the turn promptly once it judges the user is done.
    vad_stop_secs = float(os.environ.get("VAD_STOP_SECS", "0.2"))
    smart_turn_stop_secs = float(os.environ.get("SMART_TURN_STOP_SECS", "4.0"))
    # VAD sensitivity (the "hear me over the music" ruler). Silero AND-gates on confidence AND volume
    # (pipecat vad_analyzer.py: speaking = confidence>=VAD_CONFIDENCE and volume>=VAD_MIN_VOLUME), so a
    # voice partly masked by playing media can fail *either* gate and never start a turn. We start more
    # sensitive than pipecat's defaults (0.7/0.6) and pair it with a gentle brain-side ambient duck;
    # tune live: lower = hears you over louder music but risks false triggers, raise = the reverse.
    vad_confidence = float(os.environ.get("VAD_CONFIDENCE", "0.6"))
    vad_min_volume = float(os.environ.get("VAD_MIN_VOLUME", "0.4"))
    # Wall-clock turn cap: SmartTurn has none (max_duration_secs is just its model window), so a long
    # continuous ramble can hang the turn — force-complete it past this many seconds (with transcript).
    max_turn_secs = float(os.environ.get("MAX_TURN_SECS", "15"))
    logger.info(
        f"Turn detection: VAD stop_secs={vad_stop_secs}s confidence={vad_confidence} "
        f"min_volume={vad_min_volume} → SmartTurn v3 "
        f"(stop_secs={smart_turn_stop_secs}s max-silence fallback, max_turn={max_turn_secs}s cap)"
    )
    turn_analyzer = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            stop_secs=smart_turn_stop_secs, max_duration_secs=8, pre_speech_ms=500
        )
    )
    # Start strategies: half-duplex keeps the default VAD-onset start. Full-duplex (mic open
    # during TTS) gates turn-start on transcribed words via MinWords so residual TTS/media bleed
    # can't false-start a turn / barge-in (min_words applies only while the bot is speaking; a
    # single word starts a turn otherwise). Tune the interruption threshold via BARGE_IN_MIN_WORDS.
    if half_duplex:
        start_strategies = default_user_turn_start_strategies()
    else:
        min_words = int(os.environ.get("BARGE_IN_MIN_WORDS", "2"))
        start_strategies = [
            MinWordsUserTurnStartStrategy(min_words=min_words, use_interim=True)
        ]
        logger.info(f"Full-duplex barge-in gated on min_words={min_words}")
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Pinned to the pipeline rate: the mic is resampled to PIPELINE_AUDIO_RATE up front
            # (InputResampler), so the VAD must not adopt the device's native capture rate.
            vad_analyzer=config.build_vad_analyzer(
                sample_rate=config.PIPELINE_AUDIO_RATE,
                params=VADParams(
                    confidence=vad_confidence,
                    min_volume=vad_min_volume,
                    stop_secs=vad_stop_secs,
                ),
            ),
            user_mute_strategies=(
                [AlwaysUserMuteStrategy()]
                + ([BotThinkingMuteStrategy()] if mute_during_thinking else [])
            ) if half_duplex else [],
            user_turn_strategies=UserTurnStrategies(
                start=start_strategies,
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer),
                    # Safety cap so a long continuous ramble can't hang the turn (SmartTurn won't
                    # end it; this force-completes past max_turn_secs when there's transcribed text).
                    MaxTurnDurationUserTurnStopStrategy(max_turn_secs=max_turn_secs),
                ],
            ),
        ),
    )

    # Media-duck controller goes right after STT — it's the only spot that sees transcription
    # frames (the user aggregator consumes them). No-op / absent for a raw-LLM brain.
    media_duck = config.build_media_duck(llm)

    # Wake-word gate (opt-in via WAKE_WORD) sits before STT — while media plays it requires the wake
    # word before audio reaches STT (sidesteps STT-over-music) and pre-ducks on wake. None when unset.
    wake_gate = config.build_wake_word_gate(llm)
    # Tell the brain whether an acoustic gate exists: with one, going to sleep forces it active so only
    # "hey aria" can wake her (no STT on ambient TV); without one, sleep falls back to text-phrase wake.
    if hasattr(llm, "set_acoustic_wake_gated"):
        llm.set_acoustic_wake_gated(wake_gate is not None)

    # Input-stall watchdog: best-effort recovery for a frozen/silent mic capture. The restart "kicks"
    # the PortAudio stream in place (stop→start) — the least-invasive revive that doesn't tear down the
    # base-input task machinery. Goes FIRST so it times the rawest mic frames.
    async def _restart_capture(_start_frame):
        inp = transport.input()
        stream = getattr(inp, "_in_stream", None)
        if stream is None:
            return False
        stream.stop_stream()
        stream.start_stream()
        return True

    input_watchdog = config.build_input_watchdog(
        restart=_restart_capture,
        gate_state=(wake_gate.hb_state if wake_gate else None),  # heartbeat shows gate state too
    )

    pipeline = Pipeline(
        [
            transport.input(),     # mic (PipeWire → PyAudio)
            *([input_watchdog] if input_watchdog else []),  # stall watchdog (times raw mic frames)
            config.build_input_resampler(),  # normalize capture → 16 kHz (AEC source is 48 kHz-only)
            *([wake_gate] if wake_gate else []),  # wake-word gate (opt-in; mutes until wake while media plays)
            stt,                   # Whisper (local)
            *([media_duck] if media_duck else []),  # duck media on confirmed speech (gabagent brain)
            user_aggregator,       # VAD + SmartTurn v3 + user-side context
            llm,                   # Claude / OpenAI-compatible / local Ollama
            tts,                   # Kokoro (local)
            tts_gain_proc,         # attenuate Aria's voice (TTS_GAIN) so she's not louder than ducked music
            transport.output(),    # speaker
            assistant_aggregator,  # assistant-side context (after output, so it logs spoken text)
        ]
    )

    # Instrumentation (PIPELINE_OBSERVERS=1 default; 0 to silence) — stops the live-test log-grep loop.
    # MetricsLogObserver = per-service TTFB/usage; ResponseLatencyObserver = real user-perceived latency
    # (user-stop → bot-start). The on_turn_ended handler below logs the conversation-turn SPAN (which
    # includes idle/false-starts — NOT latency; labeled "span" so a big number doesn't read as a hang)
    # plus the was_interrupted half-duplex signal. All already in pipecat 1.3.0.
    observe = os.environ.get("PIPELINE_OBSERVERS", "1") not in ("0", "false", "False")
    _observers = [MetricsLogObserver(), ResponseLatencyObserver()] if observe else []
    # Tail (PIPELINE_TAIL=1) — live terminal dashboard: binds a ws server on :9292; run `pipecat tail`
    # in a second terminal to view logs/transcript/audio-levels/metrics. Off by default (binds a port).
    if os.environ.get("PIPELINE_TAIL", "0") not in ("0", "false", "False"):
        from pipecat_tail.observer import TailObserver  # lazy: pulls textual/plotext only when enabled
        _observers.append(TailObserver())
        logger.info("Tail observer on ws://localhost:9292 — run `pipecat tail` in another terminal.")
    # Idle timeout: pipecat's PipelineWorker defaults to idle_timeout_secs=300 and
    # cancel_on_idle_timeout=True, and its idle clock ONLY resets on Bot/UserSpeakingFrame.
    # For an ambient wake-word listener that is fatal: while music plays the gate mutes the mic
    # (no UserSpeakingFrame) and nobody triggers a reply (no BotSpeakingFrame), so after 5 min of
    # silence pipecat decides the pipeline is hung and cancels the whole worker — the agent "crashes"
    # for just sitting and listening. Sitting muted waiting for "hey aria" IS the intended steady
    # state here, so disable the idle-cancel by default. input_watchdog.py already monitors liveness
    # independently. PIPELINE_IDLE_TIMEOUT_SECS>0 re-enables pipecat's timer if ever wanted.
    idle_secs = float(os.environ.get("PIPELINE_IDLE_TIMEOUT_SECS", "0")) or None
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=_observers,
        idle_timeout_secs=idle_secs,
        cancel_on_idle_timeout=idle_secs is not None,
    )
    logger.info(
        "Idle timeout: "
        + (f"{idle_secs:.0f}s (pipecat cancel-on-idle ON)" if idle_secs
           else "disabled — ambient listener stays up while idle/muted")
    )
    if observe and worker.turn_tracking_observer:
        @worker.turn_tracking_observer.event_handler("on_turn_ended")
        async def _on_turn_ended(_obs, turn_count, duration, was_interrupted):  # noqa: ANN001
            # `duration` is the conversation-turn SPAN (incl. idle + transcript-less false-starts), not
            # latency — see RESPONSE | … for the real response time.
            logger.bind(transcript=True).info(
                f"TURN  | #{turn_count} span={duration:.2f}s"
                + (" INTERRUPTED" if was_interrupted else "")
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
    # Guard the TTS output stream against PipeWire module-stream-restore stranding it silent (which would
    # leave the user muted under half-duplex and break the conversation). In-app, dies with the app.
    pin_task = asyncio.create_task(config.pin_output_stream_volume())
    runner = WorkerRunner(handle_sigint=(sys.platform != "win32"))
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        pin_task.cancel()
        # Tear down an external brain (stop `gab --voice-serve`); no-op for local brains.
        await config.stop_brain(llm)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
