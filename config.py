"""Provider/model factories — the swappable layer (Pipecat 1.3.x).

`main.py` never imports a concrete STT/TTS/LLM class; it asks the factories here.
Switching any layer to a cloud provider is an env change + that provider's extra/key,
with `main.py` untouched. Provider-specific imports are done lazily inside each branch
so the optional extras (deepgram/cartesia/elevenlabs/openai) only need to be installed
when actually selected.

System prompt: we put it as the first ``system``-role message in the universal
``LLMContext`` (see main.py). Pipecat's per-provider adapters translate it correctly —
the Anthropic adapter extracts it into the API ``system`` param, OpenAI/Ollama use it
natively — so no provider branching is needed for the prompt.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


# --------------------------------------------------------------------------- env helpers
def _env(key: str, default: str | None = None) -> str | None:
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def _env_int(key: str) -> int | None:
    val = _env(key)
    return int(val) if val is not None else None


# --------------------------------------------------------------------------- system prompt
# Voice-assistant framing. The full 3-tier safety guardrail wording is added in Phase 3
# once tools exist; for the Phase-2 talk-only loop this keeps replies short and speakable.
SYSTEM_PROMPT = (
    "You are a real-time spoken assistant. You are talking to the user out loud, so:\n"
    "- Keep replies short, natural, and easy to say aloud — usually one or two sentences.\n"
    "- Never use markdown, bullet points, code blocks, or emoji; speak in plain prose.\n"
    "- Don't spell things out letter by letter or read punctuation; just talk.\n"
    "- If you don't know or didn't catch something, say so briefly and ask."
)


# --------------------------------------------------------------------------- STT factory
def build_stt():
    """Speech-to-text. whisper (local, default) | deepgram (cloud)."""
    provider = (_env("STT_PROVIDER", "whisper") or "whisper").lower()

    if provider == "whisper":
        from pipecat.services.whisper.stt import WhisperSTTService

        # `model` accepts a raw faster-whisper size string ("small.en", "base.en", …)
        # or a Model enum value. small.en is the plan default; device="auto" → CPU on
        # this AMD box (no CUDA), which is faster-than-real-time here.
        model = _env("STT_MODEL", "small.en")
        logger.info(f"STT: local Whisper (faster-whisper) model={model!r} device=auto")
        return WhisperSTTService(
            settings=WhisperSTTService.Settings(model=model), device="auto"
        )

    if provider == "deepgram":
        from pipecat.services.deepgram.stt import DeepgramSTTService  # extra: [deepgram]

        logger.info("STT: Deepgram (cloud)")
        return DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    raise ValueError(f"Unknown STT_PROVIDER={provider!r} (expected whisper|deepgram)")


# --------------------------------------------------------------------------- TTS factory
def build_tts():
    """Text-to-speech. kokoro (local, default) | cartesia | elevenlabs (cloud)."""
    provider = (_env("TTS_PROVIDER", "kokoro") or "kokoro").lower()

    if provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        # model/voices ONNX files auto-download on first use. Kokoro 1.3.x REQUIRES an
        # explicit voice — default to af_heart; override with TTS_VOICE (54 voices in
        # voices-v1.0.bin, e.g. af_bella, af_sarah, am_michael, bf_emma).
        voice = _env("TTS_VOICE", "af_heart")
        logger.info(f"TTS: local Kokoro (onnx) voice={voice}")
        return KokoroTTSService(settings=KokoroTTSService.Settings(voice=voice))

    if provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaTTSService  # extra: [cartesia]

        logger.info("TTS: Cartesia (cloud)")
        return CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            voice_id=_env("TTS_VOICE", ""),
        )

    if provider == "elevenlabs":
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService  # extra: [elevenlabs]

        logger.info("TTS: ElevenLabs (cloud)")
        return ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            voice_id=_env("TTS_VOICE", ""),
        )

    raise ValueError(f"Unknown TTS_PROVIDER={provider!r} (expected kokoro|cartesia|elevenlabs)")


# --------------------------------------------------------------------------- LLM factory
def build_llm():
    """The 'brain'. anthropic (default) | openai (any base_url) | ollama (local).

    Returns just the LLM service; `main.py` wraps it in the universal LLMContext /
    LLMContextAggregatorPair, so this stays provider-agnostic for the caller.

    BRAIN selects an *external* agent as the brain (e.g. gabagent) instead of a raw LLM:
    BRAIN=local (default) uses the LLM_PROVIDER path below; BRAIN=gabagent will return a
    BrainLLMService talking to gabagent over the Brain protocol.
    """
    brain = (_env("BRAIN", "local") or "local").lower()
    if brain == "gabagent":
        from brains.brain_llm_service import BrainLLMService
        from brains.http_brain_client import HttpBrainClient

        gab_bin = _env("GAB_BIN", os.path.expanduser("~/dev/gabagent/.venv/bin/gab"))
        port = _env("GAB_PORT", "8765")
        project = _env("GAB_PROJECT_DIR", os.getcwd())
        base_url = f"http://127.0.0.1:{port}"
        launch = [gab_bin, "--voice-serve", "--port", str(port), "--cwd", project]
        logger.info(f"Brain: gabagent (HTTP/SSE {base_url}, project={project})")
        return BrainLLMService(HttpBrainClient(base_url, launch=launch, cwd=project))
    if brain != "local":
        raise ValueError(f"Unknown BRAIN={brain!r} (expected local|gabagent)")

    provider = (_env("LLM_PROVIDER", "anthropic") or "anthropic").lower()
    model = _env("LLM_MODEL")
    base_url = _env("LLM_BASE_URL")

    # 1.3.x: configure model/generation via settings=Service.Settings(...). Passing
    # model=/params= still works but is deprecated, so we use the settings form.
    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        model = model or "claude-sonnet-4-6"  # Anthropic service default is None — set it.
        logger.info(f"LLM: Anthropic (cloud) model={model} prompt_caching=on")
        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(
                model=model,
                enable_prompt_caching=True,
                max_tokens=int(_env("LLM_MAX_TOKENS", "1024")),
            ),
        )

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        # Any OpenAI-compatible endpoint: OpenAI, OpenRouter, vLLM, LM Studio, llama.cpp…
        logger.info(f"LLM: OpenAI-compatible model={model} base_url={base_url or 'default'}")
        return OpenAILLMService(
            settings=OpenAILLMService.Settings(model=model or "gpt-4o-mini"),
            api_key=_env("OPENAI_API_KEY", "not-needed"),
            base_url=base_url,
        )

    if provider == "ollama":
        from pipecat.services.ollama.llm import OLLamaLLMService

        # Fully local brain on the RX 7900 XT (ROCm). No API key, nothing leaves the box.
        model = model or "llama3.1:8b"
        logger.info(f"LLM: local Ollama model={model} base_url={base_url or 'localhost:11434/v1'}")
        return OLLamaLLMService(
            settings=OLLamaLLMService.Settings(model=model),
            base_url=base_url or "http://localhost:11434/v1",
        )

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r} (expected anthropic|openai|ollama)")


# --------------------------------------------------------------------------- audio transport
def list_audio_devices() -> None:
    """Log PyAudio input/output devices + their indices, so the C110 can be pinned."""
    try:
        import pyaudio
    except Exception as e:  # pragma: no cover - diagnostic only
        logger.warning(f"Could not enumerate audio devices: {e}")
        return
    pa = pyaudio.PyAudio()
    try:
        logger.info("Audio devices (PyAudio indices):")
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            io = []
            if d.get("maxInputChannels"):
                io.append(f"in×{d['maxInputChannels']}")
            if d.get("maxOutputChannels"):
                io.append(f"out×{d['maxOutputChannels']}")
            logger.info(f"  [{i:2d}] {d['name']}  ({', '.join(io)})")
    finally:
        pa.terminate()


def _resolve_device_index(name: str | None, *, want_output: bool) -> int | None:
    """Resolve a PyAudio device index by case-insensitive substring match on its name.

    Returns None if `name` is unset or nothing matches (caller falls back to the OS default).
    Pinning by name survives index reshuffles; pin to avoid the silent-sink failure where the OS
    default changes mid-session (e.g. a monitor/HDMI move) and the already-open stream keeps writing
    to the now-silent original device.
    """
    if not name:
        return None
    try:
        import pyaudio
    except Exception as e:  # pragma: no cover - diagnostic only
        logger.warning(f"Cannot resolve audio device by name ({e}); using default.")
        return None
    kind = "output" if want_output else "input"
    pa = pyaudio.PyAudio()
    try:
        nlow = name.lower()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            chans = d.get("maxOutputChannels" if want_output else "maxInputChannels") or 0
            if chans > 0 and nlow in str(d.get("name", "")).lower():
                logger.info(f"Audio: matched {kind} name~={name!r} → [{i}] {d['name']}")
                return i
        logger.warning(f"Audio: no {kind} device matched name~={name!r}; using system default.")
    finally:
        pa.terminate()
    return None


def _readback_device(index: int | None, *, want_output: bool) -> None:
    """Log the device actually selected, so a silent/wrong sink is visible in the log.

    With index=None the OS default is used — which can silently change at runtime (the monitor-move
    failure). Warns if the resolved device has zero channels of the needed direction. Pipecat's
    Bot-started/stopped-speaking frames are bookkeeping around the TTS queue and fire even when
    output went nowhere, so this read-back is the only place a dead output sink surfaces.
    """
    try:
        import pyaudio
    except Exception:  # pragma: no cover - diagnostic only
        return
    kind = "output" if want_output else "input"
    pa = pyaudio.PyAudio()
    try:
        try:
            if index is not None:
                d = pa.get_device_info_by_index(index)
            else:
                d = (pa.get_default_output_device_info() if want_output
                     else pa.get_default_input_device_info())
        except Exception as e:
            logger.warning(f"Audio {kind}: could not read back device (index={index}): {e}")
            return
        chans = d.get("maxOutputChannels" if want_output else "maxInputChannels") or 0
        msg = f"Audio {kind} in use: [{d.get('index')}] {d.get('name')!r} ({kind} channels={chans})"
        if index is None:
            msg += f" — system DEFAULT (can change at runtime; pin with AUDIO_{kind.upper()}_DEVICE_NAME)"
        (logger.warning if chans == 0 else logger.info)(msg)
    finally:
        pa.terminate()


def build_transport():
    """LocalAudioTransport bound to the mic/speaker (PipeWire via PyAudio).

    Mic = Logitech C110 (mono). Pin devices explicitly with AUDIO_{INPUT,OUTPUT}_DEVICE_INDEX
    (numeric, from the `list_audio_devices()` log) or AUDIO_{INPUT,OUTPUT}_DEVICE_NAME (substring,
    resolved at startup); index wins. Unset → PyAudio's system default (the C110 mic on this box,
    but the default output can change mid-session — prefer pinning the output by name).
    """
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    in_idx = _env_int("AUDIO_INPUT_DEVICE_INDEX")
    if in_idx is None:
        in_idx = _resolve_device_index(_env("AUDIO_INPUT_DEVICE_NAME"), want_output=False)
    out_idx = _env_int("AUDIO_OUTPUT_DEVICE_INDEX")
    if out_idx is None:
        out_idx = _resolve_device_index(_env("AUDIO_OUTPUT_DEVICE_NAME"), want_output=True)
    logger.info(f"Transport: local audio  input_device_index={in_idx}  output_device_index={out_idx}")
    _readback_device(in_idx, want_output=False)
    _readback_device(out_idx, want_output=True)
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=in_idx,
            output_device_index=out_idx,
        )
    )


# --------------------------------------------------------------------------- media ducking
def build_media_duck(llm):
    """Build the MediaDuckController for an external brain, or None for a raw LLM.

    Inserted right after STT (see main.py) so it can see transcription frames — it ducks the brain's
    media on *confirmed* user speech and restores when Aria finishes. Shares the brain's session_id
    so `/media/duck` correlates, and gates on the brain's sleep state.
    """
    from brains.brain_llm_service import BrainLLMService

    if not isinstance(llm, BrainLLMService):
        return None
    from brains.media_duck import MediaDuckController

    min_words = int(_env("DUCK_MIN_WORDS", "2"))
    restore_grace = float(_env("DUCK_RESTORE_GRACE", "8.0"))
    logger.info(f"Media ducking: on confirmed speech (min_words={min_words}, restore_grace={restore_grace}s)")
    return MediaDuckController(
        llm.brain_client,
        llm.session_id,
        min_words=min_words,
        restore_grace=restore_grace,
        should_duck=lambda: not llm.is_sleeping,
    )


# --------------------------------------------------------------------------- brain lifecycle
async def start_brain(llm) -> None:
    """If `llm` wraps an external brain that needs starting (spawn + health), do it.

    No-op for plain LLM services. Called by main.py before the pipeline runs.
    """
    client = getattr(llm, "brain_client", None)
    start = getattr(client, "start", None)
    if start is not None:
        await start()


async def stop_brain(llm) -> None:
    """Tear down an external brain (close transport + stop any spawned process)."""
    client = getattr(llm, "brain_client", None)
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()
