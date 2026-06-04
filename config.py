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


# Canonical pipeline audio rate. Whisper STT and the Silero VAD both want 16 kHz; the mic is
# captured at whatever rate the device can open and resampled to this once, up front (see
# audio_resample.InputResampler + build_transport). Pin STT/VAD to this so they never adopt the
# device's native rate off the StartFrame.
PIPELINE_AUDIO_RATE = 16000


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
        # Pin to the pipeline rate so Whisper never adopts the device's native capture rate
        # (the mic is resampled to PIPELINE_AUDIO_RATE before it reaches STT). Whisper assumes
        # its input array is already at this rate — feeding it 48 kHz samples would make it
        # transcribe everything 3× too fast.
        return WhisperSTTService(
            settings=WhisperSTTService.Settings(model=model),
            device="auto",
            sample_rate=PIPELINE_AUDIO_RATE,
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


def _supported_input_rate(index: int | None, prefer: int = PIPELINE_AUDIO_RATE) -> int:
    """Return a sample rate the input device can actually open at.

    Prefer the pipeline rate (16 kHz → no resampling needed). Some PipeWire capture nodes only
    expose their native rate and reject a 16 kHz open with `-9997` — notably the `echo-cancel-source`
    (AEC Mic), which is 48 kHz only. In that case fall back to the device's native default rate;
    `InputResampler` converts it back to the pipeline rate. Returns `prefer` if probing fails (the
    transport will surface a clear error if even that doesn't open).
    """
    try:
        import pyaudio
    except Exception:  # pragma: no cover - diagnostic only
        return prefer
    pa = pyaudio.PyAudio()
    try:
        try:
            if pa.is_format_supported(
                prefer, input_device=index, input_channels=1, input_format=pyaudio.paInt16
            ):
                return prefer
        except Exception:
            pass  # prefer unsupported → fall back to the device's native rate
        try:
            d = (
                pa.get_device_info_by_index(index)
                if index is not None
                else pa.get_default_input_device_info()
            )
            return int(d.get("defaultSampleRate") or prefer)
        except Exception:
            return prefer
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
    # Capture at a rate the device can actually open; InputResampler normalizes to PIPELINE_AUDIO_RATE.
    capture_rate = _supported_input_rate(in_idx)
    if capture_rate != PIPELINE_AUDIO_RATE:
        logger.info(
            f"Audio input: device can't open at {PIPELINE_AUDIO_RATE}Hz — capturing at "
            f"{capture_rate}Hz and resampling to {PIPELINE_AUDIO_RATE}Hz (InputResampler)."
        )
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=capture_rate,
            input_device_index=in_idx,
            output_device_index=out_idx,
        )
    )


def build_input_resampler():
    """Resampler that normalizes mic capture to PIPELINE_AUDIO_RATE. Goes first in the pipeline.

    Always present (cheap, no-op when the device already opens at the pipeline rate) so the rest of
    the pipeline sees one consistent rate regardless of which mic/AEC source is bound.
    """
    from audio_resample import InputResampler

    return InputResampler(target_rate=PIPELINE_AUDIO_RATE)


def build_input_watchdog(restart=None, gate_state=None):
    """Input-stall watchdog (goes first, right after transport.input()), or None if disabled.

    Detects a mic-capture freeze — both 'no frames' and 'frames but silent' (the 2026-06-03 echo-cancel
    stall) — logs it loudly, and calls `restart` (best-effort capture kick) instead of hanging silently.
    Also emits a periodic heartbeat (with optional `gate_state`) so freeze vs lockout is one glance.
    Default ON; set `INPUT_STALL_SECS=0` to disable. See input_watchdog.InputStallDetector.
    """
    stall_secs = float(_env("INPUT_STALL_SECS", "5.0"))
    if stall_secs <= 0:
        return None
    silent_secs = float(_env("INPUT_SILENT_SECS", "8.0"))
    heartbeat_secs = float(_env("INPUT_HEARTBEAT_SECS", "10.0"))
    from input_watchdog import InputStallDetector

    logger.info(f"Input watchdog: ON (stall={stall_secs}s, silent={silent_secs}s, "
                f"heartbeat={heartbeat_secs}s, recover={'yes' if restart else 'log-only'})")
    return InputStallDetector(
        stall_secs=stall_secs, silent_secs=silent_secs, restart=restart,
        heartbeat_secs=heartbeat_secs, gate_state=gate_state,
    )


def build_vad_analyzer(*, sample_rate, params):
    """Silero VAD analyzer, optionally wrapped with the near-miss diagnostic (VAD_DEBUG=1).

    The diagnostic logs *why* an onset was missed (volume- vs confidence-gated) while tuning the
    "hear me over the music" knobs — off by default so there's no logging cost in normal runs.
    """
    if _env("VAD_DEBUG", "0") not in ("0", "false", "False", ""):
        from vad_diag import LoggingSileroVADAnalyzer

        logger.info("VAD diagnostic ON (VAD_DEBUG=1): logging near-misses to the transcript log.")
        return LoggingSileroVADAnalyzer(sample_rate=sample_rate, params=params)

    from pipecat.audio.vad.silero import SileroVADAnalyzer

    return SileroVADAnalyzer(sample_rate=sample_rate, params=params)


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
    confirm_grace = float(_env("DUCK_CONFIRM_GRACE", "2.5"))
    logger.info(
        f"Media ducking: on VAD speech onset (min_words={min_words}, "
        f"confirm_grace={confirm_grace}s, restore_grace={restore_grace}s)"
    )
    return MediaDuckController(
        llm.brain_client,
        llm.session_id,
        min_words=min_words,
        restore_grace=restore_grace,
        confirm_grace=confirm_grace,
        should_duck=lambda: not llm.is_sleeping,
        media_status=build_media_state_provider(llm),  # SHARED with the wake gate (no divergence)
    )


# --------------------------------------------------------------------------- wake word
def _wake_model_paths(names: str) -> list[str]:
    """Resolve comma-separated WAKE_WORD entries to ONNX paths.

    Each entry is either a path to a custom .onnx model (e.g. a trained "Aria" — used as-is) or a
    bundled openWakeWord pretrained name (hey_jarvis, alexa, …) resolved under the package's
    resources/models. Unknown entries are skipped with a warning.
    """
    import openwakeword

    res = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    paths = []
    for n in (x.strip() for x in names.split(",") if x.strip()):
        if n.endswith(".onnx") and os.path.exists(os.path.expanduser(n)):
            paths.append(os.path.expanduser(n))  # custom model path (e.g. trained "Aria")
            continue
        p = os.path.join(res, f"{n}_v0.1.onnx")
        if os.path.exists(p):
            paths.append(p)
        else:
            logger.warning(f"Wake word: {n!r} is neither a custom .onnx path nor a bundled model; skipping.")
    return paths


def build_media_state_provider(llm):
    """ONE shared async media-state reader for BOTH the wake gate and the media duck.

    Returns an async callable → `{"playing": bool, "kind": "audio"|"video"|None}` (the brain's neutral
    `/media/state` shape, with a value-scan fallback for an older brain).

    Why shared: the gate and duck used to each keep a *separate* 1s cache with *opposite* fail-modes
    (gate fail-open, duck fail-closed), so on a transient or cache-skew they disagreed live (gate gated
    while the duck logged "nothing playing"). One cache + one fail-mode means they can never diverge.

    Fails OPEN (`playing=False`) on any uncertainty: a transient must never lock the user out behind the
    wake word; a missed duck on a transient is benign. 1s TTL cache + in-flight coalescing so a talkative
    stretch can't spam `GET /media/state`. The duck reads `playing`; the gate also reads `kind` (so it can
    skip gating active *video* the user is watching).
    """
    import asyncio

    client = getattr(llm, "brain_client", None)
    session_id = getattr(llm, "session_id", "")
    debug = _env("WAKE_WORD_DEBUG", "0") not in ("0", "false", "False")
    state = {"v": None, "t": 0.0, "fut": None}

    async def media_status() -> dict:
        import time as _t

        now = _t.monotonic()
        if state["v"] is not None and now - state["t"] < 1.0:
            return state["v"]
        if state["fut"] is not None:  # coalesce concurrent queries onto one in-flight call
            return await state["fut"]
        ms = getattr(client, "media_state", None) if client is not None else None
        if ms is None:
            return {"playing": False, "kind": None}
        fut = asyncio.get_event_loop().create_future()
        state["fut"] = fut
        try:
            v = {"playing": False, "kind": None}
            try:
                st = await ms(session_id)
                if st:
                    playing = bool(st["playing"]) if "playing" in st \
                        else any(str(x).lower() == "playing" for x in st.values())
                    v = {"playing": playing, "kind": st.get("kind")}
            except Exception as e:  # noqa: BLE001 - fail open (not playing) + surface the type
                if debug:
                    logger.debug(
                        f"media-state: query failed (fail-open=not-playing): {type(e).__name__}: {e}"
                    )
            state["v"], state["t"] = v, _t.monotonic()
            fut.set_result(v)
            return v
        finally:
            state["fut"] = None

    return media_status


def build_wake_word_gate(llm):
    """Wake-word gate (opt-in via WAKE_WORD), or None. See wake_word.WakeWordGate.

    Inserted right after the InputResampler (16 kHz). While media plays it requires a wake word before
    commands reach STT (media_only, default on) — sidestepping STT-over-music — and pre-ducks on wake.
    """
    names = _env("WAKE_WORD", "")
    if not names:
        return None  # not configured → open-mic as before

    from wake_word import WakeWordGate
    from brains.brain_llm_service import BrainLLMService

    # Engine: openWakeWord (default), nanowakeword, or Porcupine. nano/porcupine are the wake-over-music
    # paths (the AEC double-talk clamp crushes the bare oww "Aria"); see wake_nano / wake_porcupine.
    engine = _env("WAKE_WORD_ENGINE", "oww").lower()
    threshold = float(_env("WAKE_WORD_THRESHOLD", "0.5"))
    vad_threshold = float(_env("WAKE_WORD_VAD_THRESHOLD", "0.5"))
    window_secs = float(_env("WAKE_WINDOW_SECS", "15"))
    media_only = _env("WAKE_WORD_MEDIA_ONLY", "1") not in ("0", "false", "False")
    debug = _env("WAKE_WORD_DEBUG", "0") not in ("0", "false", "False")
    debug_floor = float(_env("WAKE_WORD_DEBUG_FLOOR", "0.05"))  # low, to SEE sub-0.2 "Aria" over music
    escape_floor = float(_env("WAKE_ESCAPE_FLOOR", "0.15"))
    escape_count = int(_env("WAKE_ESCAPE_COUNT", "3"))  # 0 disables the lockout escape
    escape_secs = float(_env("WAKE_ESCAPE_SECS", "12"))
    gate_video = _env("WAKE_GATE_VIDEO", "0") not in ("0", "false", "False")  # default: don't gate video
    speex_ns = False  # openWakeWord-only; set below when that engine is selected
    extra_log = ""

    if engine in ("porcupine", "pv"):
        from wake_porcupine import PorcupineModel

        # Porcupine keyword is a platform-specific .ppn (NOT an .onnx) → resolve straight from WAKE_WORD.
        kw = os.path.expanduser(names.strip())
        if not (kw.endswith(".ppn") and os.path.exists(kw)):
            logger.warning(f"Wake word: Porcupine needs a .ppn path in WAKE_WORD; {names!r} not found; gate disabled.")
            return None
        access_key = _env("PORCUPINE_ACCESS_KEY", "")
        sensitivity = float(_env("WAKE_WORD_SENSITIVITY", "0.7"))
        key = (os.path.basename(kw).split("_")[0].replace("-", " ").strip() or "wake")
        try:
            oww = PorcupineModel([kw], access_key=access_key, sensitivity=sensitivity, key=key)
        except Exception as e:  # noqa: BLE001 - bad key / expired .ppn / platform mismatch → disable, don't crash
            logger.warning(f"Wake word: Porcupine init failed ({type(e).__name__}: {e}); gate disabled.")
            return None
        extra_log = f" sensitivity={sensitivity}"
    elif engine in ("nano", "nanowakeword"):
        paths = _wake_model_paths(names)
        if not paths:
            logger.warning("Wake word: no valid models resolved; gate disabled.")
            return None
        from wake_nano import NanoWakeWordModel

        oww = NanoWakeWordModel(paths)  # same audio contract (int16/16k/1280); same .models/.predict shape
    else:
        paths = _wake_model_paths(names)
        if not paths:
            logger.warning("Wake word: no valid models resolved; gate disabled.")
            return None
        from openwakeword.model import Model

        # Speex noise suppression: openWakeWord pre-processes the audio to suppress *stationary* background
        # (music) before the melspectrogram — its documented lever for hear-over-media false-rejects. Off by
        # default (A/B it); graceful fallback if the speexdsp-ns wheel is missing/unsupported on this box.
        speex_ns = _env("WAKE_WORD_SPEEX_NS", "0") not in ("0", "false", "False")
        try:
            oww = Model(wakeword_model_paths=paths, vad_threshold=vad_threshold,
                        enable_speex_noise_suppression=speex_ns)
        except Exception as e:  # noqa: BLE001 - speexdsp-ns unavailable → run without rather than crash
            if not speex_ns:
                raise
            logger.warning(f"Wake word: Speex NS unavailable ({type(e).__name__}: {e}); running without it.")
            speex_ns = False
            oww = Model(wakeword_model_paths=paths, vad_threshold=vad_threshold)

    brain_client = session_id = media_status = None
    if isinstance(llm, BrainLLMService):
        brain_client = llm.brain_client
        session_id = llm.session_id
        media_status = build_media_state_provider(llm)  # SHARED with the media duck (no divergence)
    logger.info(
        f"Wake word: gate ON engine={engine} models={list(oww.models.keys())} threshold={threshold} "
        f"media_only={media_only and media_status is not None} gate_video={gate_video} window={window_secs}s"
        + (f" escape={escape_count}@{escape_floor}/{escape_secs:.0f}s" if escape_count else " escape=off")
        + (" speex_ns=on" if speex_ns else "")
        + extra_log
        + (f" debug=on floor={debug_floor}" if debug else "")
    )
    return WakeWordGate(
        oww,
        threshold=threshold,
        window_secs=window_secs,
        brain_client=brain_client,
        session_id=session_id or "",
        media_status=media_status,
        media_only=media_only,
        gate_video=gate_video,
        debug=debug,
        debug_floor=debug_floor,
        escape_floor=escape_floor,
        escape_count=escape_count,
        escape_secs=escape_secs,
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
