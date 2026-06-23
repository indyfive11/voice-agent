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


def _room_id() -> str:
    """The durable per-room/device routing key, resolved ONCE so the STT request and the brain payloads
    always agree. ROOM_ID env wins; default = the hostname (a sensible per-device room name)."""
    return _env("ROOM_ID") or os.uname().nodename


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
    """Speech-to-text. whisper (local, default) | deepgram (cloud) | remote (offload to an STT service).

    `remote` keeps wake/VAD/endpointing/TTS on this device and offloads only STT to a standalone service
    (stt_service/server.py) over HTTP — for a thin client too slow for local Whisper (a Pi-4 is ~40s/
    utterance). See RemoteSTTService.
    """
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

    if provider == "remote":
        from remote_stt import RemoteSTTService

        url = _env("STT_REMOTE_URL")
        if not url:
            raise ValueError(
                "STT_PROVIDER=remote requires STT_REMOTE_URL (e.g. http://192.168.1.100:8770)"
            )
        room_id = _room_id()
        token = _env("STT_REMOTE_TOKEN")
        timeout = float(_env("STT_REMOTE_TIMEOUT", "30") or 30)
        logger.info(f"STT: remote offload url={url} room={room_id} auth={'on' if token else 'off'}")
        # Pin sample_rate to the pipeline rate — the segmented WAV we POST is at this rate, and the
        # EM service reads it from the WAV header; never adopt the device's native rate here.
        return RemoteSTTService(
            url, room_id=room_id, auth_token=token, timeout=timeout, sample_rate=PIPELINE_AUDIO_RATE
        )

    raise ValueError(f"Unknown STT_PROVIDER={provider!r} (expected whisper|deepgram|remote)")


# --------------------------------------------------------------------------- TTS factory
def build_tts():
    """TTS. kokoro (local, default) | piper (local, fast/weak-HW) | remote (offload Kokoro to a service) |
    cartesia | elevenlabs (cloud).

    `piper` = lightweight local CPU TTS for thin clients too slow for Kokoro (a Pi-4 is ~16s/utterance), but
    lower voice quality. `remote` = offload to a standalone Kokoro service (tts_service/server.py) on a fast
    host — Kokoro's good voice for a thin client that can't run it locally (the Pi-4-class reference profile:
    STT + TTS both on the brain host). See RemoteTTSService.
    """
    provider = (_env("TTS_PROVIDER", "kokoro") or "kokoro").lower()

    if provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        # model/voices ONNX files auto-download on first use. Kokoro 1.3.x REQUIRES an
        # explicit voice — default to af_heart; override with TTS_VOICE (54 voices in
        # voices-v1.0.bin, e.g. af_bella, af_sarah, am_michael, bf_emma).
        voice = _env("TTS_VOICE", "af_heart")
        # Pin Kokoro's EMIT-rate only when AUDIO_OUTPUT_SAMPLE_RATE is set (a fixed-rate output device, e.g.
        # the Pi's EMEET = 48 kHz-only). It MUST equal build_transport's audio_out_sample_rate: Kokoro
        # resamples its 24 kHz model output to this once (kokoro/tts.py), the device opens at the same rate,
        # and pipecat does NOT re-resample the TTS to the transport rate — so matching the two ends is the
        # whole fix (a mismatch plays the audio at the wrong clock = chipmunk). Unset (EM) → omit the kwarg →
        # framework default, byte-for-byte the known-good path. Never auto-probed.
        out_rate = _env_int("AUDIO_OUTPUT_SAMPLE_RATE")
        tts_kwargs = {"settings": KokoroTTSService.Settings(voice=voice)}
        if out_rate:
            tts_kwargs["sample_rate"] = out_rate  # → base TTSService _init_sample_rate (via **kwargs)
        logger.info(
            f"TTS: local Kokoro (onnx) voice={voice}"
            + (f" emit_rate={out_rate}Hz (pinned)" if out_rate else "")
        )
        return KokoroTTSService(**tts_kwargs)

    if provider == "piper":
        from pipecat.services.piper.tts import PiperTTSService  # extra: [piper]

        # Fast CPU TTS for weak hardware (Pi-4). piper voice-id namespace (en_US-lessac-medium etc.), NOT
        # Kokoro's; the .onnx voice auto-downloads on first use. Same emit-rate pin as Kokoro for a
        # fixed-rate device (EMEET 48k) — piper resamples its output to self.sample_rate, like Kokoro.
        voice = _env("TTS_VOICE", "en_US-lessac-medium")
        out_rate = _env_int("AUDIO_OUTPUT_SAMPLE_RATE")
        tts_kwargs = {"voice_id": voice}
        if out_rate:
            tts_kwargs["sample_rate"] = out_rate
        logger.info(
            f"TTS: local Piper voice={voice}"
            + (f" emit_rate={out_rate}Hz (pinned)" if out_rate else "")
        )
        return PiperTTSService(**tts_kwargs)

    if provider == "remote":
        from remote_tts import RemoteTTSService

        url = _env("TTS_REMOTE_URL")
        if not url:
            raise ValueError(
                "TTS_PROVIDER=remote requires TTS_REMOTE_URL (e.g. http://192.168.1.100:8771)"
            )
        voice = _env("TTS_VOICE", "af_heart")  # a Kokoro voice (the remote service runs Kokoro)
        room_id = _room_id()
        token = _env("TTS_REMOTE_TOKEN")
        timeout = float(_env("TTS_REMOTE_TIMEOUT", "30") or 30)
        out_rate = _env_int("AUDIO_OUTPUT_SAMPLE_RATE")
        logger.info(
            f"TTS: remote offload url={url} voice={voice} room={room_id} auth={'on' if token else 'off'}"
            + (f" emit_rate={out_rate}Hz (pinned)" if out_rate else "")
        )
        tts_kwargs = {"voice": voice, "room_id": room_id, "auth_token": token, "timeout": timeout}
        if out_rate:
            tts_kwargs["sample_rate"] = out_rate  # resample the remote Kokoro audio to the device rate (EMEET 48k)
        return RemoteTTSService(url, **tts_kwargs)

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

    raise ValueError(f"Unknown TTS_PROVIDER={provider!r} (expected kokoro|piper|cartesia|elevenlabs)")


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

        # SOP (no install-specific hardcode): GAB_BIN override → `gab` on PATH → legacy home fallback.
        import shutil
        gab_bin = (
            _env("GAB_BIN")
            or shutil.which("gab")
            or os.path.expanduser("~/dev/gabagent/.venv/bin/gab")
        )
        host = _env("GAB_HOST", "127.0.0.1")
        port = _env("GAB_PORT", "8765")
        project = _env("GAB_PROJECT_DIR", os.getcwd())
        base_url = f"http://{host}:{port}"
        # A LAN bearer token for a remote brain (Pi satellite, Topology B) — must match the brain's
        # GABAI_VOICE_AUTH_TOKEN. None on loopback (the default), where the brain runs unauthenticated.
        auth_token = _env("GAB_AUTH_TOKEN")
        # Attach vs spawn: a remote brain can't be spawned from here, so a non-loopback GAB_HOST
        # defaults to attach-only (the brain runs on the other box). Loopback defaults to connect-or-
        # spawn (HttpBrainClient.start() attaches if one is already healthy, else launches its own).
        # GAB_LAUNCH overrides either way (0 = attach-only even on loopback).
        is_remote = host not in ("127.0.0.1", "localhost", "::1")
        do_launch = _env("GAB_LAUNCH", "0" if is_remote else "1") not in ("0", "false", "False")
        # Multi-room foundation: ROOM_ID is the durable per-room/device routing key (default = hostname);
        # capabilities is this client's profile (what it does locally vs offloads), declared once at /attach.
        room_id = _room_id()
        # Key a SPAWNED brain to this room at launch (--room-id) so its TMI Tier-1 buckets under the room,
        # not the brain's implicit 'default'. This just makes explicit-at-launch the room_id the client
        # already sends per turn; default=hostname keeps EM-local on its existing bucket, no migration
        # (process-per-room: one brain per room — see the brain-topology decision).
        launch = (
            [gab_bin, "--voice-serve", "--port", str(port), "--cwd", project, "--room-id", room_id]
            if do_launch else None
        )
        logger.info(
            f"Brain: gabagent (HTTP/SSE {base_url}, project={project}, "
            f"{'spawn' if launch else 'attach'}, auth={'on' if auth_token else 'off'})"
        )
        stt_provider = (_env("STT_PROVIDER", "whisper") or "whisper").lower()
        capabilities = {
            "wake": (_env("WAKE_WORD_ENGINE", "nano") or "nano").lower() != "off",
            "vad": True,
            "stt": "remote" if stt_provider == "remote" else "local",
            "tts": True,
        }
        logger.info(f"Brain: room_id={room_id} capabilities={capabilities}")
        return BrainLLMService(HttpBrainClient(
            base_url, launch=launch, cwd=project, auth_token=auth_token,
            room_id=room_id, capabilities=capabilities,
        ))
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


async def wait_for_input_device(name: str | None, *, timeout: float, poll: float = 1.0) -> bool:
    """Block startup until an input device whose name contains `name` is enumerable, or `timeout` elapses.

    The AEC mic (`echo-cancel-source`, matched by AUDIO_INPUT_DEVICE_NAME, e.g. "AEC Mic") is created
    *asynchronously* by WirePlumber — on a cold boot/login it often isn't in the graph yet when we start.
    Plain systemd ordering can't see it (it's a PipeWire node, not a unit), so the boot-readiness wait
    belongs HERE, in the code that knows the device contract, rather than in a launch wrapper. Runtime
    death/recreation of the source is a separate concern already owned by input_watchdog.py; this is only
    the one-time cold-start wait.

    PortAudio snapshots the device list at init, so each poll uses a FRESH PyAudio instance to pick up a
    node that appeared after the previous check. Returns True as soon as a matching input device is present;
    False on timeout — the caller then falls through to build_transport, whose name-resolve simply misses and
    uses the OS default (the same graceful fail-over to the default mic the wrapper used to do, logged here).

    No-ops (returns True) when `name` is unset (nothing to wait for), `timeout` <= 0 (disabled), or PyAudio
    can't be imported (can't probe → don't block startup).
    """
    if not name or timeout <= 0:
        return True
    try:
        import pyaudio
    except Exception as e:  # pragma: no cover - diagnostic only
        logger.warning(f"AEC-mic wait: PyAudio unavailable ({e}); skipping readiness wait.")
        return True

    import asyncio
    import time

    nlow = name.lower()

    def _present() -> bool:
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                d = pa.get_device_info_by_index(i)
                if (d.get("maxInputChannels") or 0) > 0 and nlow in str(d.get("name", "")).lower():
                    return True
        finally:
            pa.terminate()
        return False

    if _present():
        return True
    logger.info(f"AEC-mic wait: input device ~={name!r} not present yet — waiting up to {timeout:.0f}s …")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(poll)
        if _present():
            logger.info(f"AEC-mic wait: input device ~={name!r} is now present.")
            return True
    logger.warning(
        f"AEC-mic wait: input device ~={name!r} did not appear within {timeout:.0f}s — falling over to "
        f"the system default mic (degraded; no echo cancellation until the AEC source returns)."
    )
    return False


async def wait_for_aec_mic() -> bool:
    """Env-driven boot wait for the AEC mic (entrypoint for main.py). Reads AUDIO_INPUT_* + AUDIO_INPUT_WAIT_SECS.

    No wait when the input is pinned by numeric index (AUDIO_INPUT_DEVICE_INDEX) — there's no name to poll for.
    Otherwise wait for AUDIO_INPUT_DEVICE_NAME up to AUDIO_INPUT_WAIT_SECS (default 60s; 0 disables the wait).
    """
    if _env_int("AUDIO_INPUT_DEVICE_INDEX") is not None:
        return True
    timeout = float(_env("AUDIO_INPUT_WAIT_SECS", "60") or 60)
    return await wait_for_input_device(_env("AUDIO_INPUT_DEVICE_NAME"), timeout=timeout)


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
    # Pin the output device OPEN-rate (and optionally channels) only when explicitly configured, for a
    # fixed-rate device (Pi EMEET = 48000). MUST equal the Kokoro emit-rate — AUDIO_OUTPUT_SAMPLE_RATE drives
    # BOTH so they can't diverge (divergence = chipmunk). Unset (EM) → omit → framework default (known-good).
    # NEVER auto-probed: auto-probing the system-default device is exactly what misfired and chipmunked EM.
    out_rate = _env_int("AUDIO_OUTPUT_SAMPLE_RATE")
    out_channels = _env_int("AUDIO_OUTPUT_CHANNELS")
    out_kwargs = {}
    if out_rate:
        out_kwargs["audio_out_sample_rate"] = out_rate
    if out_channels:
        out_kwargs["audio_out_channels"] = out_channels
    if out_kwargs:
        logger.info(f"Audio output: pinned {out_kwargs} (fixed-rate device; must match Kokoro emit_rate).")
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=capture_rate,
            input_device_index=in_idx,
            output_device_index=out_idx,
            **out_kwargs,
        )
    )


async def pin_output_stream_volume(poll_secs: float = 5.0, floor_pct: int = 60, target_pct: int = 100):
    """Keep the app's TTS output stream from being stranded silent/low by PipeWire module-stream-restore.

    The PipeWire ALSA playback sink-input (our TTS output) can come up at a STALE low/muted level —
    stream-restore replays a prior value onto it — which silences Aria. That's not just quiet audio: the
    half-duplex mic-mute keys off "bot is speaking", so a silently-stranded TTS turn leaves the user muted
    and their speech dropped (`VADController: no audio while speaking`) → the conversation breaks. We watch
    OUR sink-input (matched by this process's pid) and raise it to 100%/unmute whenever it's stranded below
    `floor_pct`. In-app (dies with the app — unlike the old /tmp watcher that orphaned then hit its loop
    limit). Best-effort: no pactl → quiet no-op. Setting it also teaches stream-restore the good value.
    """
    import asyncio
    import shutil

    if not shutil.which("pactl"):
        logger.info("Output-volume pin: pactl not found — skipping (TTS stranding guard inactive).")
        return
    pid = str(os.getpid())

    async def _pactl(*args) -> str:
        p = await asyncio.create_subprocess_exec(
            "pactl", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await p.communicate()
        return out.decode("utf-8", "replace")

    logged = False
    while True:
        try:
            listing = await _pactl("list", "sink-inputs")
            # Walk the blocks; capture vol/mute per block, lock them in when the block's
            # application.process.id matches ours (Volume/Mute appear before the property block).
            cur_idx = cur_vol = cur_mute = None
            our_idx = our_vol = our_mute = None
            for line in listing.splitlines():
                s = line.strip()
                if s.startswith("Sink Input #"):
                    cur_idx = s.split("#", 1)[1].strip()
                    cur_vol = cur_mute = None
                elif s.startswith("Mute:"):
                    cur_mute = s.split(":", 1)[1].strip()
                elif s.startswith("Volume:") and "%" in s:
                    try:
                        cur_vol = int(s.split("/")[1].strip().rstrip("%"))
                    except (IndexError, ValueError):
                        cur_vol = None
                elif s == f'application.process.id = "{pid}"':
                    our_idx, our_vol, our_mute = cur_idx, cur_vol, cur_mute
            stranded = our_idx is not None and ((our_vol is not None and our_vol < floor_pct) or our_mute == "yes")
            if stranded:
                await _pactl("set-sink-input-mute", our_idx, "0")
                await _pactl("set-sink-input-volume", our_idx, f"{target_pct}%")
                if not logged:
                    logger.info(
                        f"Output-volume pin: un-stranded TTS stream #{our_idx} "
                        f"(was vol={our_vol}% mute={our_mute}) -> {target_pct}% unmuted."
                    )
                    logged = True
        except Exception as e:  # noqa: BLE001 - never let the guard crash the app
            logger.debug(f"Output-volume pin: {type(e).__name__}: {e}")
        await asyncio.sleep(poll_secs)


async def warn_if_output_muted() -> None:
    """One-shot startup check: warn loudly if the default output SINK (the device, not our stream) is muted.

    The app never force-unmutes the user's master volume (that would be invasive), but PipeWire/WirePlumber
    can restore a muted device state across reboots — which silences Aria with NO obvious cause. On 2026-06-15
    this looked like a wake-word regression but was a stale device mute. This surfaces it in the log instead of
    leaving the user with silent dead air and no signal. Diagnostic only — never raises, never changes volume.
    """
    try:
        p = await asyncio.create_subprocess_exec(
            "pactl", "get-sink-mute", "@DEFAULT_SINK@",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await p.communicate()
        if "yes" in out.decode("utf-8", "replace").lower():
            logger.warning(
                "Heads-up: the default output sink is MUTED at startup — Aria's speech and media will be "
                "SILENT until you unmute it (system/master volume). The app does not touch your master mute."
            )
    except Exception:  # noqa: BLE001 - diagnostic only; never break startup
        pass


def build_input_resampler():
    """Resampler that normalizes mic capture to PIPELINE_AUDIO_RATE. Goes first in the pipeline.

    Always present (cheap, no-op when the device already opens at the pipeline rate) so the rest of
    the pipeline sees one consistent rate regardless of which mic/AEC source is bound.
    """
    from audio_resample import InputResampler

    return InputResampler(target_rate=PIPELINE_AUDIO_RATE)


def build_input_watchdog(restart=None, on_unrecoverable=None, gate_state=None):
    """Input-stall watchdog (goes first, right after transport.input()), or None if disabled.

    Detects a mic-capture freeze — both 'no frames' and 'frames but silent' (the 2026-06-03 echo-cancel
    stall) — logs it loudly, and calls `restart` (best-effort capture kick) instead of hanging silently.
    Also emits a periodic heartbeat (with optional `gate_state`) so freeze vs lockout is one glance.
    Default ON; set `INPUT_STALL_SECS=0` to disable. See input_watchdog.InputStallDetector.

    `on_unrecoverable` (e.g. exit the process) runs once when the in-process kicks are exhausted on a source
    that won't revive, so a supervisor (systemd `Restart=on-failure`) can do a clean full re-init instead of
    the watchdog latching inert (2026-06-22 reSpeaker silent-stall). **Opt-in** (`INPUT_STALL_EXIT_ON_FAIL=1`):
    default OFF keeps the historical log-only no-op, since exiting only helps where a supervisor restarts us
    on non-zero exit — on a bare/interactive run or a transient unit with no auto-restart, exiting would just
    leave Aria dead. Enable it on supervised deployments (the EM systemd unit has `Restart=on-failure`).
    """
    stall_secs = float(_env("INPUT_STALL_SECS", "5.0"))
    if stall_secs <= 0:
        return None
    silent_secs = float(_env("INPUT_SILENT_SECS", "8.0"))
    heartbeat_secs = float(_env("INPUT_HEARTBEAT_SECS", "10.0"))
    exit_on_fail = _env("INPUT_STALL_EXIT_ON_FAIL", "0") not in ("0", "false", "False")
    escalate = on_unrecoverable if exit_on_fail else None
    from input_watchdog import InputStallDetector

    logger.info(f"Input watchdog: ON (stall={stall_secs}s, silent={silent_secs}s, "
                f"heartbeat={heartbeat_secs}s, recover={'yes' if restart else 'log-only'}, "
                f"escalate={'exit-for-restart' if escalate else 'log-only'})")
    return InputStallDetector(
        stall_secs=stall_secs, silent_secs=silent_secs, restart=restart,
        on_unrecoverable=escalate, heartbeat_secs=heartbeat_secs, gate_state=gate_state,
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
def build_media_duck(llm, gate=None):
    """Build the MediaDuckController for an external brain, or None for a raw LLM.

    Inserted right after STT (see main.py) so it can see transcription frames — it ducks the brain's
    media on *confirmed* user speech and restores when Aria finishes. Shares the brain's session_id
    so `/media/duck` correlates, and gates on the brain's sleep state.

    `gate` (the WakeWordGate, if present) gates the duck behind the wake word: with DUCK_REQUIRE_WAKE on
    (default) media only ducks once Aria is addressed (window open), for ALL playback — so music/movies
    don't dip on ambient room speech. Set DUCK_REQUIRE_WAKE=0 to duck on any speech onset (old behavior).
    """
    from brains.brain_llm_service import BrainLLMService

    if not isinstance(llm, BrainLLMService):
        return None
    from brains.media_duck import MediaDuckController

    min_words = int(_env("DUCK_MIN_WORDS", "2"))
    restore_grace = float(_env("DUCK_RESTORE_GRACE", "8.0"))
    confirm_grace = float(_env("DUCK_CONFIRM_GRACE", "2.5"))
    sustained_secs = float(_env("DUCK_SUSTAINED_SECS", "4.0"))
    max_hold_secs = float(_env("DUCK_MAX_HOLD_SECS", "120.0"))
    # Conversation-hold: after an addressed reply over media, hold the bed ducked + floor open this long so a
    # follow-up needs no re-wake and the bed doesn't bob between turns. 0 disables (immediate bot-stopped
    # restore). Read the SAME WAKE_WINDOW_SECS the gate uses so the controller can clamp the hold to it — the
    # gate re-arms its command window ~window_secs per reply, keeping the two in lockstep (and its idle-close
    # is a free second restore guarantee). A hold > window would re-gate/bob mid-conversation, so it's clamped.
    convo_hold_secs = float(_env("DUCK_CONVO_HOLD_SECS", "8.0"))  # 2026-06-20: 15→8s, media returns sooner when no follow-up (stays < duck_watchdog 20s)
    window_secs = float(_env("WAKE_WINDOW_SECS", "15"))
    require_wake = _env("DUCK_REQUIRE_WAKE", "1") not in ("0", "false", "False")
    gate_duck = gate is not None and require_wake and hasattr(gate, "duck_allowed")
    # The ONSET duck (raw VAD onset) additionally stands down while a wake-from-sleep is PROVISIONAL
    # (is_resleep_pending): she woke on "Hey Aria" but no command has landed, so a FAILED wake-from-sleep
    # mustn't dip the music for nothing (2026-06-20 — the bed flapped on each failed wake). A real command
    # cancels the re-sleep and still ducks via the confirmed-speech path (should_duck, NOT gated on resleep).
    if gate_duck:
        should_duck = lambda: not llm.is_sleeping and gate.duck_allowed()
        # Raw VAD-onset duck uses the STRICTER gate (duck_onset_allowed): over playing media it ducks only
        # in a fresh-wake window, not the keepalive/idle tail (where the held-open window over the song the
        # user just started would otherwise make the music's own VAD onsets dip the bed — 2026-06-16). A real
        # follow-up command still ducks via the confirmed-speech path (should_duck).
        if hasattr(gate, "duck_onset_allowed"):
            should_duck_onset = lambda: (not llm.is_sleeping and not llm.is_resleep_pending
                                         and gate.duck_onset_allowed())
        else:
            should_duck_onset = lambda: (not llm.is_sleeping and not llm.is_resleep_pending
                                         and gate.duck_allowed())
    else:
        should_duck = lambda: not llm.is_sleeping
        should_duck_onset = lambda: not llm.is_sleeping and not llm.is_resleep_pending
    logger.info(
        f"Media ducking: on VAD speech onset (min_words={min_words}, "
        f"confirm_grace={confirm_grace}s, restore_grace={restore_grace}s, "
        f"sustained={sustained_secs}s, max_hold={max_hold_secs}s, "
        f"convo_hold={min(convo_hold_secs, window_secs):.0f}s, require_wake={gate_duck})"
    )
    return MediaDuckController(
        llm.brain_client,
        llm.session_id,
        min_words=min_words,
        restore_grace=restore_grace,
        confirm_grace=confirm_grace,
        sustained_secs=sustained_secs,
        max_hold_secs=max_hold_secs,
        convo_hold_secs=convo_hold_secs,
        window_secs=window_secs,
        should_duck=should_duck,
        should_duck_onset=should_duck_onset,
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
                import bot_speech

                st = await ms(session_id, bot_speaking=bot_speech.bot_speaking())
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
    # Sustained-wake: require N consecutive frames (~80ms each) >= threshold before opening. Music/asleep
    # false-positives are isolated 1-frame spikes, so >=2 rejects the blips. 3 proved too strict for real
    # wakes over a movie via the AEC mic — the score flickers above/below threshold so a clean "Aria" rarely
    # holds 3 in a row (2026-06-14 log: real attempts peaked 0.75-0.99 but "didn't sustain 3 frames", and
    # the extra frame added detection lag). 2 still rejects the 1-frame phantom yet recovers those. 1 = old
    # behaviour (a single blip wakes). The near-miss log reports "peaked N/M frames" to retune from data.
    consec_frames = int(_env("WAKE_CONSEC_FRAMES", "2"))
    # While ASLEEP, require more consecutive frames than awake before opening the window. Asleep is the
    # high-cost-of-false-positive state: a missed real wake just needs a repeat, but a false wake (a radio/TV
    # the AEC can't cancel spiking the model to ceiling) is the whole "she won't stay asleep" bug
    # (2026-06-15: talk radio re-woke her repeatedly). A stricter sustain rejects the brief 2-frame
    # coincidences that open the mic to ambient dialogue. The text wake-gate in brain_llm_service (the
    # transcript must name her and can't be a sleep phrase) is the catch-all; this just cuts phantom opens.
    # Clamped up to at least consec_frames in the gate. Was 3, but real wake-from-sleep recall over a movie
    # suffered: 2026-06-19 live, waking her from sleep over a film, 7 genuine wakes peaked 2/3 frames (one
    # short) and opened no window — the strict 3-run rejected clean "Aria"s the AEC mic passed but flickered.
    # Dropped to 2 (catches those) and paired with the gated-retry escape below so a repeated attempt opens
    # even when a single burst flickers to 1 frame, without globally weakening the single-shot bar further.
    asleep_consec_frames = int(_env("WAKE_ASLEEP_CONSEC_FRAMES", "2"))
    # Gated-retry escape: whenever the wake is being GATED (asleep OR awake-with-media-playing — both lock the
    # mic behind the wake word), if the user clearly tries AGAIN — `gated_retry_count` separate >=threshold
    # near-wakes that each fell short of the sustain, within `gated_retry_secs` — open the gate anyway. Media
    # rarely produces two near-perfect "Aria" bursts seconds apart; a user re-waking does. Catches the residual
    # 1-frame flicker (the 2026-06-19 "peaked 1/2 / 1/3" misses, asleep AND awake-over-music) the lowered bar
    # can't. Only fires while gated (open mic needs no escape). 0 disables. (Was asleep-only before 2026-06-19.)
    gated_retry_count = int(_env("WAKE_GATED_RETRY_COUNT", "2"))
    gated_retry_secs = float(_env("WAKE_GATED_RETRY_SECS", "6.0"))
    # Default 1 = gate VIDEO behind the wake word too (a movie only ducks once you say "Aria"; no duck on
    # asides). Safe to default on because the AEC echo-cancel mic keeps the movie out of the mic, so the
    # wake word is heard cleanly over it (the old over-movie lockout that kept this off is gone). Set 0 on a
    # box WITHOUT an echo-cancel source, where a movie would mask the wake word and lock you out.
    gate_video = _env("WAKE_GATE_VIDEO", "1") not in ("0", "false", "False")
    # F3 gate hardening (defense-in-depth): never clamp the mic mid-utterance, and require a media-state
    # change to persist before flipping the gate so a transient blip can't truncate a command / leak the
    # wake phrase. guard_inflight on by default; min_dwell 0 = old instant-flip behaviour.
    guard_inflight = _env("WAKE_GATE_GUARD_INFLIGHT", "1") not in ("0", "false", "False")
    min_dwell_secs = float(_env("WAKE_GATE_MIN_DWELL_SECS", "2.0"))
    # On wake-window-open, send /media/duck {mute:true} → media to full 0% for the window (vs a partial
    # duck) so no music vocal bleeds into the command's STT. Default 0 = a partial duck (media only dips):
    # with an AEC mic keeping media out of the mic, a full mute isn't needed and a partial duck is less
    # jarring. Set 1 on a box WITHOUT an echo-cancel source, where media would bleed into the command STT.
    window_mute = _env("WAKE_WINDOW_MUTE", "0") not in ("0", "false", "False")
    # Release the wake pre-duck after this many seconds if no speech follows the wake (an unused/phantom
    # wake), instead of holding media ducked for the whole window. Held while the user or Aria is speaking.
    # 6s matches the human "hear the duck, then compose and speak the command" loop — 3s released the duck
    # before the user began, so the command landed over restored audio and was missed (2026-06-15 live).
    preduck_grace = float(_env("WAKE_PREDUCK_GRACE", "6.0"))
    # Shorter pre-duck release grace for a media-KEEPALIVE-origin window. A keepalive isn't waiting on user
    # speech (the command already ran), so once Aria's announcement/reply ends the song she just started
    # should return promptly rather than linger the full wake grace — the "it dips the song I just played
    # for ~6s" annoyance (2026-06-16). Held through her reply (BotStarted cancels, BotStopped re-arms).
    preduck_grace_keepalive = float(_env("WAKE_PREDUCK_GRACE_KEEPALIVE", "1.0"))
    # Hard ceiling on the brain's media-keepalive hold (WakeHoldFrame ttl_secs). A keepalive TTL is clamped
    # to this, and the gate auto-releases at it if the brain's refreshes stop (crash/dropped turn) — so a
    # bad/large/never-refreshed hold can't pin the mic open. Must exceed the brain's media_keepalive_secs.
    hold_max_secs = float(_env("WAKE_HOLD_MAX_SECS", "120.0"))
    # V2 barge-in: while Aria is speaking, run the wake model and cut her TTS on a sustained hit (saying
    # "Aria" mid-reply interrupts and opens the floor). Opt-in, default OFF — zero behavior change when off.
    # Reuses the aria_nano wake model for now (a dedicated "stop" model is a follow-up). Threshold defaults
    # to the wake threshold; consec 2 rejects 1-frame TTS-bleed spikes the same way the wake path does.
    interrupt_enabled = _env("WAKE_INTERRUPT", "0") not in ("0", "false", "False")
    _it = _env("WAKE_INTERRUPT_THRESHOLD", "")
    interrupt_threshold = float(_it) if _it.strip() else None
    interrupt_consec_frames = int(_env("WAKE_INTERRUPT_CONSEC", "2"))
    interrupt_refractory_secs = float(_env("WAKE_INTERRUPT_REFRACTORY", "1.0"))
    interrupt_arm_delay_secs = float(_env("WAKE_INTERRUPT_ARM_DELAY", "1.0"))
    # Item C: emit a WakeEventFrame on a fresh acoustic wake so the brain can trust the audio over an STT
    # transcript that mis-expands "Hey Aria"→"how are you?" (eval: ~80% of wakes over music lose the name in
    # the text). Default on; WAKE_SIGNAL_FORWARD=0 disables the producer (brain then sees no `wake` → current
    # behavior). Mirrors GA's GABAI_VOICE_WAKE_CONFIDENCE_FILTER kill-switch.
    emit_wake_events = _env("WAKE_SIGNAL_FORWARD", "1") not in ("0", "false", "False")
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
        + (f" consec={consec_frames}" if consec_frames > 1 else "")
        + f" asleep-consec={asleep_consec_frames}"
        + (f" gated-retry={gated_retry_count}@{gated_retry_secs:.0f}s" if gated_retry_count else " gated-retry=off")
        + (" inflight-guard=on" if guard_inflight else " inflight-guard=off")
        + (f" min-dwell={min_dwell_secs:.1f}s" if min_dwell_secs > 0 else "")
        + (" window-mute=on" if window_mute else " window-mute=off")
        + (f" preduck-grace={preduck_grace:.1f}s" if preduck_grace > 0 else " preduck-grace=off")
        + f"(keepalive {preduck_grace_keepalive:.1f}s)"
        + f" hold-max={hold_max_secs:.0f}s"
        + (f" interrupt=on@{interrupt_threshold if interrupt_threshold is not None else threshold}"
           f"/consec{interrupt_consec_frames}/arm{interrupt_arm_delay_secs:.1f}s" if interrupt_enabled else "")
        + (" speex_ns=on" if speex_ns else "")
        + (" wake-signal=on" if emit_wake_events else " wake-signal=off")
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
        consec_frames=consec_frames,
        asleep_consec_frames=asleep_consec_frames,
        gated_retry_count=gated_retry_count,
        gated_retry_secs=gated_retry_secs,
        guard_inflight=guard_inflight,
        min_dwell_secs=min_dwell_secs,
        window_mute=window_mute,
        preduck_grace=preduck_grace,
        preduck_grace_keepalive=preduck_grace_keepalive,
        hold_max_secs=hold_max_secs,
        interrupt_enabled=interrupt_enabled,
        interrupt_threshold=interrupt_threshold,
        interrupt_consec_frames=interrupt_consec_frames,
        interrupt_refractory_secs=interrupt_refractory_secs,
        interrupt_arm_delay_secs=interrupt_arm_delay_secs,
        emit_wake_events=emit_wake_events,
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
    # Foundation: declare this client's room_id + capabilities once, after the brain is healthy. Tolerant
    # — a brain without /attach is a logged no-op (never blocks startup, no deploy-order coupling).
    attach = getattr(client, "attach", None)
    session_id = getattr(llm, "session_id", None)
    if attach is not None and session_id is not None:
        await attach(session_id)


async def stop_brain(llm) -> None:
    """Tear down an external brain (close transport + stop any spawned process)."""
    client = getattr(llm, "brain_client", None)
    aclose = getattr(client, "aclose", None)
    if aclose is not None:
        await aclose()
