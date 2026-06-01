# Plan: Real-Time Voice Agent for Claude (Local-First, Cloud-Ready, Tool-Enabled)

## Context

the user wants a true real-time **spoken** assistant backed by the Claude API — talk to it, it
talks back, and it can *do things* on the machine. This is "option #3" from the earlier
chat: a full voice pipeline (STT → Claude → TTS) with VAD/turn-taking, as opposed to the
mobile app (no system access) or a simple dictation frontend (no spoken replies, no fluid
turn-taking).

**Decisions captured (from clarifying questions + follow-up):**
- **STT/TTS:** Local-first, but architected so swapping in cloud providers is a one-line/config change.
- **LLM:** Swappable too — Claude is the default, but the "brain" is provider-agnostic. Config can point it at Claude (cloud), any OpenAI-compatible endpoint, or a **fully local model on the RX 7900 XT via Ollama (ROCm)**.
- **Capabilities:** Full machine control by voice — with hard safety guardrails + a verbal-confirmation gate for anything mutating.
- **Activation:** Always-listening (Silero VAD + smart turn detection, with barge-in/interruption support).

**Why a framework, not a hand-rolled pipeline:** Building the audio loop by hand (mic capture →
VAD → STT → LLM streaming → TTS → playback, plus interruption handling and turn detection) is
exactly the wheel that **Pipecat** (open-source, by Daily) reinvents well. It has first-class
`AnthropicLLMService` (streaming + tool calling + prompt caching), a `LocalAudioTransport` for
desktop mic/speaker, local `WhisperSTTService` and `KokoroTTSService`, Silero VAD, and a local
**SmartTurn v2** turn detector — all swappable for cloud services behind the same frame
interface. This gives us local-first + cloud-ready for free.

### Machine facts that shaped the design (verified this session)
- **CPU/RAM:** Ryzen 7 9800X3D + 64 GB — local Whisper + Kokoro run faster-than-real-time on CPU.
- **GPU:** AMD RX 7900 XT (24 GB), **`rocm-core` 7.2.3**. **No CUDA.** `faster-whisper` (CTranslate2) is CPU-only on AMD — that's fine here; GPU offload is an *optional* later enhancement via whisper.cpp+Vulkan.
- **Audio:** PipeWire 1.6.6; mic = **Logitech C110 webcam (mono)** — verified `wpctl` source **id 44 "Webcam C110 Mono"** (the weakest link for accuracy; a USB mic would noticeably improve STT). Input device is selectable in code — bind to id 44, not the HDMI/analog sources.
- **Python:** system `python3` is **3.14.5** — too new; key wheels (PyAudio, onnxruntime, ctranslate2) may not have 3.14 builds. **→ Use an isolated Python 3.12 venv via `uv`.**
- **Tooling status (verified live):** `portaudio` (1:19.7.0) ✓, `espeak-ng` (1.52.0) ✓, `ollama` (0.24.0, not running) ✓, ffmpeg + arecord present. **Still missing:** `uv`, a 3.12 interpreter, and `ANTHROPIC_API_KEY`.

---

## Architecture

```
                 ┌───────────────────── Pipecat pipeline ─────────────────────┐
  mic (PipeWire) │  LocalAudioTransport.input()                               │
  ──────────────►│      │  (Silero VAD + SmartTurn v2 turn detection)         │
                 │      ▼                                                      │
                 │   WhisperSTTService (faster-whisper, local CPU)            │
                 │      ▼                                                      │
                 │   user context aggregator                                  │
                 │      ▼                                                      │
                 │   AnthropicLLMService (Claude, streaming, TOOLS) ──► tools.py
                 │      ▼            (function calls: run_command, read_file, …)│
                 │   KokoroTTSService (local ONNX)                            │
                 │      ▼                                                      │
  speaker ◄──────│  LocalAudioTransport.output()  ◄── barge-in interrupts TTS │
                 │   assistant context aggregator                             │
                 └────────────────────────────────────────────────────────────┘
```

Swappability (all three layers, selected via env in `config.py`):
- STT: `WhisperSTTService`→`DeepgramSTTService`
- TTS: `KokoroTTSService`→`CartesiaTTSService`/`ElevenLabsTTSService`
- **LLM: `AnthropicLLMService`→`OpenAILLMService`(any OpenAI-compatible base_url: local llama.cpp/vLLM/LM Studio, OpenAI, OpenRouter…)→`OLLamaLLMService`(local, ROCm on the 7900 XT).**

All are drop-in within Pipecat's frame contract. **The LLM swap is now format-agnostic:** Pipecat
ships a universal `LLMContext` + `LLMContextAggregatorPair`
(`pipecat.processors.aggregators.llm_context` / `…llm_response_universal`) that adapts to
Anthropic/OpenAI/Ollama automatically — so `main.py` builds **one** context/aggregator pair
regardless of provider, and `build_llm()` returns just the service. (The older per-provider
`OpenAILLMContext` + `llm.create_context_aggregator()` path still works but is no longer needed.)

### Component choices

| Layer | Local default | Cloud-ready swap | Notes |
|-------|---------------|------------------|-------|
| Transport | `LocalAudioTransport` (PyAudio) | same (mic/speaker stay local) | input device index selectable |
| VAD | `SileroVADAnalyzer` | same | local ONNX |
| Turn-taking | `LocalSmartTurnAnalyzerV3` | same | local; reduces talk-over vs pure VAD (V2 deprecated since 0.0.106) |
| STT | `WhisperSTTService` `small.en` (CPU) | `DeepgramSTTService` | `base.en` if latency-bound |
| LLM | `AnthropicLLMService`, `claude-sonnet-4-6` | `OpenAILLMService` (any base_url) · `OLLamaLLMService` (local) | streaming + tools + prompt caching; Haiku 4.5 for min cloud latency |
| TTS | `KokoroTTSService` (kokoro-onnx) | `CartesiaTTSService` / `ElevenLabsTTSService` | Kokoro ~<0.3s synth on CPU |

**Local LLM note:** Ollama on the RX 7900 XT (24 GB, ROCm) comfortably hosts strong quantized
models (e.g. Llama 3.x / Qwen 8B–32B-class). **Tool calling reliability varies by model** — full
machine control needs a model with solid function-calling; Claude and the larger local models do
this well, smaller local models degrade tool accuracy. So the default ships as Claude; local LLM
is a fully-supported, documented swap.

---

## Implementation

Project root: **`~/dev/voice-agent/`**

### Phase 0 — System prerequisites (mostly done)
Verified live, so most of this is already satisfied:
- `portaudio` (1:19.7.0), `espeak-ng` (1.52.0), `ffmpeg`, `arecord`, and `ollama` (0.24.0) **already installed** — nothing to `pacman -S`.
- **Remaining:** install `uv` (user-level, no sudo): `curl -LsSf https://astral.sh/uv/install.sh | sh`; then `uv` will fetch the 3.12 interpreter in Phase 1. Set `ANTHROPIC_API_KEY` in `.env` (Phase 1).
- (If a future re-image is missing the libs: `sudo pacman -S --needed portaudio espeak-ng` — ask first per global rules.)

### Phase 1 — Scaffold & isolated env
- `cd ~/dev/voice-agent && uv venv --python 3.12 && source .venv/bin/activate`
- `pyproject.toml` (or `uv pip install`) deps — **pin the Pipecat version** so the fork below has a stable base and the V2→V3 / extras drift can't recur silently:
  - `pipecat-ai[anthropic,silero,whisper,local,kokoro]==<pinned>`
  - `pipecat-ai[local-smart-turn]` (SmartTurn v3 — **not** `[turn-analyzer]`, which doesn't exist) · `onnxruntime` · `python-dotenv` · `loguru`
  - dev: `pytest` (safety unit tests, Phase 3.5)
  - Optional LLM-swap extras (install when used): `pipecat-ai[openai]` (no `ollama` extra exists at 1.3.0 — `OLLamaLLMService` rides on `[openai]` + an Ollama base_url)
- **Local fork escape hatch (documented, default OFF).** Don't fork preemptively — ship on the pinned PyPI release. But wire in a *ready-to-flip* override so patching Pipecat (fixing a bug, setting a breakpoint, hacking a service) is a one-stanza change, not a re-architecture. With `uv`, redirect the source without touching the dependency declaration:
  ```toml
  # Escape hatch — uncomment to develop against a local fork instead of PyPI.
  # Clone your fork to ./vendor/pipecat (gitignored, or a submodule), then edits
  # there take effect live (editable). `uv sync` to switch; re-comment to return.
  # [tool.uv.sources]
  # pipecat-ai = { path = "./vendor/pipecat", editable = true }
  ```
  Alternative once you have committed fixes you want reproducible across machines: pin to your fork's branch instead — `pipecat-ai @ git+https://github.com/<you>/pipecat@my-fixes`. Add `vendor/` to `.gitignore`.
- `.env` (chmod 600): `ANTHROPIC_API_KEY=…`, plus provider/model toggles (see config.py). Add `.env` to `.gitignore`.
- First run auto-downloads: Whisper model, Kokoro `kokoro-v1.0.onnx` + `voices-v1.0.bin`, Silero + SmartTurn ONNX. Document the model cache path.

### Phase 2 — Core local pipeline — `main.py`
Wire the pipeline shown above. Key construction points:
- `TransportParams(audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer(), turn_analyzer=LocalSmartTurnAnalyzerV3(...))` (import `from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3`).
- Pick mic explicitly via `input_device_index` so it binds to the **C110 (verified `wpctl` source id 44)**, not an HDMI/analog input. Confirm the index PyAudio assigns to the C110 at startup and log it.
- LLM service comes from `config.build_llm()` (default Claude, `enable_prompt_caching=True`) — `main.py` never hardcodes a provider. `main.py` builds **one** universal `LLMContext(messages=[system…], tools=…)` + `LLMContextAggregatorPair(context)` (`pipecat.processors.aggregators.llm_context` / `…llm_response_universal`), provider-agnostic. System prompt lives in the context messages.
- Import paths use service submodules: `pipecat.services.whisper.stt`, `pipecat.services.kokoro.tts`, `pipecat.services.anthropic.llm`, `pipecat.services.openai.llm`, `pipecat.services.ollama.llm` (class is `OLLamaLLMService` — the odd caps are correct).
- `Pipeline([transport.input(), stt, aggregators.user(), llm, tts, transport.output(), aggregators.assistant()])`, run with `PipelineRunner` + `PipelineTask(..., allow_interruptions=True)`.
- Reference scaffold: `kwindla/macos-local-voice-agents` (same Pipecat local-voice pattern) and Pipecat's `examples/foundational`.

### Phase 3 — Tool use + safety guardrails — `tools.py`
Register tools on the LLM via `FunctionSchema`/`ToolsSchema` + `llm.register_function(name, handler)`.
Tools: `run_command`, `read_file`, `write_file`, `get_system_info`, `web_search` (optional), a
dedicated `confirm_pending_action`, and `set_listening` (mute/wake — see below).

**Three-tier safety model — fail-closed (the core of "full control" being safe).**
The default for anything not provably read-only is **ask, never run**. A command is classified
top-down; the first matching tier wins:
1. **Tier 1 — Hard denylist → refused outright, never executes, no confirmation possible.** Short-circuits everything. Encodes the user's standing safety rules:
   - VPN toggles: `vpn-full`, `vpn-split`, `vpn-bypass`, any `wg-quick`/`wg`/NM-on-wg0 change (these hang Claude's own API session — see `feedback_vpn_toggle.md`).
   - Family photo archive paths — no delete/move/overwrite.
   - System config: `sysctl`, `/etc/fstab`, swap, `systemctl` enable/disable of core services.
   - Pattern-matched (regex denylist constant); a hit → tool returns a refusal string Claude speaks aloud.
2. **Tier 3 — Auto-run, read-only (allowlist only).** Runs immediately **only if** the command matches the read-only allowlist (`ls`, `cat`, `pwd`, `date`, `df`, `git status`, `wpctl status`, …) or is `read_file`/`get_system_info`. The allowlist is the *sole* path to silent execution.
3. **Tier 2 — Confirmation gate = the default.** **Everything that is neither denied (Tier 1) nor allowlisted (Tier 3)** falls through here — including any command the classifier doesn't recognize, so unknown ≠ safe. The action is stashed in a module-level `pending_action` (single-use, auto-expires) and the tool returns "needs confirmation"; Claude speaks the exact command back and asks. Mutation patterns (write/delete/install/`sudo`/redirection) are only an *escalation hint* for how emphatically to warn — not the gate itself.
   - **Distinct confirm phrase:** `confirm_pending_action` executes the stashed action **only** when the user gives an explicit phrase like **"confirm execute"** — a bare "yes"/"yeah" does **not** fire it, so overheard speech can't trigger a pending mutation. The required phrase is stated in the system prompt and spoken by Claude when it asks.
- **Mute / sleep:** `set_listening(on=False)` ("go to sleep" / "stop listening") mutes acting on input until a wake command ("wake up" / "start listening") flips it back — keeps the always-on mic from acting on overheard speech. Optional hotkey can toggle the same flag.
- `sudo`: honor `SUDO_ASKPASS=ksshaskpass` + `sudo -A` (no blind sudo — trips faillock).
- System prompt restates the guardrails + the confirm phrase + "you are a voice assistant: keep replies short and speakable; spell nothing out; confirm before mutating."

### Phase 3.5 — Safety unit tests — `tests/test_safety.py`
The safety classifier is pure logic — testable with **zero audio hardware**, and it's the code
standing between a misheard command and the machine, so it gets a `pytest` suite (run with
`uv run pytest`). Cover:
- **Tier 1 denylist hits** refuse and never execute: VPN toggles (`vpn-split`/`vpn-bypass`/`wg-quick`), family-photo-archive paths, `sysctl`/`/etc/fstab`/swap/core-service `systemctl`.
- **Tier 3 allowlist** auto-runs (`ls`, `git status`, `read_file`, `get_system_info`).
- **Tier 2 fail-closed default:** a command that is neither denied nor allowlisted (incl. an unrecognized one) returns "needs confirmation", not execution.
- **Confirmation flow:** stash → `confirm_pending_action("confirm execute")` runs **exactly** the stashed command; a **bare "yes" does NOT execute**; single-use (second confirm is a no-op); auto-expiry.
- Run pure (no real `subprocess`) by injecting/monkeypatching the executor so tests assert *decisions*, not side effects.

### Phase 4 — Provider abstraction (STT + TTS + LLM all swappable) — `config.py`
- Env-driven factory:
  - `STT_PROVIDER` (`whisper`|`deepgram`) → `build_stt()`
  - `TTS_PROVIDER` (`kokoro`|`cartesia`|`elevenlabs`) → `build_tts()`
  - `LLM_PROVIDER` (`anthropic`|`openai`|`ollama`) + `LLM_MODEL` + optional `LLM_BASE_URL` → `build_llm()`
- `build_llm()` returns **just the LLM service** (provider/model/base_url resolved from env). `main.py` wraps it with the universal `LLMContext` + `LLMContextAggregatorPair`, which adapt per provider automatically — no per-provider aggregator branching needed.
- `LLM_PROVIDER=ollama` (or `openai` with `LLM_BASE_URL=http://localhost:11434/v1`) gives a fully local brain on the 7900 XT — set `LLM_MODEL=llama3.1:8b` (or similar tool-calling model). No API key, nothing leaves the machine.
- Switching any layer to cloud = set env + add that key; `main.py` unchanged.
- Document the optional extras (`pipecat-ai[deepgram,cartesia,elevenlabs,openai]`) as commented installs. (No `ollama` extra at 1.3.0 — Ollama uses `[openai]` + base_url.)

### Phase 5 — Latency tuning, run ergonomics, polish
- Enable Pipecat metrics (TTFB/processing) to measure end-to-end latency; tune Whisper `small.en`↔`base.en` and try `claude-haiku-4-5-20251001` if Sonnet feels slow.
- Convenience launcher: `run.sh` (activates venv, loads `.env`, `python main.py`) and an optional `~/.local/bin/voice-agent` symlink. (Optionally a `--push-to-talk` flag later; out of scope now.)
- README documenting setup, the safety model, and the cloud-swap toggles.

### Files to create
- `~/dev/voice-agent/main.py` — pipeline + runner
- `~/dev/voice-agent/config.py` — provider/model factory (cloud-ready)
- `~/dev/voice-agent/tools.py` — tool handlers + 3-tier fail-closed safety model
- `~/dev/voice-agent/tests/test_safety.py` — pytest unit tests for the safety classifier
- `~/dev/voice-agent/pyproject.toml` (or `requirements.txt`)
- `~/dev/voice-agent/.env` (secrets; gitignored) · `.gitignore` · `run.sh` · `README.md`

---

## Verification (end-to-end)

0. **Safety unit tests:** `uv run pytest` is green — required before any full-control run.
1. **Env sanity:** `uv run python -c "import pipecat, pyaudio, onnxruntime; print('ok')"` inside the 3.12 venv (proves the 3.14-wheel risk is dodged).
2. **Audio binding:** `wpctl status` confirms the C110 source; run `main.py` and confirm it captures from the mic (not HDMI) — log the chosen device.
3. **Round-trip conversation:** speak "Hello, can you hear me?" → expect a streamed transcript in logs + a spoken Kokoro reply within ~1s.
4. **Turn-taking / barge-in:** start talking while it's replying → TTS should stop (interruption works).
5. **Read-only tool:** "What's in my Downloads folder?" → runs `ls` automatically, speaks a summary.
6. **Confirmation gate + confirm phrase:** "Create a file called test.txt on my desktop" → it speaks the command back and asks. Say a bare **"yes"** → it must **not** execute (file does not appear). Say **"confirm execute"** → file appears. (Proves overheard "yes" can't fire a mutation.)
7. **Fail-closed default:** an unusual non-allowlisted command (e.g. "run my backup script") routes to the confirmation gate, not silent execution — even though no mutation keyword matched.
8. **Hard denylist:** "Switch my VPN to split tunnel" → it refuses verbally and does **not** execute (confirm no `vpn-split` ran).
9. **Mute / wake:** "go to sleep" → speaking commands afterward does nothing; "wake up" → it responds again.
10. **Latency check:** read Pipecat TTFB metrics; if >1.5s, drop Whisper to `base.en` and/or switch LLM to Haiku 4.5.
11. **Provider swap smoke test (optional):** set `TTS_PROVIDER=cartesia` + key, and separately `LLM_PROVIDER=ollama LLM_MODEL=llama3.1:8b` (with Ollama running on ROCm) → both confirm the abstraction works without touching `main.py`. Verify a tool call still fires with the local model.

---

## Risks / notes
- **Python 3.14 wheels:** the whole reason for the 3.12 venv. If `uv` can't get 3.12, fall back to 3.11.
- **C110 mic accuracy:** mono webcam mic is the biggest accuracy limiter; a cheap USB mic is the highest-ROI hardware upgrade. Noise suppression (`krisp`/`rnnoise`) is a possible later add.
- **PipeWire ↔ PyAudio:** works via the Pulse/ALSA compat layer; if device enumeration is flaky, set the input device index explicitly.
- **espeak-ng:** Kokoro's phonemizer needs it for some text; installed in Phase 0.
- **AMD GPU STT (optional future):** whisper.cpp built with Vulkan/ROCm, exposed as an OpenAI-compatible server and consumed via Pipecat's OpenAI-STT base-url — offloads STT to the 7900 XT if CPU ever becomes the bottleneck. Not needed for v1.
- **Safety:** the denylist is defense-in-depth on top of the user's global rules; it must be reviewed before first "full control" run.

## Sources
- [Pipecat (GitHub)](https://github.com/pipecat-ai/pipecat) · [Pipecat Anthropic service](https://docs.pipecat.ai/api-reference/server/services/llm/anthropic) · [Pipecat Kokoro service](https://docs.pipecat.ai/api-reference/server/services/tts/kokoro) · [LocalAudioTransport source](https://reference-server.pipecat.ai/en/latest/_modules/pipecat/transports/local/audio.html)
- [kwindla/macos-local-voice-agents (local Pipecat reference)](https://github.com/kwindla/macos-local-voice-agents) · [LocalAudioTransport+Whisper+LLM+TTS example issue](https://github.com/pipecat-ai/pipecat/issues/197)
- [LiveKit vs Pipecat comparison](https://www.cekura.ai/blogs/pipecat-vs-livekit-the-real-difference) · [Turn detection: LiveKit vs Pipecat](https://futurepulseai.blog/2025/10/24/the-turn-taking-crisis-in-voice-ai-comparing-smart-turn-detectors-livekit-vs-pipecat/)
- [whisper.cpp (AMD ROCm/Vulkan)](https://github.com/ggml-org/whisper.cpp) · [Speech-to-Text on AMD GPU (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/whisper/README.html)
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) · [Local TTS latency comparison (Kokoro/Piper)](https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-part-2)
