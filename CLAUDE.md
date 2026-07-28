# Voice Agent — project instructions

This project builds a real-time **spoken** assistant for Claude: mic → STT → **brain** → TTS with
VAD/turn-taking, a wake word, media ducking, barge-in, and voice-driven machine control.

**Read [`PLAN.md`](PLAN.md) first** — it is the current, reconciled as-built architecture (the
voice-shell / brain split, the wake/duck/half-duplex subsystem, and where we diverged from v1).
Open bugs + roadmap are in [`TRACKER.md`](TRACKER.md). The original v1 phased plan (Phase 0–5 setup,
machine facts, end-to-end verification checklist) is archived for reference at
[`docs/PLAN_v1_original.md`](docs/PLAN_v1_original.md) — deprecated, kept for provenance.

Quick orientation:
- **Architecture:** a **pluggable voice shell** — voice-agent owns audio (wake/duck/turn-taking/TTS/
  barge-in) and delegates cognition to a swappable **brain** over HTTP/SSE (`BRAIN=local` raw LLM, or
  `BRAIN=gabagent` = the full tools+safety agent). The LLM, tools, and 3-tier safety model live
  **brain-side**, not in this repo (no `tools.py`/`test_safety.py` here anymore).
- **Stack:** Pipecat (Daily). Local-first, but STT / TTS / LLM are each swappable via env in `config.py`.
- **Default:** local Whisper + Kokoro + Claude (`claude-sonnet-4-6`); LLM can swap to OpenAI-compatible or local Ollama on the RX 7900 XT.
- **Env:** isolated **Python 3.12–3.13 venv via `uv`** (`requires-python >=3.12,<3.14`; uv provisions it regardless of the 3.14 system python). 3.13 verified 2026-06; the old 3.12-only cap was solely `speexdsp-ns` (cp312 wheel) — now an optional `[speex]` extra. Needs `portaudio` + `espeak-ng`; `ANTHROPIC_API_KEY` only for the default local-Anthropic brain (none for `LLM_PROVIDER=ollama` or an external `BRAIN`).
- **Safety:** full machine control ships with a 3-tier guardrail (hard denylist → verbal-confirmation gate → read-only auto-run). Review the denylist before the first "full control" run.

## Hardware / install portability (hard SOP)

No hardware-type or installation-specific value may be a **bare constant in code**. It must run on any user's
machine with zero code edits. Every such value MUST be:
1. **A config field / env with a SAFE UNIVERSAL DEFAULT = the historical no-op** — unset/empty is valid and behaves
   exactly as today, never worse (an unconfigured install is never broken by the knob existing).
2. **Detected ONCE by an explicit, inspectable setup/detect step that WRITES config** — NOT a fragile per-startup
   auto-probe. (A per-startup audio auto-probe via pipecat `is_format_supported` is exactly what misfired and shipped
   the 1.84× chipmunk TTS.) Detect from the AUTHORITATIVE OS source: ALSA/PipeWire (`/proc/asound`, `pactl`) for audio
   rates/devices/channels, `rocminfo`/`nvidia-smi`/`lspci` for GPU, the primary route iface for the LAN IP.
3. **User-overridable afterward** (the env/config always wins over detection).
4. **Named machine-agnostically** — an env var / config key MUST NOT encode a host, machine, or reference-install
   name in its identifier (e.g. no `EM_STT_MAX_CONCURRENCY`; use `STT_SERVICE_MAX_CONCURRENCY`). Name by the ROLE
   or SUBSYSTEM the knob configures, never by the box it first ran on. Renaming a shipped key = add the agnostic
   name, read it first, keep the old name as a deprecated back-compat alias (`new or old or default`); never a hard
   break. **Applies to every new env var going forward** — pick the portable name at introduction.

**Principle: the running app reads config (dumb) · the setup step detects-and-writes (smart) · the user edits (in
control).** Genuinely-universal constants (e.g. the 16 kHz `PIPELINE_AUDIO_RATE` ↔ Whisper/Silero training) may stay
hardcoded — just comment WHY. Mirrored in `gabagent/CLAUDE.md` (cross-repo SOP).

## New-module deploy-safety (hard SOP, cross-repo)

A change that adds a **new module** which already-tracked/deployed code imports MUST do **one** of:
1. **Guard the import at the call site** so an absent module degrades to the documented no-op (fail-soft), OR
2. **Ship the new module in the same commit** (deploy-manifest) as its importer.

**Never an importer without its import target.** Rationale: satellite deploys sync **git-tracked files only**
(`git ls-files` — a blind `*.py` rsync would clobber a satellite's ARM `.venv`), so a new *untracked* module + a
*tracked* importer = a **guaranteed satellite crash** (the Pi went hard-down 2026-07-04 when an untracked
`image_display.py` shipped its tracked importers `main.py`/`config.py` to the Pi via the launch rsync →
`ModuleNotFoundError` on every start). Corollary: under a push-freeze, (1) is the freeze-safe path since (2) is
blocked. The fail-soft guard is the durable primary — it immunizes the *whole class*, not just one file; wrap
import **and** construction so an optional feature can never take down the core loop. Mirrored in
`gabagent/CLAUDE.md` (cross-repo SOP).
