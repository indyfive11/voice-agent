# Voice Agent — project instructions

This project builds a real-time **spoken** assistant for Claude: mic → STT → LLM → TTS with
VAD/turn-taking, barge-in, and voice-driven machine control.

**Read [`PLAN.md`](PLAN.md) first** — it is the approved implementation plan (architecture,
phased steps, files to create, safety model, and end-to-end verification). Start at Phase 0.

Quick orientation:
- **Stack:** Pipecat (Daily). Local-first, but STT / TTS / LLM are each swappable via env in `config.py`.
- **Default:** local Whisper + Kokoro + Claude (`claude-sonnet-4-6`); LLM can swap to OpenAI-compatible or local Ollama on the RX 7900 XT.
- **Env:** use an isolated **Python 3.12 venv via `uv`** (system python is 3.14.5; wheels lag). Needs `portaudio` + `espeak-ng` and `ANTHROPIC_API_KEY`.
- **Safety:** full machine control ships with a 3-tier guardrail (hard denylist → verbal-confirmation gate → read-only auto-run). Review the denylist before the first "full control" run.
