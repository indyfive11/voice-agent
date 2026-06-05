# Voice Agent — project instructions

This project builds a real-time **spoken** assistant for Claude: mic → STT → LLM → TTS with
VAD/turn-taking, barge-in, and voice-driven machine control.

**Read [`PLAN.md`](PLAN.md) first** — it is the approved implementation plan (architecture,
phased steps, files to create, safety model, and end-to-end verification). Start at Phase 0.

Quick orientation:
- **Stack:** Pipecat (Daily). Local-first, but STT / TTS / LLM are each swappable via env in `config.py`.
- **Default:** local Whisper + Kokoro + Claude (`claude-sonnet-4-6`); LLM can swap to OpenAI-compatible or local Ollama on the RX 7900 XT.
- **Env:** isolated **Python 3.12–3.13 venv via `uv`** (`requires-python >=3.12,<3.14`; uv provisions it regardless of the 3.14 system python). 3.13 verified 2026-06; the old 3.12-only cap was solely `speexdsp-ns` (cp312 wheel) — now an optional `[speex]` extra. Needs `portaudio` + `espeak-ng`; `ANTHROPIC_API_KEY` only for the default local-Anthropic brain (none for `LLM_PROVIDER=ollama` or an external `BRAIN`).
- **Safety:** full machine control ships with a 3-tier guardrail (hard denylist → verbal-confirmation gate → read-only auto-run). Review the denylist before the first "full control" run.
