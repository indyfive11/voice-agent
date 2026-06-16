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
