# voice-agent

A real-time **spoken** assistant shell: mic → speech-to-text → **brain** → text-to-speech, with
voice-activity detection, turn-taking, wake-word gating, and (optionally) voice-driven machine
control. Local-first; built on [Pipecat](https://github.com/pipecat-ai/pipecat).

It's a **pluggable voice shell** — it owns the audio loop and turn-taking and delegates *cognition*
to a swappable "brain" over a small HTTP/SSE protocol. Point it at a raw LLM (`BRAIN=local`) or at a
full tool-using agent. There is no code dependency on any particular brain.

## Companion project

[`gabagent`](https://github.com/indyfive11/gabagent) is the reference **brain** — a tool-using
coding/desktop agent with an escalating-tier safety model. The two are **loosely coupled — docs and
protocol only, no code dependency** in either direction. The brain↔shell contract lives in gabagent's
[`docs/VOICE_PROTOCOL.md`](https://github.com/indyfive11/gabagent/blob/master/docs/VOICE_PROTOCOL.md).
Run voice-agent with `BRAIN=local` and never touch gabagent, or wire them
together for a full voice-driven agent.

> **Brain-agnostic, with known rough edges.** The design is brain-agnostic (the `brains/` seam,
> `BRAIN=local` default), but some gabagent-specific naming has crept in (e.g. a `gabagent.duck_exclude`
> output-stream property, the `/media/*` duck contract). Renaming these to neutral terms is tracked
> for a later pass.

## Stack

- **Audio / pipeline:** Pipecat 1.3.x — local audio transport, VAD (Silero), turn-taking (SmartTurn v3),
  half-duplex with optional barge-in
- **STT:** Whisper (local) — swappable (e.g. Deepgram) via `.env`
- **TTS:** Kokoro (local) — swappable; runtime, voice-commandable output level ("Aria, lower your voice")
- **LLM (`BRAIN=local`):** Claude (`claude-sonnet-4-6`), or any OpenAI-compatible / local Ollama endpoint
- **Wake word:** openWakeWord / nanowakeword / Porcupine, behind one gate
- **Status indicator (optional):** publishes Aria's state (off / idle / sleeping / listening / thinking /
  speaking) to a tmpfs file for an external desktop "HAL eye" panel to render — a cosmetic side-channel,
  off via `ARIA_EYE_STATE=0`

Everything is selected by environment variables — see [`.env.example`](.env.example).

## Quick start

Requires **Python 3.12 or 3.13** (provisioned automatically by [`uv`](https://github.com/astral-sh/uv)) and
system `portaudio` + `espeak-ng`. The **brain is pluggable**, so the API-key requirement depends on which one
you run:

- **Default** (`BRAIN=local`, `LLM_PROVIDER=anthropic`) → needs `ANTHROPIC_API_KEY`.
- **Fully local** (`BRAIN=local`, `LLM_PROVIDER=ollama`) → no cloud key.
- **External brain** (`BRAIN=gabagent`) → the brain owns cognition; the voice shell needs **no LLM key at all**.

```bash
cp .env.example .env        # pick STT / TTS / LLM / brain (+ a key only if your brain needs one)
uv sync
./run.sh                    # or: uv run python main.py
```

`./run.sh` modes: no arg = brain from `.env`; `./run.sh local` = raw LLM; `./run.sh gab` = gabagent brain.

## Wake word

While media is playing, the agent requires a wake word before commands reach STT (sidestepping
speech-over-music mis-transcription) and pre-ducks the audio on wake. A bare openWakeWord
`wakewords/aria.onnx` ships as a starting point; train your own (e.g. "hey aria") per
[`wakewords/README.md`](wakewords/README.md) and the `wake-train/` recipe. **Speaker-specific voice
models are kept local** (not committed) — train one for your own voice.

## Safety

When driven by a tool-using brain, machine control sits behind a 3-tier guardrail: hard denylist →
verbal-confirmation gate → read-only auto-run. The guardrail is **brain-owned** — review the brain's
denylist before the first "full control" run.

## Status

Active development — the APIs and the brain protocol may still change. See [`PLAN.md`](PLAN.md) for the
architecture and roadmap.

## License

[MIT](LICENSE)
