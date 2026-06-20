# Plan: Voice Agent — as-built architecture (reconciled 2026-06-20)

This is the **current, authoritative** description of the system as built. It supersedes the original
v1 implementation plan, which is archived for reference at
[`docs/PLAN_v1_original.md`](docs/PLAN_v1_original.md) (deprecated — read this file, not that one).

The v1 Phase 0–5 foundation (Pipecat, local-first STT/Claude/TTS, `uv` venv, swappable providers)
shipped as planned. The architecture then **diverged in two major ways** that v1 never anticipated;
both are now load-bearing and are documented here. Open bugs, tuning, and feature roadmap live in
[`TRACKER.md`](TRACKER.md).

---

## What this is

A real-time **spoken** assistant: mic → STT → **brain** → TTS, with VAD/turn-taking, a wake word,
media ducking, barge-in, and voice-driven machine control. Local-first (Pipecat 1.3.0 / Daily on
PipeWire); STT / TTS / brain are each swappable via env in `config.py`.

**The core reframe vs v1:** voice-agent is a **pluggable voice shell**. It owns *audio* — capture,
wake, duck, turn-taking, TTS, barge-in — and **delegates all cognition to a swappable brain** over a
process boundary. It no longer contains the LLM, tools, or the safety model in-process.

---

## Divergence #1 — the brain is a separate process (the biggest change from v1)

**v1 plan:** the LLM (`AnthropicLLMService`), the tools (`tools.py`), and the 3-tier fail-closed
safety model (`tests/test_safety.py`) all lived **inside** voice-agent's pipeline (v1 Phase 3/3.5).

**As built:** all of that moved to a **separate `gabagent` process**, reached over an HTTP/SSE
contract. `tools.py` and `tests/test_safety.py` **no longer exist in this repo.** voice-agent talks
to whatever brain `BRAIN=` selects:

- `BRAIN=local` — a raw LLM brain (still uses `config.build_llm()`; default `claude-sonnet-4-6`,
  swappable to OpenAI-compatible or local Ollama on the RX 7900 XT).
- `BRAIN=gabagent` — the full agent: tools + escalating-tier safety + media control + an
  addressing/aside classifier. Released independently (currently **v0.5.1**, own AUR package, own
  test suite).

**Why:** the brain grew its own complexity and release cadence; coupling two independently-evolving
systems in one process was the wrong seam. The split lets the voice shell stay brain-agnostic.

**The v1 safety model is intact — it just lives brain-side now.** Hard denylist → verbal-confirmation
gate → read-only auto-run, fail-closed. Review the denylist (brain-side) before any "full control"
run, same as v1.

### The voice ↔ brain contract (`brains/`)
- `brains/brain_client.py`, `http_brain_client.py` — client to the brain's HTTP API.
- `brains/brain_llm_service.py` — a Pipecat `FrameProcessor` that stands in for the LLM service in
  the pipeline; forwards turns to the brain and streams replies back. Also hosts wake-flow logic
  (strip leading wake trigger, bare-wake prompt) and the interrupt/cancel plumbing.
- Endpoints/contract: `/respond` (turn), `/media/duck` (on/off — idempotent), `/media/state`
  (now also carries a `bot_speaking` query flag — see below), `/cancel` (barge-in floor-yield), the
  `wake_hold` SSE channel (keepalive: hold the command window open after a media command so follow-ups
  need no re-wake), the `convo_hold` SSE channel (turn-terminality hint: on a terminal reply over
  media, release the conversation-hold so the bed restores at turn end instead of holding for a
  follow-up), and the `voice_volume` SSE channel (Aria's own-voice volume control — see TTS leveling).
- **`bot_speaking` on `/media/state`:** the voice side carries `bot_speaking=true|false` on the ~1 Hz
  state poll (true while Aria's TTS is playing). The brain treats a `true` poll as a duck-watchdog
  **heartbeat refresh**, so a long reply's TTS — which produces no incoming user speech — can't outlive
  the brain's watchdog grace and pop the ducked bed mid-narration. Coupled invariant: the brain's
  `duck_watchdog_secs` must stay **above** the voice `convo_hold` (currently 20 > 8).
- **Naming debt:** the contract still carries brain-specific names (`gabagent.duck_exclude`,
  `/media/*`). Decoupling is tracked publicly as GitHub #1.

---

## Divergence #2 — the wake / duck / half-duplex subsystem (none of this was in v1)

v1's activation model was **open VAD barge-in + a verbal "go to sleep / wake up" mute**. Live
reality with the C110 mono webcam mic forced a near-complete inversion. The whole audio-front-end
below is **net-new** vs the v1 plan:

### Wake word — "Hey Aria"
- `wake_word.py` (the gate), `wake_nano.py` (speaker-specific nano model), `wake_porcupine.py`
  (alt engine). Model training pipeline in `wake-train/` (gitignored real-voice captures).
- The gate sits **upstream** of the half-duplex mute, so it sees mic audio even during the bot turn.
- Asleep/awake gating uses a fuzzy name match (recovers STT mis-spellings) + acoustic + text guards;
  a leading wake trigger is stripped before forwarding ("Hey Aria, play X" → brain hears "play X").
- **Root engineering problem (ongoing):** AEC (PipeWire `module-echo-cancel`) references Aria's TTS,
  not music — so a music bed isn't cancelled and the AEC double-talk clamp crushes the near-end
  "Aria" over loud media. This is the durable wake-over-music root cause (see TRACKER + memory).

### Media ducking
- `brains/media_duck.py` + the `/media/duck` contract. **Two duck writers**, now coordinated as
  **single-writer**:
  1. `MediaDuckController` (post-STT) — ducks on user VAD onset / confirmed speech, and is the **SOLE
     authority for the duck *off***.
  2. the wake-gate pre-duck (`wake_word.py`) — fires on a fresh wake / window-open, but **relinquishes
     silently** (clears its local flag, no `off` POST) whenever `media_duck` owns the duck, via a
     `set_media_ducked(lambda: media_duck._ducked)` callback wired in `main.py`. This fixed the
     two-writer desync where the wake gate POSTed `/media/duck off` in the reply→TTS gap and un-ducked
     music mid-reply while `media_duck` still held the duck.
- `DUCK_REQUIRE_WAKE=1` gates the duck behind the wake word so ambient room speech doesn't dip media.
- **Conversation-hold** (`DUCK_CONVO_HOLD_SECS`, default **8s**): after a **non-terminal** reply over
  playing media (one ending in "?", where Aria is waiting for an answer), the bed stays ducked + the
  floor open this long for a follow-up needing no re-wake. A **terminal** reply (the brain's
  `convo_hold release=True`, e.g. a story or a command-confirmation) restores the bed promptly at
  BotStopped instead. Stays under the brain's 20s duck-watchdog (see `bot_speaking` above).
- Hard-won invariant: **every window-hold path must own its pre-duck *release*, and must reason about
  *what* it's held open over** — an open window re-enables the ambient duck over the system's *own*
  music. (This is the F1/F3 fix family, `1ef8c71`.)

### Half-duplex turn-taking + barge-in (the v1 inversion)
- v1 assumed `allow_interruptions=True` open barge-in "just works." It didn't — the un-AEC'd mic
  **self-tripped on Aria's own voice**. So turn-taking went **half-duplex**: `turn_mute.py` MUTES the
  user during the bot turn (the opposite of v1's yield-on-barge-in).
- Interruption was then re-introduced as a controlled **interrupt-word barge-in** (V2, shipped
  `573dd41`): say "Aria" while she's speaking → a `StopWordFrame` below the mute cuts TTS *and*
  cancels the in-flight brain turn via `/cancel`. Built on the wake path because the half-duplex
  user-aggregator mute drops upstream `InterruptionFrame`/VAD during the bot turn.
- The durable fix (roadmap) is a **dedicated "stop"/"Aria stop" model** decoupled from the reused
  wake model — closes both the onset-transient and sentence-boundary self-trips.

### Endpointing — honor SmartTurn's INCOMPLETE verdict
- `turn_stop.py::SmartTurnHonoringStopStrategy` (env `TURN_HONOR_INCOMPLETE=1`, default on) — the stock
  pipecat `TurnAnalyzerUserTurnStopStrategy` ends a turn via a transcript-fallback + STT-p99 timeout
  (~0.8s) that can fire **even when SmartTurn just predicted INCOMPLETE**, cutting the user off on a
  natural mid-sentence pause ("tell me a story about…\<pause\>"). The fix gates the single stop funnel
  on `turn_analyzer.speech_triggered` (True ⟺ last verdict INCOMPLETE; cleared only on a COMPLETE
  verdict or the 4s `stop_secs` silence), so a pause is held until SmartTurn actually calls the turn.
  **Gotcha:** Whisper marks *every* segment `finalized=True`, so the guard must **not** exempt
  finalized transcripts (that made the first cut a no-op). The 15s MaxTurn cap remains the backstop.

### TTS leveling + runtime voice volume
- `tts_gain.py` — scales Aria's TTS PCM by `TTS_GAIN` (default 0.55). TTS is duck-excluded (rides at
  full level over ducked music), so it needed independent attenuation. Not in v1.
- **Runtime, voice-commandable** (was a static startup knob): "Aria, lower your voice" → the brain
  classifies a my-voice intent and emits a `voice_volume {op,value}` SSE event → `brain_llm_service`
  pushes a `VoiceVolumeFrame` DOWNSTREAM → `tts_gain` adjusts the live gain (up/down step
  `TTS_VOLUME_STEP=0.1`, down floored at `TTS_VOLUME_FLOOR=0.1`, `set` clamps to [0,1]). Distinct from
  the media-volume commands (those lower OTHER apps); this is Aria's own output. Persistence across
  restarts is deferred. `bot_speech.py` exposes the live `bot_speaking` flag (set here on
  BotStarted/BotStopped) that rides the `/media/state` poll (see contract above).

---

## Status side-channel — the Aria "HAL eye" indicator (net-new vs v1)

The voice shell publishes its semantic state to a tmpfs file that a **separate** desktop panel (a
standalone Conky "HAL eye", not in this repo) renders — so the user can *see* at a glance whether Aria
is idle, asleep, listening, thinking, or speaking.

- `aria_state.py` — an atomic writer (`tmp` + `os.replace`) to `${XDG_RUNTIME_DIR}/aria/state`
  (fallback `~/.local/state/aria/state`), JSON `{"state","level","ts"}`, enum
  `off | idle | sleeping | listening | thinking | speaking` (`error` reserved). `off` is voice-mode-down
  (eye closed); `sleeping` is the distinct asleep-but-running doze (the brain ships a matching `sleeping`
  enum; Conky renders it). It is a **cosmetic side-channel**:
  every write error is swallowed and never affects the conversation. Disable with `ARIA_EYE_STATE=0`
  (see `.env.example`); `ARIA_EYE_LEVEL_GAIN` boosts the `speaking` amplitude pulse.
- **Multi-writer, one transition each:** `main.py` (idle/off), `wake_word.py` (listening on
  window-open, rest on window-close, sleeping/idle on sleep/wake), `brains/brain_llm_service.py`
  (thinking on a turn), `tts_gain.py` (speaking + a live RMS `level` from the TTS PCM, ~20 Hz). The
  contract is **single-writer-at-a-time** — in voice mode the voice shell is the sole writer; the reader
  (Conky) fails safe to `off` if `ts` goes stale (>~5s).
- **The return-to-rest invariant:** when a transient ends, the eye settles on a process-global
  *resting state* (`aria_state.resting_state()` — `idle` awake, `sleeping` asleep), never a hardcoded
  `idle`. All four return-to-rest writers honor it — `tts_gain` BotStopped, the wake gate's
  window-open and window-close, and `brain_llm_service._run_turn` (start + finally) — so a wake-model
  false-fire while asleep can't drift the eye back to `idle` (fixed across `9cbd609` → `45e0790`).
  "Am I awake?" decisions route through `aria_state.is_resting()` (rest = `off` *or* `sleeping`), not a
  `!= "off"` literal, so the new `sleeping` rest doesn't read as awake.

---

## Smaller divergences from v1

- **Python:** v1 said "3.12 venv, fall back to 3.11." As built: **3.12–3.13** (3.13 verified
  2026-06). The 3.12-only cap was solely `speexdsp-ns` (cp312 wheel) — now an optional `[speex]`
  extra. `uv` provisions the interpreter regardless of the 3.14 system python.
- **Turn detector:** SmartTurn **V3** (v1 was updated V2→V3 before build; shipped on V3).
- **GPU STT:** v1's "optional later" whisper.cpp+Vulkan offload — **never needed**; CPU Whisper is
  faster-than-real-time. Correctly deferred indefinitely.
- **Provider swaps:** Deepgram / Cartesia / ElevenLabs / Ollama are wired per v1 but **only
  Kokoro + Whisper + Claude/gabagent are exercised live.** The cloud/local-LLM swaps are unproven
  in practice — treat as "wired, not verified."
- **Packaging (net-new vs v1):** shipped as an AUR `-git` package (`voice-agent-git`) with a per-user
  writable working-copy launcher; first public GitHub release done. Brain ships separately.

---

## Pipeline as built

```
transport.input()
  → input_watchdog              (liveness; idle-cancel disabled, music-only is steady state)
  → resampler                   (audio_resample.py)
  → wake_gate                   (wake_word.py — wake/sleep, pre-duck, keepalive window)
  → stt                         (Whisper small.en, CPU)
  → media_duck                  (brains/media_duck.py — duck on user speech)
  → user_aggregator             (Silero VAD + SmartTurn v3 + half-duplex mute, turn_mute.py)
  → llm  = BrainLLMService      (brains/brain_llm_service.py — forwards to the brain over HTTP/SSE)
  → tts                         (Kokoro)
  → tts_gain                    (tts_gain.py — attenuate Aria's level)
  → transport.output()
  → assistant_aggregator
```
Supporting modules: `turn_cap.py`, `turn_stop.py` (SmartTurn-honoring endpointing — see above),
`response_latency.py`, `vad_diag.py`, `input_watchdog.py`, `bot_speech.py` (live `bot_speaking` flag
for the duck-watchdog refresh), `aria_state.py` (the HAL-eye status side-channel — see above).

Swappability still holds (env in `config.py`): `STT_PROVIDER`, `TTS_PROVIDER`, `LLM_PROVIDER`/`BRAIN`.

---

## What's unchanged from v1 (still the source of truth there)

The original plan's **machine facts** (Ryzen 9800X3D / RX 7900 XT / PipeWire / C110 mic id 44),
**Phase 0–5 setup steps**, the **3-tier safety model design**, and the **end-to-end verification
checklist** remain valid as reference — see [`docs/PLAN_v1_original.md`](docs/PLAN_v1_original.md).
The safety verification steps (confirm-phrase gate, fail-closed default, hard denylist, mute/wake)
now exercise the **brain**, but the test intent is unchanged.

---

## Roadmap / open work

Tracked in [`TRACKER.md`](TRACKER.md). Current themes (all in the wake/duck/brain-split architecture
above, not the v1 foundation, which is done):

- **Voice-side:** wake false-positives over music (next lever: `capture_selfneg.sh` self-negatives);
  wake recall over a loud movie (held pending brain wrong-sink fix); dedicated stop-word barge-in
  model; STT mis-expanding the bare wake ("Hey Aria" → "Hey, how are you?") — next priority, needs a
  better STT engine and/or an out-of-band wake-confidence signal. *(Done this cycle: SmartTurn-honoring
  endpointing, duck single-writer, runtime voice-volume control, `bot_speaking` watchdog refresh,
  convo-hold 8s.)*
- **Brain-side (gabagent):** reconcile-vs-duck-off race stranding the music sink quiet; TIDAL search
  latency (~15s); resume-from-position-0.
- **Cross-project:** decouple gabagent-specific naming (GitHub #1); whole-home / multi-room (Home
  Assistant seam) + timers.
