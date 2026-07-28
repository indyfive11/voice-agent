# Voice brain protocol

A small, **brain-agnostic** HTTP + Server-Sent-Events contract between a **voice front-end**
(microphone, wake word, speech-to-text, text-to-speech) and a **brain** (conversation + actions).

This is the canonical spec, owned by [voice-agent](https://github.com/indyfive11/voice-agent) — the
reference **front-end**. It deliberately lives with the brain-*agnostic* party: the front-end defines
what any brain must speak, and each brain points at this one spec. Either side is swappable — anything
that speaks this protocol interoperates, so nothing provider-specific (TIDAL/Jellyfin/…) ever crosses
the boundary. [gabagent](https://github.com/indyfive11/gabagent) is one reference **brain** implementing
it; the front-end selects a brain with `BRAIN=remote` (any HTTP/SSE brain) or the reference-brain alias
`BRAIN=gabagent`.

The brain binds **loopback only** (`127.0.0.1:8765` by default). The front-end connects (or spawns the
brain), sends transcribed user utterances, and renders the streamed response as speech. (A LAN
deployment — a low-power satellite front-end talking to a brain on another box — binds the brain to the
LAN behind a bearer token instead; the wire contract below is identical either way.)

## Endpoints

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET  | `/health` | — | `{"status":"ok","mode":"voice"}` |
| POST | `/respond` | `{session_id, text, wake?}` | **SSE** stream of events (below). `409` if a turn is already in progress for the session. |
| POST | `/confirm` | `{session_id, id, approved, passphrase?}` | **SSE** continuation stream. `404` unknown session, `409` nothing awaiting confirmation / no match. |
| POST | `/cancel` | `{session_id}` | `{"ok":true}` — aborts the in-flight turn (barge-in). |
| POST | `/media/duck` | `{session_id, on, mute?}` | `{"ok":true,"ducked":[…]}` — quiet/restore local media while the user speaks (`ducked` is opaque, brain-internal). |
| GET  | `/media/state` | query: `bot_speaking=true\|false` (optional) | `{"playing":bool,"state":"playing"|"paused"|"idle","kind":"audio"|"video"|null}` |

`/media/state` doubles as a ~1 Hz heartbeat. Pass `bot_speaking=true` while the assistant's TTS is actively
playing: it lets the brain keep its duck-watchdog from auto-restoring the bed mid-reply (a long spoken answer
has no incoming user speech to keep the duck alive). Omit it or send `false` otherwise. Optional — a brain that
ignores it still works.

`/respond`'s optional **`wake`** object carries an out-of-band acoustic wake signal, for the case where a bare
wake ("Hey Aria") is fluently mis-transcribed into a question ("Hey, how are you?") — text alone can't tell that
apart from a genuine query, but the audio can. The front-end attaches it only on a *fresh* acoustic wake-open
whose own text wake-strip found no wake word:
`{ "bare_wake_likelihood": 0.9, "confidence": 0.97, "post_wake_voiced_ms": 120, "speech_dur_ms": 700 }`. Only
`bare_wake_likelihood` (the front-end's fused 0..1 confidence that the utterance is *nothing but* the wake) is
load-bearing; the rest are raw features for threshold-tuning. When it clears the brain's threshold, the brain
treats a content-free greeting as wake-only and stays silent. Optional and back-compat — absent `wake` is exact
normal behavior, and the signal can never by itself suppress a real command.

## SSE events

Each event is a single `data: {json}\n\n` frame on the `/respond` and `/confirm` streams. Empty/zero fields are
omitted. `type` is always present.

| `type` | Fields | Meaning |
|--------|--------|---------|
| `token`   | `text` | A chunk of speakable response text (stream these to TTS). |
| `status`  | `text` | A short status line (e.g. "Trying tidal…") — optional to speak. |
| `confirm` | `id, tier, method, summary, reason?, prompt_is_complete?` | A gate confirmation. `method` ∈ `spoken_yesno`\|`keyboard`\|`passphrase`. By convention `summary` is a bare action and the front-end appends the yes/no; if `prompt_is_complete` is true, speak `summary` verbatim. Reply via `POST /confirm`. |
| `blocked` | `action, reason` | An action was refused by policy. |
| `error`   | `text, summary` | Turn-level failure — `text` is speakable, `summary` is the structured cause. |
| `wake_hold` | `ttl_secs` | Keep the wake/follow-up window open for `ttl_secs` after a media-control turn, so the user can chain commands ("louder", "skip") without re-waking. |
| `convo_hold` | `release` | `release=true` on a terminal reply: restore the bed-duck at TTS-stop instead of holding it the full conversation-hold window. A reply that expects an answer (a question) omits this so the hold stays open. |
| `voice_volume` | `op, value?` | Change the **assistant's own** TTS gain (not media volume). `op` ∈ `up`\|`down`\|`set`; on `set`, `value` is an absolute level `0..1` (1.0 = full, 0.0 = silent). |
| `done`    | — | Terminal: the turn (or confirm continuation) is complete. |

## Stream properties (audio duck-exclude)

The one property the protocol reserves — stamped on the front-end's **own TTS output** sink-input
(PipeWire/PulseAudio):

| Property | Value | Stamped by | Read by |
|----------|-------|------------|---------|
| `voicebrain.duck_exclude` | `"1"` | the front-end, on its TTS output stream | the brain's local-audio logic |

- **Why.** The brain's local duck and its local-audio media-detection both scan sink-inputs. Without a marker
  they would mute the assistant's own voice (a self-duck) and count its TTS as "media playing." This property
  tells the brain "this stream is the assistant — never duck it, never treat it as media."
- **Transport.** Stamp it on **both** the native (PipeWire `PIPEWIRE_PROPS`) and libpulse (`PULSE_PROP`) paths
  — a one-path stamp has regressed before. The brain matches the rendered `voicebrain.duck_exclude = "1"` as a
  case-insensitive literal substring of the sink-input properties (the value is part of the match).
- **Legacy alias (migration).** During de-branding the brain ALSO matches the legacy `gabagent.duck_exclude`.
  A front-end may dual-stamp both through the transition and drop the legacy key once every brain it talks to
  matches the neutral one; a mismatched ordering is cushioned by a node-name safety net, but do not rely on it.
  New integrations stamp `voicebrain.duck_exclude` only.

## Health & versioning

`GET /health` returns `{"status":"ok","mode":"voice"}` today. Two fields are **reserved** for a future,
non-breaking capability handshake: `version` (a protocol version string) and `capabilities` (a list of optional
features the brain supports). Clients MUST treat their **absence as the baseline** contract described here — a
brain that omits them is a baseline brain. The handshake itself is unspecified for now; reserving the names
keeps adding it later backward-compatible.

## Design principles

- **Brain-agnostic / provider-neutral.** No provider names cross the boundary. `/media/state` is generic —
  `kind` is a media *type* (`audio`/`video`), never a provider. The brain owns the *decision* (which provider,
  duck vs pause) and the *action*; the front-end owns audio I/O and timing.
- **Locality.** The brain only auto-ducks/controls media on **this** machine (`/media/duck` and the
  `/media/state` snapshot are scoped to local audio); playback on other devices is never touched automatically.
- **`mute`.** `/media/duck {on:true, mute:true}` deepens the duck to a full mute (volume 0) for an open
  wake/command window, so a music vocal can't bleed into the transcription; `mute` defaults false → a plain
  gentle duck.
- **Two-phase confirm.** A `confirm` event pauses the turn; the front-end collects the user's yes/no and posts
  `/confirm`, and the continuation streams back on *that* response.
- **Loopback by default.** The reference deployment binds `127.0.0.1` and is not a network service. A LAN
  deployment instead binds the brain to the LAN address behind a bearer token the front-end presents on every
  request — the endpoints and event types are unchanged.

## Building your own side

- **A different front-end** (other wake word / STT / TTS): connect to the brain, POST `/respond` with the
  transcript, render `token` events as speech, handle `confirm`/`error`/`done`, and call `/media/duck` on
  speech onset/end. That's the whole integration.
- **A different brain** (other LLM / assistant): serve these endpoints. As long as `/media/state` stays
  provider-neutral and the SSE event types match, an existing front-end drives it unchanged. Select it from the
  front-end with `BRAIN=remote` (see the voice-agent README).
