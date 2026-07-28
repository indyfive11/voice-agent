# voice-agent — Roadmap (single source of truth)

*Reconciled 2026-07-13 (deep-reconcile). This is the **plan-of-record**. The as-built architecture
lives in [`PLAN.md`](PLAN.md); granular bug/tuning detail and device-specific state live in the
internal tracker (not public). When those disagree with this file, **this file wins** for
*what we are doing and why*.*

---

## Charter

**voice-agent is a voice user interface (VUI):** a thin, real-time voice shell — mic → STT → brain →
TTS, with wake, duck, turn-taking, and barge-in — that delegates **all** cognition to a swappable
brain over its HTTP/SSE protocol.

- **Brain-agnostic by protocol, not vendor** — gabagent is the *reference* brain, never a hardcode
  *(⇒ protocol de-branding, [GitHub #1](https://github.com/indyfive11/voice-agent/issues/1), is a
  core pillar, not a cleanup)*.
- **Local-first, cloud-ready** — every layer swaps by config.
- **Survives real consumer audio hardware** on low-power satellites.

**Deliverable:** a publishable, self-provisioning thin-client PoC — auto-detects a LAN brain and
HW-tiers itself.

### The three-part model (so the charter reads unambiguously)
- **BRAIN** — cognition + tools + the 3-tier safety model. Lives brain-side (gabagent is the
  reference implementation). Not in this repo.
- **VUI** — *this* project: a detachable voice shell that plugs into the brain over the protocol.
- **TUI** — a text shell over the same brain (gabagent wears its own). The VUI's peer is *a text
  shell*, not the brain itself — the symmetry lives at the **protocol seam**, not in repo thickness.

---

## Anti-drift gate — ship the PoC first

> **No new feature *domain* is entered until the thin-client PoC is published and deploy-verified on
> a clean box.**

Always permitted (not "new domains"): bug-fixes, hardware-survival hardening, and **PoC-enabling
portability work**. What waits behind the gate: net-new capability domains (new media integrations,
new platforms, expressive TTS, presence sensing, …).

*Why this gate:* 94 commits over ~5 weeks, **never publicly pushed**, with a recurring pattern of
entering a new domain (satellite → GPU-STT → image-gen → movie-recommender → installer) before the
prior north-star shipped. The center of gravity has been *surviving one flaky USB mic*. The gate
converts "the PoC" from an aspiration into the forcing function.

---

## Now — ship the thin-client PoC

1. **Installer / packaging** (cross-repo, 3-layer model, LOCKED 2026-07-12; Phase-1 built GA-side):
   shared `installkit/` (vendored primitives + HW/GPU detect + templating + token-pairing) →
   voice-agent role-provisioner (audio detect, LAN-brain discovery, role units + `.env` emit) →
   brain-side addon plugins. Delivery = `git clone → ./bootstrap.sh` (uv-based; reaches both Debian
   satellites and Arch). *Phased, each phase gated.*
   - **Progress (git-anchored, unpushed):** the entry point + satellite role-provisioner shipped
     (`af4d656`: tracked `bootstrap.sh` → `voice_agent_install.satellite`, gather→provision→
     install-unit→verify). installkit A.4 — the `render_unit` `RestartSec`/`WorkingDirectory`
     chokepoint — fixed + signed (`9b389a4` + `0b8ae18`).
   - **Decided (the maintainer, 2026-07-27) — keep the shared-`installkit` vendor model** (resolves the §10c
     vendor-vs-duplicate fork). Scope that follows: **vendor `installkit` into voice-agent at a pinned
     SHA and wire its first real consumer** — route `voice_agent_install/units.py` through
     `installkit.templating.render_unit`, killing the mount-dependency invariant now **duplicated**
     across `installkit/templating.py` + `units.py`. GA is scoping + self-pressure-testing the spec;
     VAC pressure-tests it, drives collab-adversarial to full consensus, then codes (the maintainer's gate: no
     code until GA sign-off). Constraint: the public pin can't be *real* until the push freeze lifts —
     vendor from the local SHA now, formalize the pinned public SHA + SHA-match CI at push.
   - **Status (2026-07-27): SHIPPED — published, deploy-verified.** Whole `installkit` vendored at
     `78ff1cd`; `units.py` renders via `installkit.templating`; the duplicated invariant is gone. Public
     `origin/main` (PII-clean whole history); the pin is finalized against public `installkit` with a
     **SHA-match / tree-OID vendor-check CI (green)**, a fresh-public-clone **clean-box test passed**, and
     a real-ARM **Pi deploy-test passed** (`PROVISION_EXIT 0`). The maintainer's live spoken
     wake→response on the redeployed Pi is **confirmed** (2026-07-27) — the PoC ship gate is met.
2. **Auto-detect the LAN brain** — mDNS/zeroconf or known-port probe → auto-fill brain host + token,
   replacing the hand-edited `.env`. "Fire it up and it finds the brain."
   - **Voice-host role (the discovery emitter)** — the install-time provisioner that runs *on the brain
     box*, detects its primary-route LAN IP, and turns the brain's `_voice-brain._tcp` advertiser on so a
     satellite can find it without a hand-typed IP. **SHIPPED** — `voice_agent_install/voice_host.py`
     (Layer-B caller; `bootstrap.sh --role voice-host`) invokes the brain-side seam cross-process
     (`gabagent-install --enable-voice-host --host <ip>`, never an import). The graft: Layer-B passes the
     detected LAN IP, Layer-C writes `voice_host` + `voice_advertise` together and *refuses on loopback*
     (a bare `voice_advertise:true` on a default box advertises nothing). Discovery is an enhancement, not
     a hard dependency — every failure degrades to the honest floor (type the brain's IP). Public on
     `origin/main` (`ca56619`), paired with gabagent Layer-C (`aea1306`). **Validated end-to-end on
     real hardware (2026-07-27):** brain advertises → the Pi satellite finds it by mDNS → a boot
     re-resolve corrects a stale host. The satellite role now installs the discovery extra by default
     (`5071f82`), so a fresh satellite finds its brain out of the box.
   - **Firewall-aware reachability** (a filtered mDNS responder reads identically to a dead one) — the
     ratified detect-and-report design lives in **gabagent `INSTALL_PLAN.md` §10d** (named link, not
     restated here, per the cross-repo "named links not merged content" rule). Layer-B owns the browse
     + reason vocabulary + rendered operator remedy; detect-only, never mutates a host firewall.
     **SHIPPED** — `diagnose_brain_discovery()` wired into the satellite-install path (reason vocabulary
     + firewall-aware operator remedy, non-fatal fall-through to the typed host+port floor); public on
     `origin/main`, full suite green, live-verified on the reference box against the real LAN (no firewall
     touched). *(internal: P9)*
   - **Token pairing — the last hand-typed step, removed. SHIPPED (2026-07-28).** A fresh satellite
     obtains the brain's bearer token *over the wire* during a short, operator-opened window, instead
     of the maintainer reading and typing it into `.env`. `satellite --pair` runs the CLAIM client
     (`voice_agent_install/pairing.py` — 128-bit random `client_id` persisted before first POST; a
     server-minted one-time `claim_secret` bound to the accepted peer-IP; replay-until-TTL for
     dropped-response recovery), and pairing supplies `brain_token` *before* compose/apply so a failure
     aborts before anything is written. The agnostic wire contract is owned brain-side
     (gabagent `docs/PAIRING.md`) — the maintainer's direct call for this sub-contract. Public:
     voice-agent `56d59c1` (client + 19 tests), gabagent `de8edfd` (brain lane). **Live-certified
     cross-box (2026-07-28):** real Pi → brain → token retrieved over the wire → authenticated call
     `200` with the token / `401` without, reusing the persisted `client_id` (idempotent re-run).
3. **HW-tiering** — detect-once-write-config (never a per-startup probe): strong HW runs local
   STT/TTS, weak HW (Pi-class) offloads. Per the no-hardcodes portability SOP.
4. **Protocol de-branding — [GitHub #1](https://github.com/indyfive11/voice-agent/issues/1)**
   *(promoted to a ship-blocker by the charter)*: the wire contract carried brain-specific names.
   "Brain-agnostic" is not shippable while the protocol names one brain. Neutralize the names +
   own the protocol.
   - **Status (2026-07-27): effectively CLOSED.** Three parts, all landed:
     1. **Duck-exclude stream property — neutralized.** The VUI dual-stamps both the neutral
        `voicebrain.duck_exclude` and the legacy `gabagent.duck_exclude` on both audio paths
        (`PIPEWIRE_PROPS` + `PULSE_PROP`); the brain dual-reads both keys through one shared predicate
        feeding both consumers (local-duck exclusion + media-state backstop). Additive/non-breaking.
        Public: voice-agent `ca56619` + gabagent `de3b489`.
     2. **Protocol ownership — moved to the VUI.** voice-agent now owns the canonical brain-neutral
        spec at [`docs/VOICE_PROTOCOL.md`](docs/VOICE_PROTOCOL.md) (`6b305d0`); gabagent links to it as
        one reference brain (`3706ee2`). The AI-agnostic front-end owns the agnostic contract. The
        `/media/*` endpoint names were already brain-neutral — no rename was needed.
     3. **Neutral brain selector — shipped.** `BRAIN=remote` drives any HTTP/SSE brain; `BRAIN=gabagent`
        stays as a back-compat alias (`6b305d0`).
   - **Remaining (maintainer-gated):** only the flag-day drop of the legacy `gabagent.duck_exclude` key —
     gated on the brain dual-read being live across the whole running fleet, then a coordinated cut.

**Verification debts that gate ship-confidence** (in-flight, not new work): image-gen display
joint-verify on the TV satellite; confirm-parser echo live re-drive; the post-USB-cycle in-process
re-attach hardening (currently catches cleanly ~1/3, else a ~15s restart-fallback).

---

## Next — PoC-enabling (promoted from "deferred" by the charter)

- **Portability hardening** *(internal #55)* — output-rate **autodetect-write** + kill the remaining
  path hardcodes (brain-binary path, TTS TMPDIR). This *is* "every hardcode = a ship blocker."

---

## Deferred — real, post-ship

- **Internet-outage → local-model failover** *(internal #52)* — "local-first" robustness; the PoC
  assumes a reachable LAN brain, so this waits until just after ship.
- **Per-device wake model for the Pi far-field mic** *(internal #67)* — hardware-survival quality;
  the Pi already runs. Needs a real-voice capture pass.
- **Wake precision over music** — next lever is a self-negatives capture of the user's own
  throat-clears (non-overlapping with "Hey Aria" → kills the FP without the recall cost).
- **Multi-device wake arbitration (ESP)** *(internal #66)* — "only the nearest device answers."
  Design locked; brain-side referee built flag-OFF; awaiting the threshold-zoning drive.
- **Open bug/tuning backlog** — lives in the internal tracker (wake recall over a loud movie held on
  a brain-side sink fix; brain-side reconcile-vs-duck race; TIDAL search latency; resume-from-0).

---

## Someday — beyond the PoC (new domains, gated)

- **Phone thin-client + `send_to_builder`** *(internal #68, "Vega")* — a new platform (Android);
  the long-arc endgame. Spec written; not started.
- **Presence-aware result routing** — serve the device the user is *at*. Tier-1 is software-only
  (acoustic-recency from existing VAD/wake timestamps); tier-2 adds an optional per-room mmWave
  sensor. Shares the per-device activity signal with ESP arbitration.
- **Tonality** — emotion-aware STT (soft hint) + expressive/sarcastic TTS (brain-owned tags). A TTS
  swap re-opens latency + duck/half-duplex/barge-in timing — treat as its own project. Feasibility
  study done.
- **Turbo / brain-tier eye indicator** — a distinct eye when Aria is on a faster/failover brain.
  Generalize to a `mode`/`backend` state channel. (User designs the art.)
- **Whole-home / Home Assistant seam** — multi-room control + timers; consume HA as a brain tool.
- **Stop-word barge-in model** — a dedicated "Aria stop" micro-model decoupled from the reused wake
  model (closes onset-transient + sentence-boundary self-trips). Reference: OHF-Voice LVA.
- **SOP escape-hatch sweep** — re-word absolutist internal SOPs as strong-default + gated-exception.
  Process housekeeping; kept on the roadmap per the 2026-07-13 review.

---

## Cut / killed

- **Cut this pass: none.** All idle-deferred items were reviewed 2026-07-13 and re-bucketed, not
  dropped.
- **Dead-ends — do not retry** (documented, evidence-backed):
  - **Wake model v7** — reweighting only trades recall↔precision (vocal-music residual overlaps the
    positives). v6 kept.
  - **`webrtc_noise_gain` liftable** — 5-round cross-audit took it to zero (no AEC; NS crushes recall;
    AGC mis-layered vs the AEC double-talk clamp). Shelved.
  - **Software `xvf_host REBOOT` as the reSpeaker Mode-A cure** — ruled out; only a true VBUS
    power-cycle clears the firmware wedge (now solved in hardware, see Shipped).

---

## Shipped (dated, git-anchored)

- **v1 foundation** — pluggable Pipecat voice shell, local-first STT/TTS/LLM, `uv` venv, swappable
  providers (`4d2198d`, 06-01).
- **Brain split** — LLM + tools + 3-tier safety moved to a separate brain over HTTP/SSE (present from
  commit #1; `tools.py`/`test_safety.py` no longer in-repo).
- **Wake / duck / half-duplex subsystem** — wake word "Hey Aria", media ducking (single-writer),
  half-duplex turn-taking (`e1bb62c` 06-02 → `f279250`/`a34c153`).
- **Public packaging** — README + MIT LICENSE + AUR `voice-agent-git` (`1b2742c`/`5a4c0dd`, 06-05);
  **first public GitHub push** (clean history).
- **Interrupt-word barge-in** — "Aria" mid-reply cuts TTS + cancels the brain turn (`573dd41`, 06-16).
- **HAL-eye status side-channel** — semantic state → tmpfs for a desktop panel (`3c39dd8`, 06-16).
- **Thin-client satellite + multi-room + STT/TTS offload** — the pivot that defined the PoC
  (`97a20a8`, 06-21).
- **GPU STT** — whisper.cpp/Vulkan large-v3-turbo server, live on the reference workstation
  (`7ae0fd8`, 06-22). *(Supersedes the v1 "GPU STT never needed" note.)*
- **Runtime voice-volume control** — "Aria, turn your voice up/down" via an SSE `voice_volume` channel
  (`e8225f6`, 06-20).
- **SmartTurn-honoring endpointing**, duck single-writer, convo-hold — no mid-sentence cut-offs
  (`e8225f6`/`1ef8c71`).
- **Slim the install** *(internal #72)* — dropped the dead `local-smart-turn` extra (`9a57573`, 07-27).
  The default turn analyzer (SmartTurn v3) is pure-onnx and ships in pipecat core, so the extra's
  torch/CUDA/nvidia stack was never imported at runtime — removing it takes a fresh install ~5.6 GB →
  ~0.9 GB with zero behavior change. *(Supersedes the earlier "pin torch to the CPU wheel" plan — the
  whole stack was dead, not just mis-indexed.)*
- **Long-form dictation hold** + the `send_to_builder` deferred-speak seam (`a3f6fab`, 06-24).
- **Token pairing (3c)** — the last hand-typed install step removed: a satellite claims the brain's
  bearer token over the wire during an operator-opened window (`56d59c1` client, 07-28), designed
  adversarially with the reference brain and live-certified cross-box. *(Completes "auto-fill the
  token" under Now #2.)*
- **Image-gen display consumer** — voice "make a picture" → mpv sink with locality routing
  (`4a7ad55`, 07-04); live on the workstation, TV-satellite verify open.
- **Pi media-deafness fix** — pause video on wake for a clean mic (`d7fe52f`, 07-04).
- **reSpeaker capture-stall recovery series** — ranked device matcher (`8a270c7`), reopen-on-stall
  (`dd83d48`), the ~20-min firmware-wedge proactive recycle (`843f393`/`6707f18`), and the
  **USB VBUS power-cycle self-heal** via a PPPS hub + `uhubctl` — the true Mode-A cure, **cased,
  wired, and live-verified** (`0af9989`, 07-07). *(Supersedes the tracker's "buy the hub when the maintainer
  wants" / "do when it arrives" notes.)*
- **Installer packaging — Phase-1** (text-only workstation wizard, brain-side) built + converged
  (2026-07-12); voice-agent contributes zero code until its Phase-3 lane.
