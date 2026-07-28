# Installing voice-agent

This is the **self-provisioning** install path: clone the repo, run one script, answer a couple of
prompts, and the box comes up as either a **satellite** (a thin voice client that offloads to a brain
on your LAN) or a **voice-host** (the brain box, advertising itself so satellites can find it).

> **TL;DR**
> ```bash
> git clone https://github.com/indyfive11/voice-agent && cd voice-agent
> ./bootstrap.sh                       # provision this box as a satellite (the default)
> ./bootstrap.sh --role voice-host     # …or as the brain box, with brain discovery turned on
> ```
> Re-running is safe — the dependency sync is idempotent and the provisioner *merges* into an existing
> `.env` rather than overwriting it.

If you just want to run the agent from a development checkout and edit `.env` by hand, you don't need
this guide — see **Quick start** in the [README](../README.md). `bootstrap.sh` is for standing up a
box (especially a headless satellite) with as little hand-editing as possible.

---

## What `bootstrap.sh` is (and isn't)

`bootstrap.sh` is **the** canonical install entry point. It does only what Python can't do for itself —
find or install [`uv`](https://github.com/astral-sh/uv), create the virtualenv, sync dependencies — and
then hands off to a Python provisioner (`voice_agent_install/<role>.py`) that does the prompting,
hardware detection, `.env` emission, and systemd `--user` unit install.

- It runs as **bash + coreutils only** and never imports the project, because it has to run on a box
  where the project isn't installed yet.
- It installs a systemd **`--user`** unit, so **do not run it as root** — run it as the user that will
  run the agent.
- Every install shape is derived from this path: the AUR package is a thin wrapper over the same
  steps, and a satellite's rsync tree is a *deploy* mechanism, not an install (bootstrap refuses to
  provision into a deploy mirror — see [Exit codes](#exit-codes--troubleshooting)).

### Prerequisites

- **Linux** with a systemd user session (Debian/Raspberry Pi OS and Arch are the tested targets).
- **`portaudio`** and **`espeak-ng`** system packages (audio I/O + eSpeak fallback).
- **Python is *not* a prerequisite** — `uv` provisions a matching 3.12/3.13 interpreter itself if the
  host has none in range. If `uv` is missing, bootstrap offers to install it (or pass `--yes` for an
  unattended run); answer no and install it however your distro prefers.

---

## Roles

`bootstrap.sh --role <role>` picks what the box becomes. There are two:

| Role | What it is | mDNS |
|------|-----------|------|
| `satellite` *(default)* | A thin client: local mic, wake, VAD, turn-taking, TTS, and offload of cognition (and optionally STT/TTS) to a brain over the LAN. | **Discovers** the brain (zeroconf installed by default). |
| `voice-host` | The **brain box**. Detects its own LAN IP and turns on the brain's `_voice-brain._tcp` advertiser so satellites can find it without a typed address. | **Advertises** itself. |

The roles are independent: a one-box setup can run a local brain and skip discovery entirely; a
multi-room setup runs one `voice-host` (the brain) and any number of `satellite`s.

---

## Installing a satellite (the common case)

```bash
git clone https://github.com/indyfive11/voice-agent && cd voice-agent
./bootstrap.sh                    # --role satellite is the default
```

The provisioner will:

1. **Ask for a room id** — a durable name for this box (e.g. `living-room`). It names the satellite to
   the brain and rides on every request.
2. **Find the brain.** If the brain is advertising (see [voice-host](#installing-the-brain-box-voice-host)),
   mDNS pre-fills its host/port for you to confirm. On a miss, it tells you *why* (a dead advertiser vs.
   a firewall dropping `5353/udp` read identically otherwise) and falls back to asking you to type the
   brain's `IP:port`. Discovery is an enhancement, never a hard dependency — the typed-host floor always
   works.
3. **Collect credentials.** By default you supply the brain's bearer token (and STT/TTS tokens if you
   offload those) at the prompt. To skip typing the brain token, use [`--pair`](#zero-touch-pairing---pair).
4. **Write `.env`, install the `--user` unit, enable linger, and verify** all three service legs
   (brain / STT / TTS) on authenticated routes. Linger lets the unit run without an active login
   session — essential for a headless satellite.

On success it prints how to bring the box up:

```bash
systemctl --user start voice-agent   # (the default unit name)
```

### Satellite provisioner flags

Anything after a bare `--` is passed straight through to the provisioner:

```bash
./bootstrap.sh --role satellite -- --dry-run
```

| Flag | Effect |
|------|--------|
| `--dry-run` | Compose and validate everything, write nothing. Good for a first look. |
| `--pair` | Obtain the brain token over the wire instead of typing it (see below). |
| `--skip-verify` | Write config + unit but don't probe the brain/STT/TTS legs afterward. |
| `--non-interactive` | Never prompt — fail loudly instead of guessing an unanswerable value. For automation once every value can be supplied without asking. |
| `--unit-name NAME` | Install under a different systemd unit name (default `voice-agent`). |
| `--root DIR` | Provision a specific install tree (default: the tree the script runs from). |

### Zero-touch pairing (`--pair`)

`--pair` removes the last hand-typed step — the brain's bearer token. Instead of reading the token off
the brain and typing it into `.env`, the satellite **claims it over the wire** during a short window
the operator opens on the brain:

```bash
./bootstrap.sh --role satellite -- --pair
```

All human interaction happens **brain-side** — the satellite stays headless. On the **reference brain
(gabagent)** you open the accept window with:

```bash
gab pairvoiceagent      # opens a short window; accept the pairing box that appears
```

Under the hood the satellite persists a 128-bit random `client_id`, the brain mints a one-time
`claim_secret` bound to the accepting box's IP, and the token is delivered before the provisioner
writes anything — so a satellite that *can't* pair writes and enables nothing (no half-configured,
headless wedge). Re-running with the same persisted `client_id` is idempotent.

> Pairing is a **brain** concern; the wire contract is owned brain-side and documented in the reference
> brain's spec (gabagent `docs/PAIRING.md`). MVP pairing carries the **brain auth token** only — STT/TTS
> tokens are still prompted. See also [`VOICE_PROTOCOL.md` → Token provisioning & pairing](VOICE_PROTOCOL.md).

---

## Installing the brain box (voice-host)

Run this **on the machine that runs the brain**, to turn on discovery so satellites can find it without
a typed IP:

```bash
./bootstrap.sh --role voice-host
```

It detects the box's primary-route LAN IP and enables the brain's `_voice-brain._tcp` advertiser
(cross-process into the brain's own install step — never an import). If auto-detection picks the wrong
interface, pin it:

```bash
./bootstrap.sh --role voice-host -- --host 192.168.1.50 --room-id study
```

| Flag | Effect |
|------|--------|
| `--host IP` | The LAN address to advertise on (default: auto-detect the primary-route NIC). |
| `--room-id ID` | This brain's room identity (written as `voice_room_id`). |

Discovery is off by default on a stock brain and **refuses to advertise on loopback** (a bare
"advertise" on a default single-box install would advertise nothing useful). If enabling it fails,
that's **not fatal**: satellites still work — just type the brain's IP at satellite-install time.

---

## Discovery, end to end

1. On the brain box: `./bootstrap.sh --role voice-host` → the brain advertises `_voice-brain._tcp` on
   its LAN IP.
2. On each satellite: `./bootstrap.sh` → mDNS finds the brain and pre-fills its `IP:port`; you confirm.
3. At boot, a satellite can re-resolve a stale brain address if the brain moved (opt-in per box with
   `BRAIN_REDISCOVER=1`).

mDNS returns **host and port only** — it never carries credentials, and it's off until you run the
voice-host role, so it can never be load-bearing. It just pre-fills a default you confirm. To opt a
satellite out of discovery entirely (typed-IP only), install it with `--no-mdns`:

```bash
./bootstrap.sh --role satellite --no-mdns
```

---

## `bootstrap.sh` options

```
Usage: ./bootstrap.sh [--role ROLE] [--extra NAME]... [--no-mdns] [--yes] [-- ARGS...]

  --role ROLE   satellite (default) | voice-host
  --extra NAME  Optional dependency group to install (repeatable). See pyproject.toml.
                `mdns` (zeroconf) is installed BY DEFAULT for the satellite role.
  --no-mdns     Do not install the mdns/zeroconf extra for a satellite (typed-IP only).
  --yes, -y     Do not ask before installing uv (for unattended runs).
  -- ARGS...    Everything after `--` is passed through to the role provisioner.
```

---

## Verifying and running

- **Start now:** `systemctl --user start voice-agent`
- **Watch it:** `journalctl --user -u voice-agent -f`
- **Enable at boot:** the provisioner already enables the unit and turns on linger; nothing more needed.

A satellite install verifies the brain/STT/TTS legs during provisioning. If a leg fails, the summary
names it and the config is still written — fix the service or the token and re-run (the `.env` merge is
idempotent).

---

## Exit codes & troubleshooting

`bootstrap.sh` `exec`s the provisioner, so the provisioner's exit status *becomes* bootstrap's — one
process, one status. Low codes are the provisioner's; bootstrap's own pre-handoff failures live in a
reserved `10+` band:

| Code | Meaning | Remedy |
|-----:|---------|--------|
| `0` | Provisioned. | — |
| `1` | Nothing usable was written. | See the failing step in the summary. |
| `2` | Bad args / no terminal to prompt on. | Attach a TTY, or supply values for `--non-interactive`. |
| `3` | Config + unit written, but a service leg did not answer. | Fix the service or token, re-run. |
| `10` | Bad `bootstrap.sh` usage. | Check the flag against the usage above. |
| `11` | Not a voice-agent tree. | Run from inside the cloned checkout. |
| `12` | Refused: this is a **deploy mirror** (rsync target), not an install. | Install from a fresh clone. |
| `13` | Refused: running as root. | Run as the user that will run the agent. |
| `14` | `uv` unavailable. | Install `uv`, or re-run with `--yes`. |
| `15` | Dependency sync produced no venv. | Re-run; if it persists, `uv sync` manually to see the error. |

**"It installed but did nothing / no brain discovery."** If you ran a plain `uv sync` in the tree
out-of-band, it can prune the `mdns`/zeroconf extra (discovery then fails *soft* — degrades to a manual
host prompt with no error). Re-run `./bootstrap.sh` (its sync is additive) or `uv sync --extra mdns`.

---

## See also

- [README](../README.md) — what voice-agent is, the stack, and the dev quick-start.
- [`VOICE_PROTOCOL.md`](VOICE_PROTOCOL.md) — the brain↔shell contract (including token provisioning & pairing).
- [`PLAN.md`](../PLAN.md) — as-built architecture. [`ROADMAP.md`](../ROADMAP.md) — plan of record.
