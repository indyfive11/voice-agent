#!/usr/bin/env bash
# bootstrap.sh — THE canonical install entry point for voice-agent.
#
#     git clone <this repo>  →  ./bootstrap.sh  →  uv venv  →  voice_agent_install.<role>
#
# Every other install shape is DERIVED from this path, not a rival to it: the AUR package is a thin
# wrapper over the same steps, and a satellite's rsync tree is a DEPLOY mechanism, not an install.
# If you are about to add a second way to install this project, change this file instead.
#
# SCOPE, deliberately small. This script does only what Python cannot do for itself — find or
# install `uv`, create the virtualenv, sync dependencies — and then hands off. Every decision made
# in shell is a decision the provisioner cannot unit-test, so there are as few of them as possible:
# prompting, hardware detection, config and unit installation all live in `voice_agent_install/`.
#
# It also runs in the one place nothing else can assume: a box where this project is not installed
# yet. So it is bash + coreutils only, it never imports the project, and it fails loudly with a
# remedy rather than continuing on a guess.
#
# DEVELOPERS: this provisions a box to RUN the agent, so it syncs without the dev dependency group.
# To set up a development tree, use `uv sync` directly. Running this in a checkout is ADDITIVE and
# safe — the sync is `--inexact`, so it installs what the role needs and removes nothing. It did not
# always behave that way: a plain `uv sync` reconciles the venv to the manifest, which took out the
# dev group AND any separately-installed extra. `run.sh` (`uv run`) restores default groups on the
# next launch, but nothing re-selects an extra — so a dropped `[mdns]` stayed dropped, and discovery
# fails soft, meaning the box loses brain discovery with no error at all. Hence --inexact.
#
# EXIT CODES. Because the last line `exec`s the provisioner, its exit status BECOMES this script's:
# one process, one status. So the two must not collide, and bootstrap's own pre-handoff failures live
# in a reserved 10+ band, leaving the low codes to mean exactly what the provisioner says they mean:
#   0 provisioned · 1 nothing usable was written · 2 bad args / no terminal to prompt on
#   3 config and unit ARE written but a service leg did not answer
#   10 usage · 11 not a voice-agent tree · 12 refused: deploy mirror · 13 refused: running as root
#   14 uv unavailable · 15 dependency sync did not produce a venv
# An operator (or a deploy-verify script) seeing a failure can therefore tell "you ran this in the
# wrong directory" from "provisioning failed partway", which have unrelated remedies.

set -euo pipefail

ROLE="satellite"
EXTRAS=()
ASSUME_YES=0

usage() {
    cat <<'EOF'
Usage: ./bootstrap.sh [--role ROLE] [--extra NAME]... [--yes] [-- ARGS...]

  --role ROLE   Which role to provision this box as. Default: satellite.
  --extra NAME  Optional dependency group to install (repeatable). See pyproject.toml.
                `mdns` adds zeroconf, which lets the provisioner PRE-FILL the brain host by
                discovery instead of asking for it. Off by default: discovery degrades to a
                manual prompt when absent, so an unconfigured install still works. Note that
                `run.sh` re-syncs on every launch and does not re-select extras.
  --yes         Do not ask before installing `uv`. For unattended runs.
  -- ARGS...    Everything after `--` is passed through to the provisioner.

Re-running is safe: dependency sync is idempotent, and the provisioner merges into an existing
.env rather than overwriting it.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --role)    ROLE="${2:?--role needs a value}"; shift 2 ;;
        --extra)   EXTRAS+=("--extra" "${2:?--extra needs a value}"); shift 2 ;;
        --yes|-y)  ASSUME_YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --)        shift; break ;;
        *)         echo "bootstrap: unknown option: $1" >&2; usage >&2; exit 10 ;;
    esac
done

# --- where are we -----------------------------------------------------------------------------
# Resolved from THIS SCRIPT's own location, never from the caller's cwd — same rule as
# voice_agent_install/paths.py, and for the same reason: two install trees can coexist on one box,
# and operating on the one you are not running is a silent, extremely confusing failure.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

if [ ! -f "$ROOT/main.py" ] || [ ! -d "$ROOT/voice_agent_install" ]; then
    echo "bootstrap: $ROOT does not look like a voice-agent checkout (no main.py / voice_agent_install/)." >&2
    echo "           Run this script from inside the tree you cloned." >&2
    exit 11
fi

# Two layouts are supported and no others: a git checkout, and the XDG data dir the packaged install
# uses. This third shape — the right files, no .git, not the XDG path — is a DEPLOY ARTIFACT: an rsync
# mirror of a checkout, with a snapshot of pyproject.toml/uv.lock from whenever it was last seeded.
# Syncing there resolves against that stale manifest and prunes whatever was installed out-of-band.
# Same classification voice_agent_install/paths.py makes; refusing here is what keeps it a mirror.
XDG_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/voice-agent"
# Resolve the same way $ROOT was (physical, symlinks followed) or the compare is logical-vs-physical: a
# symlinked $HOME or ~/.local/share would make a legitimate XDG install fail and be refused as a mirror.
# When the directory does not exist yet the raw string stands and the compare correctly fails.
XDG_ROOT="$(cd -- "$XDG_ROOT" 2>/dev/null && pwd -P || printf '%s' "$XDG_ROOT")"
if [ ! -d "$ROOT/.git" ] && [ "$ROOT" != "$XDG_ROOT" ]; then
    echo "bootstrap: $ROOT is not a git checkout and is not $XDG_ROOT." >&2
    echo "           This looks like a deploy mirror (rsync target), not an install. Installing into it" >&2
    echo "           would sync against its stale pyproject.toml/uv.lock and prune packages it needs." >&2
    echo "           Install from a fresh clone instead, or pass --root to the provisioner deliberately." >&2
    exit 12
fi

# Everything provisioned here belongs to ONE user: the .env this writes, the venv it syncs, and the
# systemd **--user** unit the role is meant to install. Run as root, the .env and venv come out
# root-owned and unusable by the account that actually runs the agent, and there is no user session
# for a --user unit to live in. Refuse rather than half-succeed.
if [ "$(id -u)" -eq 0 ]; then
    echo "bootstrap: do not run this as root." >&2
    echo "           voice-agent installs a systemd --user unit; run it as the user that will run the agent." >&2
    exit 13
fi

# --- uv ---------------------------------------------------------------------------------------
# uv installs itself to ~/.local/bin, which is not on PATH in a fresh non-login shell on Debian, so
# "not on PATH" and "not installed" are different states and only one of them needs the network.
if ! command -v uv >/dev/null 2>&1 && [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    UV_INSTALL_CMD='curl -LsSf https://astral.sh/uv/install.sh | sh'
    echo "bootstrap: uv is not installed. It manages the Python toolchain and this project's venv."
    echo "           Install command: $UV_INSTALL_CMD"
    if [ "$ASSUME_YES" -ne 1 ]; then
        # Downloading and executing a remote script is a real change to this machine, so it is asked
        # for explicitly and shown first. Answer no and install uv however your distro prefers.
        if [ ! -t 0 ]; then
            echo "bootstrap: no terminal to ask on. Install uv, or re-run with --yes." >&2
            exit 14
        fi
        printf 'Run it now? [y/N] '
        read -r reply
        case "$reply" in [yY]|[yY][eE][sS]) ;; *) echo "bootstrap: aborted; install uv and re-run." >&2; exit 14 ;; esac
    fi
    command -v curl >/dev/null 2>&1 || { echo "bootstrap: curl is required to install uv." >&2; exit 14; }
    eval "$UV_INSTALL_CMD"
    PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || {
        echo "bootstrap: uv still not on PATH after install — open a new shell and re-run." >&2; exit 14; }
fi

# --- venv + dependencies ------------------------------------------------------------------------
# `uv sync` creates .venv and provisions a matching interpreter itself if the host has none in the
# requires-python range, which is why this script never checks for a system python.
# --no-dev: `[dependency-groups] dev` installs by default and a provisioned box has no use for the test
# toolchain. --inexact: `uv sync` otherwise RECONCILES the venv to the manifest, which makes it a
# DESTRUCTIVE verb in a tree that is already provisioned — it removes anything not named here, including
# extras installed out-of-band. That is how a stray run in a working tree silently dropped `zeroconf`:
# discovery.py imports it lazily inside a try and degrades to "manual host only" without raising, so the
# box loses brain discovery with no error and no symptom until someone wonders why. An installer may add
# what a role needs; it may never quietly take away capabilities it was not asked about.
SYNC_FLAGS=(--no-dev)
if uv sync --help 2>/dev/null | grep -q -- '--inexact'; then
    SYNC_FLAGS+=(--inexact)
else
    # Never prune silently. If this uv cannot be additive, say so rather than narrowing the box.
    echo "bootstrap: WARNING — this uv has no --inexact, so the sync will REMOVE packages not named in" >&2
    echo "           pyproject.toml, including extras installed separately (e.g. the mdns/zeroconf" >&2
    echo "           extra). Upgrade uv, or re-pass '--extra <name>' for anything you need kept." >&2
fi
echo "bootstrap: syncing dependencies into $ROOT/.venv (this can take a while on first run)…"
uv sync "${SYNC_FLAGS[@]}" "${EXTRAS[@]+"${EXTRAS[@]}"}"

VENV_PYTHON="$ROOT/.venv/bin/python"
[ -x "$VENV_PYTHON" ] || { echo "bootstrap: expected an interpreter at $VENV_PYTHON after sync." >&2; exit 15; }

# --- hand off -------------------------------------------------------------------------------------
# Everything past this line is Python's job. exec, so the provisioner owns the terminal (it prompts)
# and its exit status is this script's exit status.
echo "bootstrap: provisioning role '$ROLE'…"
exec "$VENV_PYTHON" -m "voice_agent_install.$ROLE" "$@"
