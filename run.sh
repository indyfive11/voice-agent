#!/usr/bin/env bash
# voice-agent launcher — one-word start.
#
#   voice-agent          start with the brain configured in .env (BRAIN=…)
#   voice-agent gab      gabagent brain
#   voice-agent debug    gabagent brain + brain-side debug log (joint debugging)
#   voice-agent local    local LLM brain (Ollama/Anthropic/OpenAI per .env)
#
# Set any env inline as a prefix, e.g.  GAB_PORT=9000 voice-agent gab
set -euo pipefail

# Resolve the project dir even when invoked via a symlink on PATH.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

case "${1:-}" in
  gab|gabagent) export BRAIN=gabagent;                              shift ;;
  debug)        export BRAIN=gabagent GABAI_VOICE_DEBUG_LOG=1;      shift ;;
  local)        export BRAIN=local;                                 shift ;;
  -h|--help)    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
esac

# --frozen --no-sync: run the venv EXACTLY as it was provisioned — never re-lock, never re-sync at start.
# The lock is shared across satellites (x86_64 EM, aarch64 Pi, laptop); a bare `uv run` would (a) re-lock
# on any pyproject drift, diverging that shared lock from a runtime box, and (b) re-sync dev deps in/out
# every start. run.sh is a PURE RUNNER; deploy-time reconcile is owned by the deploy step (pi-voice-launch:
# bounded, guarded `uv sync --locked --no-dev --inexact`). Fresh boxes: bootstrap.sh builds the venv first.
# (Tests run via `uv run python -m pytest`, which syncs dev deps on demand — unaffected by --no-sync here.)
exec uv run --frozen --no-sync python main.py "$@"
