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

exec uv run python main.py "$@"
