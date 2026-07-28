#!/usr/bin/env bash
# installer-parity.sh — config-knob parity gate (Installer-Parity SOP, CLAUDE.md).
#
# DELTA-scoped: only env knobs ADDED since the push base must be documented in .env.example (the ~84
# pre-existing internal knobs are grandfathered — see .installer-parity-ignore). Plus a FULL-TREE
# CANARY so a refactor that breaks the detector regex can't pass by matching nothing.
#
# Checks (BLOCK = exit 1):
#   1. ignore-list lint      — every entry needs a `# reason` (BLOCK)
#   2. canary                — full-tree scan matches >= CANARY_FLOOR known keys, else false-green (BLOCK)
#   3. new-knob doc parity   — env key added in diff, absent from .env.example and not ignored (BLOCK)
#   4. default-flip          — added literal default != documented .env.example value (WARN, non-blocking)
#
# Base resolution: $1, else $PARITY_BASE, else merge-base with @{upstream}, else with origin/main.
# No base resolvable -> canary + lint still run; the delta checks WARN-skip (never a silent pass).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

CANARY_ONLY=0
[[ "${1:-}" == "--canary-only" ]] && { CANARY_ONLY=1; shift; }

CANARY_FLOOR="${CANARY_FLOOR:-100}"
ENV_EXAMPLE=".env.example"
IGNORE_FILE=".installer-parity-ignore"
SCAN_FILES=$(git ls-files '*.py' run.sh bootstrap.sh)
fail=0

# env-key extraction: os.environ.get("K"| os.getenv("K"| getenv("K"| _env("K"| environ["K"]
# matches os.environ.get / environ.get / environ.setdefault / os.getenv / getenv / _env( / environ["K"]
# (prefix-agnostic on environ.* so `from os import environ; environ.get("K")` can't evade the detector).
# A pathological ALIAS (`from os import environ as e; e.get(...)`) is not followed — no regex can — but a
# wholesale idiom shift crashes the canary count below its floor and blocks. The canary is that backstop.
key_re='(environ\.get|environ\.setdefault|getenv|_env)\(['"'"'"][A-Z][A-Z0-9_]{2,}|environ\[['"'"'"][A-Z][A-Z0-9_]{2,}'
extract_keys() { grep -hoE "$key_re" 2>/dev/null | grep -oE '[A-Z][A-Z0-9_]{2,}$'; }

# ---- documented + ignored sets ----
documented_keys() { grep -oE "^#?[[:space:]]*[A-Z][A-Z0-9_]+=" "$ENV_EXAMPLE" 2>/dev/null | grep -oE '[A-Z][A-Z0-9_]+'; }
ignored_tokens() { grep -vE '^[[:space:]]*(#|$)' "$IGNORE_FILE" 2>/dev/null | awk '{print $1}'; }

# ---- 1. ignore-list lint (reason mandatory) ----
if [ -f "$IGNORE_FILE" ]; then
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ "$line" != *"#"* ]]; then
      echo "BLOCK  ignore-list entry without a '# reason': $line" >&2; fail=1
    fi
  done < "$IGNORE_FILE"
fi

# ---- 2. canary (full-tree) ----
n_scanned=$(echo "$SCAN_FILES" | xargs grep -hoE "$key_re" 2>/dev/null | grep -oE '[A-Z][A-Z0-9_]{2,}$' | sort -u | wc -l)
if [ "$n_scanned" -lt "$CANARY_FLOOR" ]; then
  echo "BLOCK  canary: env-knob scan matched only $n_scanned keys (floor $CANARY_FLOOR)." >&2
  echo "       The detector regex likely no longer matches the code's env-read idiom (false green)." >&2
  echo "       Update key_re in $0 (and this floor) to the new idiom." >&2
  fail=1
else
  echo "OK     canary: scan matched $n_scanned known env keys (>= floor $CANARY_FLOOR)"
fi

# ---- base resolution for the delta checks ----
base="${1:-${PARITY_BASE:-}}"
if [ "$CANARY_ONLY" -eq 1 ]; then
  base=""   # stateless invocation (suite pytest): canary + lint only, never the git-diff delta
  echo "     (--canary-only: delta doc-parity intentionally skipped)"
elif [ -z "$base" ]; then
  base=$(git merge-base @ '@{upstream}' 2>/dev/null) \
    || base=$(git merge-base @ origin/main 2>/dev/null) || base=""
fi

if [ -z "$base" ]; then
  echo "WARN   no push base resolvable — delta checks skipped (canary + lint still enforced)" >&2
else
  mapfile -t doc  < <(documented_keys | sort -u)
  mapfile -t ign  < <(ignored_tokens | sort -u)
  is_known() { local k="$1"; printf '%s\n' "${doc[@]}" "${ign[@]}" | grep -qxF "$k"; }

  # ---- 3. new-knob doc parity (added lines only) ----
  added=$(git diff "$base" -- '*.py' run.sh bootstrap.sh 2>/dev/null | grep '^+' | grep -v '^+++' | extract_keys | sort -u)
  for k in $added; do
    is_known "$k" || { echo "BLOCK  new env knob '$k' read in code but absent from $ENV_EXAMPLE" >&2
      echo "       -> document it in $ENV_EXAMPLE (with a safe no-op default) OR add to $IGNORE_FILE with a reason." >&2
      fail=1; }
  done

  # ---- 4. default-flip (WARN, non-blocking) ----
  while IFS= read -r addln; do
    kv=$(echo "$addln" | grep -oE "(os\.environ\.get|os\.getenv|getenv)\(['\"][A-Z][A-Z0-9_]{2,}['\"][[:space:]]*,[[:space:]]*['\"][^'\"]*['\"]") || continue
    k=$(echo "$kv" | grep -oE "[A-Z][A-Z0-9_]{2,}" | head -1)
    codedef=$(echo "$kv" | grep -oE "['\"][^'\"]*['\"][[:space:]]*\)?$" | tail -1 | tr -d "\"')" )
    docval=$(grep -oE "^${k}=.*" "$ENV_EXAMPLE" | head -1 | cut -d= -f2-)
    [ -z "$docval" ] && continue
    if [ "$codedef" != "$docval" ]; then
      printf 'WARN   default-flip: %s code-default=%q but %s shows %q (verify no-op or ack in %s)\n' \
        "$k" "$codedef" "$ENV_EXAMPLE" "$docval" "$IGNORE_FILE" >&2
    fi
  done < <(git diff "$base" -- '*.py' 2>/dev/null | grep '^+' | grep -v '^+++')
fi

if [ "$fail" -ne 0 ]; then
  echo "" >&2; echo "installer-parity: config-knob gate FAILED — see BLOCK lines above." >&2
  exit 1
fi
echo "installer-parity: config-knob gate OK"
exit 0
