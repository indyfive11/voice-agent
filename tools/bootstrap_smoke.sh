#!/usr/bin/env bash
# bootstrap_smoke.sh — reachability backstop (Installer-Parity SOP, CLAUDE.md, Tier 3).
#
# Grep + .env.example prove a surface was EDITED; only a from-scratch install proves it is REACHED
# ("off-box = reachability"). This is the cheap, always-runnable half: it materialises ONLY the
# git-tracked tree (exactly what a satellite rsync `git ls-files` and a fresh AUR `git clone` ship —
# untracked files DO NOT exist there) and asserts the install-relevant surfaces are reachable in it,
# WITHOUT a multi-GB dependency sync.
#
#   --full   additionally run the real `bootstrap.sh --role satellite` (needs network + time). Off by
#            default so this stays a fast standing gate; wire --full into a pre-release check.
#
# Exit 0 = reachable · 1 = a tracked-tree gap (a surface that won't reach a fresh install).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo "smoke: materialising the git-TRACKED tree (satellite deploy = git ls-files; untracked files do NOT"
echo "       ship — exactly the New-Module trap) -> $work"
# Working content of every tracked file, modes preserved (matches the satellite git-ls-files deploy AND a
# fresh clone once committed). An untracked new module is intentionally ABSENT until it is git-added.
git ls-files -z | tar --null --files-from=- -cf - | tar -xf - -C "$work"

repo="$OLDPWD"
cd "$work"
rc=0
# 1. canonical entry points present in the materialised tree, and git TRACKS them executable so a fresh
#    clone gets them +x. (Check the tracked mode, not filesystem -x: mktemp lands on /tmp which is often
#    noexec, where `test -x` is a false negative regardless of the real mode bits.)
[[ -f bootstrap.sh ]] || { echo "FAIL bootstrap.sh missing in tracked tree" >&2; rc=1; }
[[ -f run.sh ]]       || { echo "FAIL run.sh missing in tracked tree" >&2; rc=1; }
for f in bootstrap.sh run.sh; do
    mode=$(git -C "$repo" ls-files --stage "$f" | awk '{print $1}')
    [[ "$mode" == 100755 ]] || { echo "FAIL $f not tracked executable (mode=$mode) — fresh clone won't be +x" >&2; rc=1; }
done
# 2. the parity manifest is reachable from a fresh checkout (tools/aur_parity resolves it)
python3 - <<'PY' || rc=1
import tomllib, pathlib, sys
m = tomllib.load(open("pyproject.toml","rb")).get("tool",{}).get("voice-agent",{}).get("install")
if not m or not m.get("system_pkgs") or not m.get("user_writable_dirs"):
    print("FAIL manifest not reachable/complete in fresh checkout", file=sys.stderr); sys.exit(1)
print(f"OK   manifest reachable: {m}")
PY
# 3. the mdns discovery module a satellite needs is a TRACKED file (not an untracked stray)
if git -C "$OLDPWD" ls-files --error-unmatch voice_agent_install/discovery.py >/dev/null 2>&1; then
    [[ -f voice_agent_install/discovery.py ]] || { echo "FAIL discovery.py tracked but absent from archive" >&2; rc=1; }
    echo "OK   discovery.py reaches a fresh install"
fi
# 4. the parity tooling itself reaches a fresh install (untracked = would not ship)
for t in scripts/installer-parity.sh tools/aur_parity.py; do
    [[ -f "$t" ]] || { echo "FAIL $t absent from tracked tree (untracked? git-add it)" >&2; rc=1; }
done

if [[ $FULL -eq 1 ]]; then
    echo "smoke --full: running real bootstrap.sh --role satellite (this pulls deps)…"
    ./bootstrap.sh --role satellite --yes -- --help || { echo "FAIL full bootstrap smoke" >&2; rc=1; }
fi

[[ $rc -eq 0 ]] && echo "bootstrap-smoke: OK (tracked tree reaches every install surface)"
exit $rc
