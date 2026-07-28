#!/usr/bin/env python3
"""aur_parity.py — cross-repo AUR bridge for the Installer-Parity SOP (CLAUDE.md).

Asserts the SEPARATE AUR package repo (voice-agent-git PKGBUILD) stays in sync with the code's declared
install manifest (`[tool.voice-agent.install]` in pyproject.toml):

  * PKGBUILD `depends`  MUST be a superset of manifest.system_pkgs  (a required system lib not packaged
    = a clean-box install that crashes at runtime).
  * the launcher's `rsync --exclude` set MUST be a superset of manifest.user_writable_dirs  (a
    user-writable dir NOT excluded = `rsync -a --delete` WIPES the user's data on package upgrade).
  * the launcher MUST provision the venv on first run via `bootstrap.sh --sync-only`  (run.sh is a pure
    runner and never syncs, so a launcher that only mirrors + `exec`s run.sh ships an EMPTY .venv that
    crashes on first import — a BEHAVIOR gap the two superset checks above are structurally blind to).

Two invocation contexts (maintainer+GA consensus, the v0.8.0 hole):
  * SUITE / general  (default): AUR clone absent -> SKIP-LOUD (portable CI has no clone; never a silent
    pass, never a tracked hardcoded path).
  * BUMP  (--bump, i.e. the AUR publish recipe): AUR clone absent -> FAIL. At bump time the clone is
    present by definition (you are editing the PKGBUILD in it), so absence == misconfiguration, not N/A.

A PKGBUILD that can't be parsed is a FAIL LOUD, never a green.

Clone location: $VOICE_AGENT_AUR_DIR, else ~/dev/voice-agent-aur.
Exit: 0 = OK or legitimate skip · 1 = parity FAIL / parse FAIL / bump-without-clone.
"""
from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _manifest() -> dict:
    with (REPO / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    m = data.get("tool", {}).get("voice-agent", {}).get("install", {})
    return {
        "system_pkgs": list(m.get("system_pkgs", [])),
        "user_writable_dirs": list(m.get("user_writable_dirs", [])),
    }


def _aur_dir() -> Path:
    return Path(os.environ.get("VOICE_AGENT_AUR_DIR", str(Path.home() / "dev" / "voice-agent-aur")))


def _norm(tok: str) -> str:
    return tok.strip().strip("/").strip()


def _parse_pkgbuild(text: str) -> tuple[set[str], set[str]]:
    """Return (depends, launcher_excludes). Raises ValueError on an unparseable PKGBUILD."""
    # [^)\n]* is bounded to the depends LINE on purpose: a missing closing paren must FAIL LOUD (no
    # match -> parse error), never silently over-capture across newlines into the next array (which
    # would "parse" a malformed PKGBUILD and could pass on a coincidental token overlap). depends is a
    # single-line array here; a deliberately multi-line depends would need this widened with intent.
    dm = re.search(r"^depends=\(([^)\n]*)\)", text, re.MULTILINE)
    if not dm:
        raise ValueError("no depends=(...) array found (malformed/missing closing paren?)")
    depends = {
        _norm(a or b) for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", dm.group(1)) if (a or b)
    }

    excludes: set[str] = set()
    for a, b, c in re.findall(r"--exclude=(?:'([^']*)'|\"([^\"]*)\"|(\S+))", text):
        tok = a or b or c
        if tok:
            excludes.add(_norm(tok))
    if not excludes:
        raise ValueError("no rsync --exclude tokens found (launcher heredoc missing or renamed)")
    return depends, excludes


def _launcher_provisions(text: str) -> bool:
    """True if the AUR launcher provisions the venv on first run (calls `bootstrap.sh --sync-only`).

    The gap this guards (found by the first from-scratch AUR install): run.sh is a PURE RUNNER
    (`uv run --no-sync`) and never provisions, so a launcher that only mirrors the tree and `exec`s
    run.sh leaves a fresh install with an EMPTY .venv -> ModuleNotFoundError on the very first import.
    First-run provisioning MUST route through the tree's `bootstrap.sh --sync-only` (single source of
    truth for the sync + its role extras). This checks BEHAVIOR, not a dep/exclude superset — the class
    the depends/excludes checks are structurally blind to.
    """
    return re.search(r"bootstrap\.sh[^\n]*--sync-only", text) is not None


def main(argv: list[str]) -> int:
    bump = "--bump" in argv or os.environ.get("BUMP") == "1"
    ctx = "bump" if bump else "suite"
    aur = _aur_dir()
    pkgbuild = aur / "PKGBUILD"

    if not pkgbuild.is_file():
        msg = f"AUR clone not found at {aur} (set VOICE_AGENT_AUR_DIR to override)"
        if bump:
            print(f"FAIL   [{ctx}] {msg}", file=sys.stderr)
            print("       At bump time the clone must be present — this is a misconfiguration.", file=sys.stderr)
            return 1
        print(f"SKIP-LOUD [{ctx}] {msg} — cross-repo parity NOT verified here (run `make aur-parity` on a box "
              f"with the AUR clone, and it is hard-required in the AUR bump recipe).", file=sys.stderr)
        return 0

    man = _manifest()
    text = pkgbuild.read_text(encoding="utf-8")
    try:
        depends, excludes = _parse_pkgbuild(text)
    except ValueError as e:
        print(f"FAIL   PKGBUILD parse error at {pkgbuild}: {e}", file=sys.stderr)
        print("       Refusing to pass a PKGBUILD I can't verify (would be a false green).", file=sys.stderr)
        return 1

    fail = 0
    missing_deps = [p for p in man["system_pkgs"] if _norm(p) not in depends]
    if missing_deps:
        print(f"FAIL   PKGBUILD depends is missing declared system_pkgs: {missing_deps}", file=sys.stderr)
        print(f"       depends has: {sorted(depends)}", file=sys.stderr)
        fail = 1

    missing_excl = [d for d in man["user_writable_dirs"] if _norm(d) not in excludes]
    if missing_excl:
        print(f"FAIL   DATA-LOSS: launcher rsync --exclude is missing user_writable_dirs: {missing_excl}", file=sys.stderr)
        print(f"       Without the exclude, `rsync -a --delete` WIPES these on `voice-agent --update`.", file=sys.stderr)
        print(f"       excludes has: {sorted(excludes)}", file=sys.stderr)
        fail = 1

    if not _launcher_provisions(text):
        print("FAIL   FIRST-RUN CRASH: the AUR launcher never provisions the venv "
              "(no `bootstrap.sh --sync-only`).", file=sys.stderr)
        print("       run.sh is a pure runner (uv run --no-sync); without a first-run sync a fresh install "
              "has an EMPTY .venv and crashes on the first import (ModuleNotFoundError).", file=sys.stderr)
        fail = 1

    if fail:
        print(f"\naur-parity [{ctx}]: FAILED — reconcile {pkgbuild} with the manifest.", file=sys.stderr)
        return 1
    print(f"aur-parity [{ctx}]: OK — depends ⊇ system_pkgs {man['system_pkgs']}, "
          f"excludes ⊇ user_writable_dirs {man['user_writable_dirs']}, launcher provisions via bootstrap --sync-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
