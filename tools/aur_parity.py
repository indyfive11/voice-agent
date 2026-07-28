#!/usr/bin/env python3
"""aur_parity.py — cross-repo AUR bridge for the Installer-Parity SOP (CLAUDE.md).

Asserts the SEPARATE AUR package repo (voice-agent-git PKGBUILD) stays in sync with the code's declared
install manifest (`[tool.voice-agent.install]` in pyproject.toml):

  * PKGBUILD `depends`  MUST be a superset of manifest.system_pkgs  (a required system lib not packaged
    = a clean-box install that crashes at runtime).
  * the launcher's `rsync --exclude` set MUST be a superset of manifest.user_writable_dirs  (a
    user-writable dir NOT excluded = `rsync -a --delete` WIPES the user's data on package upgrade).

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
    dm = re.search(r"^depends=\(([^)]*)\)", text, re.MULTILINE)
    if not dm:
        raise ValueError("no depends=(...) array found")
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
    try:
        depends, excludes = _parse_pkgbuild(pkgbuild.read_text(encoding="utf-8"))
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

    if fail:
        print(f"\naur-parity [{ctx}]: FAILED — reconcile {pkgbuild} with the manifest.", file=sys.stderr)
        return 1
    print(f"aur-parity [{ctx}]: OK — depends ⊇ system_pkgs {man['system_pkgs']}, "
          f"excludes ⊇ user_writable_dirs {man['user_writable_dirs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
