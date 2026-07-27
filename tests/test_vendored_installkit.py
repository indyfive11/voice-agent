"""Guards on the VENDORED copy of installkit (Layer A), vendored into this tree at a pinned SHA.

Two independent risks, two tests:

1. RESOLUTION (the sibling-split trap). ``import installkit`` must resolve to the copy INSIDE the
   voice-agent tree, never the developer's editable sibling at ``~/dev/installkit``. If it resolved to
   the sibling, host tests would exercise a different tree than the one that rsyncs to the Pi
   (``git ls-files '*.py'``) — a stale/wrong vendored copy could ship green.

2. ISOLATION (what makes installkit vendorable at all). A vendored module that grew a third-party or
   app import would drag an undeclared dep onto the Pi, or couple the two products. This mirrors
   installkit's own ``test_import_isolation.py`` over the vendored copy so a future edit to OUR copy
   cannot ship such an import green.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import installkit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_DIR = Path(installkit.__file__).resolve().parent
_STDLIB = set(getattr(sys, "stdlib_module_names", set()))
_ALLOWED_ROOTS = _STDLIB | {"installkit"}
_FORBIDDEN_APPS = {"gabagent", "voice_agent", "voice-agent"}


def _module_files() -> list[Path]:
    return sorted(_PKG_DIR.rglob("*.py"))


def test_installkit_resolves_to_the_vendored_copy_in_this_tree():
    """Not the editable sibling ~/dev/installkit — the copy the Pi rsync actually ships."""
    assert _PKG_DIR.is_relative_to(_REPO_ROOT), (
        f"installkit resolved to {_PKG_DIR}, OUTSIDE the voice-agent tree ({_REPO_ROOT}). "
        f"A sibling/editable install would test a different tree than the one that deploys to the Pi."
    )


def test_vendored_installkit_has_module_files():
    assert _module_files(), "vendored installkit should contain .py modules"


def test_vendored_installkit_imports_only_stdlib_or_itself():
    """No third-party and no consuming-app import anywhere in the vendored copy."""
    offenders: list[str] = []
    for f in _module_files():
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import within installkit — fine
                    continue
                roots = [(node.module or "").split(".")[0]]
            for r in roots:
                if r and (r not in _ALLOWED_ROOTS or r in _FORBIDDEN_APPS):
                    offenders.append(f"{f.name}: {r}")
    assert not offenders, "vendored installkit may only import stdlib/itself:\n" + "\n".join(offenders)
