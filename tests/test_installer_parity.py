"""Installer-parity gates (Installer-Parity SOP, CLAUDE.md).

STATELESS gates only — safe to run in the suite AND in CI with no git diff base. The DELTA-scoped
config-knob doc-parity gate lives in scripts/installer-parity.sh (it needs a push base); here we only
exercise its stateless halves (the detector CANARY + the ignore-list lint) so a broken detector or a
reason-less ignore entry fails the suite too.

The cross-repo AUR bridge (tools/aur_parity.py) is exercised in SUITE mode: with no AUR clone it must
SKIP-LOUD (return 0), never hard-fail portable CI. Its BUMP-context hard-fail is enforced by the AUR
bump recipe (`make aur-parity BUMP=1`), not here.
"""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], **kw):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, **kw)


def test_manifest_present_and_wellformed():
    with (REPO / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    m = data.get("tool", {}).get("voice-agent", {}).get("install")
    assert m is not None, "missing [tool.voice-agent.install] manifest — the parity single-source-of-truth"
    assert m.get("system_pkgs"), "manifest.system_pkgs must be non-empty (declared app system libs)"
    assert m.get("user_writable_dirs"), "manifest.user_writable_dirs must be non-empty (data-loss guard)"
    for key in ("system_pkgs", "user_writable_dirs"):
        assert all(isinstance(x, str) and x for x in m[key]), f"{key} must be non-empty strings"


def test_uv_lock_is_consistent_with_pyproject():
    """A dep added to pyproject without `uv lock` = a stale lock that breaks `uv sync --locked` on the
    satellite. `uv lock --check` is offline + fast."""
    r = _run(["uv", "lock", "--check"])
    assert r.returncode == 0, f"uv.lock out of sync with pyproject.toml — run `uv lock`.\n{r.stderr}"


def test_bootstrap_sync_is_additive_inexact():
    """The --inexact invariant as a GATE, not a comment: a bare `uv sync` reconciles the venv to the
    manifest and silently drops the mdns/zeroconf extra -> the box loses brain discovery with no error."""
    text = (REPO / "bootstrap.sh").read_text(encoding="utf-8")
    assert "SYNC_FLAGS+=(--inexact)" in text, "bootstrap.sh must add --inexact to its sync flags"
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#") or "uv sync" not in s or "--help" in s:
            continue
        assert ("SYNC_FLAGS" in s) or ("--inexact" in s), (
            f"non-additive `uv sync` in bootstrap.sh (missing --inexact / SYNC_FLAGS): {s!r}"
        )


def test_no_untracked_importable_modules():
    """New-Module Deploy-Safety SOP as a gate: an untracked *.py/*.sh a tracked importer needs won't
    ship (satellite = `git ls-files`; AUR = fresh clone) -> guaranteed clean-box crash. BLOCKS."""
    ignore = _ignored_paths()
    r = _run(["git", "ls-files", "--others", "--exclude-standard", "*.py", "*.sh"])
    stray = [p for p in r.stdout.split() if p and p not in ignore]
    assert not stray, (
        f"untracked code files that a fresh clone / satellite rsync will NOT ship: {stray}\n"
        f"-> git-track them, fail-soft the import, or add to .installer-parity-ignore with a reason."
    )


def test_config_knob_canary_and_ignore_lint():
    """Stateless halves of scripts/installer-parity.sh: the detector CANARY (regex still matches the
    code's env idiom) + the ignore-list reason lint. Delta doc-parity self-skips with no base."""
    r = _run(["bash", "scripts/installer-parity.sh", "--canary-only"])
    assert r.returncode == 0, f"canary/ignore-lint failed:\n{r.stdout}\n{r.stderr}"


def test_aur_bridge_suite_mode_skips_loud_without_clone(monkeypatch, tmp_path):
    """With no AUR clone the bridge must SKIP-LOUD (rc 0), never hard-fail portable CI."""
    monkeypatch.setenv("VOICE_AGENT_AUR_DIR", str(tmp_path / "no-such-aur"))
    monkeypatch.delenv("BUMP", raising=False)
    r = _run([sys.executable, "tools/aur_parity.py"])
    assert r.returncode == 0, f"suite-mode bridge should skip-loud, not fail:\n{r.stderr}"
    assert "SKIP-LOUD" in r.stderr


def _ignored_paths() -> set[str]:
    f = REPO / ".installer-parity-ignore"
    if not f.is_file():
        return set()
    out = set()
    for ln in f.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if s and not s.startswith("#"):
            out.add(s.split()[0])
    return out
