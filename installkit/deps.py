"""Dependency-engine — SYSTEM layer. Pure stdlib.

Owns the parts of provisioning that are the SAME regardless of app: which OS package manager we're on,
whether uv/rg/git are present, and turning "these system packages are missing" into an install command.
Python-runtime provisioning is NOT here — it's a per-installer CALLBACK the caller supplies (gabagent →
its venv/package; voice-agent → `uv sync` with role extras), so we never smuggle one app's Python model
into the other. This layer only reports and, when explicitly asked, provisions SYSTEM packages.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

__all__ = [
    "detect_distro",
    "package_manager",
    "have",
    "uv_present",
    "SystemReadiness",
    "check_system_readiness",
    "install_command",
]

# distro-family → (package-manager binary, install subcommand). Kept tiny and data-driven so adding a
# distro is one line, not a code branch.
_PKG_MANAGERS: dict[str, tuple[str, list[str]]] = {
    "arch": ("pacman", ["pacman", "-S", "--needed"]),
    "debian": ("apt", ["apt", "install", "-y"]),
    "fedora": ("dnf", ["dnf", "install", "-y"]),
}

# Canonical tool name → per-family package name, when they differ. Absent ⇒ the tool name IS the package.
_PKG_NAMES: dict[str, dict[str, str]] = {
    "rg": {"debian": "ripgrep", "arch": "ripgrep", "fedora": "ripgrep"},
}


def detect_distro() -> str | None:
    """Return the distro FAMILY ('arch' | 'debian' | 'fedora') from /etc/os-release, else None.

    Uses the authoritative OS source (os-release ID / ID_LIKE), not a guess. None means 'unknown distro'
    — the caller degrades to manual guidance rather than running the wrong package manager."""
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            fields = {}
            for line in f:
                if "=" in line:
                    k, _, v = line.partition("=")
                    # os-release keys are conventionally UPPERCASE (ID, ID_LIKE); normalize to lower
                    # so the lookups below match regardless of the file's casing.
                    fields[k.strip().lower()] = v.strip().strip('"').lower()
    except OSError:
        return None
    ids = [fields.get("id", "")] + fields.get("id_like", "").split()
    for token in ids:
        if token in ("arch", "archlinux", "cachyos", "endeavouros", "manjaro"):
            return "arch"
        if token in ("debian", "ubuntu", "raspbian", "linuxmint", "pop"):
            return "debian"
        if token in ("fedora", "rhel", "centos"):
            return "fedora"
    return None


def package_manager(distro: str | None = None) -> str | None:
    """The install binary for a distro family ('pacman'/'apt'/'dnf'), or None if unknown/absent."""
    distro = distro or detect_distro()
    if distro in _PKG_MANAGERS:
        binary = _PKG_MANAGERS[distro][0]
        return binary if shutil.which(binary) else None
    return None


def have(cmd: str) -> bool:
    """Is an executable on PATH? (thin, testable wrapper over shutil.which)."""
    return shutil.which(cmd) is not None


def uv_present() -> bool:
    """uv is the delivery spine's Python provisioner. bootstrap.sh ensures it BEFORE Python exists;
    this is the in-Python check for anything that runs after."""
    return have("uv")


def _pkg_name(tool: str, distro: str | None, extra_names: dict[str, dict[str, str]] | None = None) -> str:
    """Resolve `tool` to its per-distro package name. A caller-supplied `extra_names` (same
    tool→{distro: pkg} shape) is merged OVER the shared table, so an app can teach the engine its own
    package names WITHOUT leaking them into Layer A's app-agnostic `_PKG_NAMES` (import-isolation spirit)."""
    if not distro:
        return tool
    mapping = dict(_PKG_NAMES.get(tool, {}))
    if extra_names and tool in extra_names:
        mapping.update(extra_names[tool])
    return mapping.get(distro, tool)


def install_command(
    tools: list[str],
    distro: str | None = None,
    extra_names: dict[str, dict[str, str]] | None = None,
) -> list[str] | None:
    """Build the (sudo-free) argv that would install `tools` on this distro, or None if we can't tell.

    `extra_names` lets a caller (e.g. the voice-agent installer) supply its own tool→{distro: pkg} map for
    packages Layer A shouldn't know about (portaudio19-dev, etc.); unset ⇒ the shared table only, i.e. no
    change for the MVP. The caller decides whether to run it (and how to elevate) — never an implicit sudo."""
    distro = distro or detect_distro()
    if distro not in _PKG_MANAGERS or not tools:
        return None
    base = list(_PKG_MANAGERS[distro][1])
    return base + [_pkg_name(t, distro, extra_names) for t in tools]


@dataclass
class SystemReadiness:
    """The dry-run capability report the wizard shows before it writes anything."""
    distro: str | None
    pkg_manager: str | None
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra_names: dict[str, dict[str, str]] | None = None  # caller pkg-name overrides, carried for suggested_install

    @property
    def ok(self) -> bool:
        return not self.missing

    def suggested_install(self) -> list[str] | None:
        return install_command(self.missing, self.distro, self.extra_names) if self.missing else None


# The text-only workstation MVP's genuine system needs: git + ripgrep (rg is a HARD agent dependency).
# Python/uv are handled by bootstrap.sh before this code runs. Kept intentionally small — the MVP is
# text-only; audio/media system deps belong to later per-role / per-plugin installers, not here.
WORKSTATION_TOOLS = ["git", "rg"]


def check_system_readiness(
    tools: list[str] | None = None,
    distro: str | None = None,
    extra_names: dict[str, dict[str, str]] | None = None,
) -> SystemReadiness:
    """One-shot dry run: which of `tools` are present vs missing on this box. Never installs, never
    raises — pure inspection (idempotency- and portability-friendly). `extra_names` (caller pkg-name map)
    is carried onto the result so `suggested_install()` resolves app-specific package names."""
    distro = distro or detect_distro()
    tool_list = tools if tools is not None else WORKSTATION_TOOLS
    present = [t for t in tool_list if have(t)]
    missing = [t for t in tool_list if not have(t)]
    return SystemReadiness(
        distro=distro,
        pkg_manager=package_manager(distro),
        present=present,
        missing=missing,
        extra_names=extra_names,
    )


def run_install(argv: list[str], *, elevate: bool = True) -> int:
    """Run a system-package install command, optionally via sudo. Returns the exit code. Only called
    with explicit user consent from the caller (mode='auto' or a confirmed prompt) — never implicitly."""
    cmd = (["sudo"] + argv) if (elevate and not _is_root()) else argv
    try:
        return subprocess.run(cmd).returncode
    except (OSError, subprocess.SubprocessError):
        return 1


def _is_root() -> bool:
    import os
    return hasattr(os, "geteuid") and os.geteuid() == 0
