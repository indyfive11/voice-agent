"""Where this install lives — the DUAL-PATH resolver.

the maintainer's call (2026-07-21): voice-agent supports **two** install layouts, explicitly, and the installer
detects which one it is running in rather than forcing a single shape:

1. **Checkout** — a ``git clone`` the operator runs from (``~/dev/voice-agent``). This is what we
   develop against and, load-bearing, it is where the ``aria-stt``/``aria-tts`` units'
   ``WorkingDirectory`` already points. It is also the only channel that reaches a Debian satellite,
   since the AUR package is Arch-only.
2. **XDG / packaged** — ``~/.local/share/voice-agent``, which the AUR package's post-install text
   already documents ("bootstraps ~/.local/share/voice-agent + a uv venv … edit
   ~/.local/share/voice-agent/.env").

Both were already in the wild before this module existed; the bug was that nothing *knew* which one
it was in, so anything that wrote config had to guess a path. Detection order is deliberate: we ask
"where is the code that is running right now", never "which directory exists" — two layouts can
coexist on one box (a developer with the AUR package installed), and writing the `.env` of the tree
you are NOT running is a silent, extremely confusing failure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Layout", "detect_layout", "XDG_DIR_NAME"]

XDG_DIR_NAME = "voice-agent"


@dataclass(frozen=True)
class Layout:
    """A resolved install layout.

    ``kind`` ∈ {``checkout``, ``xdg``}. ``root`` is the directory holding ``main.py`` — the thing a
    systemd unit must use as its ``WorkingDirectory`` and the thing ``.env`` sits next to.
    ``venv_python`` is the interpreter a generated unit should exec DIRECTLY: going through
    ``run.sh``/``uv run`` puts a dependency resolve on the boot path and needs ``uv`` on a
    ``--user`` unit's PATH, which it is not.
    """

    kind: str
    root: Path
    reason: str

    @property
    def env_path(self) -> Path:
        return self.root / ".env"

    @property
    def venv_python(self) -> Path:
        return self.root / ".venv" / "bin" / "python"

    @property
    def is_git_checkout(self) -> bool:
        return (self.root / ".git").exists()


def _xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))


def detect_layout(*, module_file: str | None = None, override: str | None = None) -> Layout:
    """Resolve the layout of the install we are RUNNING IN.

    ``override`` (``--root``) always wins — it is what makes the whole provisioner testable against a
    sandbox directory, and it is the operator's escape hatch when detection is wrong.

    Otherwise the root is derived from this file's own location: ``voice_agent_install/paths.py`` sits
    one directory below the project root in both layouts. That is the only signal that cannot lie
    about which tree is executing. We then *label* it by comparing against the known XDG path, purely
    so messages can say something useful to the operator — the label never changes where we write.
    """
    if override:
        root = Path(override).expanduser().resolve()
        return Layout(kind="checkout" if (root / ".git").exists() else "xdg", root=root,
                      reason="explicit --root")

    here = Path(module_file or __file__).resolve()
    root = here.parent.parent

    xdg_root = (_xdg_data_home() / XDG_DIR_NAME).resolve()
    if root == xdg_root:
        return Layout(kind="xdg", root=root, reason="running from the XDG data dir (packaged install)")
    if (root / ".git").exists():
        return Layout(kind="checkout", root=root, reason="running from a git checkout")
    # A tree that is neither a checkout nor the XDG path: the reference Pi is exactly this — an rsync
    # target that is not a git repo at all. Treat it as a checkout-shaped install (it has the same
    # layout) but say so, because "not a git repo" means `git clone`-based updates do not apply and an
    # operator debugging it deserves to know that up front.
    return Layout(kind="checkout", root=root,
                  reason="running from a non-git tree (rsync/manual deploy) — updates are not `git pull`")
