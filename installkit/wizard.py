"""Wizard PRIMITIVES — pure stdlib, no third-party deps.

Deliberately dependency-free: installkit is vendored to a Debian satellite where rich/prompt-toolkit may
not exist yet at bootstrap time. The gabagent-side (Layer C) is free to render with rich on top of these;
the primitives themselves only need a terminal. Only the primitives are shared — the CONTROL FLOW is not
(gabagent's installer is an option-loop, voice-agent's a linear role provisioner), so each installer keeps
its own flow and just calls these.
"""
from __future__ import annotations

import os
import sys

__all__ = [
    "WizardCancelled",
    "supports_color",
    "panel",
    "heading",
    "note",
    "prompt",
    "choose",
    "confirm",
    "save_confirm",
]


class WizardCancelled(Exception):
    """Raised when the user aborts (EOF / Ctrl-C) at a prompt with no usable default."""


def supports_color() -> bool:
    """True when it's safe to emit ANSI — a real tty and not NO_COLOR/dumb-term."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") in ("", "dumb"):
        return False
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if supports_color() else text


def panel(title: str, subtitle: str = "") -> None:
    """A minimal boxed banner (stdlib-only stand-in for a rich Panel)."""
    lines = [title] + ([subtitle] if subtitle else [])
    width = max(len(s) for s in lines) + 4
    print(_c("┌" + "─" * width + "┐", "36"))
    for i, s in enumerate(lines):
        style = "1" if i == 0 else "2"
        print(_c("│", "36") + "  " + _c(s.ljust(width - 4), style) + "  " + _c("│", "36"))
    print(_c("└" + "─" * width + "┘", "36"))


def heading(text: str) -> None:
    print()
    print(_c(text, "1"))


def note(text: str) -> None:
    print(_c(text, "2"))


def prompt(text: str, default: str = "") -> str:
    """Read one line. Empty input returns `default`. EOF/Ctrl-C returns `default` if one exists,
    else raises WizardCancelled — the caller decides whether an abort is fatal."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"{text}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        if default:
            return default
        raise WizardCancelled from None
    return val or default


def choose(text: str, options: list[str], default_index: int = 0) -> int:
    """Numbered single-select menu. Returns the chosen 0-based index. Blank input takes the default;
    an unusable abort raises WizardCancelled. Re-prompts on out-of-range / non-numeric input."""
    if not options:
        raise ValueError("choose() needs at least one option")
    if not 0 <= default_index < len(options):
        default_index = 0
    heading(text)
    for i, opt in enumerate(options, 1):
        marker = _c(str(i), "36")
        print(f"  {marker}  {opt}")
    print()
    while True:
        raw = prompt(f"Select [1-{len(options)}]", default=str(default_index + 1))
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        note(f"Enter a number between 1 and {len(options)}.")


def confirm(text: str, default: bool = True) -> bool:
    """Yes/no prompt. Blank input takes `default`; an abort takes `default` (never fatal)."""
    hint = "Y/n" if default else "y/N"
    try:
        raw = prompt(f"{text} ({hint})", default="")
    except WizardCancelled:
        return default
    if not raw:
        return default
    return raw[0].lower() == "y"


def save_confirm(path: str, default: bool = True) -> bool:
    """The shared 'about to write a file — proceed?' gate. Callers own the actual write."""
    return confirm(f"Write configuration to {path}?", default=default)
