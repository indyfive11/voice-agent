"""Comment-preserving ``.env`` read / merge / write.

WHY THIS EXISTS. Neither layer has an env *parser* — `installkit.templating` renders and writes but
never reads, so the only available "update" was a blind template overwrite. On a live satellite that
is destructive: the reference Pi's ``.env`` carries 42 keys, of which only ~16 are role-essential.
A re-provision that emitted the role keys alone would silently delete the operator's hand-tuning
(``WAKE_WORD_MEDIA_ONLY``, ``INPUT_USB_RESET_VIDPID``, the wake thresholds…), flatten every comment
explaining WHY those values are what they are, and drop a deliberately-empty ``KEY=`` — which does
not mean the same thing as an absent key.

So the contract here is **merge, never clobber**:

- an existing key's value is replaced IN PLACE, keeping its position and its trailing comment;
- a key we do not manage is left byte-identical;
- new keys are appended in a clearly-marked block;
- ``KEY=`` (present, empty) is preserved as distinct from absent — see :func:`parse`.

The parser is deliberately small and dumb: this file format is ours, it is written by us and
hand-edited by an operator, and it never needs to grow shell-expansion semantics. It reads what
``config.py``'s ``_env()`` will read and nothing more.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

__all__ = ["EnvLine", "parse", "merge", "render", "write", "backup"]

# KEY=value, tolerating leading whitespace and an optional `export ` prefix. The value half is taken
# verbatim to end-of-line: quoting/escaping is the reader's problem, not ours, and re-quoting a value we
# did not write is how a working config gets mangled.
_ASSIGN = re.compile(r"^(\s*)(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


@dataclass(frozen=True)
class EnvLine:
    """One physical line. ``key is None`` for comments and blanks — those are carried through
    untouched so an operator's notes survive a re-provision."""

    raw: str
    key: Optional[str] = None
    value: Optional[str] = None


def parse(text: str) -> list[EnvLine]:
    """Parse ``.env`` text into ordered lines, preserving everything we do not understand.

    A line is only treated as an assignment if it matches ``KEY=``; anything else (comment, blank,
    junk) is retained verbatim. ``KEY=`` with an empty right-hand side yields ``value == ""`` — which
    is NOT the same as the key being absent, and callers must keep that distinction (an empty value
    is how several knobs here mean "explicitly off").
    """
    lines: list[EnvLine] = []
    for raw in text.splitlines():
        m = _ASSIGN.match(raw)
        if m and not raw.lstrip().startswith("#"):
            lines.append(EnvLine(raw=raw, key=m.group(2), value=m.group(3)))
        else:
            lines.append(EnvLine(raw=raw))
    return lines


def _strip_inline_comment(value: str) -> tuple[str, str]:
    """Split a raw value into ``(value, trailing)`` where ``trailing`` is an unquoted ``  # …`` tail.

    The reference ``.env`` really does carry these (``STT_MODEL=small.en          # drop to base.en
    if Pi-4 STT latency is too high``), and losing the note when we rewrite the value would throw away
    the reason the value exists. Only splits on a ``#`` that is outside quotes — a ``#`` inside a
    quoted value is data.
    """
    in_single = in_double = False
    for i, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # Only a comment if there's whitespace before it (or it starts the value).
            if i == 0 or value[i - 1].isspace():
                return value[:i].rstrip(), value[i:]
    return value, ""


def merge(existing: str, updates: dict[str, Optional[str]]) -> tuple[str, dict[str, tuple[str, str]]]:
    """Apply ``updates`` to ``existing`` text. Returns ``(new_text, changes)``.

    ``changes`` maps key → ``(old, new)`` for keys whose value actually changed, so a caller can show
    a real key-level diff at the save-confirm rather than asking the operator to trust a rewrite.
    A value of ``None`` in ``updates`` means "leave unset" — an ABSENT key is left absent, and an
    existing key is left ALONE rather than blanked. Deleting an operator's key is never something a
    detection step should do implicitly; the portability SOP's "unset = the historical no-op" is about
    what we *write*, not a licence to remove what they wrote.
    """
    lines = parse(existing)
    seen: set[str] = set()
    changes: dict[str, tuple[str, str]] = {}
    out: list[str] = []

    for line in lines:
        if line.key is None or line.key not in updates:
            out.append(line.raw)
            continue
        new_value = updates[line.key]
        seen.add(line.key)
        if new_value is None:
            out.append(line.raw)  # explicitly not touching it
            continue
        old_value, trailing = _strip_inline_comment(line.value or "")
        if old_value == new_value:
            out.append(line.raw)
            continue
        changes[line.key] = (old_value, new_value)
        indent = _ASSIGN.match(line.raw).group(1)  # type: ignore[union-attr]
        sep = "  " if trailing else ""
        out.append(f"{indent}{line.key}={new_value}{sep}{trailing}")

    appended = {k: v for k, v in updates.items() if v is not None and k not in seen}
    if appended:
        if out and out[-1].strip():
            out.append("")
        out.append(f"# --- written by voice-agent-install ({time.strftime('%Y-%m-%d')}) ---")
        for key in sorted(appended):
            out.append(f"{key}={appended[key]}")
            changes[key] = ("", appended[key])

    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    return text, changes


def render(pairs: dict[str, Optional[str]], *, header: str = "") -> str:
    """Render a fresh ``.env`` from scratch (no existing file). ``None`` values are omitted entirely —
    an unwritten key is the historical no-op, whereas ``KEY=`` would assert an empty value."""
    out: list[str] = []
    if header:
        out.extend(f"# {ln}" if ln else "#" for ln in header.splitlines())
        out.append("")
    for key in sorted(k for k, v in pairs.items() if v is not None):
        out.append(f"{key}={pairs[key]}")
    return "\n".join(out) + "\n"


def backup(path: str) -> Optional[str]:
    """Copy ``path`` aside before we touch it. Returns the backup path, or ``None`` if there was no
    file to back up. Timestamped rather than a single ``.bak`` so a second provision cannot destroy
    the evidence from the first."""
    if not os.path.exists(path):
        return None
    dest = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    with open(path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    os.chmod(dest, 0o600)
    return dest


def write(path: str, text: str, *, mode: int = 0o600) -> None:
    """Write ``text`` to ``path`` atomically at ``mode`` (0600 by default — this file carries bearer
    tokens). Atomic because a half-written ``.env`` is an unbootable satellite, and the failure would
    land exactly when the operator is least able to debug it."""
    tmp = f"{path}.tmp-{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        os.chmod(path, mode)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
