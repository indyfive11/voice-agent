"""Artifact templating engine (A.4) — pure stdlib.

One dumb renderer + a family of thin, atomic writers for the artifacts a provisioned box needs: an
`.env`, a systemd `--user` unit, a `.desktop` launcher. The engine RETURNS rendered text; a separate
writer does the filesystem side (so the wizard can diff/preview before committing).

Three properties are load-bearing:

  1. BOOT-SAFETY IS STRUCTURAL, not remembered (render_unit). The renderer simply does NOT expose the
     systemd shapes that can block boot on a remote host — no `Type=` parameter (hardcoded `simple`), no
     `Requires=` (remote deps can only be Wants=/After=), no `ExecStartPre` (no remote probe before
     start). The Tier-0 boot-network SOP moves from "obey the rule" to "the API can't express the
     violation." The one thing the engine cannot police is the CONTENT of ExecStart (a caller could
     smuggle a remote-wait loop inside it) — for a `--user` `Type=simple` unit that is a self-hang of
     one service, not a boot-block, but the contract forbids it: the sanctioned re-resolve is the in-app
     async probe, never an ExecStart loop.

  2. ATOMIC + MODE-SAFE WRITES (write_atomic). Temp file created ALREADY at the target mode (never a
     world-readable window for a secret/.env), temp lives in the TARGET'S directory (same-fs atomic
     rename), and an identical existing file is a no-op (idempotent re-provisioning).

  3. SECRETS ARE NEVER TEMPLATED. Templates carry `${PLACEHOLDERS}`, never resolved hosts/tokens; the
     A.5 engine mints/reconciles secrets and the caller writes them via the 0600 `.env` writer here.

App-agnostic by construction — no app names, no install-specific constants. Every variable arrives from
the caller's mapping (portability SOP: no bare constants in shared code).
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from string import Template

__all__ = [
    "render",
    "write_atomic",
    "render_env",
    "write_env",
    "render_unit",
    "write_unit",
    "render_desktop",
    "write_desktop",
]


def _no_newline(value: str, field: str) -> str:
    """Reject a raw newline in any value bound for a newline-delimited Key=value grammar (.env, systemd
    unit, .desktop). A '\\n'/'\\r' would let a caller smuggle a second directive onto its own line —
    e.g. a `Requires=`/`ExecStartPre=` injected into a unit, defeating the render_unit chokepoint that
    is supposed to make that shape UNEXPRESSIBLE. The guard is what makes "structurally impossible"
    true rather than true-by-convention. Returns the value unchanged when clean."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field} contains a newline — would inject a second directive/line")
    return value


# --------------------------------------------------------------------------------------------------- #
# Core: dumb substitution + atomic write
# --------------------------------------------------------------------------------------------------- #

def render(template: str, mapping: dict[str, str], *, strict: bool = True) -> str:
    """Substitute ${NAME} / $NAME from `mapping` via string.Template.

    strict=True (default) → substitute(): a placeholder with no key raises KeyError (never a silent
    blank — a blank host/token ships a broken unit). ⚠ A LITERAL '$' in the body — systemd
    '$XDG_RUNTIME_DIR', shell '${HOME}', '$()' — MUST be written '$$' (string.Template's own escape) or
    it is read as a placeholder and raises (hole H1). Say so in every template-author contract.

    strict=False → safe_substitute(): unknown placeholders are left verbatim (partial / preview render).

    NOTE: the dedicated render_env/render_unit/render_desktop below do NOT go through Template — they emit
    caller strings verbatim — so a literal '$' in an ExecStart passed to render_unit needs no escaping.
    The '$$' rule applies only when you drive this generic render() with a hand-written template body.
    """
    t = Template(template)
    return t.substitute(mapping) if strict else t.safe_substitute(mapping)


def write_atomic(path: str, content: str | bytes, *, mode: int = 0o644) -> bool:
    """Write `content` to `path` via temp-file + atomic rename. Returns True if the file changed, False
    if byte-identical (idempotent no-op — a re-run leaves an unchanged file untouched: no mtime churn,
    no spurious diff).

    Hardening (hole H2):
      - the temp file is created at 0600 by mkstemp then set to `mode` BEFORE any content is written, so
        a secret (mode=0o600) is never briefly world-readable (the post-rename-chmod 0644-leak class).
      - the temp is created IN THE TARGET'S DIRECTORY so os.replace() is a same-filesystem atomic rename
        (cross-fs it silently degrades to a non-atomic copy, or raises EXDEV).
    """
    p = Path(path)
    new = content.encode("utf-8") if isinstance(content, str) else content
    try:
        if p.read_bytes() == new:
            return False  # identical — don't touch the file at all
    except OSError:
        pass  # absent / unreadable → fall through and write
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)  # mkstemp gives 0600; set exact bits (umask-independent) before writing
        with os.fdopen(fd, "wb") as f:
            f.write(new)
            f.flush()
            os.fsync(f.fileno())  # content on disk before the rename → never a half-written artifact
        os.replace(tmp, str(p))  # atomic within one filesystem
        return True
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------------------------------- #
# .env writer
# --------------------------------------------------------------------------------------------------- #

# Values matching this are safe to emit bare (no dotenv-special char): tokens [A-Za-z0-9_-], IPs, ports,
# provider enums, plain URLs. Anything else (space, '#', quote, '$', …) gets quoted below.
_ENV_BARE_OK = re.compile(r"^[A-Za-z0-9_.:/@+=-]*$")


def _quote_env_value(v: str) -> str:
    """Quote a value for python-dotenv (voice-agent's config.py reads .env via load_dotenv) — hole H3.

    Newlines are REJECTED (a raw '\\n' breaks the KEY=value line and orphans the rest of the file).
    Otherwise: bare when safe; else SINGLE-quoted — python-dotenv treats a single-quoted value literally
    (no ${VAR} interpolation, no escape decoding), which cleanly covers spaces ("EMEET …" device names),
    '#', '=', and '$'. The rare value containing a single quote falls back to double-quotes with '\\' and
    '"' escaped (double-quoted dotenv values DO interpolate '${VAR}', but the satellite key set —
    tokens/IPs/ports/URLs/device-names — never contains '$')."""
    _no_newline(v, "env value")  # a raw \n orphans the rest of the file (same guard as unit/desktop)
    if _ENV_BARE_OK.match(v):
        return v
    if "'" not in v:
        return "'" + v + "'"
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_env(pairs: dict[str, str | None]) -> str:
    """KEY=value lines, keys SORTED for a deterministic (idempotent-diff) file. A None or empty-string
    value is OMITTED entirely — the portability SOP's 'unset = historical no-op': a bare 'KEY=' reads
    back as empty-string (a set-but-empty override), NOT as absent, which is a different, wrong meaning.
    Values are quoted per _quote_env_value (H3)."""
    lines = [f"{k}={_quote_env_value(str(pairs[k]))}"
             for k in sorted(pairs)
             if pairs[k] is not None and pairs[k] != ""]
    return "\n".join(lines) + "\n" if lines else ""


def write_env(path: str, pairs: dict[str, str | None]) -> bool:
    """Write an .env at 0600 (it carries tokens) via the atomic writer."""
    return write_atomic(path, render_env(pairs), mode=0o600)


# --------------------------------------------------------------------------------------------------- #
# systemd --user unit writer — the boot-safety chokepoint
# --------------------------------------------------------------------------------------------------- #

_RESTART_VALUES = {"no", "on-failure", "on-abnormal", "on-success", "always"}
# Bounded start-timeout ceiling (chokepoint-2): even though a Type=simple unit is "started" at fork and
# never blocks boot, a bounded TimeoutStartSec caps the blast radius of any future unit shape.
_MAX_TIMEOUT_START_SEC = 30

# Restart pacing floor/ceiling (chokepoint-4). The floor is the load-bearing half: systemd's own default
# is `DefaultRestartUSec=100ms`, so a renderer that OMITS RestartSec inherits sub-second pacing from
# per-box state in ~/.config/systemd/user.conf — state this engine cannot see and did not write. At 100ms
# a service that fails instantly at boot (losing a race with the sound stack, say) burns its entire
# StartLimitBurst in a fraction of a second and latches `failed` PERMANENTLY, needing a human to run
# `reset-failed`. That is the same remote-hands dependency the boot-safety rule exists to remove, so the
# sub-second restart storm is made UNEXPRESSIBLE here rather than merely discouraged. The ceiling stops
# the opposite error: pacing so slow the unit never realistically recovers.
_MIN_RESTART_SEC = 1
_MAX_RESTART_SEC = 300

# The latch window a caller gets for FREE (chokepoint-5). `start_limit_burst` attempts paced
# `restart_sec` apart span restart_sec × (burst − 1) seconds; once the burst is exceeded inside
# `start_limit_interval` the unit latches `failed` until a human runs `reset-failed`. A chokepoint is only
# worth having if the caller who passes NOTHING gets the safe artifact — a guard that fires solely on
# explicitly-bad values still lets the default through. 60s is the measured shape of the failure this
# guards: a service racing the user sound stack at cold boot on slow hardware, where recovery means USB
# re-enumeration, not a retry. Below this floor the unit can latch before its dependencies finish coming up.
# The value is the ONLY measured one available — a hand-tuned satellite unit on that hardware chose
# 15s × (5−1) = 60s — so the defaults below sit exactly ON the floor rather than comfortably above it, and
# tuning the pacing DOWN is refused by construction. An unmeasured "half the default" floor was considered
# and rejected: a number nobody measured, defended by prose, is the drift this module keeps finding.
_MIN_LATCH_WINDOW_SEC = 60


def render_unit(
    *,
    description: str,
    exec_start: str,
    working_directory: str | None = None,
    environment_file: str | None = None,
    wants: tuple[str, ...] = (),
    after: tuple[str, ...] = (),
    restart: str = "on-failure",
    restart_sec: int = 15,
    start_limit_burst: int = 5,
    start_limit_interval: int = 600,
    timeout_start_sec: int = 15,
    condition_path_exists: tuple[str, ...] = (),
) -> str:
    """Render a systemd `--user` unit whose shape CANNOT express the Tier-0 boot-network violation.

    Structural guarantees (the whole point — a caller literally cannot render the unsafe shapes):
      - Type is HARDCODED `simple` and is NOT a parameter. A Type=oneshot/notify unit blocks dependents
        until it signals ready → that is the boot-block hole; it stays unreachable (chokepoint-1).
      - A remote-touching dependency can only be Wants=/After=, NEVER Requires= (no Requires= param), so
        a down *service* can never cascade-fail this unit. **This guarantee stops at service units and you
        must respect the exception:** systemd derives an implicit `RequiresMountsFor=` — a genuine
        Requires=/After= on mount units — from every path-taking directive, `WorkingDirectory=` included
        (systemd.exec(5); measured: `WorkingDirectory=/tmp` → `RequiresMountsFor=/tmp`, `After=…tmp.mount`).
        On a local filesystem that is inert. Pointed at NFS, autofs, sshfs or removable media it is a hard
        mount dependency that this renderer cannot see, cannot express and cannot forbid — the string leaves
        here clean and systemd adds it afterwards, so no test of rendered text can catch it. **Keep the
        install root on a local filesystem.**
      - There is NO ExecStartPre parameter → no remote probe can run before start. The sanctioned boot
        re-resolve is the in-app async probe, never a unit-level wait (chokepoint-3). What the engine
        CANNOT police is the ExecStart *content*; for a --user Type=simple unit an embedded remote-wait
        loop is a self-hang of one service, not a boot-block, but the contract forbids it.
      - Restart is bounded by StartLimitIntervalSec/StartLimitBurst (no restart storm) and TimeoutStartSec
        is bounded ≤ 30s (chokepoint-2). The rate limiter only actually engages while restarts fall inside
        the window, so `restart_sec < start_limit_interval` is ENFORCED — otherwise the second attempt
        lands outside the window, the burst counter never accumulates, and "bounded" would be false.
      - RestartSec is ALWAYS EMITTED and is floored at 1s (chokepoint-4). Omitting it would silently
        inherit `DefaultRestartUSec` (100ms) from box-local user.conf, which makes the rendered unit's
        boot-safety behaviour vary per machine — see the constant block above. `working_directory` is
        offered for the same reason: without it, a caller whose ExecStart needs a cwd is pushed into
        `/bin/sh -c 'cd … && exec …'`, and that wrapper reintroduces the PATH/shell failure class the
        direct-exec contract exists to kill.

    The defaults are the ones a unit racing a slow start-dependency needs, not the smallest ones that pass
    validation: an unparameterised render must be the safe artifact, because a caller who already knows the
    right numbers does not need a chokepoint. They sit EXACTLY on the chokepoint-5 floor (15 × 4 = 60s), so
    the guard has no margin and tuning either value *down* on its own is refused. That is deliberate — the
    floor is the safety line, not a suggestion — and faster restarts stay expressible by widening the
    burst instead: `restart_sec=5, start_limit_burst=13` is also 60s and renders fine. A caller on hardware
    with no slow start-dependency is not stuck; they just have to say so explicitly.
    """
    if not description or not exec_start:
        raise ValueError("render_unit requires both description and exec_start")
    if restart not in _RESTART_VALUES:
        raise ValueError(f"unsupported Restart= value: {restart!r}")
    if not 0 < timeout_start_sec <= _MAX_TIMEOUT_START_SEC:
        raise ValueError(f"timeout_start_sec must be 1..{_MAX_TIMEOUT_START_SEC} (boot-safety chokepoint-2)")
    if not _MIN_RESTART_SEC <= restart_sec <= _MAX_RESTART_SEC:
        raise ValueError(f"restart_sec must be {_MIN_RESTART_SEC}..{_MAX_RESTART_SEC} "
                         f"(boot-safety chokepoint-4: sub-second pacing latches the unit `failed`)")
    # Both restart invariants are VACUOUS under Restart=no: systemd never restarts, so StartLimit* is
    # inert and there is no latch to guard against. Enforcing them anyway would force a caller to pass a
    # meaningless start_limit_burst — a lie about restart behaviour — to render a shape that was already
    # safe. `no` stays in _RESTART_VALUES, so the contract and the guard agree.
    if restart != "no":
        if restart_sec >= start_limit_interval:
            raise ValueError(f"restart_sec ({restart_sec}) must be < start_limit_interval "
                             f"({start_limit_interval}): every retry would fall outside the window, the "
                             f"burst counter would never accumulate, and the rate limit could never engage")
        latch_window = restart_sec * max(start_limit_burst - 1, 0)
        if latch_window < _MIN_LATCH_WINDOW_SEC:
            raise ValueError(f"restart_sec × (start_limit_burst - 1) = {latch_window}s is below the "
                             f"{_MIN_LATCH_WINDOW_SEC}s floor (boot-safety chokepoint-5): the unit would "
                             f"latch `failed` before a slow start-dependency finishes coming up. Raise "
                             f"either value.")
    # Absolute path, or the bare `~`. A relative WorkingDirectory renders as valid-looking text and is
    # REJECTED by systemd at load (`LoadState=bad-setting`, the directive dropped) — the mirror of the
    # newline guard: newlines make the text wrong, a relative path makes text that is right and dead on
    # arrival. `~` is a SPECIAL VALUE meaning the user's home, NOT a prefix: `~` loads and `~/sub` is a
    # load error (measured), so the natural home-relative spelling must be refused too. systemd allows a
    # relative value only against RootDirectory=, which this engine deliberately does not offer.
    if working_directory is not None and not (working_directory.startswith("/") or working_directory == "~"):
        raise ValueError(f"working_directory must be an absolute path or the bare '~', got "
                         f"{working_directory!r} — systemd treats '~' as a special value, not a prefix, so "
                         f"'~/sub' is rejected at load time exactly like a relative path")
    # Reject newlines in EVERY field that becomes a line — else a '\n' smuggles a raw Requires=/
    # ExecStartPre= back into the section, reopening the shape the chokepoint forbids by omission.
    _no_newline(description, "description")
    _no_newline(exec_start, "exec_start")
    if working_directory is not None:
        _no_newline(working_directory, "working_directory")
    if environment_file is not None:
        _no_newline(environment_file, "environment_file")
    for d in wants:
        _no_newline(d, "wants[]")
    for d in after:
        _no_newline(d, "after[]")
    for c in condition_path_exists:
        _no_newline(c, "condition_path_exists[]")

    unit = ["[Unit]", f"Description={description}"]
    unit += [f"Wants={d}" for d in wants]            # Wants=, never Requires= — no cascade-fail
    unit += [f"After={d}" for d in after]
    unit += [f"ConditionPathExists={c}" for c in condition_path_exists]  # absent optional dep ⇒ clean skip
    # StartLimit* live in [Unit] on modern systemd (moved from [Service]); bound the restart rate here.
    unit += [f"StartLimitIntervalSec={start_limit_interval}", f"StartLimitBurst={start_limit_burst}"]

    service = ["[Service]", "Type=simple"]
    # Ordered so the file reads in the order the unit is set up. NOT because ExecStart resolves against it:
    # systemd requires an absolute path or a bare filename searched in a fixed list, so a relative ExecStart
    # is a load error no matter what WorkingDirectory= says.
    if working_directory is not None:
        service.append(f"WorkingDirectory={working_directory}")
    service.append(f"ExecStart={exec_start}")
    if environment_file is not None:
        # '-' prefix: a missing env file is non-fatal — the unit still starts and the app fails soft on
        # absent config, rather than the unit refusing to launch.
        service.append(f"EnvironmentFile=-{environment_file}")
    # RestartSec is never omitted — see chokepoint-4. An absent value is not a neutral default here, it
    # is a delegation to per-box state.
    service += [f"Restart={restart}", f"RestartSec={restart_sec}",
                f"TimeoutStartSec={timeout_start_sec}"]

    install = ["[Install]", "WantedBy=default.target"]
    return "\n".join(unit + [""] + service + [""] + install) + "\n"


def write_unit(unit_dir: str, name: str, **kw: object) -> bool:
    """Write <unit_dir>/<name>.service (caller supplies unit_dir, e.g. ~/.config/systemd/user)."""
    fname = name if name.endswith(".service") else f"{name}.service"
    return write_atomic(str(Path(unit_dir) / fname), render_unit(**kw), mode=0o644)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------- #
# .desktop launcher writer (specced whole; called only in the full-host role, per the 3a/3b split)
# --------------------------------------------------------------------------------------------------- #

def render_desktop(
    *,
    name: str,
    exec_cmd: str,
    icon: str | None = None,
    terminal: bool = False,
    categories: tuple[str, ...] = (),
) -> str:
    """Render a freedesktop `.desktop` Application entry."""
    if not name or not exec_cmd:
        raise ValueError("render_desktop requires both name and exec_cmd")
    # Same newline guard: a '\n' in exec_cmd injects a second Exec= line, etc.
    _no_newline(name, "name")
    _no_newline(exec_cmd, "exec_cmd")
    if icon:
        _no_newline(icon, "icon")
    for cat in categories:
        _no_newline(cat, "categories[]")
    lines = ["[Desktop Entry]", "Type=Application", f"Name={name}", f"Exec={exec_cmd}"]
    if icon:
        lines.append(f"Icon={icon}")
    lines.append(f"Terminal={'true' if terminal else 'false'}")
    if categories:
        lines.append("Categories=" + ";".join(categories) + ";")  # trailing ';' per the spec
    return "\n".join(lines) + "\n"


def write_desktop(apps_dir: str, filename: str, **kw: object) -> bool:
    """Write <apps_dir>/<filename>.desktop (caller supplies apps_dir, e.g. ~/.local/share/applications)."""
    fname = filename if filename.endswith(".desktop") else f"{filename}.desktop"
    return write_atomic(str(Path(apps_dir) / fname), render_desktop(**kw), mode=0o644)  # type: ignore[arg-type]
