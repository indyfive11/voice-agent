"""The satellite's systemd ``--user`` unit — its SHAPE and pacing intent.

This module owns *which* unit a voice satellite needs (Layer B, app-domain intent). It does NOT render
it: the safe render is delegated to ``installkit.templating.render_unit`` (Layer A, the vendored
boot-safety chokepoint whose API cannot even express a Tier-0 boot-network violation — no ``Type=``, no
``Requires=``, no ``ExecStartPre``, ``RestartSec`` floored, newlines rejected). ``satellite_unit()``
returns the kwargs; the caller renders with ``render_unit(**satellite_unit(...))``.

GROUND TRUTH, snapshotted from the reference satellite 2026-07-21 before writing a line of this
(INSTALL_PLAN §9 item 3 requires it, and the snapshot immediately earned its keep — the live unit
turned out to be *transient*, i.e. there is no persistent unit on any satellite today):

    FragmentPath=/run/user/1000/systemd/transient/aria-voice.service
    Transient=yes
    ExecStart=/usr/bin/bash -c 'exec ./run.sh gab > pi-run.log 2>&1'
    (plus --setenv=PATH=$HOME/.local/bin:… injected by the launcher; no Environment=,
     no EnvironmentFile=, no WorkingDirectory=, no Restart=, and Linger=no)

That unit works ONLY because a human on another box conjures it with the right PATH after logging
in. Everything below is the difference between that and a unit that survives a cold boot alone.
Each choice here fixes a specific way the naive translation fails:

- **ExecStart runs the venv interpreter directly**, not ``run.sh``. ``run.sh`` is ``exec uv run python
  main.py``: ``uv`` lives in ``~/.local/bin``, which is NOT on a ``--user`` unit's PATH, so the naive
  unit dies rc=127 with nothing in the log about PATH. Going direct also takes ``uv``'s per-launch
  dependency resolve off the boot path — that resolve can touch the network, and a network operation
  on the boot path is a Tier-0 boot-safety violation.
- **Because we bypass ``run.sh``, ``BRAIN`` must be in ``.env``.** ``run.sh gab`` is what exported
  ``BRAIN=gabagent``; drop the wrapper and that argv is gone. The role provisioner writes it. The unit
  needs no ``EnvironmentFile=``: ``WorkingDirectory=<root>`` puts cwd at the root, and the app
  self-loads ``<root>/.env`` via ``load_dotenv(override=True)`` at import (``main.py``/``config.py``).
- **Sound-server ordering.** The transient unit never raced PipeWire because a human started it long
  after login. A linger-started boot unit does race it, and losing that race is the ``-9997``
  output-failure class already in the tracker. Hence ``Wants=``/``After=`` the sound stack — ``Wants=``
  not ``Requires=`` so a box running bare ALSA still starts the agent.
- **``RestartSec`` is load-bearing, not boilerplate.** ``main.py`` deliberately ``os._exit(1)``s so a
  supervisor re-initialises a wedged mic. With systemd's 100ms default, three mic stalls inside the
  start-limit window latch the unit ``failed`` PERMANENTLY — the satellite is bricked until someone
  SSHes in and runs ``reset-failed``. installkit's renderer floors this by construction; we pass the
  measured pacing (``15 × (5−1) = 60s`` latch window) explicitly so the intent is declared here, not
  inherited from installkit's identical default.
- **No ``Requires=`` on anything remote, ever.** The brain is on another box; boot-safety says a
  remote dependency may never gate startup. installkit's ``render_unit`` has no ``Requires=`` parameter
  at all, so this is guaranteed structurally rather than remembered.

**AND THE EXCEPTION THAT IS NOT OURS TO REFUSE — KEEP THE INSTALL ROOT ON A LOCAL FILESYSTEM.**
Omitting ``Requires=`` does NOT mean a rendered unit carries none. systemd derives an implicit
``RequiresMountsFor=`` — a genuine ``Requires=``/``After=`` on mount units — from every path-taking
directive, and ``WorkingDirectory=`` is one (``systemd.exec(5)``; measured on a live ``--user``
manager: ``WorkingDirectory=/tmp`` → ``RequiresMountsFor=/tmp``, ``After=… tmp.mount``). We always pass
``working_directory=root``, so every emitted unit carries one. On a local disk that is inert. Point it
at NFS, autofs, sshfs or removable media and the satellite's unit gains a hard mount dependency on a
network filesystem — the Tier-0 cascade-failure shape the rest of this docstring exists to forbid,
arrived at by a path nobody would think to look for. It is not hypothetical for the XDG layout, whose
root sits under ``$HOME``, and a home directory on NFS is an ordinary deployment. The string leaves
here clean and systemd adds the dependency afterwards, so **no test of rendered text can catch it** —
which is exactly why it is written down instead of guarded. (installkit's ``render_unit`` documents the
identical caveat at its own chokepoint; neither layer can mitigate it.)
"""

from __future__ import annotations

__all__ = [
    "satellite_unit",
    "SATELLITE_SOUND_UNITS",
    "DEFAULT_RESTART_SEC",
    "DEFAULT_START_LIMIT_BURST",
    "DEFAULT_START_LIMIT_INTERVAL",
]

# The user-session sound stack. Wants= (not Requires=) so a box running bare ALSA — no PipeWire at all
# — still starts the agent instead of failing on a dependency it never needed.
SATELLITE_SOUND_UNITS = ("pipewire.service", "pipewire-pulse.service", "wireplumber.service")

# Restart pacing. 15s is chosen against main.py's watchdog: long enough that a genuine restart loop is
# visibly slow rather than a CPU-spinning hot loop, short enough that a real recovery (USB re-enumeration
# after the power-cycle rung) completes well inside one interval. The start limit is then sized so the
# unit tolerates several watchdog restarts in a row WITHOUT latching failed: 15 × (5−1) = 60s, which is
# exactly installkit's chokepoint-5 latch-window floor — we pass it explicitly rather than lean on the
# renderer's (identical) default, so the value is asserted as this role's declared intent.
DEFAULT_RESTART_SEC = 15
DEFAULT_START_LIMIT_BURST = 5
DEFAULT_START_LIMIT_INTERVAL = 600


def satellite_unit(*, root: str, venv_python: str,
                   description: str = "Aria voice satellite") -> dict[str, object]:
    """Return the kwargs for ``installkit.templating.render_unit`` describing the satellite's unit:
    run the agent, survive a cold boot, never gate on the brain.

    No ``environment_file`` is passed — the app self-loads ``<root>/.env`` from cwd, and cwd is
    ``WorkingDirectory=root``. Pacing is passed explicitly so this role declares its own intent (the
    values equal installkit's safe defaults, but relying on the default would make the value-asserts in
    the tests meaningless).
    """
    return dict(
        description=description,
        exec_start=f"{venv_python} main.py",
        working_directory=root,
        wants=SATELLITE_SOUND_UNITS,
        after=SATELLITE_SOUND_UNITS,
        restart="on-failure",
        restart_sec=DEFAULT_RESTART_SEC,
        start_limit_burst=DEFAULT_START_LIMIT_BURST,
        start_limit_interval=DEFAULT_START_LIMIT_INTERVAL,
    )
