"""systemd ``--user`` unit generation for the voice roles.

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
  ``BRAIN=gabagent``; drop the wrapper and that argv is gone. The role provisioner writes it.
- **Sound-server ordering.** The transient unit never raced PipeWire because a human started it long
  after login. A linger-started boot unit does race it, and losing that race is the ``-9997``
  output-failure class already in the tracker.
- **``RestartSec`` is load-bearing, not boilerplate.** ``main.py`` deliberately ``os._exit(1)``s so a
  supervisor re-initialises a wedged mic. With systemd's 100ms default, three mic stalls inside the
  start-limit window latch the unit ``failed`` PERMANENTLY — the satellite is bricked until someone
  SSHes in and runs ``reset-failed``, which is precisely the remote-hands dependency this whole
  exercise exists to remove.
- **No ``Requires=`` on anything remote, ever.** The brain is on another box; boot-safety says a
  remote dependency may never gate startup. An unreachable brain is a degraded start, not a failed
  boot.

**AND THE EXCEPTION THAT IS NOT OURS TO REFUSE — KEEP THE INSTALL ROOT ON A LOCAL FILESYSTEM.**
Omitting ``Requires=`` from this generator does NOT mean a rendered unit carries none. systemd derives
an implicit ``RequiresMountsFor=`` — a genuine ``Requires=``/``After=`` on mount units — from every
path-taking directive, and ``WorkingDirectory=`` is one (``systemd.exec(5)``; measured on a live
``--user`` manager: ``WorkingDirectory=/tmp`` → ``RequiresMountsFor=/tmp``, ``After=… tmp.mount``).
``UnitSpec.working_directory`` is REQUIRED, so **every unit this module emits carries one.**

On a local disk that is inert. Point it at NFS, autofs, sshfs or removable media and the satellite's
unit gains a hard mount dependency on a network filesystem — the Tier-0 cascade-failure shape the rest
of this docstring exists to forbid, arrived at by a path nobody would think to look for. It is not
hypothetical for the XDG layout, whose root sits under ``$HOME``, and a home directory on NFS is an
ordinary deployment. The string leaves here clean and systemd adds the dependency afterwards, so **no
test of rendered text can catch it** — which is exactly why it is written down instead of guarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["UnitSpec", "render_unit", "SATELLITE_SOUND_UNITS"]

# The user-session sound stack. Wants= (not Requires=) so a box running bare ALSA — no PipeWire at all
# — still starts the agent instead of failing on a dependency it never needed.
SATELLITE_SOUND_UNITS = ("pipewire.service", "pipewire-pulse.service", "wireplumber.service")

# Restart pacing. 15s is chosen against main.py's watchdog: long enough that a genuine restart loop is
# visibly slow rather than a CPU-spinning hot loop, short enough that a real recovery (USB re-enumeration
# after the power-cycle rung) completes well inside one interval. The start limit is then sized so the
# unit tolerates several watchdog restarts in a row WITHOUT latching failed.
DEFAULT_RESTART_SEC = 15
DEFAULT_START_LIMIT_BURST = 5
DEFAULT_START_LIMIT_INTERVAL = 600


@dataclass(frozen=True)
class UnitSpec:
    """Everything needed to render one ``--user`` service unit."""

    description: str
    exec_start: str
    working_directory: str
    wants: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    restart: str = "on-failure"
    restart_sec: int = DEFAULT_RESTART_SEC
    start_limit_burst: int = DEFAULT_START_LIMIT_BURST
    start_limit_interval: int = DEFAULT_START_LIMIT_INTERVAL
    wanted_by: str = "default.target"


def _check_no_newline(value: str, field_name: str) -> str:
    """A newline in a rendered field would inject arbitrary directives into the unit. Reject rather
    than escape — every value here comes from detection or an operator prompt, and there is no
    legitimate multi-line unit field in this generator."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must not contain a newline: {value!r}")
    return value


def render_unit(spec: UnitSpec) -> str:
    """Render ``spec`` to systemd unit text.

    Note what is deliberately ABSENT: no ``Requires=``, no ``ExecStartPre`` that touches the network,
    and no ``TimeoutStartSec`` long enough to stall a boot. The unit's whole remote story is "start,
    and let the app's in-process retry handle a brain that is not up yet."
    """
    for name, value in (("description", spec.description), ("exec_start", spec.exec_start),
                        ("working_directory", spec.working_directory)):
        _check_no_newline(value, name)

    lines = ["[Unit]", f"Description={_check_no_newline(spec.description, 'description')}"]
    if spec.wants:
        lines.append("Wants=" + " ".join(spec.wants))
    if spec.after:
        lines.append("After=" + " ".join(spec.after))
    # StartLimit* belong in [Unit] on modern systemd; Restart*/RestartSec belong in [Service].
    lines.append(f"StartLimitIntervalSec={int(spec.start_limit_interval)}")
    lines.append(f"StartLimitBurst={int(spec.start_limit_burst)}")

    lines += ["", "[Service]", "Type=simple",
              f"WorkingDirectory={spec.working_directory}",
              f"ExecStart={spec.exec_start}"]
    for key in sorted(spec.environment):
        lines.append(f'Environment="{key}={_check_no_newline(spec.environment[key], key)}"')
    lines += [f"Restart={spec.restart}", f"RestartSec={int(spec.restart_sec)}"]

    lines += ["", "[Install]", f"WantedBy={spec.wanted_by}", ""]
    return "\n".join(lines)


def satellite_unit(*, root: str, venv_python: str, description: str = "Aria voice satellite") -> UnitSpec:
    """The satellite's unit: run the agent, survive a cold boot, never gate on the brain."""
    return UnitSpec(
        description=description,
        exec_start=f"{venv_python} main.py",
        working_directory=root,
        wants=SATELLITE_SOUND_UNITS,
        after=SATELLITE_SOUND_UNITS,
        # XDG_RUNTIME_DIR is normally set for a --user unit by the user manager itself; we do NOT
        # hardcode /run/user/<uid> here, because pinning a uid is exactly the installation-specific
        # constant the portability SOP forbids.
        environment={},
    )
