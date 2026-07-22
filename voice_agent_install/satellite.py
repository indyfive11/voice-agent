"""The satellite role provisioner, and the entry point ``bootstrap.sh`` hands off to.

INSTALL-TIME ONLY. Nothing in the running application may import this package: the satellite sync
manifest is ``git ls-files '*.py'`` + ``run.sh``, so a runtime importer drags every module here onto
that manifest and any module NOT tracked becomes a guaranteed satellite crash. The one sanctioned
runtime import is ``config.py``'s boot re-resolve of ``discovery`` alone.

Run as ``python -m voice_agent_install.satellite`` — which is what ``bootstrap.sh`` execs after it has
provisioned ``uv`` and the venv. Until this module had a ``main()``, that exec imported the module,
ran nothing, and exited 0: a clean-box install that reported SUCCESS having provisioned nothing.
Every failure path below therefore exits NON-ZERO, and the summary names the step that failed —
an installer whose only signal is ``$?`` cannot be trusted if ``$?`` can be 0 for a no-op.

This is also the module GA's tree-audit was pointing at: ``discover_brain``/``default_providers`` and
``detect_output_sample_rate`` were built, tested, and imported by nothing outside tests. A seam with
no caller is unbuilt work that looks shipped. This calls them.

SHAPE (3b). Credentials are supplied by prompt — three secrets, which is the honest floor. The
interactive claim handshake that removes the typing is 3c and deliberately not here: everything
below is independent of *how* credentials arrive, and it is where a clean-box install actually dies,
so it ships first rather than waiting on the pairing design.

WHAT DISCOVERY DOES AND DOES NOT BUY US. mDNS returns host and port only, and it is off by default on
a stock brain (``voice_advertise: bool = False``), so it can never be load-bearing. It is used here
purely to pre-fill a default the operator confirms. The STT/TTS URLs are COMPOSED from the brain host
— note *composed*, not "derived for free": ``_repoint_remote_services`` only rewrites an existing
absolute URL and skips on empty, so nothing in the codebase would construct these at first provision,
and an unwritten ``STT_REMOTE_URL`` is a hard startup error (``config.py:130``).
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import audio_detect, discovery, envfile, profile, units, verify
from .paths import Layout, detect_layout

__all__ = ["SatelliteAnswers", "compose_env", "service_url", "DEFAULT_STT_PORT", "DEFAULT_TTS_PORT",
           "StepResult", "Console", "gather", "provision", "main",
           "EXIT_OK", "EXIT_PROVISION_FAILED", "EXIT_VERIFY_FAILED", "EXIT_USAGE"]

# Exit codes are a CONTRACT with bootstrap.sh and with any deploy-verify script, so they distinguish
# the cases an operator would act on differently. A single non-zero would collapse "nothing was
# written" and "everything was written but the brain is unreachable" — opposite remedies.
EXIT_OK = 0
EXIT_PROVISION_FAILED = 1   # nothing usable was written; re-run after fixing the reported step
EXIT_USAGE = 2              # bad arguments / no terminal to prompt on
EXIT_VERIFY_FAILED = 3      # config + unit ARE written; a service leg did not answer

DEFAULT_UNIT_NAME = "aria-voice.service"

# These are voice-agent's OWN default service ports, declared server-side with an env override
# (`stt_service/server.py:235`, `tts_service/server.py:303`) — structurally identical to
# `discovery.DEFAULT_BRAIN_PORT = 8765`. An application's own default port with an override is a safe
# universal default under the portability SOP, NOT an installation-specific hardcode; the values that
# would be hardcodes (which host, which token) are supplied per-install.
DEFAULT_STT_PORT = 8770
DEFAULT_TTS_PORT = 8771


@dataclass
class SatelliteAnswers:
    """Everything the provisioner needs. Assembled from detection + prompts; kept as plain data so the
    whole compose step is testable without a terminal, a network, or a sound card."""

    brain_host: str
    brain_port: int
    room_id: str
    brain_token: Optional[str]
    stt_token: Optional[str]
    tts_token: Optional[str]
    input_device: str
    output_device: str
    wake_word: str
    wake_engine: str = "oww"
    output_sample_rate: Optional[int] = None
    stt_host: Optional[str] = None
    tts_host: Optional[str] = None
    stt_port: int = DEFAULT_STT_PORT
    tts_port: int = DEFAULT_TTS_PORT
    extra: dict[str, str] = field(default_factory=dict)


def service_url(host: str, port: int) -> str:
    """Compose an absolute service URL. Always an IP/host we were given — never a ``.local`` name:
    a ``.local`` in a service URL makes every connect eat a synchronous ``getaddrinfo``/nss-mdns
    lookup (~5s with Avahi down), which lands inside the boot budget."""
    return f"http://{host}:{port}"


def compose_env(answers: SatelliteAnswers) -> dict[str, Optional[str]]:
    """Build the full key→value map for a satellite ``.env``.

    Returns values as strings (or ``None`` for "leave unset"). Every key in
    ``profile.REQUIRED_KEYS`` is guaranteed present — :func:`validate` enforces it, and the caller
    refuses to write a config that fails that check rather than emitting a file that starts and then
    misbehaves.
    """
    stt_host = answers.stt_host or answers.brain_host
    tts_host = answers.tts_host or answers.brain_host

    env: dict[str, Optional[str]] = dict(profile.profile_for("satellite"))
    env.update({
        "GAB_HOST": answers.brain_host,
        "GAB_PORT": str(answers.brain_port),
        "ROOM_ID": answers.room_id,
        "STT_REMOTE_URL": service_url(stt_host, answers.stt_port),
        "TTS_REMOTE_URL": service_url(tts_host, answers.tts_port),
        "AUDIO_INPUT_DEVICE_NAME": answers.input_device,
        "AUDIO_OUTPUT_DEVICE_NAME": answers.output_device,
        "WAKE_WORD": answers.wake_word,
        "WAKE_WORD_ENGINE": answers.wake_engine,
    })

    # Tokens: written only when we have one. An absent token is NOT the same as an empty one — the
    # services read empty as "run open" (`stt_service/server.py:183-184`), so writing `KEY=` would
    # assert an open service we have not verified is open.
    for key, value in (("GAB_AUTH_TOKEN", answers.brain_token),
                       ("STT_REMOTE_TOKEN", answers.stt_token),
                       ("TTS_REMOTE_TOKEN", answers.tts_token)):
        env[key] = value if value else None

    # The chipmunk guard's write half. `None` means leave UNSET, which is the historical no-op for a
    # resampling sound server; a fixed-rate hardware endpoint gets its ALSA-authoritative rate pinned.
    # Never a guess in either direction — an unclassifiable device yields None upstream.
    env["AUDIO_OUTPUT_SAMPLE_RATE"] = str(answers.output_sample_rate) if answers.output_sample_rate else None

    env.update(answers.extra)
    return env


def validate(env: dict[str, Optional[str]]) -> list[str]:
    """Return a list of problems, empty if the config is provisionable.

    This is the gate that stops the two silent-misconfiguration classes we know about: a missing key
    that deletes a whole subsystem (``WAKE_WORD`` → the gate returns None and the mic is open), and a
    missing key that flips a default into the wrong topology (``GAB_HOST`` → loopback → ``GAB_LAUNCH``
    → try to spawn a local brain).
    """
    problems = []
    for key in profile.REQUIRED_KEYS:
        if not (env.get(key) or "").strip():
            problems.append(f"{key} is required for a satellite but is unset")
    host = (env.get("GAB_HOST") or "").strip().lower()
    if host in ("127.0.0.1", "localhost", "::1"):
        problems.append("GAB_HOST is loopback — a satellite must point at another box (and loopback "
                        "would flip GAB_LAUNCH to 1 and try to spawn a brain this box has no binary for)")
    if (env.get("WAKE_WORD_MEDIA_ONLY") or "") != "0":
        problems.append("WAKE_WORD_MEDIA_ONLY must be 0 on a satellite, or the mic is open whenever "
                        "nothing is playing and the wake gate is never exercised")
    return problems


def apply(layout: Layout, env: dict[str, Optional[str]], *,
          confirm: Callable[[str, dict[str, tuple[str, str]]], bool]) -> tuple[Optional[str], dict]:
    """Merge ``env`` into the layout's ``.env`` and write it, after ``confirm`` approves the diff.

    Returns ``(backup_path, changes)``. Merge — never overwrite: an operator's hand-tuned keys and the
    comments explaining them survive a re-provision, and a backup is taken before any write so a bad
    provision is always recoverable.
    """
    path = str(layout.env_path)
    try:
        with open(path, encoding="utf-8") as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ""

    if existing:
        text, changes = envfile.merge(existing, env)
    else:
        text, changes = envfile.render(env), {k: ("", v) for k, v in env.items() if v is not None}

    if not changes:
        return None, {}
    if not confirm(path, changes):
        return None, {}

    backup_path = envfile.backup(path)
    envfile.write(path, text)
    return backup_path, changes


# =================================================================================================== #
# The driver — everything below is the CALLER the modules above were missing.
#
# Design rule: every side effect (terminal, subprocess, network) enters through an injected seam, so
# the whole flow is testable without a box. The parts that touch reality are small and named; the
# parts that decide are pure. That is deliberate — the last time this package shipped, its tests
# measured only the pure half and could not observe that nothing invoked it.
# =================================================================================================== #


@dataclass
class StepResult:
    """One provisioning step's outcome. ``detail`` is written for a human staring at a failed install,
    so it names the remedy rather than the symptom."""

    name: str
    ok: bool
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{'OK  ' if self.ok else 'FAIL'} {self.name}: {self.detail}"


@dataclass
class Console:
    """The IO seam. Secrets go through ``secret_fn`` (no echo) — an install that prints three tokens
    into the operator's scrollback has leaked them to every subsequent `screen`/`tmux` scrollback and
    to anyone reading over their shoulder."""

    input_fn: Callable[[str], str] = input
    secret_fn: Callable[[str], str] = getpass.getpass
    out: Callable[[str], None] = print
    interactive: bool = True

    def ask(self, text: str, default: str = "") -> str:
        if not self.interactive:
            if default:
                return default
            raise _NonInteractive(text)
        suffix = f" [{default}]" if default else ""
        answer = self.input_fn(f"{text}{suffix}: ").strip()
        return answer or default

    def secret(self, text: str) -> Optional[str]:
        """Empty answer → None ("leave unset"), which is NOT the same as an empty token: an empty
        value asserts the service runs open, and we have not verified that."""
        if not self.interactive:
            raise _NonInteractive(text)
        return self.secret_fn(f"{text} (blank = leave unset): ").strip() or None


class _NonInteractive(RuntimeError):
    """Raised when a prompt is needed but the run was told not to prompt. Never swallowed: a
    provisioner that silently substitutes a default for an unanswerable question is how a box ends up
    configured with values nobody chose."""


def _unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def gather(console: Console, *, layout: Layout, proc_asound: str = "/proc/asound",
           providers: Optional[Sequence[Callable[[], Optional[discovery.BrainEndpoint]]]] = None
           ) -> SatelliteAnswers:
    """Detection + prompts → the full answer set. Pure of side effects apart from the console.

    ``providers`` overrides the discovery seam. Default is the maintainer-locked ``(mdns, manual)`` order;
    a test supplies its own so the flow never opens a socket, and a future provider slots in here
    without touching this function.
    """
    room_id = console.ask("Room id for this satellite (e.g. the room it lives in)", "")
    while not room_id:
        room_id = console.ask("Room id is required (it names this box to the brain)", "")

    # Discovery runs AFTER room_id because the mDNS provider filters adverts by room, and the manual
    # provider is the floor beneath it. A miss is not fatal — manual_prompt is inside the seam.
    console.out("Looking for a LAN brain (mDNS, then manual)…")
    endpoint = discovery.discover_brain(
        providers if providers is not None
        else discovery.default_providers(room_id=room_id, input_fn=console.input_fn)
    )
    if endpoint is None:
        raise _ProvisionError("no brain host was supplied — a satellite must point at another box")
    console.out(f"  brain: {endpoint.host}:{endpoint.port} (via {endpoint.source})")

    # Three secrets, from two different services. This is the honest floor; 3c's pairing handshake is
    # what removes the typing, and nothing below depends on HOW they arrive.
    brain_token = console.secret("Brain auth token (the brain's GABAI_VOICE_AUTH_TOKEN)")
    stt_token = console.secret("STT service token (STT_SERVICE_AUTH_TOKEN)")
    tts_token = console.secret("TTS service token (TTS_SERVICE_AUTH_TOKEN)")

    cards = audio_detect.list_alsa_cards(proc_asound=proc_asound)
    if cards:
        console.out("ALSA cards on this box:")
        for idx, card_id, desc in cards:
            console.out(f"  {idx}: {card_id} — {desc}")
    input_device = console.ask("Input (mic) device name", "")
    while not input_device:
        input_device = console.ask("Input device is required (the wake gate listens on it)", "")
    output_device = console.ask("Output (speaker) device name", "")
    while not output_device:
        output_device = console.ask("Output device is required", "")

    wake_word = console.ask("Wake word", "aria")

    # The chipmunk guard's read half. Never a guess in either direction: an unclassifiable device
    # yields None, which leaves AUDIO_OUTPUT_SAMPLE_RATE unset — the historical no-op.
    decision = audio_detect.detect_output_sample_rate(output_device, proc_asound=proc_asound)
    console.out(f"  output rate: {decision.reason}")

    return SatelliteAnswers(
        brain_host=endpoint.host, brain_port=endpoint.port, room_id=room_id,
        brain_token=brain_token, stt_token=stt_token, tts_token=tts_token,
        input_device=input_device, output_device=output_device, wake_word=wake_word,
        output_sample_rate=decision.sample_rate,
    )


class _ProvisionError(RuntimeError):
    """A step could not be completed. Carries operator-facing text, never a stack trace."""


def _default_run(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=30, check=False)


def install_unit(layout: Layout, *, unit_name: str = DEFAULT_UNIT_NAME,
                 run: Callable[[Sequence[str]], subprocess.CompletedProcess] = _default_run,
                 unit_dir: Optional[Path] = None) -> list[StepResult]:
    """Render, write, enable and linger the satellite's ``--user`` unit.

    This is the half that did not exist: ``units.py`` rendered a string with no writer, so a
    provisioned box had a config file and no way to start itself at boot.
    """
    steps: list[StepResult] = []
    target = (unit_dir or _unit_dir())
    target.mkdir(parents=True, exist_ok=True)
    path = target / unit_name

    text = units.render_unit(units.satellite_unit(root=str(layout.root),
                                                  venv_python=str(layout.venv_python)))
    path.write_text(text, encoding="utf-8")
    # Assert the ARTIFACT, not the call: the whole class of bug this package keeps hitting is a step
    # that reports success without producing anything.
    steps.append(StepResult("unit-write", path.is_file() and path.stat().st_size > 0, str(path)))

    reload_rc = run(["systemctl", "--user", "daemon-reload"])
    steps.append(StepResult("daemon-reload", reload_rc.returncode == 0,
                            "systemd re-read the unit" if reload_rc.returncode == 0
                            else f"systemctl --user daemon-reload failed: {reload_rc.stderr.strip()}"))

    enable_rc = run(["systemctl", "--user", "enable", unit_name])
    steps.append(StepResult("unit-enable", enable_rc.returncode == 0,
                            f"{unit_name} starts at login" if enable_rc.returncode == 0
                            else f"systemctl --user enable {unit_name} failed: {enable_rc.stderr.strip()}"))

    # Linger is what makes "at login" mean "at boot" on a headless satellite nobody logs into. Its
    # absence is not cosmetic — without it the box comes up silent after a power cut, which is the
    # exact remote-hands dependency this installer exists to remove. So a failure here is a FAILED
    # step with the remedy in it, never a warning that scrolls past.
    user = getpass.getuser()
    linger_rc = run(["loginctl", "enable-linger", user])
    steps.append(StepResult("linger", linger_rc.returncode == 0,
                            f"{user} lingers → the unit starts at boot without a login"
                            if linger_rc.returncode == 0
                            else f"could not enable linger; run: sudo loginctl enable-linger {user}"))
    return steps


def run_verify(env: dict[str, Optional[str]], *,
               probe_brain=None, probe_stt=None, probe_tts=None) -> list[StepResult]:
    """Probe all three legs on AUTHENTICATED routes, after deciding absent-vs-empty for each token.

    A missing token is an ERROR rather than "assume the service is open": both services read an empty
    token as run-open, so an authenticated probe passes either way and the install would certify a
    credential it never had.

    The probes resolve at CALL time, not as default arguments. A default is bound once when the
    function is defined, so `probe_brain=verify.probe_brain` looks like an injection seam and is not
    one — the caller's substitution is silently ignored. Caught by a test that patched the module and
    watched the real probe run anyway.
    """
    probe_brain = probe_brain or verify.probe_brain
    probe_stt = probe_stt or verify.probe_stt
    probe_tts = probe_tts or verify.probe_tts
    steps: list[StepResult] = []
    for leg, key in (("brain", "GAB_AUTH_TOKEN"), ("stt", "STT_REMOTE_TOKEN"), ("tts", "TTS_REMOTE_TOKEN")):
        kind = verify.classify_credential(env.get(key), present_in_config=key in env and env[key] is not None)
        if kind == "missing":
            steps.append(StepResult(f"{leg}-credential", False,
                                    f"{key} was not supplied — cannot prove {leg} is deliberately open"))
    if any(not s.ok for s in steps):
        return steps

    host, port = (env.get("GAB_HOST") or ""), int(env.get("GAB_PORT") or 0)
    results = [
        probe_brain(host, port, env.get("GAB_AUTH_TOKEN"), room_id=env.get("ROOM_ID") or "install-check"),
        probe_stt(env.get("STT_REMOTE_URL") or "", env.get("STT_REMOTE_TOKEN")),
        probe_tts(env.get("TTS_REMOTE_URL") or "", env.get("TTS_REMOTE_TOKEN")),
    ]
    steps.extend(StepResult(f"{r.leg}-probe", r.ok, r.detail) for r in results)
    return steps


def _confirm_diff(console: Console) -> Callable[[str, dict], bool]:
    """Show the operator exactly what changes, with secrets masked, and let them refuse."""
    def confirm(path: str, changes: dict[str, tuple[str, str]]) -> bool:
        console.out(f"\nAbout to write {path}:")
        for key in sorted(changes):
            old, new = changes[key]
            if "TOKEN" in key or "KEY" in key:
                old, new = ("***" if old else ""), ("***" if new else "")
            console.out(f"  {key}: {old or '(unset)'} → {new or '(unset)'}")
        if not console.interactive:
            return True   # --non-interactive is an explicit "don't ask me", not "don't write"
        return console.ask("Write it? [y/N]", "n").lower().startswith("y")
    return confirm


def provision(console: Console, *, root: Optional[str] = None, dry_run: bool = False,
              skip_verify: bool = False, unit_name: str = DEFAULT_UNIT_NAME,
              proc_asound: str = "/proc/asound",
              providers: Optional[Sequence[Callable[[], Optional[discovery.BrainEndpoint]]]] = None,
              run: Callable[[Sequence[str]], subprocess.CompletedProcess] = _default_run,
              unit_dir: Optional[Path] = None) -> tuple[int, list[StepResult]]:
    """The whole flow. Returns ``(exit_code, steps)`` — never raises for an operator-caused failure."""
    steps: list[StepResult] = []
    layout = detect_layout(override=root)
    console.out(f"Install layout: {layout.kind} at {layout.root} ({layout.reason})")

    try:
        answers = gather(console, layout=layout, proc_asound=proc_asound, providers=providers)
    except (_ProvisionError, _NonInteractive) as e:
        steps.append(StepResult("gather", False, str(e)))
        return EXIT_PROVISION_FAILED, steps
    except (KeyboardInterrupt, EOFError):
        steps.append(StepResult("gather", False, "cancelled by the operator — nothing was written"))
        return EXIT_PROVISION_FAILED, steps

    env = compose_env(answers)
    problems = validate(env)
    if problems:
        steps.extend(StepResult("validate", False, p) for p in problems)
        return EXIT_PROVISION_FAILED, steps
    steps.append(StepResult("validate", True, f"{len(profile.REQUIRED_KEYS)} required keys present"))

    if dry_run:
        console.out("\n--dry-run: composed and validated, writing nothing.")
        console.out(envfile.render({k: v for k, v in env.items() if not ("TOKEN" in k and v)}))
        return EXIT_OK, steps

    apply(layout, env, confirm=_confirm_diff(console))

    # Assert the ARTIFACT rather than the call's return. `apply` returns (None, {}) both when the
    # operator declined AND when a re-provision found nothing to change — indistinguishable from
    # here — so the only trustworthy question is whether the file on disk is now a valid satellite
    # config. This is also what makes a re-run idempotent instead of "already done, therefore failed".
    try:
        written = envfile.parse(Path(layout.env_path).read_text(encoding="utf-8"))
        on_disk = {line.key: line.value for line in written if line.key}
    except FileNotFoundError:
        on_disk = {}
    missing = [k for k in profile.REQUIRED_KEYS if not (on_disk.get(k) or "").strip()]
    if missing:
        steps.append(StepResult("env-write", False,
                                f"{layout.env_path} is missing {', '.join(missing)} — declined, or the "
                                f"write did not land. Nothing was enabled."))
        return EXIT_PROVISION_FAILED, steps
    mode = oct(Path(layout.env_path).stat().st_mode & 0o777)
    steps.append(StepResult("env-write", True, f"{layout.env_path} ({mode}) has every required key"))

    steps.extend(install_unit(layout, unit_name=unit_name, run=run, unit_dir=unit_dir))
    if any(not s.ok for s in steps):
        return EXIT_PROVISION_FAILED, steps

    if skip_verify:
        steps.append(StepResult("verify", True, "skipped at the operator's request (--skip-verify)"))
        return EXIT_OK, steps

    verify_steps = run_verify(env)
    steps.extend(verify_steps)
    if any(not s.ok for s in verify_steps):
        # Config and unit ARE written; a leg did not answer. Distinct code: the remedy is to fix the
        # service or the token, not to re-provision this box.
        return EXIT_VERIFY_FAILED, steps
    return EXIT_OK, steps


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent_install.satellite",
        description="Provision this box as a voice satellite: write .env, install a --user unit, "
                    "enable linger, and verify all three service legs on authenticated routes.")
    parser.add_argument("--root", help="install root to provision (default: the tree this module runs from)")
    parser.add_argument("--dry-run", action="store_true", help="compose and validate, write nothing")
    parser.add_argument("--non-interactive", action="store_true",
                        help="never prompt; fail instead of guessing an unanswerable value")
    parser.add_argument("--skip-verify", action="store_true",
                        help="do not probe the brain/STT/TTS legs after writing")
    parser.add_argument("--unit-name", default=DEFAULT_UNIT_NAME, help=f"(default: {DEFAULT_UNIT_NAME})")
    args = parser.parse_args(list(argv) if argv is not None else None)

    console = Console(interactive=not args.non_interactive)
    if console.interactive and not sys.stdin.isatty():
        console.out("No terminal to prompt on. Re-run attached to a TTY, or pass --non-interactive "
                    "once every value can be supplied without asking.")
        return EXIT_USAGE

    code, steps = provision(console, root=args.root, dry_run=args.dry_run,
                            skip_verify=args.skip_verify, unit_name=args.unit_name)

    console.out("\n--- provisioning summary ---")
    for step in steps:
        console.out(f"  {'OK  ' if step.ok else 'FAIL'} {step.name}: {step.detail}")
    if code == EXIT_OK:
        console.out("\nProvisioned. `systemctl --user start " + args.unit_name + "` to bring it up now.")
    elif code == EXIT_VERIFY_FAILED:
        console.out("\nConfig and unit are written, but a service leg did not answer (see FAIL above). "
                    "Fix the service or the token and re-run; the .env merge is idempotent.")
    else:
        console.out("\nNOT provisioned. Nothing was enabled. Fix the FAIL above and re-run.")
    return code


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m`
    raise SystemExit(main())
