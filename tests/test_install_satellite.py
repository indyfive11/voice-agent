"""Tests for the satellite provisioner: env merge, profile invariants, unit generation, leg probes.

Scope note, because it matters more than usual here: these are IN-PROCESS tests. Per the
verification-scope rule, in-process proves the pure logic and NOTHING about whether a real box comes
up — it cannot observe apt package names, `uv sync` on the target arch, whether the emitted unit
loads, linger taking effect, PipeWire reachability from a session-less unit, or real multicast. Those
need the off-box tier. What these DO pin is every silent-misconfiguration path we found by reading
the code, so that a regression reintroducing one fails here instead of on a satellite.
"""

import pytest

from installkit import templating
from voice_agent_install import envfile, profile, units
from voice_agent_install import satellite as sat
from voice_agent_install import verify
from voice_agent_install.paths import Layout


# ------------------------------------------------------------------ envfile: merge, never clobber
def test_merge_preserves_unmanaged_keys_and_comments():
    existing = (
        "# tuned by hand 2026-06 — do not lose this\n"
        "WAKE_WORD_THRESHOLD=0.75\n"
        "\n"
        "GAB_HOST=192.168.1.99\n"
        "INPUT_USB_RESET_VIDPID=2886:001a\n"
    )
    text, changes = envfile.merge(existing, {"GAB_HOST": "192.168.1.100"})

    assert "# tuned by hand 2026-06 — do not lose this" in text
    assert "WAKE_WORD_THRESHOLD=0.75" in text
    assert "INPUT_USB_RESET_VIDPID=2886:001a" in text
    assert "GAB_HOST=192.168.1.100" in text
    assert changes == {"GAB_HOST": ("192.168.1.99", "192.168.1.100")}


def test_merge_keeps_inline_comment_when_value_changes():
    """The reference .env carries `STT_MODEL=small.en   # drop to base.en if ...` — the note explains
    WHY the value is what it is, so rewriting the value must not silently delete the reasoning."""
    existing = "STT_MODEL=small.en          # drop to base.en if Pi-4 STT latency is too high\n"
    text, _ = envfile.merge(existing, {"STT_MODEL": "base.en"})
    assert "STT_MODEL=base.en" in text
    assert "# drop to base.en if Pi-4 STT latency is too high" in text


def test_merge_none_leaves_existing_key_untouched():
    """None means 'leave unset', which must never be read as 'delete what the operator wrote'."""
    existing = "AUDIO_OUTPUT_SAMPLE_RATE=48000\n"
    text, changes = envfile.merge(existing, {"AUDIO_OUTPUT_SAMPLE_RATE": None})
    assert "AUDIO_OUTPUT_SAMPLE_RATE=48000" in text
    assert changes == {}


def test_parse_distinguishes_empty_value_from_absent():
    lines = envfile.parse("EXPLICITLY_EMPTY=\n")
    assert lines[0].key == "EXPLICITLY_EMPTY"
    assert lines[0].value == ""  # present-but-empty, NOT None


def test_merge_appends_new_keys_without_touching_the_rest():
    existing = "EXISTING=1\n"
    text, changes = envfile.merge(existing, {"BRAND_NEW": "2"})
    assert "EXISTING=1" in text
    assert "BRAND_NEW=2" in text
    assert changes["BRAND_NEW"] == ("", "2")


def test_write_is_0600(tmp_path):
    """The file carries three bearer tokens."""
    import os
    p = tmp_path / ".env"
    envfile.write(str(p), "GAB_AUTH_TOKEN=secret\n")
    assert oct(os.stat(p).st_mode & 0o777) == "0o600"


# ------------------------------------------------------------------ the silent-misconfiguration gates
def _answers(**over):
    base = dict(
        brain_host="192.168.1.100", brain_port=8765, room_id="bedroom",
        brain_token="t1", stt_token="t2", tts_token="t3",
        input_device="EMEET", output_device="EMEET",
        wake_word="wakewords/aria.onnx",
    )
    base.update(over)
    return sat.SatelliteAnswers(**base)


def test_compose_writes_every_required_key():
    env = sat.compose_env(_answers())
    assert not sat.validate(env)
    for key in profile.REQUIRED_KEYS:
        assert (env.get(key) or "").strip(), f"{key} missing"


def test_unset_wake_word_is_rejected():
    """config.py:1494-95 — an unwritten WAKE_WORD does not degrade the gate, it DELETES it and leaves
    the mic open, with no error. This must be impossible to produce."""
    env = sat.compose_env(_answers(wake_word=""))
    assert any("WAKE_WORD is required" in p for p in sat.validate(env))


def test_media_only_must_be_zero():
    """Default '1' means wake is required only while media plays — a quiet box is an open mic, and an
    acceptance test would pass having never exercised the gate."""
    env = sat.compose_env(_answers())
    env["WAKE_WORD_MEDIA_ONLY"] = "1"
    assert any("WAKE_WORD_MEDIA_ONLY must be 0" in p for p in sat.validate(env))


def test_loopback_brain_host_is_rejected():
    """Loopback flips GAB_LAUNCH to 1 (config.py:298) and the satellite tries to spawn a brain binary
    it does not have."""
    env = sat.compose_env(_answers(brain_host="127.0.0.1"))
    problems = sat.validate(env)
    assert any("loopback" in p for p in problems)


def test_gab_launch_is_always_zero():
    assert sat.compose_env(_answers())["GAB_LAUNCH"] == "0"


def test_brain_is_written_because_the_unit_bypasses_run_sh():
    """The generated unit execs the venv python directly, so `run.sh gab`'s argv no longer exports it."""
    assert sat.compose_env(_answers())["BRAIN"] == "gabagent"


def test_service_urls_are_composed_not_assumed():
    """Nothing in the codebase constructs these at first provision (_repoint_remote_services only
    rewrites an EXISTING absolute URL), and an unwritten STT_REMOTE_URL is a hard startup error."""
    env = sat.compose_env(_answers())
    assert env["STT_REMOTE_URL"] == "http://192.168.1.100:8770"
    assert env["TTS_REMOTE_URL"] == "http://192.168.1.100:8771"


def test_services_can_live_on_a_third_box():
    env = sat.compose_env(_answers(stt_host="192.168.1.200"))
    assert env["STT_REMOTE_URL"] == "http://192.168.1.200:8770"
    assert env["TTS_REMOTE_URL"] == "http://192.168.1.100:8771"


def test_absent_token_is_unset_not_empty():
    """Empty means 'service is open' to both services; absent means 'we have no credential'. Writing
    KEY= would assert an open service we never verified."""
    env = sat.compose_env(_answers(stt_token=None))
    assert env["STT_REMOTE_TOKEN"] is None


def test_sample_rate_unset_unless_detected():
    assert sat.compose_env(_answers())["AUDIO_OUTPUT_SAMPLE_RATE"] is None
    assert sat.compose_env(_answers(output_sample_rate=48000))["AUDIO_OUTPUT_SAMPLE_RATE"] == "48000"


# ------------------------------------------------------------------ unit generation
def _unit(**kw):
    return templating.render_unit(**units.satellite_unit(
        root="/home/x/voice-agent",
        venv_python="/home/x/voice-agent/.venv/bin/python", **kw))


def test_unit_execs_venv_python_not_uv():
    """`run.sh` is `exec uv run python main.py`; uv lives in ~/.local/bin which is NOT on a --user
    unit's PATH → rc 127. Going direct also keeps a dependency resolve off the boot path."""
    text = _unit()
    assert "ExecStart=/home/x/voice-agent/.venv/bin/python main.py" in text
    assert "uv run" not in text
    assert "run.sh" not in text


def test_unit_orders_after_the_sound_server():
    """A linger-started boot unit races PipeWire; losing that race is the -9997 output-failure class."""
    text = _unit()
    assert "After=" in text and "pipewire.service" in text
    assert "Wants=" in text  # Wants, not Requires — a bare-ALSA box must still start


def test_unit_never_requires_anything():
    """Boot-safety: no remote or hard dependency may gate startup."""
    assert "Requires=" not in _unit()


def test_unit_restart_sec_prevents_the_permanent_failed_latch():
    """main.py deliberately os._exit(1) so a supervisor re-inits a wedged mic. At systemd's 100ms
    default, a few stalls latch the unit `failed` forever and the box needs remote hands."""
    text = _unit()
    assert f"RestartSec={units.DEFAULT_RESTART_SEC}" in text
    assert units.DEFAULT_RESTART_SEC >= 15
    assert "StartLimitBurst=5" in text


def test_unit_rejects_newline_injection():
    # installkit's render_unit raises ValueError on a newline in any rendered field (its message text
    # differs from the old local renderer's, so assert the TYPE, not the message).
    with pytest.raises(ValueError):
        templating.render_unit(description="x\nExecStartPre=/bin/rm -rf /",
                               exec_start="/bin/true", working_directory="/tmp")


# ------------------------------------------------------------------ credential classification
def test_missing_credential_is_not_treated_as_open():
    """Both services read an empty token as 'run open', so an absent token in OUR config is
    indistinguishable from a deliberately-open service unless we classify it separately."""
    assert verify.classify_credential(None, present_in_config=False) == "missing"
    assert verify.classify_credential("", present_in_config=True) == "open"
    assert verify.classify_credential("abc", present_in_config=True) == "guarded"


def test_silence_wav_is_a_valid_riff():
    wav = verify.silence_wav(seconds=0.1, rate=16000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) == 44 + int(0.1 * 16000) * 2


# ------------------------------------------------------------------ layout detection
def test_explicit_root_wins(tmp_path):
    layout = Layout(kind="checkout", root=tmp_path, reason="test")
    assert layout.env_path == tmp_path / ".env"
    assert layout.venv_python == tmp_path / ".venv" / "bin" / "python"


def test_apply_merges_and_backs_up(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GAB_HOST=10.0.0.1\nOPERATOR_TUNED=keepme\n")
    layout = Layout(kind="checkout", root=tmp_path, reason="test")

    backup, changes = sat.apply(layout, {"GAB_HOST": "192.168.1.100"}, confirm=lambda p, c: True)

    assert backup is not None and "OPERATOR_TUNED=keepme" in open(backup).read()
    assert "OPERATOR_TUNED=keepme" in env_path.read_text()
    assert "GAB_HOST=192.168.1.100" in env_path.read_text()
    assert changes == {"GAB_HOST": ("10.0.0.1", "192.168.1.100")}


def test_apply_writes_nothing_when_declined(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GAB_HOST=10.0.0.1\n")
    layout = Layout(kind="checkout", root=tmp_path, reason="test")

    backup, changes = sat.apply(layout, {"GAB_HOST": "192.168.1.100"}, confirm=lambda p, c: False)

    assert backup is None and changes == {}
    assert env_path.read_text() == "GAB_HOST=10.0.0.1\n"


# =================================================================================================== #
# The DRIVER — the caller these modules did not have.
#
# These are still in-process, with every side effect injected, so they prove the flow and NOTHING
# about whether a real box comes up. What they CAN observe that the previous 26 could not: that the
# entry point exists, runs, and exits non-zero when it provisions nothing. That is the specific
# failure this suite was blind to — `python -m voice_agent_install.satellite` used to import the
# module, run nothing and exit 0, and every green test agreed.
# =================================================================================================== #

import subprocess
import sys
from pathlib import Path

from voice_agent_install import discovery, profile


class FakeConsole(sat.Console):
    """A scripted console: answers come off a list, secrets off another, output is captured."""

    def __init__(self, answers, secrets=("btok", "stok", "ttok")):
        self._answers = list(answers)
        self._secrets = list(secrets)
        self.lines = []
        super().__init__(input_fn=self._next, secret_fn=lambda p: self._next_secret(),
                         out=self.lines.append, interactive=True)

    def _next(self, prompt=""):
        return self._answers.pop(0) if self._answers else ""

    def _next_secret(self):
        return self._secrets.pop(0) if self._secrets else ""


def _endpoint(host="10.0.0.9", port=8765):
    return lambda: discovery.BrainEndpoint(host, port, "test")


def _ok_run(argv):
    return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")


def _fail_run_for(match):
    def run(argv):
        rc = 1 if match in " ".join(argv) else 0
        return subprocess.CompletedProcess(list(argv), rc, stdout="", stderr="boom")
    return run


ANSWERS = ["living_room", "mic-name", "spk-name", "aria", "y"]   # room, in, out, wake, confirm


def _probes_ok(monkeypatch):
    monkeypatch.setattr(sat.verify, "probe_brain",
                        lambda *a, **k: sat.verify.LegResult("brain", True, "ok"))
    monkeypatch.setattr(sat.verify, "probe_stt",
                        lambda *a, **k: sat.verify.LegResult("stt", True, "ok"))
    monkeypatch.setattr(sat.verify, "probe_tts",
                        lambda *a, **k: sat.verify.LegResult("tts", True, "ok"))


# --- the P10 regression: the entry point must not succeed while doing nothing --------------------- #

def test_module_run_as_main_does_not_exit_zero_having_provisioned_nothing():
    # THE regression. Before main() existed this exited 0 with no output and ignored argv entirely,
    # so bootstrap.sh reported a successful install of nothing — and a DoD checking $? agreed.
    out = subprocess.run([sys.executable, "-m", "voice_agent_install.satellite", "--non-interactive"],
                         capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode != 0
    assert "NOT provisioned" in out.stdout


def test_module_exposes_a_real_cli():
    out = subprocess.run([sys.executable, "-m", "voice_agent_install.satellite", "--help"],
                         capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0
    for flag in ("--root", "--dry-run", "--non-interactive", "--skip-verify"):
        assert flag in out.stdout


# --- the flow -------------------------------------------------------------------------------------- #

def test_provision_writes_env_unit_and_enables_linger(tmp_path, monkeypatch):
    _probes_ok(monkeypatch)
    units_dir = tmp_path / "systemd"
    calls = []

    def run(argv):
        calls.append(list(argv))
        return _ok_run(argv)

    console = FakeConsole(ANSWERS)
    code, steps = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),),
                                run=run, unit_dir=units_dir, proc_asound=str(tmp_path / "nope"))

    assert code == sat.EXIT_OK, [str(s) for s in steps]
    # ARTIFACTS, not step status — the class of bug here is a step reporting success with no output.
    env_text = (tmp_path / ".env").read_text()
    for key in profile.REQUIRED_KEYS:
        assert f"{key}=" in env_text
    assert oct((tmp_path / ".env").stat().st_mode & 0o777) == "0o600"
    unit = (units_dir / sat.DEFAULT_UNIT_NAME).read_text()
    assert "ExecStart=" in unit and str(tmp_path) in unit
    assert ["systemctl", "--user", "daemon-reload"] in calls
    assert ["systemctl", "--user", "enable", sat.DEFAULT_UNIT_NAME] in calls
    assert any(c[:2] == ["loginctl", "enable-linger"] for c in calls)


def test_dry_run_writes_absolutely_nothing(tmp_path):
    console = FakeConsole(ANSWERS)
    code, _ = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),), dry_run=True,
                            proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_OK
    assert not (tmp_path / ".env").exists()


def test_dry_run_output_never_contains_a_token(tmp_path):
    console = FakeConsole(ANSWERS, secrets=("SUPERSECRET1", "SUPERSECRET2", "SUPERSECRET3"))
    sat.provision(console, root=str(tmp_path), providers=(_endpoint(),), dry_run=True,
                  proc_asound=str(tmp_path / "nope"))
    assert "SUPERSECRET" not in "\n".join(console.lines)


def test_declining_the_diff_provisions_nothing_and_enables_nothing(tmp_path):
    calls = []
    console = FakeConsole(["living_room", "mic-name", "spk-name", "aria", "n"])   # <- declines
    code, steps = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),),
                                run=lambda a: (calls.append(list(a)), _ok_run(a))[1],
                                unit_dir=tmp_path / "systemd", proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_PROVISION_FAILED
    assert calls == []                     # nothing enabled after a refusal
    assert not (tmp_path / "systemd" / sat.DEFAULT_UNIT_NAME).exists()


def test_linger_failure_fails_the_install(tmp_path, monkeypatch):
    # Without linger a headless satellite never starts after a power cut — the exact remote-hands
    # dependency this installer removes. It must not scroll past as a warning.
    _probes_ok(monkeypatch)
    console = FakeConsole(ANSWERS)
    code, steps = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),),
                                run=_fail_run_for("enable-linger"), unit_dir=tmp_path / "systemd",
                                proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_PROVISION_FAILED
    linger = [s for s in steps if s.name == "linger"][0]
    assert not linger.ok and "enable-linger" in linger.detail


def test_unreachable_service_is_a_distinct_exit_code_with_artifacts_intact(tmp_path, monkeypatch):
    # Config + unit ARE written; a leg did not answer. Collapsing this into EXIT_PROVISION_FAILED
    # would tell the operator to re-provision when the remedy is to fix the service or the token.
    monkeypatch.setattr(sat.verify, "probe_brain",
                        lambda *a, **k: sat.verify.LegResult("brain", False, "unreachable"))
    monkeypatch.setattr(sat.verify, "probe_stt",
                        lambda *a, **k: sat.verify.LegResult("stt", True, "ok"))
    monkeypatch.setattr(sat.verify, "probe_tts",
                        lambda *a, **k: sat.verify.LegResult("tts", True, "ok"))
    console = FakeConsole(ANSWERS)
    code, _ = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),), run=_ok_run,
                            unit_dir=tmp_path / "systemd", proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_VERIFY_FAILED
    assert (tmp_path / ".env").exists()
    assert (tmp_path / "systemd" / sat.DEFAULT_UNIT_NAME).exists()


def test_missing_token_is_an_error_not_an_assumption_that_the_service_is_open(tmp_path):
    # An absent token is indistinguishable from "deliberately open" to an authenticated probe, so it
    # must fail BEFORE probing rather than certify a credential the install never had.
    console = FakeConsole(ANSWERS, secrets=("", "", ""))     # all three left blank
    code, steps = sat.provision(console, root=str(tmp_path), providers=(_endpoint(),), run=_ok_run,
                                unit_dir=tmp_path / "systemd", proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_VERIFY_FAILED
    assert any(s.name.endswith("-credential") and not s.ok for s in steps)


def test_no_brain_host_fails_rather_than_writing_a_broken_config(tmp_path):
    console = FakeConsole(ANSWERS)
    code, steps = sat.provision(console, root=str(tmp_path), providers=(lambda: None,),
                                run=_ok_run, proc_asound=str(tmp_path / "nope"))
    assert code == sat.EXIT_PROVISION_FAILED
    assert not (tmp_path / ".env").exists()


def test_reprovision_is_idempotent_and_preserves_hand_tuned_keys(tmp_path, monkeypatch):
    _probes_ok(monkeypatch)
    (tmp_path / ".env").write_text("# hand-tuned\nWAKE_WORD_THRESHOLD=0.42\n")
    for _ in range(2):
        code, _ = sat.provision(FakeConsole(ANSWERS), root=str(tmp_path), providers=(_endpoint(),),
                                run=_ok_run, unit_dir=tmp_path / "systemd",
                                proc_asound=str(tmp_path / "nope"))
        assert code == sat.EXIT_OK
    text = (tmp_path / ".env").read_text()
    assert "WAKE_WORD_THRESHOLD=0.42" in text and "# hand-tuned" in text
