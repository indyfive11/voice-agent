"""LocalSinkDucker — the Pi-side media-duck belt (attenuate the LOCAL media sink-input via pactl).

Verifies it finds the matched media stream, ducks its PipeWire node volume, restores the exact prior,
no-ops when disabled / unmatched, and is idempotent. The pactl subprocess is injected (no real audio).
"""

import asyncio

from local_duck import LocalSinkDucker

# Real `pactl list sink-inputs` shape from the Pi (2026-06-23 probe): Mopidy + the TTS/speech-dispatcher
# and a Chromium stream that must NOT be ducked when match='mopidy'.
SAMPLE = (
    "Sink Input #98\n"
    "\tVolume: mono: 65536 / 100% / 0.00 dB\n"
    '\t\tapplication.name = "speech-dispatcher-dummy"\n'
    '\t\tnode.name = "speech-dispatcher-dummy"\n'
    "Sink Input #20137\n"
    "\tVolume: front-left: 65536 / 100% / 0.00 dB,   front-right: 65536 / 100% / 0.00 dB\n"
    '\t\tapplication.name = "Mopidy"\n'
    '\t\tnode.name = "Mopidy"\n'
    "Sink Input #20164\n"
    "\tVolume: front-left: 65536 / 80% / 0.00 dB\n"
    '\t\tapplication.name = "Chromium"\n'
    '\t\tnode.name = "Chromium"\n'
)


def _sample(mopidy_vol=100, mopidy_idx="20137"):
    return (
        "Sink Input #98\n"
        "\tVolume: mono: 65536 / 100% / 0.00 dB\n"
        '\t\tapplication.name = "speech-dispatcher-dummy"\n'
        f"Sink Input #{mopidy_idx}\n"
        f"\tVolume: front-left: 65536 / {mopidy_vol}% / 0.00 dB\n"
        '\t\tapplication.name = "Mopidy"\n'
        '\t\tnode.name = "Mopidy"\n'
        "Sink Input #20164\n"
        "\tVolume: front-left: 65536 / 80% / 0.00 dB\n"
        '\t\tapplication.name = "Chromium"\n'
    )


class _FakeRunner:
    def __init__(self, list_output=SAMPLE, list_rc=0):
        self.calls = []
        self._out = list_output       # may be reassigned between duck/restore to simulate a song switch
        self._rc = list_rc

    async def __call__(self, *args, timeout=2.0):
        self.calls.append(args)
        if args[:2] == ("list", "sink-inputs"):
            return self._rc, self._out
        return 0, ""

    def sets(self):
        return [a for a in self.calls if a and a[0] == "set-sink-input-volume"]


def _mk(runner, **kw):
    return LocalSinkDucker(enabled=kw.pop("enabled", True), duck_pct=kw.pop("duck_pct", 18),
                           match=kw.pop("match", "mopidy"), restore_level=kw.pop("restore_level", 100),
                           runner=runner)


def test_ducks_only_the_matched_stream():
    r = _FakeRunner()
    d = _mk(r)
    asyncio.run(d.duck(True))
    # Only Mopidy (#20137) is ducked → set to 18%; speech-dispatcher (#98) and Chromium (#20164) untouched.
    assert r.sets() == [("set-sink-input-volume", "20137", "18%")]


def test_restores_exact_prior():
    r = _FakeRunner()
    d = _mk(r)
    asyncio.run(d.duck(True))
    asyncio.run(d.duck(False))
    assert ("set-sink-input-volume", "20137", "100%") in r.sets()  # restored to the read prior, not a guess


def test_disabled_is_noop():
    r = _FakeRunner()
    d = _mk(r, enabled=False)
    asyncio.run(d.duck(True))
    assert r.calls == []  # never even lists sink-inputs


def test_no_match_no_action():
    r = _FakeRunner()
    d = _mk(r, match="mpd")  # nothing named mpd in the sample
    asyncio.run(d.duck(True))
    assert r.sets() == []


def test_idempotent_duck_on_does_not_clobber_prior():
    r = _FakeRunner()
    d = _mk(r)
    asyncio.run(d.duck(True))
    asyncio.run(d.duck(True))  # second on must NOT re-read/re-save (would save the already-ducked 18 as prior)
    assert len(r.sets()) == 1  # only the first duck set the volume
    asyncio.run(d.duck(False))
    assert ("set-sink-input-volume", "20137", "100%") in r.sets()  # still restores the TRUE prior


def test_restore_without_duck_is_noop():
    r = _FakeRunner()
    d = _mk(r)
    asyncio.run(d.duck(False))  # never ducked → nothing to restore
    assert r.sets() == []


def test_substring_match_is_case_insensitive():
    r = _FakeRunner()
    d = _mk(r, match="MOPIDY")
    asyncio.run(d.duck(True))
    assert r.sets() == [("set-sink-input-volume", "20137", "18%")]


def test_pactl_list_failure_is_safe_noop():
    r = _FakeRunner(list_rc=1)
    d = _mk(r)
    asyncio.run(d.duck(True))  # list failed → no sets, no raise
    assert r.sets() == []


def test_stranded_read_at_duck_level_restores_to_baseline_not_18():
    # REGRESSION (2026-06-23 stuck-mute): module-stream-restore replayed the ducked 18% onto the new song,
    # so the belt reads 18 as the "prior". It must NOT save 18 (→ restore-to-18 strands the music quiet) —
    # the guard falls back to the baseline so the music comes back to 100%.
    r = _FakeRunner(list_output=_sample(mopidy_vol=18))  # Mopidy stream stranded at the duck level
    d = _mk(r, restore_level=100)
    asyncio.run(d.duck(True))
    asyncio.run(d.duck(False))
    assert ("set-sink-input-volume", "20137", "100%") in r.sets()  # restored to baseline, NOT 18
    # the only 18% set is the duck-on; the restore must not also be 18 (that's the strand)
    assert [s for s in r.sets() if s[2] == "18%"] == [("set-sink-input-volume", "20137", "18%")]


def test_restore_refinds_new_sink_input_index_after_song_switch():
    # A song switch can spawn a NEW sink-input index; restore must re-find the CURRENT Mopidy stream by name
    # and lift IT (not the dead duck-time index) to the prior.
    r = _FakeRunner(list_output=_sample(mopidy_vol=100, mopidy_idx="20137"))
    d = _mk(r)
    asyncio.run(d.duck(True))                       # ducks idx 20137 → 18
    r._out = _sample(mopidy_vol=18, mopidy_idx="30000")  # song switch: new index, stranded at 18
    asyncio.run(d.duck(False))                      # restore must target the NEW index 30000
    assert ("set-sink-input-volume", "30000", "100%") in r.sets()
    assert not any(s[1] == "20137" and s[2] == "100%" for s in r.sets())  # not the dead index
