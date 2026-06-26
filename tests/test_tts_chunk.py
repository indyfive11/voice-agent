"""Tests for the shared TTS chunker (tts_chunk.split_for_synth) + the server's byte-identical delegator.

The chunker is the felt-latency cliff fix (GA↔VAC collab 2026-06-26): pipecat hands run_tts ONE
sentence, stock Kokoro synthesizes it whole before any audio, so a long run-on = a multi-second
first-audio. split_for_synth sub-splits a long sentence at clause boundaries so first-audio lands on
the first clause — while leaving short/normal sentences whole (prosody-safe).
"""

import re

import pytest

from tts_chunk import split_for_synth


# --- gating: short/normal sentences pass through WHOLE (the common case, natural prosody) ---------
def test_empty_and_whitespace_return_empty():
    assert split_for_synth("") == []
    assert split_for_synth("   \n  ") == []


def test_short_sentence_below_threshold_is_whole():
    s = "Sure, I can do that."
    assert split_for_synth(s, split_threshold=140) == [s]


def test_at_threshold_boundary_is_whole():
    s = "x" * 140
    assert split_for_synth(s, split_threshold=140) == [s]


def test_just_over_threshold_with_clauses_splits():
    # 3 clauses, total > threshold → sub-split so first-audio is the first clause
    s = ("First we boot the device, then we wait for the network to settle, "
         "and finally we resolve the audio sink before opening the stream.")
    assert len(s) > 60
    chunks = split_for_synth(s, split_threshold=60, max_chars=160)
    assert len(chunks) > 1
    # first chunk is a short clause, not the whole thing → fast first-audio
    assert len(chunks[0]) < len(s)
    # nothing lost: rejoining recovers the words in order
    assert " ".join(chunks).split() == s.split()


# --- sub-sentence (clause) splitting -------------------------------------------------------------
def test_clause_split_on_commas():
    s = "We can go to the store, pick up some milk, and then head straight home again."
    chunks = split_for_synth(s, split_threshold=0, min_len=1, max_chars=0)
    assert len(chunks) == 3
    assert chunks[0].endswith(",")


def test_max_chars_cap_breaks_comma_less_runon():
    # no internal punctuation at all → only the hard max-char cap can break it
    s = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
    chunks = split_for_synth(s, split_threshold=0, max_chars=30, min_len=1)
    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks).split() == s.split()


def test_min_len_merges_tiny_leading_fragment_forward():
    # "Oh." is tiny — must merge forward so the first synth chunk isn't a 1-word clip
    s = "Oh. That is a genuinely interesting question to think through carefully."
    chunks = split_for_synth(s, split_threshold=0, min_len=24, max_chars=0)
    assert chunks[0].startswith("Oh.")
    assert len(chunks[0]) >= 24


def test_split_threshold_zero_always_considers_splitting():
    s = "Short, but split."
    # threshold 0 disables the gate → splits on the comma even though it's short
    assert len(split_for_synth(s, split_threshold=0, min_len=1, max_chars=0)) == 2


# --- clause=False reproduces the historical sentence-only behavior (no drift) ---------------------
_OLD_SENT_RE = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")


def _old_split_sentences(text, *, min_len=24):
    """The pre-refactor tts_service._split_sentences, verbatim, as the regression oracle."""
    parts = [p.strip() for p in _OLD_SENT_RE.findall(text) if p.strip()]
    merged = []
    for p in parts:
        if merged and len(merged[-1]) < min_len:
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)
    if not merged:
        t = text.strip()
        return [t] if t else []
    return merged


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "One sentence only.",
        "Hi. How are you? I am fine!",
        "Yes.",
        "A short one. Then a much longer second sentence that keeps going and going.",
        "No terminal punctuation here",
        "Tiny. Now a sufficiently long follow-on clause to exceed the merge threshold easily.",
    ],
)
def test_server_delegator_is_byte_identical_to_old(text):
    # how tts_service/server.py now calls it
    new = split_for_synth(text, min_len=24, clause=False, max_chars=0, split_threshold=0)
    assert new == _old_split_sentences(text, min_len=24)


def test_server_delegator_import_path():
    # the server delegates to the same shared function (guards against drift)
    from tts_service.server import _split_sentences

    assert _split_sentences("Hi. There.") == split_for_synth(
        "Hi. There.", min_len=24, clause=False, max_chars=0, split_threshold=0
    )
