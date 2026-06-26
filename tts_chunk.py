"""Shared TTS chunker — split a reply into synth chunks for fast first-audio.

WHY (collab finding, GA↔VAC 2026-06-26): pipecat already aggregates the LLM token stream to
SENTENCE granularity *before* calling ``run_tts`` (``TextAggregationMode.SENTENCE`` is the 1.3.0
default), so a sentence-level split downstream is a no-op. The felt-latency cliff is a single
LONG sentence synthesized whole before any audio emits (kokoro-onnx ``create_stream`` is
fake-streaming — it yields the whole input as one chunk at full-synth time). The fix is to split
that one sentence at SUB-sentence (clause) boundaries and synth+flush each, so first-audio lands
on the first clause instead of the whole sentence.

ONE implementation, imported by BOTH the in-pipeline ``StreamingKokoroTTSService`` (laptop, local
Kokoro) AND ``tts_service/server.py`` (remote Kokoro, Pi) — no drift. Pure functions, no heavy
deps, so the standalone service imports it without dragging pipecat in.

Length-GATED by default for the daily driver: a sentence at or below ``split_threshold`` chars
passes through WHOLE (today's behavior, natural prosody — Kokoro renders each chunk with
sentence-final intonation, so blind clause-splitting every sentence sounds choppy). Only longer
run-ons are sub-split. ``split_threshold=0`` disables the gate (always sub-split — the aggressive
profile, e.g. a slow remote satellite).
"""

from __future__ import annotations

import re

# Sentence-only terminals (reproduces the historical tts_service._split_sentences when clause=False).
_SENT_DELIMS = ".!?"
# Clause boundaries: sentence terminals PLUS punctuation that carries a natural prosodic pause.
_CLAUSE_DELIMS = ".!?;:,—–"


def _delim_re(delims: str) -> re.Pattern[str]:
    """Build the proven "run up to and including delimiter(s), or trailing remainder" pattern for a
    delimiter set (mirrors the original tts_service _SENT_RE, generalized over the delimiter class)."""
    esc = re.escape(delims)
    return re.compile(rf"[^{esc}]*[{esc}]+|\S[^{esc}]*$")


def split_for_synth(
    text: str,
    *,
    min_len: int = 24,
    max_chars: int = 0,
    split_threshold: int = 0,
    clause: bool = True,
) -> list[str]:
    """Split text into synth chunks for incremental first-audio.

    Args:
        text: the text to split (typically ONE sentence, as pipecat hands ``run_tts``).
        min_len: merge a tiny leading chunk forward into the next so we don't synth choppy 1-2 word
            clips (each synth has fixed overhead). The point is a short FIRST chunk without
            over-fragmenting the rest.
        max_chars: hard cap — force-break a chunk longer than this on a word boundary (so a
            comma-less run-on still yields early audio). ``0`` = no cap.
        split_threshold: sentences with ``len(text) <= split_threshold`` pass through WHOLE (prosody
            -safe common case). ``0`` disables the gate (always sub-split).
        clause: ``True`` splits on clause boundaries (commas/semicolons/dashes too); ``False``
            splits on sentence terminals only (``.!?`` — reproduces the historical behavior).

    Returns:
        A non-empty list of chunks for non-empty input (the whole text as one item when short or
        when it has no internal boundaries).
    """
    t = text.strip()
    if not t:
        return []
    # Length gate: short/normal text passes through whole (natural prosody, today's behavior).
    if split_threshold and len(t) <= split_threshold:
        return [t]

    pat = _delim_re(_CLAUSE_DELIMS if clause else _SENT_DELIMS)
    parts = [p.strip() for p in pat.findall(t) if p.strip()]

    # Hard max-char cap: break a too-long chunk (e.g. a comma-less run-on) on a word boundary.
    capped: list[str] = []
    for p in parts:
        while max_chars and len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars  # no space in range — hard cut
            head = p[:cut].strip()
            if head:
                capped.append(head)
            p = p[cut:].strip()
        if p:
            capped.append(p)

    # Merge tiny leading fragments forward so the FIRST chunk isn't a 1-2 word clip.
    merged: list[str] = []
    for p in capped:
        if merged and len(merged[-1]) < min_len:
            merged[-1] = f"{merged[-1]} {p}".strip()
        else:
            merged.append(p)

    return merged or [t]
