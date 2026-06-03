"""Offline wake-word A/B harness — a CLEAN, repeatable test of "Aria over music", no human/AEC/switch.

Why this exists: live wake-over-music numbers are confounded by (1) how the user says "Aria" each time
(distance/clarity — the WAKE log is blind to it), (2) the AEC's variable cancellation, and (3) PipeWire
graph disruption on track switches. So a live 0.36 vs 0.06 vs 0.00 tells us almost nothing. This harness
removes ALL of that: take ONE fixed "Aria" recording, mix it with a music bed at controlled signal-to-noise
ratios, and score it with openWakeWord — toggling Speex noise suppression. Same input every run → the only
thing that changes is the knob, so the resulting score curve is a real A/B instead of vibes.

It measures the MODEL + Speex (not the acoustic AEC path — for AEC residual use the zero-effort
`pw-record --target echo-cancel-source` capture). Pair the two and you've isolated every layer.

Record the inputs at 16 kHz mono (or anything — we resample):
    arecord -f S16_LE -r 16000 -c 1 aria.wav     # say "Aria" once, clean, then Ctrl-C
    # music.wav: a few seconds of a track (pw-record/parec a sink monitor, or any music file)

    uv run python tools/wakeword_ab.py --wake aria.wav --music music.wav --model wakewords/aria.onnx
    uv run python tools/wakeword_ab.py --wake aria.wav --music music.wav --save-mix 0   # also write the
                                                                                        # 0 dB mix to hear

The table: rows = SNR (voice-to-music, dB; +inf = no music, negative = music louder than voice — the
masking regime), columns = peak score with speex off / on. If Speex lifts the negative-SNR rows, it's
confirmed; if not, the model is the bottleneck (retrain "hey aria" with music in the negatives).
"""

from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

import openwakeword
from openwakeword.model import Model

RATE = 16000
CHUNK = 1280  # 80 ms @ 16 kHz — openWakeWord's frame
PREROLL_SECS = 1.5  # music context before the word (the embedding buffer needs lead-in)
TAIL_SECS = 1.0


def _read_wav_16k_mono(path: str) -> np.ndarray:
    """Read a WAV → mono int16 @ 16 kHz (downmix + resample as needed)."""
    with wave.open(os.path.expanduser(path), "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if sw != 2:
        sys.exit(f"{path}: need 16-bit PCM (got sampwidth={sw}); re-record with -f S16_LE")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)  # downmix
    if sr != RATE:
        x = _resample(x, sr, RATE)
    return x


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    try:
        import soxr  # high quality (same lib pipecat uses)
        return soxr.resample(x, src, dst)
    except Exception:  # noqa: BLE001 - fall back to linear; fine for a diagnostic music bed
        n_out = int(round(len(x) * dst / src))
        return np.interp(np.linspace(0, len(x), n_out, endpoint=False), np.arange(len(x)), x)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def _music_bed(music: np.ndarray, length: int) -> np.ndarray:
    """Loop/trim the music to exactly `length` samples (a continuous bed under the word)."""
    if len(music) == 0:
        return np.zeros(length)
    reps = int(np.ceil(length / len(music)))
    return np.tile(music, reps)[:length]


def _build_mix(wake: np.ndarray, music: np.ndarray, snr_db: float) -> np.ndarray:
    """Place `wake` in a music bed; scale the music so voice-to-music RMS = snr_db. inf → no music."""
    pre, tail = int(PREROLL_SECS * RATE), int(TAIL_SECS * RATE)
    total = pre + len(wake) + tail
    voice = np.zeros(total)
    voice[pre:pre + len(wake)] = wake

    if np.isinf(snr_db):
        mix = voice
    else:
        bed = _music_bed(music, total)
        wr, mr = _rms(wake), _rms(bed)
        if mr > 0:
            target_mr = wr / (10 ** (snr_db / 20.0))  # music RMS for this voice-to-music ratio
            bed = bed * (target_mr / mr)
        mix = voice + bed
    return np.clip(mix, -32768, 32767).astype(np.int16)


def _peak_score(model_path: str, speex: bool, pcm: np.ndarray) -> float:
    """Fresh model per call (no buffer cross-talk); feed in 1280-sample frames; return the peak score."""
    oww = Model(wakeword_model_paths=[model_path], vad_threshold=0.0,
                enable_speex_noise_suppression=speex)
    key = next(iter(oww.models.keys()))
    peak = 0.0
    for i in range(0, len(pcm) - CHUNK + 1, CHUNK):
        peak = max(peak, float(oww.predict(pcm[i:i + CHUNK]).get(key, 0.0)))
    return peak


def _resolve_model(name: str) -> str:
    if name.endswith(".onnx") and os.path.exists(os.path.expanduser(name)):
        return os.path.expanduser(name)
    pkg = os.path.dirname(openwakeword.__file__)
    p = os.path.join(pkg, "resources", "models", f"{name}_v0.1.onnx")
    if not os.path.exists(p):
        sys.exit(f"no model {name!r} (give a bundled name like hey_jarvis or a path to a .onnx)")
    return p


def _save_wav(path: str, pcm: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline wake-word A/B: Aria over music, Speex off/on.")
    ap.add_argument("--wake", required=True, help="WAV of a single clean 'Aria'")
    ap.add_argument("--music", default=None, help="WAV of background music (looped under the word)")
    ap.add_argument("--model", default="wakewords/aria.onnx", help="custom .onnx path or bundled name")
    ap.add_argument("--snrs", default="inf,20,10,5,0,-5,-10",
                    help="comma SNRs in dB (voice-to-music; negative = music louder)")
    ap.add_argument("--save-mix", type=float, default=None, metavar="SNR",
                    help="also write /tmp/wakeword_ab_mix.wav at this SNR (to listen to what the model hears)")
    args = ap.parse_args()

    model_path = _resolve_model(args.model)
    wake = _read_wav_16k_mono(args.wake)
    music = _read_wav_16k_mono(args.music) if args.music else np.zeros(0)
    snrs = [float("inf") if s.strip() in ("inf", "+inf") else float(s) for s in args.snrs.split(",")]

    print(f"model={os.path.basename(model_path)}  wake={args.wake} ({len(wake)/RATE:.2f}s)  "
          f"music={args.music or '(none)'}  frames={CHUNK}@{RATE}\n")
    print(f"{'SNR(dB)':>8} {'speex=off':>10} {'speex=on':>10}   note")
    print("-" * 48)
    for snr in snrs:
        mix = _build_mix(wake, music, snr)
        off = _peak_score(model_path, False, mix)
        on = _peak_score(model_path, True, mix)
        label = "inf" if np.isinf(snr) else f"{snr:+.0f}"
        note = "no music (baseline)" if np.isinf(snr) else ("music louder" if snr < 0 else "")
        print(f"{label:>8} {off:>10.3f} {on:>10.3f}   {note}")
        if args.save_mix is not None and (np.isinf(snr) and np.isinf(args.save_mix) or snr == args.save_mix):
            out = "/tmp/wakeword_ab_mix.wav"
            _save_wav(out, mix)
            print(f"          (wrote {out} — play it to hear what the model scores)")
    if not args.music:
        print("\n(no --music → only the baseline row is meaningful; pass --music to test over-music)")


if __name__ == "__main__":
    main()
