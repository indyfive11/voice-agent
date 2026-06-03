"""Standalone wake-word probe — validates openWakeWord detection over playing music, live.

Run BEFORE wiring wake-word into the pipeline: it captures the same AEC Mic the agent uses (via
PipeWire/`parec`, which also resamples to 16 kHz natively) and prints live "hey jarvis" detection
scores. Play music, say "hey jarvis" over it, and watch whether the score crosses the threshold
reliably — the make-or-break assumption for the wake-word-gated architecture (detection must survive
over media even though full STT doesn't).

    uv run python tools/wakeword_probe.py            # hey_jarvis, threshold 0.5
    uv run python tools/wakeword_probe.py alexa 0.4  # different pretrained model / threshold

Capture is done by `parec` (PulseAudio/PipeWire) in a subprocess — no PyAudio/soxr in-process (they
crash next to onnxruntime). Ctrl-C to quit. Hand-to-the user (needs the mic; no mic in CI).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

import openwakeword
from openwakeword.model import Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "hey_jarvis"
THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
SOURCE = os.environ.get("AEC_SOURCE", "echo-cancel-source")  # the AEC'd PipeWire source
RATE = 16000
CHUNK = 1280  # samples @16k = 80ms (openWakeWord's preferred frame)


def main() -> None:
    pkg = os.path.dirname(openwakeword.__file__)
    model_path = os.path.join(pkg, "resources", "models", f"{MODEL}_v0.1.onnx")
    if not os.path.exists(model_path):
        sys.exit(f"no pretrained model {MODEL!r} at {model_path}")
    oww = Model(wakeword_model_paths=[model_path], vad_threshold=0.0)  # probe raw scores
    key = next(iter(oww.models.keys()))
    print(f"model: {key}  threshold={THRESHOLD}  source={SOURCE}  (say '{MODEL.replace('_', ' ')}')")

    cmd = [
        "parec", f"--device={SOURCE}", f"--rate={RATE}",
        "--channels=1", "--format=s16le", "--latency-msec=80",
    ]
    print("capture:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print("\nlistening — play music and say the wake word over it. Ctrl-C to quit.\n")

    nbytes = CHUNK * 2
    rolling_max = 0.0
    last_print = 0.0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                print("capture ended (is the source name right? try AEC_SOURCE=...)")
                break
            pcm = np.frombuffer(buf, dtype=np.int16)
            score = float(oww.predict(pcm).get(key, 0.0))
            rolling_max = max(rolling_max, score)
            now = time.time()
            if score >= THRESHOLD:
                print(f"  *** DETECTED  score={score:.3f}")
                rolling_max = 0.0
            elif now - last_print >= 1.0:  # heartbeat: shows it's alive + the noise floor
                print(f"  ... max(1s)={rolling_max:.3f}")
                rolling_max = 0.0
                last_print = now
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
