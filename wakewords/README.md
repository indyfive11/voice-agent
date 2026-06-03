# Custom wake-word models

Drop self-trained openWakeWord models here (e.g. `hey_aria.onnx`) and point the agent at them:

```bash
# in .env
WAKE_WORD=wakewords/hey_aria.onnx
```

`config._wake_model_paths` loads a `.onnx` path as-is, so a custom model works exactly like the
bundled pretrained ones (`hey_jarvis`, `alexa`, …). Comma-separate to load several.

## Training "hey aria" (~1 hr, Google Colab, no recordings needed)

1. Open the openWakeWord training notebook:
   https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb
2. Wake phrase: **"hey aria"** (two+ syllables — far lower false-accept rate than bare "Aria";
   that's why the pretrained set is all "hey X"/"alexa").
3. Export **ONNX** (our openWakeWord build is ONNX-only).
4. Save the file here as `hey_aria.onnx`.
5. Validate over music before wiring it in:
   `uv run python tools/wakeword_probe.py wakewords/hey_aria.onnx`
6. Set `WAKE_WORD=wakewords/hey_aria.onnx` in `.env` and relaunch. Tune `WAKE_WORD_THRESHOLD`
   (raise for fewer false fires, lower for fewer misses).

Optional: train a **custom verifier model** on your own voice (openWakeWord's second-stage filter) to
cut false activations from other speakers.

These models are small and self-trained, so they're versioned (see the `.gitignore` exception) —
the `*.onnx` ignore is for downloaded model *caches*, not these.
