# Training custom wake-word models

openWakeWord has no pretrained "hey nova" / "nova". Train them (synthetic speech, no recordings needed).

## Steps (per phrase: run once for `hey nova`, once for `nova`)

1. Open the openWakeWord automatic training notebook (GitHub: `dscripka/openWakeWord` -> README ->
   "Training New Models" -> `notebooks/automatic_model_training.ipynb`, "Open in Colab"). Use a GPU runtime.
2. In the config cell set `target_phrase` to `["hey nova"]` (or `["nova"]`) and `model_name` to `hey_nova`
   (or `nova`). Phonetic spelling tricks (e.g. "hey no-va") can be tried if the TTS mispronounces it.
3. Raise `n_samples` (e.g. 20000+) and `steps` for better quality; keep `target_false_positives_per_hour`
   low (0.2-0.5) because "nova" is a common word. Run all cells (~1 h on a free GPU).
4. Download the output `.onnx` (the notebook also writes a `.tflite`; we run onnx, ignore the tflite).
5. Save it as `wakeword/models/hey_nova.onnx` / `wakeword/models/nova.onnx` (any `*.onnx` there is loaded).
6. Run `uv run python main.py` and tune `--threshold` (start 0.5; lower for misses, raise for false
   wakes). Say the phrase from normal distance, plus ~10 minutes of ordinary talk/TV to check false wakes.

## Notes

- A one-word wake word ("nova") will false-trigger far more than "hey nova"; consider shipping only
  `hey_nova` and using `nova` with a higher threshold, or as a second opinion.
- Optional quality step: a custom verifier model (see openWakeWord docs) trained on your own voice.
- Models need 16 kHz mono int16 input, which is what `listener.py` produces; no extra config.
