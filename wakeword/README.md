# Nova wake-word listener

Listens on the default mic for "Hey Nova" / "Nova" (openWakeWord). On detection it streams the mic to
ElevenLabs Scribe, prints the transcript, and goes back to listening. Startup framework only: nothing
downstream of the transcript yet. Independent of `../router`; the only coupling is reading
`ELEVENLABS_API_KEY` from `../router/.env` (or the environment, or `NOVA_ENV_FILE=/path/to/.env`).

```
uv sync
uv run python main.py --placeholder          # pipeline test: say "hey jarvis", then speak
uv run python main.py                        # real run: needs models/hey_nova.onnx and/or models/nova.onnx
uv run python main.py --list-devices
uv run python main.py --placeholder --wav utterance.wav --max-wakes 1   # no mic: 16 kHz mono 16-bit WAV
uv run pytest                                # offline: no mic, no network
```

## Layout

| file | role |
|---|---|
| `detector.py` | `WakeDetector`: openWakeWord `Model` (onnx) + score threshold + cooldown |
| `listener.py` | `MicFrames` (16 kHz mono int16, 80 ms = 1280-sample frames), `WavFrames`, `Listener` loop |
| `stt.py` | `Transcriber`: Scribe realtime websocket (`scribe_v2_realtime`, manual commit), batch fallback (`scribe_v2`); `SilenceEndpointer` (1.5 s) |
| `config.py` | audio constants, env config, API key loading |
| `main.py` | CLI |

Config (env, overridable by flags): `WAKEWORD_MODELS_DIR` (default `models/`), `WAKEWORD_THRESHOLD` (0.5),
`WAKEWORD_COOLDOWN_S` (2.0). Endpointing: `--silence` (1.5 s) and `--rms-threshold` (400, int16 RMS; raise it
if your mic's noise floor keeps the utterance from ending, lower it if quiet speech is missed).

STT `--stt-mode auto` (default) streams in realtime and falls back to "record until silence, then batch
transcribe" if the realtime session fails. `realtime` / `batch` force one path.

## Placeholder mode

Custom models don't exist yet. `--placeholder` loads the pretrained **`hey_jarvis`** model (downloaded with
`openwakeword.utils.download_models`) so wake -> STT -> print can be tested end to end. It prints a
`PLACEHOLDER MODE` banner and is not the real wake word. Without `--placeholder`, `main.py` refuses to start
if `models/` has no `.onnx` files.

## Training "hey_nova" and "nova" (see `train_notes.md`)

Short version: use the openWakeWord automatic-training Colab, one run per phrase, download the resulting
`.onnx`, and drop it in `wakeword/models/` as `hey_nova.onnx` / `nova.onnx`. Every `.onnx` in that folder is
loaded; the file stem is the model name shown in `[wake]` lines.
