# Nova wake-word listener

Listens on the default mic for "Hey Nova" / "Nova" (openWakeWord). On detection it streams the mic to
ElevenLabs Scribe, prints the transcript, and goes back to listening (standalone mode). With `--handoff`
it instead starts the Nova voice conversation on `../router/voice_server.py` (see **Demo setup**).
Standalone mode reads `ELEVENLABS_API_KEY` from `../router/.env` (or the environment, or
`NOVA_ENV_FILE=/path/to/.env`); `--handoff` / `--dry-run` never need the key.

```
uv sync
./run.sh                                     # demo: = uv run python main.py --handoff (needs voice_server running)
uv run python main.py --dry-run              # only print detections (no voice_server, no key)
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
| `handoff.py` | `VoiceServerClient` (HTTP) and `WakeService`: wake -> POST /voice/start -> release mic -> poll status -> cooldown -> resume |
| `main.py` | CLI |
| `run.sh` | one-command `--handoff` start |

Config (env, overridable by flags): `WAKEWORD_MODELS_DIR` (default `models/`), `WAKEWORD_THRESHOLD` (0.3),
`WAKEWORD_COOLDOWN_S` (2.0), `WAKEWORD_VOICE_URL` (`http://127.0.0.1:8765`). Endpointing: `--silence` (1.5 s) and `--rms-threshold` (400, int16 RMS; raise it
if your mic's noise floor keeps the utterance from ending, lower it if quiet speech is missed).

STT `--stt-mode auto` (default) streams in realtime and falls back to "record until silence, then batch
transcribe" if the realtime session fails. `realtime` / `batch` force one path.

## Demo setup (wake word -> Unity voice conversation)

Flow: this process listens for `hey_nova`; on a detection it `POST`s `/voice/start` to `router/voice_server.py`,
**closes the microphone** (voice_server opens the mic itself for the conversation, STT -> router -> TTS), polls
`GET /voice/status` until `ended` / `idle` / `error`, waits a cooldown, reopens the mic and listens again. Nothing here
talks to Unity.

**Unity visibility.** Staging's `VoiceSessionClient` checks `GET /voice/status` every `idleCheckSeconds` (default 5 s)
while idle and starts polling when it sees an active state, so a session started by this wake process shows up in Unity
within about 5 s (the orb/HUD may lag the first moments of the conversation). Lower `idleCheckSeconds` on the client for a
snappier demo. The mic, STT and TTS run in Python regardless.

**Mic ownership.** voice_server owns the microphone during a conversation, so before *every* mic open (startup, after a mic
error, after a conversation) this process calls `GET /voice/status` and, while the state is `loading` / `listening` /
`processing` / `processing` / `speaking`, keeps the mic closed and polls every second (sleeping, no hot loop). An unreachable server does not block the
mic. Residual race: if something else (e.g. the telephone gesture) starts a conversation *after* the status check while this
process already holds the mic, it is only noticed at the next periodic status check (about 1 s), so the two can briefly
compete for the device. Likewise `idle` after `/voice/stop` only means the server cleared its state, not that its worker
released the mic; voice_server refuses `/voice/start` (`started:false`) until the old worker is gone, so a start refused in a
non-busy state is retried with backoff (0.5 s doubling to 4 s, 8 attempts) with the mic closed, and only then given up on.
After our own `--conversation-timeout` stop the mic reopens after the cooldown without being able to tell whether the old
worker has finished with the device.

1. **Start voice_server** (needs its own venv and `ELEVENLABS_API_KEY` in `router/.env`; from the repo root):
   `router/.venv/bin/python -m router.voice_server` (listens on `127.0.0.1:8765`; `--no-speak` prints replies instead of speaking).
   Wait until `GET /voice/status` shows `"router_ready": true`.
2. **Plug in the model**: put `hey_nova.onnx` in `wakeword/models/` (the `.onnx` files are gitignored, so they are never committed;
   or pass `--model /path/to/hey_nova.onnx`). With no `.onnx` the program exits with "drop hey_nova.onnx into wakeword/models/".
3. **Start the wake word**: `cd wakeword && uv sync && ./run.sh` (same as `uv run python main.py --handoff`).
   The first run downloads openWakeWord's shared melspectrogram/embedding models (~2 MB) into the venv, so it needs internet once (timeouts apply; the files are validated and moved into place atomically, so a failed or truncated download is simply retried on the next start).
4. Say "hey nova". You should see `[wake] hey_nova (0.xx) - starting conversation`, then `conversation state: ...` lines.
   After the conversation the mic reopens (after `--handoff-cooldown`, default 3 s).

Options: `--voice-url URL` (non-default voice_server), `--conversation-timeout S` (300; then `POST /voice/stop`),
`--handoff-cooldown S` (3), `--cooldown S` (detector re-trigger guard, 2), `--dry-run` / `--no-handoff` (print detections only: use this to
tune the threshold without a server), `--wav FILE` (feed a 16 kHz mono WAV instead of the mic), `-v`.
If a conversation is started by something else (e.g. the Unity telephone gesture) the mic is released for its duration too.

**Threshold tuning.** Default `--threshold 0.3`. The first demo model has roughly 0.46 recall and ~0.7 false wakes/hour at
a threshold around there: lower (0.2) catches more "hey nova" but wakes on noise more; raise (0.5+) for fewer false wakes but you
will have to repeat yourself. Try `--dry-run --threshold X` and say the phrase / run the room's usual noise to pick a value.

**Mic selection.** `uv run python main.py --list-devices`, then `--device INDEX_OR_NAME` (`./run.sh --device 3`). Note voice_server records
from the *system default* input device (it has no device option), so make the same mic the default in your OS if you pick one here.

**RMS note.** `--rms-threshold` (int16 RMS, default 400) only affects standalone mode's end-of-utterance detection. In handoff mode
voice_server does the endpointing, so it is irrelevant here; if standalone utterances never end, your mic noise floor is above 400, raise it.

**Troubleshooting.**
- `voice_server not reachable ... Connection refused`: start voice_server (step 1) or fix `--voice-url`. Wake words are retried 3x with
  1 s / 2 s backoff, then dropped; listening continues.
- `microphone error ... reopening in N s`: wrong/busy/unplugged device; the mic is retried with backoff (1 s doubling to 30 s). Check `--list-devices`.
- Nothing happens on "hey nova": run with `--dry-run -v`, lower `--threshold`, check the mic level/device, speak 0.5-1 m from the mic.
- It wakes by itself: raise `--threshold`.
- After a conversation it does not wake: wait out `--handoff-cooldown`; check `GET /voice/status` is not stuck in `listening`
  (`--conversation-timeout` then stops it).

Not covered by tests: a real mic, a real voice_server conversation and Unity (tests use a fake HTTP server and fake mic; a `hey nova` WAV was also run through
the real `router.voice_server` HTTP layer with a fake `converse()`).

## Placeholder mode

Custom models don't exist yet. `--placeholder` loads the pretrained **`hey_jarvis`** model (downloaded with
`openwakeword.utils.download_models`) so wake -> STT -> print can be tested end to end. It prints a
`PLACEHOLDER MODE` banner and is not the real wake word. Without `--placeholder`, `main.py` refuses to start
if `models/` has no `.onnx` files.

## Training "hey_nova" and "nova" (see `train_notes.md`)

Short version: use the openWakeWord automatic-training Colab, one run per phrase, download the resulting
`.onnx`, and drop it in `wakeword/models/` as `hey_nova.onnx` / `nova.onnx`. Every `.onnx` in that folder is
loaded; the file stem is the model name shown in `[wake]` lines.
