# Nova router

ModernBERT chooses the route for a request
Python tools read telemetry and the local LLM handles supported conversation
The voice server connects this to Unity

## Setup

Use Python 3.12 or newer
From the repo folder install the locked environment

```sh
uv sync --project router --locked
```

Keep model files outside the repo
Use `NOVA_MODELS_DIR` or `--model` to select a checkpoint
A base model without a trained classifier head reports `untrained`
Keep the trained label order matched to `labels.json`

Put keys and database settings in the repo-root `.env` (next to `router/`)
Never commit that file
Microphone and Windows playback need `sounddevice`

## Run with Unity

Start PostgreSQL and Ollama first
Run these in separate terminals from the repo folder

```cmd
.\router\.venv\Scripts\python.exe simulator.py
```

```cmd
.\router\.venv\Scripts\python.exe -m router.voice_server --model "C:\Users\cardin\models\Nova\router\ModernBERT-v4.2" --datasource postgres
```

Change the checkpoint path for your machine
Keep the simulator running so readings stay fresh
Use `--datasource sim` for voice without PostgreSQL
That mode does not supply the shared HUD telemetry

The server uses localhost on port 8765
Unity reads voice status and telemetry from it
Wrist demo buttons update the simulator oxygen value through the server
No database credentials go to Unity

## Demo controls

The simulator consumes oxygen at the mission time scale
Use the wrist **REFILL** button for a normal conversation demo
Use **-10 O2** and **+10 O2** to show changing readings
Critical readings take priority over conversation
Alert thresholds and response wording live in `turn_policy.json`
They are demo settings and have not been validated for a real suit

Restart the simulator and server after code updates
Do not use `--new-mission` unless you want to end the active mission and start another
Existing mission history is kept

## Checks

```cmd
.\router\.venv\Scripts\python.exe -m pytest router/tests/test_datasource.py router/tests/test_voice_server.py router/tests/test_db_config.py
```

These checks do not use the microphone or paid speech APIs
The full suite also contains model and release checks
Passing synthetic tests does not establish performance on real speech
Keep released datasets and their provenance records unchanged

## More detail

[Unity voice and wrist setup](../Assets/Scripts/Voice/README.md)
[Local LLM demo](docs/live-demo.md)
[Response generation](docs/post-tool-generation.md)
[Reference datasets and units](docs/reference-data.md)
[Trip clarification](docs/trip-clarification-implementation.md)
[Dataset handoff](docs/v3-trip-handoff.md)

Use `python -m router.train --help` for training options
Use `python -m router.evaluate --help` for evaluation options
Training and model promotion are separate from running this demo
