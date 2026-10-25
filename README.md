# Nova

**Nova is an offline-first voice assistant for astronaut gear.** Say "Hey Nova", ask a question, and a small on-device classifier decides whether to run a tool (oxygen, navigation, map, data, safety) or hand the question to a language model. The answer is spoken back. A Unity front end for Meta Quest 3 shows a simulation of the HUD, map and terrain.

Built by team **Jeviathans**

## Introduction

An astronaut's hands are full and connectivity is unreliable, so the assistant has to be hands-free, quick and predictable. Nova's design keeps the decision-making small and local, and uses a larger model only where it is needed.

```
"Hey Nova" -> wake word (openWakeWord, local)
                  |
                  v
           speech to text (ElevenLabs Scribe)
                  |
                  v
        ModernBERT intent classifier (local, 6 classes)
        confidence score on every prediction
          |                |                  |
     >= 0.9            0.5 - 0.9            < 0.5 or "llm"
          |                |                  |
     run the tool    clarifying question   Gemini Flash
          |                                   |  second tool check,
          |                                   |  then a short reply
          v                                   v
      template reply ---------------->  number verifier
                                              |
                                              v
                                  text to speech (ElevenLabs)
```

**Speech-to-Speech Pipeline**

- **Wake word:** openWakeWord with a custom `hey_nova` model, run through ONNX.
- **Speech to text:** ElevenLabs Scribe (realtime, with a batch fallback).
- **Router:** ModernBERT-base fine-tuned as a six-way classifier: `call_safety`, `call_navigation`, `call_map`, `call_data`, `call_o2` and `llm`. Everything else collapses into `llm`. Temperature scaling calibrates the confidence scores, and two thresholds (0.9 and 0.5) decide whether to act, ask or defer.
- **Tools:** Python tools read simulated telemetry (or PostgreSQL). Replies are fixed templates, and a verifier checks that every spoken number comes from the tool result. The data is meant to be similar to live streamed sensor data.
- **LLM fallback:** Gemini Flash-Lite answers general questions and double-checks whether a tool applies. It replies in at most two short sentences and never writes numbers itself, because code fills them in.
- **Safety scope:** Medical, health and safety-related questions are not answered by the LLM. They get a fixed "Please contact base." reply, so there is no risk of a hallucinated answer.
- **Text to speech:** ElevenLabs speaks every line.

## Purpose

- Show that a small, fast classifier can route spoken requests to tools on constrained, offline-capable hardware.
- Avoid LLM hallucinations in a setting where wrong numbers are dangerous: tools produce the facts, the LLM only handles open-ended questions, and it is blocked from medical and safety topics.
- Give the router a confidence score so that it acts, asks or defers, instead of guessing.
- Demonstrate the full loop, from voice to tool to spoken answer, in a Unity HUD on Quest 3.

## Running

Requirements: Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), Unity 6000.3.9f1 for the headset front end, and a microphone and speaker.

### 1. Install and configure

```sh
uv sync --project router --locked
```

Create a `.env` file in the repo root (it is gitignored, never commit it):

```
ELEVENLABS_API_KEY=...
GEMINI_API_KEY=...
```

Model files live outside git. Point to them with `NOVA_MODELS_DIR` (default `/home/isaach/models/Nova/router`) or pass `--model`. The ModernBERT-base download script is `router/download_base.py`.

### 2. Try it without a microphone

Text-only REPL over the full router, tools and LLM stack:

```sh
cd router
uv run python -m router.demo_live                       # real checkpoint
uv run python -m router.demo_live --fake-router         # keyword router, no checkpoint needed
uv run python -m router.demo_live --fake-router --no-llm  # fully offline
```

### 3. Voice demo

Start the voice server (add `--no-llm` for template replies only, `--datasource postgres` for database telemetry):

```sh
uv run python -m router.voice_server --model "$NOVA_MODELS_DIR/<checkpoint>"
```

Start the wake-word listener in its own environment. It needs `wakeword/models/hey_nova.onnx`, which is gitignored:

```sh
cd wakeword
./run.sh
```

Say "Hey Nova", then ask something like "check my oxygen", "take me to the airlock" or "why is the sky grey?".

### 4. Unity

See [Unity setup](#unity-setup) below and the [router guide](router/README.md) for the full PostgreSQL and Windows steps.

### Tests

```sh
NOVA_MODELS_DIR=/home/isaach/models/Nova/router uv run --project router pytest router/tests
```

Some tests assume the older eight-class labels and need updating (see limitations).

## Known limitations

This is a demo. Please read these before relying on any number in it.

- **Not tested on real speech at scale.** The end-to-end voice path works in our own testing, but accuracy and error rates have not been measured on varied speakers, accents or noisy environments.
- **Synthetic training data.** The classifier was trained on template-generated phrasings with simulated speech noise. Held-out synthetic data scores about 0.94 macro-F1. That is not a real-world accuracy number.
- **Overconfident mistakes still happen.** On dev data about 1.25% of examples at the 0.9 threshold were confidently wrong, above our 0.5% target. The confidence temperature and thresholds were tuned on the same dev set, so these numbers are optimistic. The 0.5 to 0.9 band is unmeasured.
- **Not fully offline.** Only the wake word and router run locally. Speech to text, text to speech and the Gemini fallback need internet.
- **The local LLM and document search were cut.** A local Qwen model and an SQLite document search were replaced by the Gemini API and are not implemented. Retrieval is only an injection point.
- **Gemini is lightly tested.** The Gemini path is configured but has not been run extensively. The model is stateless, with no conversation memory.
- **No barge-in.** Nova can't be interrupted while speaking.
- **All safety values are placeholders.** Alert tiers, feasibility numbers and the risk-term list are demo values awaiting expert sign-off. This is not safety-qualified software.

## Future scope

- Collect and test on real push-to-talk and noisy-environment speech, with multiple speakers. Report per-class accuracy and the high-confidence error rate.
- Separate calibration data from dev data, and derive the thresholds properly.
- Build the arbiter for the mid-confidence band and measure it.
- Export the router to ONNX INT8 and measure latency and memory on target hardware.
- Compare against other backbones such as DistilBERT, MiniLM and ModernBERT-large, with several seeds.
- Bring back a local LLM so the whole stack can run without a network, and replace cloud speech with on-device speech models.
- Add a document search over trusted sources (SQLite or RAG) for vehicle and base maintenance, and for general astronomy questions, so answers rely on cited material. Keep medical and safety topics blocked.
- Support multi-intent requests, conversation memory and interrupting Nova mid-reply.
- Replace placeholder safety values with ones reviewed by domain experts.
- Finish the Unity model of vehicles and bases, and test standalone performance on Quest 3.

## Unity setup

## Open the project

1. Use Unity 6000.3.9f1
2. Open the repo in Unity Hub
3. Open `Assets/Scenes/SampleScene.unity`
4. Press Play

For Quest Link connect the headset and use the Windows build profile
For standalone Quest use the Android profile and Build and Run
Standalone voice audio is not connected yet

## Jezero terrain

Terrain files are generated locally and stay out of Git

1. Get `M20_JezeroCrater_CTXDEM_20m.tif` from [USGS](https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic)
2. Put it in `Data/Source/`
3. Install the conversion tools

```sh
pip install numpy pillow
python Tools/convert_jezero.py
```

4. Run **Tools > Jezero > Import Terrain** in Unity

If terrain is missing add `Assets/JezeroTerrain/Generated/Jezero Terrain.prefab`
Use position `(-25494, 2567, -51220)` with zero rotation and unit scale
The player stays near the world origin to reduce tracking jitter
Terrain data comes from USGS Astrogeology and Mangold et al 2021

## HiRISE area

Get the [elevation model](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_dtm_mosaic) and [surface image](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic) from USGS
Keep their filenames and put both in `Data/Source/`

```sh
python Tools/convert_hirise.py
```

Run **Tools > Jezero > Import HiRISE Test Area**
It updates the existing terrain prefab so do not add another copy
The patch covers about 557 by 633 meters around spawn
It uses 1 meter elevation data and tinted 25 centimeter grayscale imagery
The surrounding landscape keeps the CTX terrain
Run **Import Terrain** to return to CTX only

Use **Tools > Jezero > Build Surface Detail** to rebuild dust and gravel and rocks
Adjust **Jezero Rock Field** on **HiRISE Test Area** then rebuild
The default rock radius is 80 meters
Only boulders have colliders
Textures and rocks are procedural detail rather than mapped Martian features
Generated terrain assets stay in `Assets/JezeroTerrain/Generated/`

## Controls

- Left joystick to move
- Right joystick to snap turn
- WASD and mouse for desktop testing
- Left pinch to open the wrist computer
- Right pinch to open the hand map
- Telephone pose or V to toggle voice

## Voice setup

ModernBERT picks tool routes and its checkpoint files stay outside Git
Pass your local checkpoint path when starting the server
Gemini writes replies and backs BERT up when it isn't sure which tool to use
Add `--no-llm` to the server for offline replies only
Keep shared keys in the root `.env`

The optional wake listener hears Hey Nova and starts the same Unity voice session
It needs a separate environment and the custom `hey_nova.onnx` model

[Router and Gemini setup](router/README.md)

[Wake word on Windows](wakeword/README.md#windows-setup)

## Shared readings

Keep shared keys and settings in the root `.env`
The HUD and wrist use the same PostgreSQL telemetry as NOVA
Keep `simulator.py` and the voice server running with `--datasource postgres`
Missing or stale readings show **TELEMETRY UNAVAILABLE**
Use the wrist oxygen buttons for demos
Refill sets oxygen to 100 percent and the simulator keeps consuming it

[Voice setup and oxygen demo](Assets/Scripts/Voice/README.md)
[Minimap](Assets/Scripts/Minimap/README.md)
[Navigation](Assets/Scripts/Navigation/README.md)
[Hand gestures](Assets/Scripts/Gestures/README.md)

## Testing

Check readability and tracking in Quest Link
Check terrain edges and boulder collision
Standalone performance still needs testing on Quest 3
The target is 72 FPS

For a local HUD demo turn off **Use Server Telemetry** on **Suit Telemetry**
Leave **Simulate** enabled and use F8 to cycle demo alerts
Prefabs are in `Assets/HelmetHUD`

## Use of AI in Development

Throughout the creation of the project, Generative AI was used for: writing the generator and templates for our synthetic training data, creating documentation (not including this section), expanding upon human written code, refactoring code, and code reviews.

AI was not used for: major design decisions (architecture, tool set, thresholds, safety scope), closing PRs, or any video or image generation. The product itself uses ModernBERT, Gemini and ElevenLabs at runtime, as described above.
