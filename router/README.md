# Nova router

The active contract is the [six-class router and voice path](#six-class-routing-and-voice-sessions)
(`call_safety`, `call_navigation`, `call_map`, `call_data`, `call_o2`, `llm`). It needs a newly
trained head at `$NOVA_MODELS_DIR/binary`; old binary/eight-class checkpoints are incompatible.
The sections below preserve historical v1/v2 results and architecture notes; their distinct
OOD/NO_TOOL/call_llm policies are superseded by the six-class section. The base model alone has
a random head flagged `untrained: true`. Checkpoints are git-ignored.

Output of `Router.predict(text)`:
`{label, confidence, topk:[{label,p}], latency_ms, untrained, warning?}` - softmax probs
(optionally divided by `temperature`). Runs on CPU; `device='auto'` picks MPS (Mac), then CUDA, then CPU.

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

Put keys and database settings in the root `.env`
Never commit that file
Microphone and Windows playback use `sounddevice` which is included in the router dependencies
To update an existing Windows venv run `.\router\.venv\Scripts\python.exe -m pip install -e .\router` from the repo folder

## Run with Unity

Start PostgreSQL first
Run these in separate terminals from the repo folder

```cmd
.\router\.venv\Scripts\python.exe simulator.py
```

```powershell
.\router\.venv\Scripts\python.exe -m router.voice_server --model "$HOME\models\Nova\router\ModernBERT-v4.2" --datasource postgres
```

This example uses the local Windows checkpoint
Change `--model` if your checkpoint is somewhere else
The server defaults to `sim` when `--datasource` is omitted
Server `--device` selects the torch device rather than the mic
`--timeout` sets how many seconds of silence end a conversation
Keep the simulator running so readings stay fresh
Use `--datasource sim` for voice without PostgreSQL
That mode does not supply the shared HUD telemetry

The server uses localhost on port 8765
Unity reads voice status and telemetry from it
Wrist demo buttons update the simulator oxygen value through the server
No database credentials go to Unity

## BERT model

ModernBERT chooses tool routes and Gemini does not retrain it
Model files stay outside Git and need to be copied separately
Pass the trained checkpoint folder with `--model`
Its label order must match `router/labels.json`
Without `--model` the server uses `NOVA_MODELS_DIR/v1`
The current fallback is `/home/isaach/models/Nova/router/v1` which is a Linux path

ModernBERT-v4.2 is still marked as a candidate in its metadata
Using it does not promote or retrain the model

Check readiness from PowerShell after starting the server

```powershell
Invoke-RestMethod http://127.0.0.1:8765/voice/status
```

Wait for `router_ready` to be true
A failed load reports `router_error`

## Gemini

Keep `GEMINI_API_KEY` and `ELEVENLABS_API_KEY` in the root `.env`
Each teammate needs their own local file
Ollama is optional and is not needed for the Gemini setup

The voice server uses Gemini for replies and as a second routing layer
Gemini can choose a tool when BERT does not
Critical alerts bypass this second routing layer
Gemini is checked when each conversation starts and falls back to offline replies if it's down
Add `--no-llm` to the server to skip Gemini entirely
Model names and routing policy in `turn_policy.json` are still provisional

Voice and wake word can use another file through `NOVA_ENV_FILE`
The simulator and Stage 1 API still use the root `.env`
Environment variables take priority over file values

## Wake word

The listener starts a conversation through the same voice server used by Unity
It releases the mic during the conversation and resumes afterward
The custom `hey_nova.onnx` model is not in Git
See [wake-word setup](../wakeword/README.md#windows-setup) for the separate environment and Windows commands

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
[LLM demo details](docs/live-demo.md)
[Response generation](docs/post-tool-generation.md)
[Reference datasets and units](docs/reference-data.md)
[Trip clarification](docs/trip-clarification-implementation.md)
[Dataset handoff](docs/v3-trip-handoff.md)

Edit `labels.json` (`{"labels": [...]}`). Order = class index; the head size is `len(labels)`, so a fine-tuned
checkpoint is only valid with the label order it was trained on. Current labels are **placeholders**
(`call_safety ... call_x`, `NO_TOOL`, `OOD`) to be replaced by the real 10 classes (get_alerts/get_state unmerged).

## Trained vs untrained marker

`Router.load` looks for `<model_dir>/router_meta.json`. `{"trained": true, ...}` clears the warning; missing or
`false` => untrained (the default for the base model). `train.py` writes it (with the temperature fitted on dev).

## Data, training, results

`data/gen_synthetic_v1.py` reproduces the frozen v1 `data/{train,dev,test}.jsonl` byte-for-byte
(200/40/40 rows per label). v1 generator: many
phrasings per class (verbs `take/bring/get/lead/guide/walk me`, `go/head/walk`, politeness wrappers, word order,
with and without `to/the/my`, contractions, fillers, STT-style slips and typos); a **shared place list**
(home, homebase, home base, base, habitat, airlock, rover, lander, crater, ridge, landing pad ...) is substituted
into navigation, map, safety, O2, NO_TOOL ("I'm home") and OOD ("fix the wifi at home") templates so a place word
never decides the class; `call_llm` covers general space/science/how-it-works questions, `OOD` is unrelated topics
(food, sports, Earth weather, shopping, gibberish). v1 uses literal-disjoint template pools, **not** disjoint expanded frames or slot-fill families. Texts equal to a line
of `demo_calls*.json` or `paraphrase_test.jsonl` are never emitted (those files are read for exclusion only).

`data/paraphrase_test.jsonl` is 64 hand-written utterances (8 per class, incl. `take me home`, `take me to home`,
`go to home`, `directions to homebase`) kept as a **regression/development set**, not independent validation; `demo_calls*.json` likewise.

Trained checkpoints are kept **outside git**: `$NOVA_MODELS_DIR` if set, else `/home/isaach/models/Nova/router`
(`router.core.models_dir()`). `train.py` (`--out-dir`), `pipeline.py`/`evaluate.py`/`eval_sets.py` (`--checkpoint`) and
`demo.py` (`--model`) default to that directory; pass an explicit path to override per call. Never write checkpoints
into the repo (`checkpoints/` stays in `.gitignore` only as a guard). The base model stays in `$NOVA_MODELS_DIR/ModernBERT-base`, outside git.

```sh
export NOVA_MODELS_DIR=${NOVA_MODELS_DIR:-/home/isaach/models/Nova/router}
mkdir -p "$NOVA_MODELS_DIR"
uv run python data/gen_synthetic_v1.py  # reproduces v1 data; no training needed
uv run python data/gen_synthetic.py     # writes data/v2; does not touch v1
# v2 was already trained once (2 epochs); do not rerun training for this review.
uv run python -m router.eval_sets --checkpoint "$NOVA_MODELS_DIR/v1" --show-errors   # --checkpoint defaults to $NOVA_MODELS_DIR/v1
```

Accuracy (argmax, calibrated temperature; 2 epochs, 1,200 rows for v0 vs 1,600 for v1, one run each):

| set | n | v0 | v1 |
|---|---|---|---|
| old test (v0 generator) | 240 | 229 (95.4%) | 234 (97.5%) |
| new test (v1 generator) | 320 | 270 (84.4%) | 306 (95.6%) |
| paraphrase_test (hand-written) | 64 | 54 (84.4%) | 64 (100%) |
| demo_calls | 15 | 15 | 15 |
| demo_calls_47 | 47 | 45 (95.7%) | 47 (100%) |

v0 failed `take me home`, `bring me back home`, `go to home` (-> NO_TOOL) and put `why is the sky red on mars`,
`why does mars look red` in OOD; v1 fixes all of them. Caveats: paraphrase_test and the demo sets are tiny
(64/47 rows, so +-5 points of noise) and were looked at while building v1 (I did not train on them, but I knew
the failures v0 had); dev/test share the generator family with train. Known v1 gaps: verbless fragments such as bare
`back to the base` (predicted NO_TOOL at 0.55), very short gibberish (`zxqv` -> NO_TOOL at 0.62), and
`okay im home please` (NO_TOOL vs navigation is ambiguous). None of this is evidence for the 95% target on real speech.

## Place resolver (`places.py`, `places.json`)

`resolve_place(text, threshold=0.85) -> {place_id, matched_text, score} | None`, also `from router import resolve_place`.
Gazetteer `places.json` maps canonical id -> aliases (`homebase`: home, home base, base, base camp ...). Text is
normalised, 1..N-word windows are compared to aliases with `difflib` on space-free strings (so `home base` ==
`homebase`, `homebse` ~ 0.93). Windows containing stopwords (`to the my back over ...`) are skipped; a non-exact match
must share its first letter with the alias (`over` != `rover`, `bridge` != `ridge`). Ties: higher score, longer window,
earlier position, then alphabetical id. `creator` returns **None**, not `crater`, at the default threshold (covered by a regression test).

## Pipeline stub (`pipeline.py`, `pipeline_config.json`)

```sh
uv run python -m router.pipeline --text "take me home" --text "hi"       # JSON line per utterance
```

`Pipeline(router, llm_fn=..., arbiter_fn=...).handle(text)` uses calibrated top-1 confidence and
`pipeline_config.json` (default `tau_hi=0.90`, `tau_lo=0.50`):

- Tool top class, `c >= tau_hi`: tool route (no real tool execution).
- Tool top class, `tau_lo <= c < tau_hi`: injectable `arbiter_fn(text, topk)` gets the top **three**
  `{label,p}` entries and may return one of those labels or `"none"`. Only agreement with the router's top
  label permits a tool route. Disagreement, none, absent arbiter, or arbiter exception emits a deterministic
  clarification question, never another tool. **The LLM cannot override the router.** No real arbiter is wired.
- `c < tau_lo`: existing LLM-route clarification policy. NO_TOOL/call_llm/OOD otherwise keep their distinct
  policies, including the out-of-scope template for OOD. `llm_fn` defaults to a placeholder, not a real call.

Navigation/map use `resolve_destination`, not the single-best-match `resolve_place`: extract role-bearing
spans (to/toward/back to; from = origin; not/instead of = exclusion) **before fuzzy matching**. Thus
`go from the rover to the airlock` resolves airlock and `to the rover not home` resolves rover.
Several distinct remaining destinations produce `route:"clarify"`, `reason:"ambiguous_place"` and a
fixed slot question, never a positional guess. This is a small deterministic grammar, not a general parser;
unknown/unmatched places still leave args empty, as do map zoom requests without a place.

## Turn wrapper (`turn.py`, `tools.py`, `verifier.py`, `turn_policy.json`, `demo_turn.py`)

```sh
uv run python -m router.demo_turn            # 16 scripted turns through the real v1 router (CPU, ~8 s)
uv run python -m router.demo_turn --fake     # same turns with a scripted fake router
```

`TurnHandler(router, alert_source, registry, llm_fn=..., arbiter_fn=...).handle(text)` order: (1) poll
`alert_source()` (sim snapshot) - a **critical** alert speaks a fixed, snapshot-verified template first, then normal read-only routing continues.
No interim/LLM generation, menus, refill/print offers or messages are allowed while it persists; greetings and
scene diagnostics still work. Pending offers/clarifications clear. `lines` holds alert then answer, each independently
verified against its own source; `spoken` joins them with a space for older clients. A new/changed critical alert
at the final re-poll replaces the answer. The alert repeats on every utterance; (2) `Pipeline.handle`; (3) required-slot check: navigation/map without a resolved place
-> deterministic slot question, ambiguous place -> "which place" question, never a tool call (`pipeline.py` now returns
`clarify/missing_slot` for a tool whose `required_slots` are unfilled; note map now requires a place, so "zoom out"
alone asks where); (4) at most one call through `ToolRegistry` (max 8 tools, 1 call per turn, fixed schemas; six fake
tools over a fabricated `SimState`, the sixth being `send_message`, which the router can never select); (5) template from the tool_result; low confidence and OOD get template replies, only
call_llm/NO_TOOL use the injectable `llm_fn` (default stub); (6) `verifier.verify_numbers` on every spoken line - a
number not in the source (tool_result; the sim snapshot for alerts; nothing for LLM output) replaces the line with a
safe fallback. All wording, alert tiers, sim thresholds and alert lines live in `turn_policy.json`, **every value a
PLACEHOLDER pending user sign-off** (the loader refuses unlabelled sections). Mid-band clarification wording still comes
from `pipeline_config.json`. Demo note: on `Get me the coolant level.` v1 gives navigation/.896 (clarify) while v2 gave call_data/.751.

### Turn follow-up: return-to-home feasibility, refill-call offer, LLM scope (2026-10-24)

**Every new value below is a PLACEHOLDER in `turn_policy.json` pending user sign-off** (sections `feasibility`,
`confirmation`, `send_message`, `llm_scope`); nothing is validated against a real suit, EVA rule or document.

- **Feasibility** (`turn.assess_feasibility`, navigation only): outbound = ceil(distance_m / 60 m/min); return = the same
  distance back (0 when the destination is in `home_places`, i.e. `homebase`); required = outbound + return + 10 min
  reserve; O2 minutes remaining = the sim's `minutes_remaining` (floored; fallback `o2_pct` x 2.4 min/% if absent; neither
  present -> no claim at all). Outcome: `cannot_reach` (remaining < outbound) | `cannot_return` (remaining < outbound +
  return) | `tight` (remaining < required) | `comfortable`. All numbers are whole minutes written into
  `tool_result["data"]["feasibility"]`, so the verifier checks the spoken line against them. The route line is always
  spoken and the tool always runs: turn.py reports and offers, the safety loop owns go/no-go. Wart: the fabricated
  `eta_min` in the fake navigation table can differ by 1 min from the computed outbound leg (airlock, landing_pad,
  ice_shelf, dune_field); a real navigation tool should supply one number.
- **Refill-call offer** (stateful): `tight`/`cannot_return`/`cannot_reach` append "Do you want me to call <place> and
  check if they have oxygen refills?" and store `TurnHandler.pending_confirmation` (only if that line was actually
  spoken). Next turn, after the alert poll and before the router, a deterministic matcher (word lists in policy; every
  word must be a yes/no/filler word, both at once = neither) answers it: yes -> exactly one `send_message(recipient=
  destination, urgency, message)` with urgency from `send_message.urgency_by_outcome` and a template message (numbers
  verified against the feasibility data first; an unverifiable message is not sent), no -> ack, anything else ->
  pending cleared and normal routing. Every turn consumes the pending offer; it also expires after
  `confirmation.expiry_turns` (2) turns. A critical alert clears it. Not built: a reply channel (`send_message` is a
  fake that returns "queued"; nobody answers).
- **LLM scope** (placeholder policy, no documents exist): scope is decided by **router class + retrieval score + risk
  terms, never by the LLM judging itself.** `NO_TOOL` text containing a risk term (prefix match: oxygen, pressure, suit,
  medical, radiation, dose, emergency, battery, repair, ...) gets the "I can't answer that from memory" template and no
  LLM call; other chitchat reaches the stub ungrounded, output number-verified. `call_llm` is grounded only: the
  injectable `retriever(text) -> [{"text","score"}]` (none supplied, an error, or nothing >= `retrieval_min_score` 0.5
  -> "I don't have a document for that yet.", no LLM call); otherwise the top 3 passages go to `llm_fn(text, policy,
  passages=[...])` and the verifier also accepts numbers written in those passages (digits only; a spelled-out number
  in a passage does not count). Risk terms are not applied to `call_llm` (retrieval gates it). Breaking change: the
  old "call_llm goes straight to the stub" behavior is gone.

## Live LLM demo (`llm_generator.py`, `demo_live.py`, 2026-10-24)
`uv run python -m router.demo_live [--fake-router] [--no-llm] [--script FILE]` runs the turn stack on the offline FixtureDatasource with an
HTTP chat generator with Gemini as the current default behind chitchat and call_llm
It falls back to the FillerGenerator when the service is unavailable and does not wire up the second routing layer
Config: PLACEHOLDER `live_llm` in `turn_policy.json`. Only tested against a fake local server. See `docs/live-demo.md`.

## Speech (`tts.py`, 2026-10-24)

The final spoken line (`result["spoken"]`: after the number verifier and blocked-line substitution, every route incl.
alerts) can also be spoken aloud. Text output is unchanged. Off by default (tests make no API call).

- Setup: `uv sync` (adds `elevenlabs`, `python-dotenv`); put `ELEVENLABS_API_KEY=...` in the root `.env` (gitignored; the
  path is resolved from the file, not the cwd). Optional overrides: `ELEVENLABS_VOICE_ID` (default `VmPkIaPdP997cjIfBWRB`),
  `ELEVENLABS_MODEL_ID` (default `eleven_v4`, accepted by the API; the key lacks `models_read`, so the list endpoint is not usable).
- One client per process (`tts.init()` at startup, else lazy on first `speak`); PCM 24 kHz piped to `paplay` (else `aplay`);
  20 s network and 60 s playback timeouts; empty text is a no-op; errors are `SpeechError` without the key.
- `TurnHandler(..., tts_fn=tts.speak)` calls `tts_fn(spoken, route)` once per turn; an exception is caught and stored as
  `result["tts_error"]`, the turn is otherwise unchanged.
- Live checks: `uv run python -m router.tts "Nova speech test."`; `uv run python -m router.demo_turn --speak --max-turns 3`.

## Speech-to-text (`stt.py`, 2026-10-24)
- `uv run --with sounddevice python -m router.stt` (conversation: listen, route, speak reply, repeat until 10 s of silence or Ctrl-C; `--once`, `--no-speak`, `--model DIR`): loads the router checkpoint on a background thread, streams the mic to
  ElevenLabs Scribe realtime (`scribe_v2_realtime`, server VAD, 1 s silence ends the utterance), then runs the first transcript
  through `TurnHandler`. `stt.listen_and_route()` is the same as a function; `stt.transcribe(frames)` is the STT part alone.
- Same key handling as `tts.py` (the root `.env`, never in errors). `sounddevice` is included in `pyproject.toml`.
- Tests: `tests/test_stt.py` (fake client). Verified live with TTS-generated audio; mic capture itself not tried.
- Not built: streaming synthesis, interrupting speech for a critical alert, Unity/Quest audio path, voice-activity handling.

## v2 family split and verification (2026-10-24, Linux CPU)

`data/gen_synthetic.py` now partitions frame-defining slot fills and destination forms by split and
checks expanded frames after substitution. Content slots, noise pools, STT slip pairs and extra places are
also partitioned. The closed core place gazetteer/determiners and wake words are deliberate shared entity
vocabulary, not claimed to be held-out; tests document these exceptions. Tests also regenerate v1
byte-for-byte and v2 exactly. No demo/paraphrase file was edited or copied into training data; exact
normalized evaluation texts remain excluded. Semantic near-duplicates may remain in synthetic language.

v2 already exists at `$NOVA_MODELS_DIR/v2`: two epochs, T=1.359 (v1 T=1.501). It was **not retrained**
during resume. Same-set comparison (argmax, newly run on CPU):

| set | n | v1 correct | v2 correct |
|---|---:|---:|---:|
| **v2** family-held-out test | 320 | 308 (96.25%) | 306 (95.63%) |
| paraphrase_test (regression) | 64 | 64 (100%) | 63 (98.44%) |
| demo_calls_47 (regression) | 47 | 47 (100%) | 45 (95.74%) |
| demo_calls (regression) | 15 | 15 (100%) | 15 (100%) |

v2 is not an across-the-board improvement. Its family split is harder than v1's literal-template split;
the v2 test is held out from **v2**, not guaranteed independent of v1's old template families. Neither
these synthetic counts nor the development-adjacent paraphrase/demo sets establish real-speech accuracy.

### Action-tool risk report

```sh
uv run python -m router.eval_sets --checkpoint "$NOVA_MODELS_DIR/v2" --data-dir data/v2 --device cpu
uv run python -m router.report_gate --checkpoint "$NOVA_MODELS_DIR/v2" --device cpu --report-dir reports
# Repeat both with v1 for the same v2 splits. No training or real LLM calls.
```

Action tools are **all five tools: call_safety, call_navigation, call_map, call_data, call_o2** (changed
2026-10-24; earlier numbers counted only the first three): every tool call is spoken as fact, so a false
activation of any of them misleads the astronaut. Reports go to `router/reports/` (gitignored).
High-confidence-wrong (HCW) means predicted tool != true class and calibrated confidence >= tau_hi.
Rate denominator is **all utterances in the split**, not accepted predictions. The report also gives
HCW/accepted, false activations vs wrong-tool mistakes, other argmax errors, coverage and Wilson bounds.
Below-threshold arbiter behavior is not scored; no arbiter is supplied here.

Sweep values: .50, .60, .70, .80, .85, .90, .92, .94, .95, .96, .97, .98, .99.
Select the lowest grid value meeting the 0.5% combined-action HCW target on **DEV ONLY**; test sweep is
informational, never a tuning input. Results on `data/v2` (320 dev / 320 test):

Per-tool HCW (hcw/accepted) with the five-tool action set, rerun 2026-10-24 on `data/v2` (v1 T=1.501, v2 T=1.359):

| model, tau_hi | split | safety | navigation | map | data | o2 | combined (rate; Wilson95+) |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 @ .97 (dev-chosen) | dev | 0/24 | 1/25 | 0/31 | 0/23 | 0/17 | 1/320 (0.312%; 1.75%) |
| v1 @ .97 | test | 0/27 | 0/23 | 0/29 | 0/12 | 0/28 | 0/320 (0%; 1.19%) |
| v2 @ .98 (dev-chosen) | dev | 1/18 | 0/21 | 0/32 | 0/27 | 0/17 | 1/320 (0.312%; 1.75%) |
| v2 @ .98 | test | 0/29 | 0/17 | 0/25 | 0/20 | 0/25 | 0/320 (0%; 1.19%) |
| v1 @ .90 (shipped) | dev | 0/28 | 2/37 | 0/36 | 2/30 | 0/27 | 4/320 (1.25%; 3.17%) |
| v1 @ .90 | test | 1/35 | 0/31 | 0/33 | 0/26 | 0/31 | 1/320 (0.312%; 1.75%) |
| v2 @ .90 (shipped) | dev | 1/25 | 1/34 | 0/37 | 2/35 | 0/32 | 4/320 (1.25%; 3.17%) |
| v2 @ .90 | test | 0/35 | 0/26 | 0/33 | 0/34 | 0/35 | 0/320 (0%; 1.19%) |

Adding call_data/call_o2 moved the DEV-only suggestion for v1 from .92 to .97 (two dev call_data HCWs at .90/.92);
v2's suggestion stays .98. Correct-action recall at gate: v1 59.5%/59.5% at .97, v2 57%/58% at .98 (dev/test);
at .90 77%/77.5% (v1) and 79.5%/81.5% (v2). **tau_hi stays at the requested .90**; the .97/.98 values are
empirical suggestions, not a safety certificate. At .90 both models are over the 0.5% target on dev (4/320).
**Mid-band arbiter-agree-wrong is unmeasured** (no real arbiter), so every rate here is a LOWER BOUND on the real
executed-wrong rate. With 0/320 test errors the Wilson 95% upper bound is still **1.186%**, above 0.5%.
Untouched push-to-talk data is required. (The older paraphrase_test / demo_calls_47 zero-HCW statements were computed
for the three-tool action set and have not been recomputed for five tools.)

**Dev data is used three times** (epoch selection, temperature fit, tau_hi choice), so dev-based confidence and gate
numbers are optimistic. When real data arrives, carve out a separate **calibration split** (temperature + tau_hi)
distinct from the epoch-selection split and from the final untouched test.

Resume verification: `uv run pytest -q`: **179 passed, no skips**, including the six base-model smoke tests.
No tracked pycache/egg-info artifacts. All model loads and inference were Linux CPU. Pipeline examples at
config .90/.50 with no arbiter: v2 `okay im home please` -> NO_TOOL/.819/LLM stub;
`Get me the coolant level.` -> call_data/.751/clarify; `go from the rover to the airlock` -> navigation/.990/airlock;
`take me home` -> navigation/.994/homebase; `hi` -> NO_TOOL/.975/LLM stub.

## Stubbed (not implemented)

- `export_onnx_stub.py` - ONNX INT8 export interface; prints "not implemented".
- Real tool integration (tools are fakes over a fabricated sim), real LLM and arbiter calls, deterministic arg extraction
  beyond place, voice input: out of scope.
- Number verifier is a simple digit/number-word check (`verifier.py`); a real verifier (units, ordinals, fractions,
  times, semantics) and real LLM-output validation remain unbuilt. Real arbiter, 10-class labels and a v3 model unbuilt.
- Actual MPS execution and ONNX INT8 on the target Mac remain unvalidated; real push-to-talk data remains uncollected.

## Notes

- `predict_batch` per-item `latency_ms` = batch wall time / batch size. Single `predict` latency = tokenize + forward.
- 2026-10-24 verification: Python 3.14.7, torch 2.14.1+cpu, transformers 5.20.2; **Linux CPU**
  benchmark (50 runs, 3 warm-ups): p50 **104.9 ms**, p95 131.7 ms. Not a Mac/MPS measurement;
  re-measure on the target Mac. At that earlier checkpoint nine tests passed (six model smoke tests and three packaging checks);
  the CLI still reports `untrained`. macOS 14 ARM64 dependency selection passed a cross-platform
  `uv sync --locked --dry-run`; actual MPS execution still needs verification on the Mac.

## Six-class routing and voice sessions

ModernBERT uses these ordered labels (head indices):
`call_safety`, `call_navigation`, `call_map`, `call_data`, `call_o2`, `llm`.
**Binary and eight-class checkpoints are incompatible**; train a new six-class head before
using the live voice path. No training or new checkpoint is included here.
The checkpoint directory deliberately remains **`$NOVA_MODELS_DIR/binary`**
(default `/home/isaach/models/Nova/router/binary`) for entrypoint compatibility;
the directory name does not describe the new six-class head. Missing checkpoints
fail explicitly with a retraining-required error; checkpoint metadata must match label order.

```sh
# Run from the repository root, only when training is authorized:
router/.venv/bin/python -m router.train --data-dir router/data \
  --out-dir /home/isaach/models/Nova/router/binary --epochs 3 --device cpu
python -m router.pipeline --checkpoint "$NOVA_MODELS_DIR/binary" --text "what color is the sky"
```

`colab/train_binary_router.ipynb` retains its filename but trains the six-class head
into `/content/models/binary` and exports `binary.zip`. Its GPU command uses
`--base /content/models/ModernBERT-base --batch-size 16 --eval-batch-size 32
--device cuda --require-cuda` (bf16 only on supported GPUs).

Top-level and v2 JSONL data retain `fine_label` and rebuild `label` from it:
all five tool intents stay distinct; `call_llm`, `NO_TOOL`, and `OOD` merge into
`llm`. The top-level paraphrase set is shared by v1/v2 (there is no separate v2
paraphrase file). Generators reproduce the same mapping and preserve the original
texts/order. `llm` therefore has three times the examples of each tool, not a newly
balanced corpus. Frozen `v3-trip`/`v4-trip` files remain untouched; their eight-class
evaluation/release tooling is not compatible with the active labels/config and needs
separate migration before reuse. Retraining, calibration and real speech validation
remain necessary; this is not a safety-qualified model.

### Dispatch and confidence

BERT selects the tool directly using `pipeline_config.json`'s `tool_labels`;
there is **no keyword selector**. At `confidence >= tau_hi` (currently 0.9),
a `call_*` prediction bypasses the LLM and uses the original place resolver and
required-slot checks. Missing/ambiguous places ask a question rather than execute.
Below `tau_lo` (currently 0.5), tool predictions fall to the LLM self-check instead
of executing blindly. The original middle band `[tau_lo, tau_hi)` remains
conservative: an explicitly supplied arbiter must agree with BERT or the pipeline
asks an intent clarification; with no arbiter (normal voice path), it clarifies.
Thresholds are unchanged and have not been calibrated for this untrained head.

An `llm` prediction at any confidence uses the generator's structured self-check
(`tool` plus a catalog name, or `none`). A recovered catalog tool goes through the
same slot/place checks; model-supplied arguments/free text are never executed or
spoken. Failed/malformed checks fall through to answer generation, not an action.
General answers use existing generation/verifier gates with Mars-base context
(showcase sky color: grey). Unknown safety requests are refused ("Please contact
base."); matching safety documents may be answered. Quick responses are unchanged,
and there is no interim "Let me think" speech. Offline FillerGenerator is a
deterministic showcase, not a live LLM.

### Voice follow-up and daily context

The Gemini adapter uses `https://generativelanguage.googleapis.com/v1beta/openai`,
model `gemini-3.5-flash-lite` (override with `NOVA_LLM_MODEL`), with
`reasoning_effort: minimal` when thinking is disabled. Set `GEMINI_API_KEY` or
`NOVA_LLM_API_KEY`; absent credentials explicitly select the offline filler.
Live Gemini compatibility/latency has not been measured.

Completed tool calls and generated statements close the session immediately;
questions and pending clarification/confirmation retain the normal ten-second
follow-up window. There is no five-second post-generation window.
`router.stt.converse` returns when that window ends; the integrated wakeword path
is `python wakeword/main.py --router-checkpoint <six-class-checkpoint>` in an
environment with both dependency sets. The existing transcriber captures follow-ups
and returns to wake detection on silence/channel closure. `--transcript-only`
preserves transcript-only operation; `--no-speak` routes without playback.
Live mic, Scribe, TTS and a trained six-class checkpoint are not exercised by offline tests.

Each turn appends only its raw user query to `router/memory/YYYY-MM-DD.md` using
the local date. Self-check and generation load the entire current day's file as
untrusted context, never numeric evidence. Files are gitignored, persistent across
sessions and have no automatic retention limit; protect/delete them as appropriate.

### Offline validation of this correction

Focused pipeline/showcase/daily-memory/data/training/STT/HTTP-generator tests:
**219 passed**; wakeword tests: **29 passed**. Full router diagnostic:
**1460 passed, 104 failed, 6 errors**. Residual failures are legacy demo/session/interim,
OOD/NO_TOOL/retrieval-only expectations and frozen v3/v4 contracts coupled to the
active labels/config. These are not claimed green or silently skipped; the frozen
releases were intentionally not migrated. No live Gemini calls, training, audio or
trained-head accuracy measurements were performed.
