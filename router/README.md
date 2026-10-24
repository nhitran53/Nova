# Nova router (ModernBERT-base text classifier)

Text input only (voice later). The base model alone has a random head and every prediction is flagged
`untrained: true`; use a fine-tuned checkpoint (`/home/isaach/models/Luna/router/v1`, see "Data, training, results") for real labels.
Checkpoints are git-ignored.

Output of `Router.predict(text)`:
`{label, confidence, topk:[{label,p}], latency_ms, untrained, warning?}` - softmax probs
(optionally divided by `temperature`). Runs on CPU; `device='auto'` picks MPS (Mac), then CUDA, then CPU.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
cd router
uv sync --locked                  # local .venv, editable router package, includes pytest
```

Python >=3.12 is supported; `.python-version` selects 3.14. The resolved torch release has
Python 3.14 wheels on Linux x86_64 and macOS ARM64, so no Python downgrade is needed.
Linux uses the explicit PyTorch CPU index; macOS uses the default PyPI wheels (MPS included).
The latest torch's macOS wheels require Apple Silicon and macOS 14 or newer (including the target M5 Mac);
older macOS and Intel Macs are not supported by that upstream release.
`pyproject.toml` declares dependencies and `uv.lock` records exact versions for both platforms;
`requirements.txt` has been removed. To deliberately refresh dependencies, run
`uv lock --upgrade` followed by `uv sync --locked` and re-run the checks below.

The base model lives **outside the repo**, in `$NOVA_MODELS_DIR/ModernBERT-base` (default
`/home/isaach/models/Luna/router/ModernBERT-base`); it is not downloaded during sync. **Only for a fresh machine
without the model**, download it into a new/empty directory (pass `--revision SHA` to pin; the resolved commit and
file sha256s are printed):

```sh
uv run python -I download_base.py
```

The downloaded model dir is untrusted data: keep scripts out of it, load with `trust_remote_code`
off (default), and use safetensors. `.gitignore` covers `.venv/`, `models/`, `checkpoints/`, caches and build artifacts.

Base model recorded on 2026-10-24: `answerdotai/ModernBERT-base` @ `8949b909ec900327062f0ebf497f51aef5e6f0c8`,
`model.safetensors` sha256 `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`.

## Run (from Nova/router)

```sh
uv run python -m router.cli --text "check my oxygen level"  # pretty; add --json for JSON
uv run python -m router.cli                               # REPL, empty line quits
uv run pytest                                            # offline after model download
uv run python -m router.bench_latency --runs 50 --device cpu
```

After activating `.venv` (`source .venv/bin/activate`), plain `python -m router.cli` also works.
From `Nova/`, use `uv run --project router python -m router.cli --text "check my oxygen level"`.
The package remains flat in this directory; `core.py` holds the implementation so it does not
shadow the installed `router` package when running commands here. `from router import Router`
remains supported. Default model path is `$NOVA_MODELS_DIR/ModernBERT-base` (see above), independent of the checkout.

Use `--model <dir|HF id>` or `ROUTER_MODEL_DIR` to point at another checkpoint, `--temperature T`, `--device`.
Set `HF_HUB_OFFLINE=1` to explicitly disable Hub requests when using the local model.

## Swapping labels

Edit `labels.json` (`{"labels": [...]}`). Order = class index; the head size is `len(labels)`, so a fine-tuned
checkpoint is only valid with the label order it was trained on. Current labels are **placeholders**
(`call_safety ... call_x`, `NO_TOOL`, `OOD`) to be replaced by the real 10 classes (get_alerts/get_state unmerged).

## Trained vs untrained marker

`Router.load` looks for `<model_dir>/router_meta.json`. `{"trained": true, ...}` clears the warning; missing or
`false` => untrained (the default for the base model). `train.py` writes it (with the temperature fitted on dev).

## Data, training, results

`data/gen_synthetic.py` writes `data/{train,dev,test}.jsonl` (200/40/40 rows per label). v1 generator: many
phrasings per class (verbs `take/bring/get/lead/guide/walk me`, `go/head/walk`, politeness wrappers, word order,
with and without `to/the/my`, contractions, fillers, STT-style slips and typos); a **shared place list**
(home, homebase, home base, base, habitat, airlock, rover, lander, crater, ridge, landing pad ...) is substituted
into navigation, map, safety, O2, NO_TOOL ("I'm home") and OOD ("fix the wifi at home") templates so a place word
never decides the class; `call_llm` covers general space/science/how-it-works questions, `OOD` is unrelated topics
(food, sports, Earth weather, shopping, gibberish). Train/dev/test use disjoint template pools. Texts equal to a line
of `demo_calls*.json` or `paraphrase_test.jsonl` are never emitted (those files are read for exclusion only).

`data/paraphrase_test.jsonl` is 64 hand-written utterances (8 per class, incl. `take me home`, `take me to home`,
`go to home`, `directions to homebase`) used **only for evaluation**; `demo_calls*.json` likewise.

Trained checkpoints are kept **outside git**: `$NOVA_MODELS_DIR` if set, else `/home/isaach/models/Luna/router`
(`router.core.models_dir()`). `train.py` (`--out-dir`), `pipeline.py`/`evaluate.py`/`eval_sets.py` (`--checkpoint`) and
`demo.py` (`--model`) default to that directory; pass an explicit path to override per call. Never write checkpoints
into the repo (`checkpoints/` stays in `.gitignore` only as a guard). The base model stays in `models/ModernBERT-base`.

```sh
export NOVA_MODELS_DIR=${NOVA_MODELS_DIR:-/home/isaach/models/Luna/router}
mkdir -p "$NOVA_MODELS_DIR"
uv run python data/gen_synthetic.py
nohup uv run python -m router.train --epochs 2 --out-dir "$NOVA_MODELS_DIR/v1" > /tmp/train_v1.log 2>&1 &   # ~12 min, CPU
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
earlier position, then alphabetical id. Known fuzz false positive: `creator` ~ `crater` (0.92); raise `threshold` if needed.

## Pipeline stub (`pipeline.py`, `pipeline_config.json`)

```sh
uv run python -m router.pipeline --text "take me home" --text "hi"       # JSON line per utterance
```

`Pipeline(router, llm_fn=...).handle(text)` -> tool route `{route:"tool", tool, args, confidence, label}` for the labels
listed under `tool_labels` in `pipeline_config.json` (navigation/map also resolve `args.place`; other tools `args: {}`),
otherwise `{route:"llm", reason, prompt_policy, response, ...}` with a distinct policy per class (`OOD`: "politely say
out of scope") or the `low_confidence` clarify policy when confidence < `tau_lo` (default 0.5, overridable). No LLM is
called; `llm_fn(text, prompt_policy)` defaults to a placeholder string. Tool/label routing is config, not code.

## Stubbed (not implemented)

- `export_onnx_stub.py` - ONNX INT8 export interface; prints "not implemented".
- Tool execution, real LLM call, deterministic arg extraction beyond place, voice input: out of scope.

## Notes

- `predict_batch` per-item `latency_ms` = batch wall time / batch size. Single `predict` latency = tokenize + forward.
- 2026-10-24 verification: Python 3.14.7, torch 2.14.1+cpu, transformers 5.20.2; **Linux CPU**
  benchmark (50 runs, 3 warm-ups): p50 **104.9 ms**, p95 131.7 ms. Not a Mac/MPS measurement;
  re-measure on the target Mac. All nine tests passed (six model smoke tests and three packaging checks);
  the CLI still reports `untrained`. macOS 14 ARM64 dependency selection passed a cross-platform
  `uv sync --locked --dry-run`; actual MPS execution still needs verification on the Mac.
