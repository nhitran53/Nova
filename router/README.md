# Nova router (ModernBERT-base text classifier)

Scaffold only. **No training has been done**: the classification head is random, so every
prediction is flagged `untrained: true` with a warning, and labels/confidences are meaningless.
Text input only (voice later).

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

The existing `models/ModernBERT-base` is reused, not downloaded during sync. **Only for a
fresh checkout without the model**, download it into a new/empty directory:

```sh
uv run python -I download_base.py
```

The downloaded model dir is untrusted data: keep scripts out of it, load with `trust_remote_code`
off (default), and use safetensors. `.gitignore` covers `.venv/`, `models/`, caches and build artifacts.

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
remains supported. Use the editable `uv sync` installation to keep the default model path
pointing at the existing checkout's `models/` directory.

Use `--model <dir|HF id>` or `ROUTER_MODEL_DIR` to point at another checkpoint, `--temperature T`, `--device`.
Set `HF_HUB_OFFLINE=1` to explicitly disable Hub requests when using the local model.

## Swapping labels

Edit `labels.json` (`{"labels": [...]}`). Order = class index; the head size is `len(labels)`, so a fine-tuned
checkpoint is only valid with the label order it was trained on. Current labels are **placeholders**
(`call_safety ... call_x`, `NO_TOOL`, `OOD`) to be replaced by the real 10 classes (get_alerts/get_state unmerged).

## Trained vs untrained marker

`Router.load` looks for `<model_dir>/router_meta.json`. `{"trained": true, ...}` clears the warning; missing or
`false` => untrained (the default for the base model). The future training script must write it.

## Stubbed (not implemented)

- `train_stub.py` - argparse + JSONL `{text,label}` interface; prints "not implemented".
- `export_onnx_stub.py` - ONNX INT8 export interface; prints "not implemented".
- Deterministic arg extraction, fallback chain, voice input: out of scope.

## Notes

- `predict_batch` per-item `latency_ms` = batch wall time / batch size. Single `predict` latency = tokenize + forward.
- 2026-10-24 verification: Python 3.14.7, torch 2.14.1+cpu, transformers 5.20.2; **Linux CPU**
  benchmark (50 runs, 3 warm-ups): p50 **104.9 ms**, p95 131.7 ms. Not a Mac/MPS measurement;
  re-measure on the target Mac. All nine tests passed (six model smoke tests and three packaging checks);
  the CLI still reports `untrained`. macOS 14 ARM64 dependency selection passed a cross-platform
  `uv sync --locked --dry-run`; actual MPS execution still needs verification on the Mac.
