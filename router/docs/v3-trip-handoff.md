# v3-trip teammate handoff: isolated experiment, not promotion

## Source handoff reviewed — not a trained release

Final independent targeted review on 2026-10-24 found all six tooling/packaging findings
resolved and marked the source handoff READY. Earlier provisional transfer artifacts are
superseded. The supervisor builds and verifies the final source transfer after recording
this review; use its external SHA256 receipt to identify that archive. This review is
**not trained-model approval or promotion permission**. Run the bounded teammate GPU smoke
first and review its evidence before the separate explicit full-training step below.
CUDA compatibility, memory fit, timing, quality and release eligibility remain unproven.

Run from the **extracted repository root** (the directory containing `router/`). This
bundle is a snapshot of dirty source, not a clone recipe. No training has been run by
this handoff. No v1/v2 checkpoint, base weights, Qwen runtime, credentials or environment
is bundled. v2 **data** is included only for the legacy audit. Read
`router/data/v3-trip/README.md`, `router/docs/trip-clarification-implementation.md`,
`research/trip-feasibility-and-clarification.md`, and `NOTES.md` first.

## Transfer trust and contents

Builder/verifier use Python stdlib only, no project install required:

```bash
python3 router/scripts/v3-trip/transfer.py build --root . --destination /absolute/external/new-transfer
python3 router/scripts/v3-trip/transfer.py verify /absolute/external/new-transfer.tar.gz
sha256sum /absolute/external/new-transfer.tar.gz
```

The source-file allowlist is explicit in `transfer.py`: root README/NOTES, router
pyproject/uv.lock/README, selected trip research doc; immediate router `.py`/`.json`,
tests `.py`, docs `.md`, handoff scripts `.py`/`.sh`, legacy data `.py`/`.jsonl`,
v2 `.jsonl`, v3-trip `.py`/`.json`/`.jsonl`/`.md`. Source HTTP adapters remain ordinary
Python, not a bundled Qwen runtime. Everything else is excluded: `.env*`, credentials,
virtualenvs, caches, logs, reports, model weights/binaries, Qwen installs, `.git` history,
`.pi` sessions and unrelated repository areas. Selected symlinks/path escapes are
rejected, never dereferenced; excluded files are never opened. Embedded credential
patterns in allowed content abort without printing matches. This conservative scan is
not a proof of absence of every possible secret. Review the scope before transferring.
Manifest records content hashes/sizes, HEAD/branch and dirty **paths**, not a secret-bearing
patch. Repackaging an extracted bundle without `.git` inherits its origin HEAD/branch and
records current changed paths against the incoming manifest, explicitly marked extracted. Archive verification rejects links, duplicates, excluded/escaping paths and inventory
or hash mismatches. Hashes are integrity checks, not signatures; exchange archive SHA256
through a trusted channel.

On teammate machine, use the trusted copy of verifier before extraction (Python 3.12):

```bash
python3 /trusted/transfer.py verify /path/to/transfer.tar.gz
mkdir -p "$HOME/work/Nova-v3-trip"  # must be empty
# Only extract the verified archive, not arbitrary untrusted tar input.
tar -xzf /path/to/transfer.tar.gz -C "$HOME/work/Nova-v3-trip"
cd "$HOME/work/Nova-v3-trip"
python3 router/scripts/v3-trip/transfer.py verify .
python3 router/data/v3-trip/release.py --check
```

Extraction includes `TRANSFER-MANIFEST.json`. A git clone alone cannot recover these
uncommitted files. Do not run `git reset`, overwrite an existing checkout, or regenerate
frozen data to make an audit pass.

## WSL2 / RTX 5070 Laptop 8GB setup

On **Windows**, install a current NVIDIA Windows driver explicitly supporting your
RTX 5070 Laptop and WSL CUDA (NVIDIA's driver selector / CUDA on WSL guide), update WSL
(`wsl --update`), and use WSL2 Ubuntu with Python 3.12 and `python3.12-venv` installed.
Do **not** install a Linux NVIDIA display driver inside WSL. In Ubuntu, `nvidia-smi`
(or `/usr/lib/wsl/lib/nvidia-smi`) must show the GPU. If absent, fix Windows/WSL first;
CPU fallback is not acceptable. A displayed CUDA version is a driver ceiling, not proof
that a Python wheel supports Blackwell or this encoder.

Use a new environment outside the repository. **Do not use ordinary `uv sync`:**
`router/pyproject.toml` and `uv.lock` select Linux CPU torch. They are preserved for
provenance, deliberately bypassed below. These pip commands do not consume uv sources.
Install from the official CUDA 12.8 (or newer supported Blackwell) PyTorch wheel index;
consult https://pytorch.org/get-started/locally/ and the NVIDIA compatibility matrix if
this index/version fails. No promise of a particular version fitting 8GB is made.

```bash
python3.12 -m venv "$HOME/venvs/nova-v3-trip-cu128"  # choose a new name if it exists
export V3_PY="$HOME/venvs/nova-v3-trip-cu128/bin/python"
"$V3_PY" -m pip install --upgrade pip
"$V3_PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$V3_PY" -m pip install 'transformers>=4.48,<5' safetensors huggingface_hub numpy 'elevenlabs>=2.74.1' 'python-dotenv>=1.2.4' pytest
"$V3_PY" -m pip check
```

Dependencies are resolved on the teammate host, **not claimed locked in advance**.
The smoke script records installed versions (`pip freeze`), Python/platform,
torch/CUDA/compiled architectures, transformers and GPU/driver information. Keep the
install terminal output; do not dump environment variables, HF cache credentials,
`.env`, pip configuration or tokens into evidence. No HF login is needed for the public
base. Reproduction uses that recorded freeze plus the same explicit CUDA wheel index;
`uv.lock` is historical CPU provenance, not the training environment lock.

## Pinned base, preflight, bounded smoke — no full training yet

The existing `router/README.md` records official public
`answerdotai/ModernBERT-base` at
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. Download into a **new** external directory:

```bash
export V3_BASE="$HOME/models/Nova/router/ModernBERT-base-v3-pinned"
# Keep this terminal's printed revision and per-file SHA256 listing for return.
"$V3_PY" -m router.download_base "$V3_BASE" --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8
```

Save its printed file hashes/revision separately for return. Alternatively copy an
already downloaded base separately, without credentials/cache files, verify the same
revision and hashes, and set `V3_BASE` to it. Never bundle v1/v2 candidates as the base.

```bash
mkdir -p "$HOME/nova-v3-runs"
export V3_RUN="$HOME/nova-v3-runs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash router/scripts/v3-trip/run.sh smoke
```

The script refuses an existing run directory and requires external absolute paths.
It resolves the run path through existing parents before any write and rejects resolved
locations inside the checkout, including `..` and symlink-parent aliases. It freezes dependencies, records GPU information and captures another current source
snapshot. Preflight must pass real CUDA/backward and data audits. Smoke has a 240-second
external timeout (+10-second kill grace), one warmup + three optimizer steps, batch 4,
length 64, bf16 and gradient checkpointing; it saves **no checkpoint**. Tee keeps failure
codes; `exit-codes.tsv` and logs survive failure. Timeout/OOM can prevent a smoke JSON
report: retain the log and code (124/137 etc.), mark failed, do not train. A successful
CUDA tensor probe alone is insufficient: actual ModernBERT forward/backward must pass.

Inspect `smoke.json` for success, finite losses, peak allocated/reserved memory,
per-step times, overhead and extrapolated three-epoch optimizer time. Driver/wheel/
encoder compatibility and 8GB fit are established only by this trial; if it fails, stop,
review with the owner, and start a new uniquely named run after an approved correction.
The historical **unmeasured 5–30 minute** idea referred to a few thousand rows, not a
benchmark. This candidate has only **416 training rows**, 98 correlated authored families.
Smoke supplies the first real estimate; extrapolation excludes dev, calibration, scoring,
checkpoint I/O and thermal/power variation. No training time or quality has been measured
by the packaging worker.

## Explicit full experiment (only after smoke review)

Obtain human numerical action-risk and coverage targets first; no default policy target
is release authority. Supply the approved HCW rate as a fraction to the gate command.
Also record required coverage, per-action constraints and rationale in an external
`human-review.md`; this script cannot invent or approve those values.

```bash
# Keep V3_PY, V3_BASE, V3_RUN from the successful reviewed smoke.
export V3_RISK_TARGET=...  # replace with the human-approved numeric fraction, not literal ...
bash router/scripts/v3-trip/run.sh train
```

This is a deliberately separate, potentially long-running teammate command. The handoff
worker does not execute it. Checkpoint destination is **only**
`$HOME/models/Nova/router/v3-trip`, must not exist; no defaults, thresholds or pointers
are changed. Three epochs, seed 0, dev checkpoint selection; temperature is independently
fit on `calibration.jsonl` by trainer. The separate calibration CLI writes a report only,
not another silent metadata change. Gate requires independent `threshold` split and
reports proposed vs unchanged current tau separately. Test, frozen authored acceptance
seeds, and regression get split-named, nonoverwrite reports. `test` retains the evaluator's
default `label` field; `heldout_scenarios` and `regression` explicitly pass
`--label-field expected_label`. Missing or unknown target labels fail rather than silently
falling back to another field or relabelling. Legacy labels are not scored as v3. Real Router + TurnHandler acceptance runs even after evaluation/gate failures;
its failed report and nonzero status are evidence, not reasons to erase/retry reports.
Training failure also attempts acceptance (possibly an incomplete-checkpoint failure)
and preserves the training exit code. A machine/process kill may prevent any report;
return the incomplete logs and mark the run failed.
If a train fails and leaves the reserved `$HOME/models/Nova/router/v3-trip`, rename it to `v3-trip.failed-<run>`, never delete it.

## Return evidence and release blockers

Return checkpoint **with tokenizer, config and `router_meta.json`**, not just weights;
base revision/hash listing; the complete external run directory (source snapshot,
environment/freeze/GPU information, stdout/stderr logs, exit codes, smoke resources/time,
calibration/gate/evaluation/acceptance reports). `acceptance.json` contains actual
single-turn and stateful replay transcripts; do not replace them with fixture specifications
or claim an injected menu is real encoder coverage. Add `human-review.md` recording
approved targets, coverage decisions, outstanding gaps and reviewer identity, without
secrets. Create a checksummed return inventory after the run, even if failed:

```bash
python3 router/scripts/v3-trip/evidence.py --run-dir "$V3_RUN" --checkpoint "$HOME/models/Nova/router/v3-trip"
# On receipt, with the corresponding run and checkpoint paths:
python3 router/scripts/v3-trip/evidence.py --run-dir "$V3_RUN" --checkpoint "$HOME/models/Nova/router/v3-trip" --verify
```

Transfer the returned checkpoint separately from the source archive. Inspect the generated
`RETURN-MANIFEST.json` and verify its SHA256 inventory on receipt before evaluation/use.
The return evidence collector accepts only known report/log/metadata extensions plus
`source/TRANSFER-MANIFEST.json` and `source.tar.gz` integrity records, never `.env`, links,
caches, credentials or arbitrary directories. Creation and receipt verification validate
both the source folder payload against its manifest and the actual archive bytes; the
return inventory also pins the source manifest and archive hashes/sizes. Deleted or
corrupted returned source therefore fails verification, not just changed reports. Do not place credentials in
logs/review notes. Failed/incomplete runs remain failed; a manifest is not an acceptance gate.

Acceptance menu coverage uses a dedicated ambiguous probe requiring human adjudication; "no menu produced" is a named limitation for human adjudication, not a failure.

Reject missing CUDA, absent/failed smoke, failed required coverage,
unsupported terrain/clearance claims, more than one tool call per turn, unsafe/missing
telemetry that fails open, consent/preemption/freshness failures. Review the full transcripts
and acceptance report; a CLI zero is not human release signoff or numerical coverage approval.
Synthetic families, 15 acceptance seeds and six replays are neither blind human trials nor
real speech. **Never promote synthetic-only**: real speech/session diversity, human annotation
and independent review/signoff are required. No automatic deployment is provided.

Rollback is default isolation: stop using the external candidate; keep prior runtime defaults,
v1/v2 files and policy unchanged. Retain evidence for audit, do not copy candidate metadata
into default paths or edit tau_hi as part of this handoff. Escalate runtime/policy/data/training
bugs to the owner; this seam changes only packaging and instructions.
