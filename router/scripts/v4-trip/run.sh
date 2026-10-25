#!/usr/bin/env bash
# Run from repository root. Never source an .env or activate the repo environment.
set -euo pipefail
[[ -f router/scripts/v4-trip/run.sh ]] || { echo "Run from repository root (router/scripts/v4-trip/run.sh must exist)" >&2; exit 2; }
mode=${1:?usage: bash router/scripts/v4-trip/run.sh smoke|train|eval|gate|accept}
: "${V4_PY:?absolute dedicated Python interpreter required}"
: "${V4_RUN:?new external evidence directory for smoke, same directory for train}"
: "${V4_BASE:=$HOME/models/ModernBERT-base}"
: "${V4_CHECKPOINT:=$HOME/models/Nova/router/v4-trip}"
: "${V4_RISK_TARGET:=0.05}" # Offline HCW target only; never changes runtime thresholds.
[[ $V4_PY = /* && $V4_RUN = /* && $V4_BASE = /* && $V4_CHECKPOINT = /* ]] || exit 2
repo=$(pwd -P)
# Resolve existing parent symlinks and '..' before any evidence directory/write.
V4_RUN=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$V4_RUN")
case "$V4_RUN/" in "$repo/"*) echo 'Evidence must be outside repo' >&2; exit 2;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# No shell tracing, environment dumps, credentials, speech or network runtime.
run() {
  local label=$1; shift
  [[ -d $V4_RUN && ! -e $V4_RUN/$label.log ]] || { echo 'New evidence log required; refusing overwrite' >&2; return 2; }
  local rc=0
  local arg redact=0
  for arg in "$@"; do
    if (( redact )); then printf '%s ' '[external-path-redacted]' >> "$V4_RUN/commands.txt"; redact=0
    else printf '%q ' "$arg" >> "$V4_RUN/commands.txt"; [[ $arg != --sealed && $arg != --exclude-ids ]] || redact=1; fi
  done
  printf '\n' >> "$V4_RUN/commands.txt"
  "$@" 2>&1 | tee "$V4_RUN/$label.log" || rc=$?
  printf '%s\t%s\n' "$label" "$rc" >> "$V4_RUN/exit-codes.tsv"
  return "$rc"
}
# v4-trip-1, 1.1 and 1.2 schemas share data_contract v4-trip-1
# in preflight, checkpoint metadata, train/eval/gate/accept compatibility checks.
data=router/data/v4-trip
case "$mode" in
smoke)
  mkdir "$V4_RUN"
  "$V4_PY" -m pip freeze > "$V4_RUN/dependencies.freeze.txt"
  "$V4_PY" -m pip check > "$V4_RUN/dependencies.check.txt"
  "$V4_PY" -c 'import platform,sys,torch,transformers; print(sys.version); print(platform.platform()); print("torch",torch.__version__,"CUDA",torch.version.cuda,"transformers",transformers.__version__); print("architectures",torch.cuda.get_arch_list())' > "$V4_RUN/environment.txt"
  run gpu nvidia-smi
  sealed_args=()
  [[ -z ${V4_SEALED:-} ]] || sealed_args=(--sealed "$V4_SEALED")
  python3 router/scripts/v4-trip/transfer.py build --root . --destination "$V4_RUN/source" "${sealed_args[@]}" > "$V4_RUN/source-snapshot.json"
  run preflight timeout 120 "$V4_PY" -m router.preflight --data-dir "$data" --require-cuda --report "$V4_RUN/preflight.json"
  run smoke timeout --signal=TERM --kill-after=10 240 "$V4_PY" -m router.smoke --base "$V4_BASE" --data-dir "$data" --report "$V4_RUN/smoke.json" --steps 3 --batch-size 4 --max-length 64 --gradient-checkpointing --mixed-precision bf16
  ;;
train)
  [[ -d $V4_RUN && ! -e $V4_RUN/train.log ]] || exit 2
  "$V4_PY" - "$V4_RUN" "$V4_BASE" "$V4_RISK_TARGET" <<'PY'
import json,sys,math,hashlib
from pathlib import Path
p=Path(sys.argv[1])
target=float(sys.argv[3])
assert math.isfinite(target) and 0 <= target <= 1, 'human target must be a finite fraction'
assert json.loads((p/'preflight.json').read_text())['passed'], 'preflight must pass'
s=json.loads((p/'smoke.json').read_text())
assert s['success'] and s['actual_encoder_forward_backward'], 'CUDA smoke must pass'
assert Path(s['settings']['base']).resolve() == Path(sys.argv[2]).resolve(), 'base changed since smoke'
assert s['settings']['batch_size'] == 4 and s['settings']['max_length'] == 64, 'smoke settings differ'
manifest=json.loads((p/'source/TRANSFER-MANIFEST.json').read_text())
for name, record in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == record['sha256'], 'source changed since smoke: ' + name
PY
  python3 router/scripts/v4-trip/transfer.py verify "$V4_RUN/source" > "$V4_RUN/source-verified.json"
  checkpoint="$V4_CHECKPOINT"
  [[ ! -e $checkpoint ]] || { echo 'Checkpoint exists; refusing overwrite' >&2; exit 2; }
  # Explicit long-running teammate action; never called by smoke/setup/packaging.
  train_rc=0
  run train "$V4_PY" -m router.train --out-dir "$checkpoint" --base "$V4_BASE" --data-dir "$data" --require-cuda --device cuda --epochs 3 --batch-size 4 --eval-batch-size 4 --max-length 64 --seed 0 --mixed-precision bf16 --gradient-checkpointing || train_rc=$?
  (( train_rc == 0 )) || exit "$train_rc"
  # Evaluation and external acceptance are explicit separate subcommands.
  ;;
eval)
  for split in dev calibration threshold test; do
    run "evaluate-$split" "$V4_PY" -m router.evaluate --checkpoint "$V4_CHECKPOINT" --data-dir "$data" --split "$split" --device "${V4_DEVICE:-cuda}" --batch-size 4 --report-dir "$V4_RUN/evaluations"
  done
  run evaluate-seen "$V4_PY" -m router.evaluate --checkpoint "$V4_CHECKPOINT" --data-dir "$data" --split seen_during_development --label-field expected_label --device "${V4_DEVICE:-cuda}" --report-dir "$V4_RUN/seen-diagnostics"
  ;;
gate)
  run gate "$V4_PY" -m router.report_gate --checkpoint "$V4_CHECKPOINT" --data-dir "$data" --device "${V4_DEVICE:-cuda}" --selection-split threshold --target "$V4_RISK_TARGET" --report-dir "$V4_RUN/gate"
  ;;
accept)
  : "${V4_SEALED:?external sealed JSONL required; never packaged}"
  extra=()
  [[ -z ${V4_EXCLUDE_IDS:-} ]] || extra=(--exclude-ids "$V4_EXCLUDE_IDS")
  run acceptance "$V4_PY" -m router.acceptance --checkpoint "$V4_CHECKPOINT" --data-dir "$data" --sealed "$V4_SEALED" "${extra[@]}" --device "${V4_DEVICE:-cuda}" --report "$V4_RUN/acceptance.json"
  ;;
*) echo 'Unknown mode' >&2; exit 2;;
esac
