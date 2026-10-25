#!/usr/bin/env bash
# Run from repository root. Never source an .env or activate the repo environment.
set -euo pipefail
[[ -f router/scripts/v3-trip/run.sh ]] || { echo "Run from repository root (router/scripts/v3-trip/run.sh must exist)" >&2; exit 2; }
mode=${1:?usage: bash router/scripts/v3-trip/run.sh smoke|train}
: "${V3_PY:?absolute dedicated Python interpreter required}"
: "${V3_RUN:?new external evidence directory for smoke, same directory for train}"
: "${V3_BASE:?absolute local pinned ModernBERT directory required}"
[[ $V3_PY = /* && $V3_RUN = /* && $V3_BASE = /* ]] || exit 2
repo=$(pwd -P)
# Resolve existing parent symlinks and '..' before any evidence directory/write.
V3_RUN=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$V3_RUN")
case "$V3_RUN/" in "$repo/"*) echo 'Evidence must be outside repo' >&2; exit 2;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# No shell tracing, environment dumps, credentials, speech or network runtime.
run() {
  local label=$1; shift
  local rc=0
  printf '%q ' "$@" >> "$V3_RUN/commands.txt"
  printf '\n' >> "$V3_RUN/commands.txt"
  "$@" 2>&1 | tee "$V3_RUN/$label.log" || rc=$?
  printf '%s\t%s\n' "$label" "$rc" >> "$V3_RUN/exit-codes.tsv"
  return "$rc"
}
data=router/data/v3-trip
case "$mode" in
smoke)
  mkdir "$V3_RUN"
  "$V3_PY" -m pip freeze > "$V3_RUN/dependencies.freeze.txt"
  "$V3_PY" -m pip check > "$V3_RUN/dependencies.check.txt"
  "$V3_PY" -c 'import platform,sys,torch,transformers; print(sys.version); print(platform.platform()); print("torch",torch.__version__,"CUDA",torch.version.cuda,"transformers",transformers.__version__); print("architectures",torch.cuda.get_arch_list())' > "$V3_RUN/environment.txt"
  run gpu nvidia-smi
  python3 router/scripts/v3-trip/transfer.py build --root . --destination "$V3_RUN/source" > "$V3_RUN/source-snapshot.json"
  run preflight timeout 120 "$V3_PY" -m router.preflight --data-dir "$data" --require-cuda --report "$V3_RUN/preflight.json"
  run smoke timeout --signal=TERM --kill-after=10 240 "$V3_PY" -m router.smoke --base "$V3_BASE" --data-dir "$data" --report "$V3_RUN/smoke.json" --steps 3 --batch-size 4 --max-length 64 --gradient-checkpointing --mixed-precision bf16
  ;;
train)
  : "${V3_RISK_TARGET:?human-approved action HCW target required; do not invent one}"
  [[ -d $V3_RUN && ! -e $V3_RUN/train.log ]] || exit 2
  "$V3_PY" - "$V3_RUN" "$V3_BASE" "$V3_RISK_TARGET" <<'PY'
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
  python3 router/scripts/v3-trip/transfer.py verify "$V3_RUN/source" > "$V3_RUN/source-verified.json"
  checkpoint="$HOME/models/Nova/router/v3-trip"
  [[ ! -e $checkpoint ]] || { echo 'Checkpoint exists; refusing overwrite' >&2; exit 2; }
  # Explicit long-running teammate action; never called by smoke/setup/packaging.
  train_rc=0
  run train "$V3_PY" -m router.train --out-dir "$checkpoint" --base "$V3_BASE" --data-dir "$data" --require-cuda --device cuda --epochs 3 --batch-size 4 --eval-batch-size 4 --max-length 64 --seed 0 --mixed-precision bf16 --gradient-checkpointing || train_rc=$?
  if (( train_rc != 0 )); then
    run acceptance "$V3_PY" -m router.acceptance --checkpoint "$checkpoint" --data-dir "$data" --device cuda --report "$V3_RUN/acceptance.json" || true
    exit "$train_rc"
  fi
  # Keep collecting acceptance evidence even if an evaluation fails; return first failure.
  status=0
  collect() { local rc=0; run "$@" || rc=$?; if (( status == 0 && rc != 0 )); then status=$rc; fi; }
  collect calibration "$V3_PY" -m router.calibrate --checkpoint "$checkpoint" --data-dir "$data" --device cuda --batch-size 4 --report "$V3_RUN/calibration.json"
  collect gate "$V3_PY" -m router.report_gate --checkpoint "$checkpoint" --data-dir "$data" --device cuda --batch-size 4 --selection-split threshold --target "$V3_RISK_TARGET" --report-dir "$V3_RUN/gate"
  for split in test heldout_scenarios regression; do
    label_args=()
    case "$split" in heldout_scenarios|regression) label_args=(--label-field expected_label);; esac
    collect "evaluate-$split" "$V3_PY" -m router.evaluate --checkpoint "$checkpoint" --data-dir "$data" --split "$split" "${label_args[@]}" --device cuda --batch-size 4 --report-dir "$V3_RUN/evaluations"
  done
  collect acceptance "$V3_PY" -m router.acceptance --checkpoint "$checkpoint" --data-dir "$data" --device cuda --report "$V3_RUN/acceptance.json"
  exit "$status"
  ;;
*) echo 'Unknown mode' >&2; exit 2;;
esac
