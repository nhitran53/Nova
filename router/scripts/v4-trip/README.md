# Candidate-only v4 tooling

Run `bash router/scripts/v4-trip/run.sh smoke|train|eval|gate|accept` from the
repository root, in a dedicated Colab/local environment with the existing router
dependencies. No additional heavy dependencies are introduced. Smoke performs
three CUDA optimizer steps; train is separate and requires the passing smoke
and unchanged source snapshot. Neither is part of CPU tooling validation.

Set absolute `V4_PY`, `V4_RUN` (new external evidence directory for smoke), and
`V4_BASE` (local pinned ModernBERT base). `V4_CHECKPOINT` defaults to
`$HOME/models/Nova/router/v4-trip`, must be new for train. `V4_DEVICE` defaults
to cuda for eval/gate/accept; CPU is supported for these diagnostics.
`V4_RISK_TARGET` defaults to **0.05**, the offline action-HCW selection target,
not a runtime policy or a safety claim. `gate` selects on threshold.jsonl only;
`train` fits temperature on calibration.jsonl, never dev. Human-owned tau_hi /
tau_lo in pipeline_config.json are not changed.

`accept` requires `V4_SEALED` pointing to an external JSONL. Optional
`V4_EXCLUDE_IDS` points to a JSON list or newline-delimited list of overlapping
ids. Exclusions apply to headline classifier and behavior scores, not seen
reference diagnostics. Missing coverage fails closed. The 15 old seeds and
8 controls are seen-during-development only, never independent evidence.
No sealed texts are emitted unless the direct acceptance CLI is explicitly
invoked with `--show-text`. Reports contain aggregate counts and failure ids.

The parallel `transfer.py build --root . --destination /external/new-source
[--sealed /external/input.jsonl]` packages allowlisted v4 data/tooling and v3
compatibility sources. The optional sealed input must be outside the repository;
only its SHA256 is included in the receipt/manifest, never its path or contents.
It is not transferred to Colab: supply it separately for acceptance. Existing
v3 scripts/defaults are unchanged. Never promote this candidate implicitly.
