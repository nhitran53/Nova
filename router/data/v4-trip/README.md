# v4.2 trip-routing candidate data

**Agent-authored synthetic, not human speech or an independent safety benchmark.**
No training, runtime changes, packaging, promotion or measured generalization
claim is part of this data revision. Schema: `v4-trip-1.2`; version: `v4.2`.

## Revision and counts

`families.json` assigns whole families to splits before deterministic expansion.
The 148 v4.1 families and their templates/splits are preserved. Sixteen new
training-only families add 64 manually authored templates / 256 rows. Total:
**164 families, 656 source templates, 2,624 rows**, at most 16 rows per family.
Train is balanced at **192 rows per label**, 48 source templates per label.
Expansion variants are correlated, not independently collected speech.

| Split | Families | Rows | Navigation | Safety | Map | Data | O2 | LLM | NO_TOOL | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 96 | 1536 | 192 | 192 | 192 | 192 | 192 | 192 | 192 | 192 |
| dev | 12 | 192 | 16 | 32 | 16 | 32 | 16 | 32 | 16 | 32 |
| calibration | 18 | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| threshold | 18 | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| test | 20 | 320 | 48 | 32 | 48 | 32 | 48 | 32 | 48 | 32 |

New lowercase/fragment forms include trip cancellation (`NO_TOOL`), short current
radiation/heart-rate/pressure/vitals/temperature/BPM readings (`call_data`), and
oxygen/air remaining, remaining O2 and tank level (`call_o2`). Quoted dialogue,
someone-else hypotheticals, and “not asking ... just ...” scope-selection twins
contrast non-requests with actual navigation, current readings, explanations,
map/safety requests and unrelated music/messaging tasks. A quote alone does not
make an utterance non-action: an explicit own request remains positive.

### What is still held out (and the trade-off)

Train now includes **contrastive correction-style semantics** in the eight-label
`contrastive_scope_selection` construction: rejected intent followed by selected
intent, quoted-mention versus own-request scope, and third-person versus own
telemetry scope. Broad correction semantics are **no longer held out**.

The original eight `contrastive_correction` families (128 test rows) remain
entirely test-only. Precisely, their explicit repair surfaces beginning **“I
meant”, “Correction:”, “Actually”, “Let me rephrase:”** are absent from train;
all their original family IDs and template frames remain unseen there. Test's
other 192 rows cover nine shared construction types. This is a narrower
surface/construction holdout, not an unseen semantic capability or independent
author. Adding train contrasts may reduce non-action false activations, but that
is an unmeasured hypothesis, not a model result. Train now has 16 constructions;
each covers at least five labels; scope selection covers all eight.

## Intent semantics and excluded rows

Eight-class order is unchanged. `call_navigation` covers destination guidance,
distance, feasibility and oxygen trip budgets/projections (including truncated
missing-destination questions); it is never travel clearance. Unknown/missing
places need runtime resolution/clarification, not invented coordinates.
`call_safety` is generic current oxygen/pressure health assessment. `call_o2`
is current oxygen amount/percentage/time left, not a trip budget. `call_data`
is current sensor readings (BPM, radiation rate/dose, temperature, suit pressure,
levels/vitals), not projected non-oxygen readings. A label does not create sensor
support: never fabricate a reading. `call_map` displays maps; `call_llm` explains;
`NO_TOOL` includes cancellation, statements and non-requests; `OOD` is unrelated.

Unsupported battery/water/fuel/food/power trip budgets, projected non-oxygen
arrival readings and two-intent projected questions remain **outside every
scored split**. Runtime guard/clarification handles them. The nine unlabelled
`clarification_probe.jsonl` rows remain non-scored, not training/calibration or
threshold inputs. `seen_during_development.jsonl` preserves the 23 old v3
seed/control references, never independent acceptance. Both files and their
locks are unchanged. No runtime behavior is evaluated here.

## Regeneration and audit

From repository root (CPU, stdlib only):

```sh
router/.venv/bin/python router/data/v4-trip/release.py --check
router/.venv/bin/python -m pytest -q router/tests/test_v4_trip_release.py
# Explicit regeneration; no training or frozen-file rewriting:
router/.venv/bin/python router/data/v4-trip/release.py
```

Seed 41024 controls per-split shuffle; stored splits are authoritative. Four
variants rotate existing `PLACES`/`RESOURCES`; slotless templates rotate empty,
hey/um/please prefixes. These are not a gazetteer. `--out-dir` changes only output
location, not source root; copy the entire source directory for an isolated
source audit. Frozen locks are checked, never rewritten.

Checks reject cross-split exact/normalized/family/template-ID/delexicalized-frame
overlap and within-split normalized duplicates. All scored splits exclude old
v3 seeds by normalized text. Near-duplicate review uses max(delexicalized token
Jaccard, character-5-gram Jaccard), review >=0.72, hard >=0.985: **96 review row
pairs, six template-pair groups, zero hard pairs, max 0.818182**, all same-label.
Review flags are retained transparently; lexical similarity cannot certify
semantic independence. Token row-presence purity: **16 pure train tokens at
>=16 rows; zero non-allowlisted pure tokens at >=24 rows** (v4.1: 14 / zero).
The existing explicit intent-word allowlist and gates are unchanged. Details and
all residual pure tokens are in `manifest.json` / `REVISION_REPORT.md`.

The counts-only `overlap_check.py /path/to/other.jsonl` CLI remains available.
External content is not generator input; its text/hashes/path are not emitted.
It does not replace semantic review.

## SUPERSEDED v4.1 evidence and downstream schema blocker

**All v4.1 release hashes are SUPERSEDED by v4.2**, not current locks.
`SUPERSEDED_V4_1.json` records old source, generator, manifest, scored-artifact,
near-report and documentation SHA256s; it is embedded in the current manifest.
The complete old README, revision report and tooling report remain as
`V4_1_README.md`, `V4_1_REVISION_REPORT.md`, `V4_1_TOOLING_COMPATIBILITY.md`.
Their v4.1 statements/hashes are historical only. `LEGACY_PROVENANCE.md` retains
the initial v4 ledger and adds this revision. All old family/template content
is retained; unchanged dev/calibration/threshold/test bytes retain the same hash
but their v4.1 release attribution is superseded.

`router/training_tools.py` accepts only `v4-trip-1` / `v4-trip-1.1`, not the new
`v4-trip-1.2`; downstream v4 contract loading therefore needs an approved schema
acceptance bump before training/evaluation. It is deliberately **not edited**
in this data-only task. See `TOOLING_COMPATIBILITY.md` for grep-only findings.
