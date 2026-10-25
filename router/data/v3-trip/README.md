# v3-trip-1 — reviewable data release candidate (not a model release)

This directory is the **data seam only**. No training, inference benchmark, calibration,
threshold/default/checkpoint change, or runtime/policy change is part of this release.
Human annotation review and independent code/data review remain required. This is not a
safety benchmark, evidence of trained behavior, or operational EVA clearance.

## Provenance and independence

All sources and paraphrases here are **agent-authored synthetic data**, not recordings,
licensed manual excerpts, human-heldout annotations, or independently collected speech.
`families.json` is the authoritative authored source: 201 linguistic scenario/template
families, each explicitly assigned a split **before** its two authored paraphrases and
at most three deterministic content fills are expanded. No v2 rows were copied/relabelled.
The maximum six rows from one family are correlated examples, not six independent trials.
`procedure_family` identifies that same bounded synthetic request frame; there are no
external repair/manual procedure sources. Broad action concepts necessarily recur across
splits; the auditor cannot establish semantic independence of differently authored frames.

The original six `heldout_scenarios.jsonl` seeds retain their text, expected labels and IDs;
their provenance is now explicitly `agent_authored_seed_not_human_heldout`. These seeds
were already visible during prior implementation and are acceptance/regression material,
not blind human validation. Nine new seeds complete label/contrast coverage. Freeze locks
are established at this release, not retroactive proof these texts were unseen.

Core entities intentionally recur: home/home base/homebase/base camp, base 2/base2/second
base/base b, tower/comms tower; sensors and scientific topics also recur. The alias surface
forms differ by split (see `PLACES`), except home. Unknown numbered suffixes `base3` and
`crater 7` occur only in test, and `base two` is frozen acceptance with no current resolved
place. Unknown named places, absent destinations, and truncated/malformed requests have
explicit unresolved-slot metadata. This tests generalization, not guaranteed resolver
behavior. Source review and later real-encoder/slot replay are still necessary.

## Label and metadata contract

Ordered class indices are unchanged from `router/labels.json`:

1. `call_safety`: current generic oxygen-plus-pressure/suit health check, **not trip clearance**.
2. `call_navigation`: local destination distance, guidance, reachability/return assessment,
   including **destination oxygen budgets**. Modes: `assessment`, `distance`, `guidance`.
   Classification is not a tool-execution grant. Unknown/missing/malformed destinations
   require clarification and no call until a valid destination is supplied.
3. `call_map`: map display/marker requests, not route guidance.
4. `call_data`: current non-oxygen telemetry readouts (not generic explanations).
5. `call_o2`: current oxygen amount/percentage/time only, not a destination budget.
6. `call_llm`: general science/concept explanations; astronomical distance remains general
   knowledge, not local navigation. Pure unsupported terrain/hazard-assurance requests
   also use this label with `response_mode=unsupported_hazard`, **zero tools, no clearance,
   scope refusal or clarification**. This proposed annotation requires human review/replay;
   it does not authorize consequential LLM guidance or prove runtime lexical gates cover
   every paraphrase. Mixed guidance-plus-hazard requests are excluded from classifier
   train/calibration/scoring and kept as a human-adjudication replay challenge.
7. `NO_TOOL`: greetings, thanks, acknowledgments, identity and pure negated/cancelled requests.
   Mixed replacement requests are not treated as simple cancellation examples.
8. `OOD`: unrelated tasks, purchases, bookings and non-destination idioms.

This preserves unrelated class semantics and index order, not every legacy annotation.
Train/dev/calibration/threshold/test JSONL rows expose `text` and `label` plus audit metadata
(`split`, family/template/procedure IDs, provenance and optional response/slot expectations).
**Only `text` is classifier input**; never concatenate labels, modes, destinations or expected
outcomes into encoder input. Destination IDs are expected semantics, not verified parser
outputs. Per-row expected dispositions are specifications for later evaluation, not observed
runtime behavior. Future trainer schema handling belongs to the next seam.

## Counts and intended use

| Artifact | Rows | Families | Intended use |
|---|---:|---:|---|
| train.jsonl | 416 | 98 | Small diverse synthetic candidate training pool |
| dev.jsonl | 112 | 26 | Model/checkpoint selection only |
| calibration.jsonl | 104 | 26 | Later independent temperature fit only |
| threshold.jsonl | 104 | 26 | Later threshold selection, **not** temperature fitting |
| test.jsonl | 106 | 25 | Locked evaluation; not tuning |
| heldout_scenarios.jsonl | 15 | 15 | Frozen authored single-turn acceptance seeds |
| regression.jsonl | 8 | 8 | Frozen new authored all-label regression controls |
| replay_scenarios.jsonl | 6 | 6 | Frozen stateful replay specifications, not training rows |

Each classifier split represents all eight labels, with at least 12 train families per
label and three families per label in every other split. Counts are intentionally not
class-balanced: slot-bearing families have more rows than slotless families. Exact per-label
counts and SHA-256 hashes are in `manifest.json`; row count must not be reported as independent
sample size. Six to twenty-four examples per evaluation class are far too few for reliable
per-action risk bounds or a safety claim. A useful candidate experiment is possible; real
speech, speaker/session diversity, human review, and larger independent evidence remain gaps.

Replay scripts specify synthetic fixture conditions, bounded menu selection, retained slots,
critical interruption, cancellation and a mixed-hazard challenge. The menu probe "Check my suit status and show directions home." is a multi-intent utterance
with no single classifier label, not a trip-feasibility request, and requires human adjudication.
Acceptance consumes actual Router→TurnHandler menus only, adapts to the offered options,
and never injects state or counts injection hints as encoder evidence. No menu produced
is reported as `unmet_menu_probe` / `requires_human_adjudication`, a limitation rather
than a failure; omitted or unsuccessful executable checks still fail closed.
No v3 candidate has yet been trained
or replayed; see the tooling interfaces below.

## Reproduce and audit (stdlib, offline, no models)

From the repository root:

```sh
python router/data/v3-trip/release.py
python router/data/v3-trip/release.py --check
python router/data/v3-trip/release.py --out-dir /tmp/nova-v3-trip-check
python router/data/v3-trip/release.py --check --out-dir /tmp/nova-v3-trip-check
router/.venv/bin/python -m pytest router/tests/test_v3_trip_release.py -q -p no:cacheprovider
```

Seed **31024** controls only stable within-split ordering, never split membership. Generation
has no dependency on model/network/credentials, time, random hash order or external packages.
The source file and generator content hashes, generated split hashes/counts, frozen hashes,
protected-file hashes and legacy-audit hash are recorded in `manifest.json`. `--check` performs
all audits then byte-compares artifacts without writing. Regeneration writes only generated
split files, `legacy_audit.json`, and `manifest.json`, never frozen or legacy files. To verify
frozen files independently, compare their SHA-256 digests with `frozen_hashes.json`. Changing
that lock is an explicit review/versioning event; no generator command refreshes it.

The auditor rejects cross-split exact text and NFKC/casefold/Unicode-word/punctuation/whitespace
normalized text matches, family/template/procedure ID overlaps, and normalized authored
frame overlap. It also rejects within-split normalized duplicates in classifier files,
missing labels, unexpected labels/index order, protected regression text in any classifier
split, and changed frozen bytes. Replay **initial utterances** participate in leakage checks;
context-dependent followup fragments (`first`, `cancel`, etc.) deliberately do not masquerade
as standalone classifier rows. Detection is lexical/metadata-based: synonyms, paraphrase
semantics and incorrectly assigned family IDs require human review. No assertion of real
speaker/session separation is possible for this synthetic corpus.

## Legacy incompatibility and migration boundary

`legacy_audit.json` hashes/inventories v2 train/dev/test and protected demo/paraphrase files,
with explicitly heuristic regex candidate counts by **old** label. This is triage, not an
exhaustive semantic audit, automatic relabeller, scorer, or indication every flagged row is
wrong. Historical v2 destination feasibility lives under both safety and o2; broad terrain
hazard examples need adjudication rather than blind navigation remapping. Generic health,
oxygen amount/time, map, telemetry, explanation, chitchat and OOD distinctions must survive.

Do **not** silently score `demo_calls*.json`, `paraphrase_test.jsonl`, or old v2 files against
this new contract and call it v3 accuracy. Retain their historical labels/hashes/results;
if a later comparison needs migrated expectations, create a separately versioned adjudicated
mapping with original file hash, row ID/text hash, old label, new label, rationale and human
review status. No protected legacy bytes were changed. Frozen new regression rows are not
copies of legacy regression rows. Historical v2 temperatures, coverage and HCW are not valid
release evidence after this semantic change.

## Remaining gates / separate handoffs

Human review of family boundaries, annotations (especially unsupported hazards and ambiguous
health wording), evaluation sizes, risk/coverage targets and real telemetry/geometry assumptions
is required. Real speech and blind human-authored scenarios are absent. Synthetic request
frames do not cover all negation, STT, combined intent or terrain language. Nothing here validates
oxygen reserves, speed/burn, freshness, return geometry, route optimization, critical monitoring,
message consent or physical safety. No human numerical release target is inferred.

The tooling seam is available through stable `--help` interfaces: `router.preflight`,
`router.smoke` (CUDA-only bounded optimizer trial; never saves a checkpoint), `router.train`
(v3 requires explicit new output, dev selection and independent calibration), `router.calibrate`
(report-only independent fit), `router.evaluate` (split-named exclusive reports),
`router.report_gate --selection-split threshold` (offline suggestion separate from unchanged
runtime threshold), and `router.acceptance` (real encoder/registry/fixture replay; fails on
failed required coverage). All are invoked with `python -m` from the repository root.
Use distinct new report paths/directories per run; existing artifacts are refused, not overwritten.
GPU compatibility, 8GB fit, runtime estimates, full training, candidate quality, packaging and
promotion remain unverified/separate gates. No training command or checkpoint promotion is
authorized by this data release.
