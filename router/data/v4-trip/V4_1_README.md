# v4-trip: construction-crossed synthetic routing data

**Candidate data only. Agent-authored synthetic, not human speech/safety benchmark.**
No training, threshold change, runtime change, packaging or promotion is part of this
release. Improved generalization is a hypothesis, not a measured result. Authoring
uses the owner's taxonomy, the v3 audit design and 23 already-seen v3 seed/control
references; there is no independent acceptance set in this directory.

## Approved scope adjustment and design

The original 60–100 families × at most 12 rows could not satisfy the requested
minimum 1,900 rows. The owner approved the initial 128-family release and then
this **v4.1 revision in place**, allowing up to about 170 families. The dataset
name remains `v4-trip`; manifest schema is `v4-trip-1.1`, version `v4.1`.
This revision has **148 families, 592 manually authored templates, 2,368 rows**,
with at most 16 rows per family. No model or external paraphraser generated these
utterances. Twenty new mission-adjacent families supply natural near misses,
acknowledgements/cancellations and counterexamples to incidental token cues.

`families.json` is the authoring source and contains the split assignment **before
expansion**. Never split generated rows. Globally, all 16 construction types
cross all eight labels. This does **not** mean all eight labels occur for each
construction in training: 14 training constructions cover five labels each and
`embedded_indirect` covers six. The exact within-train construction × label row
contingency is recorded in `manifest.json` and `REVISION_REPORT.md`.
There are 80/12/18/18/20 families in train/dev/calibration/threshold/test.
Train contains 15 constructions; dev/calibration/threshold/test contain
9/11/9/10 respectively. `contrastive_correction` remains entirely test-only
(eight families, 128 rows); the other 192 test rows cover nine constructions.

V4.1 replaces the initial cyclic allocation with explicitly stored whole-family
assignments, chosen with seed 41024 to meet train crossing and equal label counts.
The generator does not recompute allocation: the stored split is authoritative.
The original 128 families remain; 20 new families are training-only (two per tool
label, four each for NO_TOOL/OOD). Train has 40 source templates per label;
expansion is not independent speech collection.

| Split | Rows | Navigation | Safety | Map | Data | O2 | LLM | NO_TOOL | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1280 | 160 | 160 | 160 | 160 | 160 | 160 | 160 | 160 |
| dev | 192 | 16 | 32 | 16 | 32 | 16 | 32 | 16 | 32 |
| calibration | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| threshold | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| test | 320 | 48 | 32 | 48 | 32 | 48 | 32 | 48 | 32 |

Every label occurs in every split, max:min label count is at most 2, and no
label exceeds 25%. Expansion rotates four reference/resource variants per
construction using `PLACES` and `RESOURCES` in `release.py`; the same variants
are used for its contrast families across labels. Forms include base 2, base
two, base2, base-2, second base, home, homebase, base b, base 3, unknown numbered
craters and ASR-like base too. These are surface forms, **not a new gazetteer**;
runtime must resolve/clarify them. No destination/resource spelling is used as
a split identifier. Oxygen/O2/air/tank occur in navigation budgets and in
contrasts; every navigation family has all six requested twin labels sharing
reference and resource vocabulary. Some explicitly contrast a sensor with
oxygen, or use resource-themed music in OOD: these deliberately artificial
minimal contrasts are not natural speech evidence.

Frames include sufficiency, ability modals, conditional projection, conditional
feasibility, see/check-if, imperatives, wh quantities, ellipsis, truncation,
hypotheticals, negation, comparisons, ASR-like omissions/run-ons, politeness,
embedded indirect requests, and contrastive corrections. Statement/non-request
and negated/no-plan readings of the same vocabulary are NO_TOOL counterparts,
not positive tool requests. I/we and filler variants are present. Four slotless
truncation expansions use deterministic filler prefixes; no label is assigned
by the filler. Train has 40 source templates per label, not hundreds of
independently collected examples. Repeated slot expansions are correlated.

## Intent semantics (unchanged eight-class order)

- `call_navigation`: destination guidance, distance, feasibility and oxygen
  budgets/projections. “make it” is covered. Truncated destination questions
  including “will I have enough” remain navigation; missing destination metadata
  directs runtime clarification, not a tool call with an invented place.
- `call_safety`: generic current oxygen/pressure **health assessment**, not travel.
- `call_o2`: current oxygen amount/percentage or time remaining, not a trip budget.
- `call_data`: current BPM, radiation rate, cumulative radiation dose, temperature,
  suit-pressure readings, and generic “levels/vitals.” Intent is labeled even
  where a tool does not expose the reading; **never fabricate a measurement**.
- `call_map`: map display rather than route/distance or budget assessment.
- `call_llm`: explanations, not telemetry or operational guarantees.
- `NO_TOOL`: statements, explicit cancellation, non-requests and negated no-plan.
- `OOD`: unrelated tasks, including “make it as a singer.”

Navigation rows are classifier candidates, never clearance. Place metadata is
surface provenance, not validated coordinates. Missing/unknown places require
runtime handling. No confidence, oxygen threshold, resource model, policy or
label registry was changed.

### Owner-decision items: clarification has no classifier label

The owner requires **CLARIFY**, not an oxygen-only answer, for (a) projected
arrival readings of radiation/BPM/temperature, (b) two-intent questions, including
combined oxygen and another projected reading, and (c) trip budgets on battery,
water, fuel, food or power. These are intentionally **absent from all five scored
classifier splits**. `clarification_probe.jsonl` has nine non-scored examples,
no classifier `label`, and is not training, temperature-calibration or threshold
selection input. The open owner decision is how the eight-class classifier and
runtime should represent/implement these intents without a clarification class.
Do not quietly classify them as oxygen navigation or claim these probes pass.
Owner review must also approve handling unavailable current pressure/cumulative
dose readings and unknown/ASR place aliases; the label does not enable a tool.

## Regeneration and audit (CPU, stdlib only)

From repository root:

```sh
router/.venv/bin/python router/data/v4-trip/release.py --check
router/.venv/bin/python -m pytest -q router/tests/test_v4_trip_release.py
# Explicit regeneration only; does not train or change authoring/frozen files:
router/.venv/bin/python router/data/v4-trip/release.py
```

Seed 41024 controls per-split shuffle. JSON serialization, row IDs and template
IDs are deterministic. Manifest records split/label/construction counts, source
and generator SHA256, scored artifact hashes, frozen hashes and source-v3 hashes.
`frozen_hashes.json` locks only the 23 seen references and the nine non-scored
clarification probes; those files are not regenerated/rewritten by the builder.
`--out-dir` writes only generated artifacts. To audit an isolated full copy,
run that copy's release script: `--out-dir` alone is an output/byte-comparison
location, not a new source root. See `LEGACY_PROVENANCE.md` for source/evidence
limitations and `TOOLING_COMPATIBILITY.md` before any downstream training.

Checks reject cross-split exact, NFKC/casefold/punctuation-normalized, family,
template-ID and delexicalized template-frame overlap, and normalized duplicates
within a split. Construction *types* intentionally cross splits; individual
source template frames must not. Exact/normalized overlap with either old v3
seed/control file is rejected in **all five** splits, stronger than train-only.

`near_duplicate_report.json` reports cross-split pairs using the maximum of
**delexicalized token Jaccard and character-5-gram Jaccard**. Known place forms,
numbered craters, resource names and template slots are delexicalized first.
Review threshold is **>=0.72**; hard failure is **>=0.985**, not automatic removal.
Defaults yield **96 flagged row pairs, six template-pair groups, zero hard
pairs, max score 0.818182**, all same-label. They are related scope/correction frames, retained transparently for
human review. These common semantics reduce independence; low overlap is not a
proof of semantic independence. Thresholds can be configured with
`--near-threshold` and `--hard-threshold`. Nondefault thresholds change report
and manifest bytes: use a separate `--out-dir`, then check with the same options;
do not replace canonical release evidence without review.

## V4.1 lexical diagnostics

`release.py` reports normalized token **row presence**, not repeated occurrences.
The manifest lists every 100%-label-pure train token with at least 16 rows.
Pure tokens fell from **34 to 14**; zero non-allowlisted pure tokens reach the
hard gate of 24 rows. Only the explicit `INTENT_WORD_ALLOWLIST` constant exempts
intent words; incidental words are never exempted. The named former incidental
cues now cross labels, or disappear (`action`). Natural counterexamples include
map screens in data requests, percentages in explanations, and minutes in dinner
timers. Residual pure tokens below 24 are reported, not hidden or auto-dropped.
NO_TOOL self-announcing phrases are 0% in each split under the report's explicit
phrase diagnostic. Most source data remains synthetic contrast framing, not a
claim of natural mission speech. See `REVISION_REPORT.md` for 30 seeded training
samples (including ten new OOD/NO_TOOL rows), exact diagnostics and current hashes;
`LEGACY_PROVENANCE.md` marks all previous v4 release hashes superseded.

## External overlap check (supervisor-owned inputs)

```sh
router/.venv/bin/python router/data/v4-trip/overlap_check.py /path/to/other.jsonl
```

Read-only standalone CLI takes JSONL objects with `text`. It emits **counts only**:
external row count and, per v4 split, exact/normalized/near matching external-row
counts and near-pair counts. Near includes exact/normalized matches (not exclusive
buckets). It uses the same delexicalized metric, default 0.72, configurable with
`--near-threshold`. It never prints or writes external text, its hashes, examples,
or its path; malformed input yields only `{"error_count": 1}` and exit 1.
External content is not used by the generator. Human semantic overlap review is
still necessary; this lexical metric cannot certify independence.
