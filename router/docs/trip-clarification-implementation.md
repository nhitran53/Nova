# Trip feasibility and bounded clarification — implementation evidence

## Status

Partial delivery, **not the full user done criterion**. User approved navigation semantics and
explicit selection among the current BERT menu; supervisor authorized deterministic implementation
and new data preparation only. Router training remains blocked pending user resource permission.
No first-turn keyword tool remapping, thresholds/default checkpoint changes, training, paid API,
TTS, server operations, staging or commits were performed.

## Implemented contract

Each TurnHandler owns bounded clarification state: original utterance/place resolution, ordered
offered identities, original router output, policy version, creation turn and monotonic time.
PLACEHOLDER expiry is two followup turns / 120 seconds (human sign-off still required).
Exact names/identities and ordinal/index selections dispatch only an offered identity; schema and
place resolution are revalidated and current alert telemetry/registry geometry are used. The
explicit selection gate is `explicit_offered_choice`; confidence is the original classifier's
confidence, NOT a new prediction for the chosen option (`confidence_label` and
`original_router_confidence` identify its provenance). Missing-slot continuation uses `retained_slot`.

This is the narrow approved exception to BERT first-turn selection, not an LLM tool selector.
Unknown/unoffered/mixed choices, cancel and expiry call nothing; unanswered fragments do not reach
chitchat. A clear new request is reclassified through the normal BERT gates, never dispatched as a
menu choice. Critical alerts clear state before routing. Message consent is separate: navigation
can offer contact, but selection cannot itself send a message. Slot state is separate from choice
state; ambiguous-place menus authorize only offered places. No telemetry/geometry is saved in
clarification state.

Navigation now reports assessment/distance/guidance modes without implying physical movement or
an optimized route, adds oxygen required/available/margin to verified tool data, and explicitly
reports unavailable oxygen assessment. Oxygen/pressure generic safety remains; choosing it from
a trip menu runs that generic check and explicitly denies trip clearance. A confident first-turn
safety selection with a destination is still restricted without silently switching tools.
Attached unknown `base3` can no longer fuzzy-match `base2`. New PLACEHOLDER general-knowledge prompt
explains answer/null-refusal versus refusal/empty-text; postprocessing still honors model refusal.

## Actual v2 transcript (CPU, fabricated SimState, no generator/network)

Ran `router/.venv/bin/python -m router.pipeline --checkpoint /home/isaach/models/Nova/router/v2`
for the three exact first-turn phrases, then actual-v2 TurnHandler replay:

- `can i make it to base 2`: BERT call_safety **0.9436079859733582**, tau_hi.
  Turn route destination_safety, **zero calls**; “A general safety check does not assess that trip.
  Ask for navigation to check the destination.” **Recognition remains wrong for approved semantics.**
- `can i make it home`: BERT call_safety **0.8860114812850952**, no arbiter;
  menu `[call_safety, call_navigation]`, **zero calls**.
- Followup `navigation`: explicit offered choice, **one navigation({place: homebase})**;
  assessment data distance=120 m, ETA=2 min, outbound=2, return=0, reserve=10,
  required=12, available=212, margin=200; comfortable. Verified spoken assessment plus
  navigation information, no motion/optimized-route claim, no LLM.
- Separate same home menu, followup `a safety check`: **one safety({})**, tool status_word=go,
  no LLM; scope explicitly says generic oxygen/pressure only, not destination-trip clearance.
- `can i make it base2`: BERT call_llm **0.8122778534889221**; route no_document,
  **zero calls**, no LLM. **Also unmet recognition, not overridden.**

These are fabricated SimState distances, not live measurements. FixtureDatasource has separate
Euclidean geometry; DB-backed navigation may fail if destination rows do not exist.

## Validation

- Final full offline suite before last new-intent-prefix tightening: **1174 passed in 29.87 s**.
- After that tightening: three targeted interrupt/mixed-selection tests **3 passed in 5.08 s**
  (two newly added tests, one existing affected test). No unnecessary full-suite repeat afterward.
- Earlier full pass initially exposed two expectation mismatches: demo heading/menu wording and
  TTS `hi` now routes greeting under existing speaking-scope work. Updated targeted expectations;
  TTS generic LLM test now uses `thanks`. No skips or relaxed safety/number gates.
- New tests cover names/identities/ordinals, either choice, unknown/both/cancel/unoffered,
  turn/time bounds, missing oxygen, oxygen shortage, current geometry/telemetry, unknown base3,
  missing/ambiguous places, critical preemption, new intent, no message consent and refusal priority.

## Remaining blockers/limits

First-turn destination recognition requires approved versioned ModernBERT data migration/training
and calibration; see data/v3-trip/README.md (preparation only, six authored heldout seeds, no training
release). Do not promote current v2 as satisfying the user criterion. CPU training resources and
release approval remain human-owned; existing recipe conflates model-selection and calibration dev.

The existing feasibility model is still demo-only: reserve/speed/O2 fallback, mirrored return leg
rather than destination-to-home geometry, current activity burn, no obstacle/hazard routing, no
position/mission envelope or continuous safety monitor. No freshness thresholds were changed.
Refill confirmation still uses its pre-existing stored distance (fresh oxygen), a separate remaining
geometry-freshness limitation. No real Postgres/Qwen response-quality validation was performed.
Pending state is per handler/session; callers must not reuse handlers across missions/sessions.
Independent review is required. New wording and bounds remain PLACEHOLDER pending sign-off.
