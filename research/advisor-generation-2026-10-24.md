**Advisor report: post-tool-call generation scaffolding (2026-10-24).** I had no write tool, so the full artifact for `/home/isaach/Nova/research/advisor-generation-2026-10-24.md` is below. Nothing is committed.

## Inherited decisions
- **Hard rules:** the safety loop owns go/no-go; critical alerts bypass the LLM; every spoken number traces to a tool_result or an explicitly configured source; the LLM never overrides the router; at most 8 tools and 1 call per turn.
- **Ownership:** alert tiers, wording and thresholds are human-owned and live in `turn_policy.json` as PLACEHOLDERs. Existing values must not change.
- **Passage-number provenance is open.** The user leans yes; the manager's caveat is that safety numbers come from live tools only. It must stay configurable and must not change silently.
- **Also open:** the `uses_llm` / `spoken` keys, and whether a "yes" after a changed situation should re-ask.

## Diagnosis: is the architecture sound?
Mostly yes, and it matches the brief.
- **Slot placeholders with hidden values.** The model emits `{slot}` placeholders, never sees the values, and code fills them. This is the right core idea, because the model cannot misquote a number it never saw.
- **Templates on tool routes.** This gives zero latency and leaves nothing new to audit. The "template_only" request exposure is a good model space that doesn't change behaviour.
- **Retry and fallback.** At most 2 attempts, then a fallback to known-good lines, with no retry on exceptions. That is conservative and correct.
- **Verifier runs twice.** It runs inside `postprocess` and again in `_speak`, which is the right defence in depth.

What the main agent missed:

1. **Drift (the important one): `generation.levels` is a second, conflicting threshold system.** `compute_labels` uses new bands that don't match `sim_fields`:
   - **O2:** the label says "low" below 30, but alerts say warning below 30 and critical below 16. So O2 at 14% gets the label "low", not "critical".
   - **BPM:** "elevated" starts above 120 while the warning starts above 150, and "low" starts below 50 while critical starts below 35.
   - **New ideas nobody signed off:** "nominal" and "time_margin_level" ("ample" from 120 minutes up) are new safety-flavoured judgments.
   - **Why it matters:** the labels go straight into the prompt (`FACTS: o2_level=nominal`), and the filler prose speaks them ("Oxygen is 87.5 percent, nominal"). That is a reassurance statement on a safety reading from a source the human doesn't own, and it can contradict the alert loop. It is labelled PLACEHOLDER, but it breaks the rule that tiers and thresholds are human-owned and single-sourced.
   - **Fix:** derive labels from `sim_fields`, with tiers `critical | warning | normal` only. Otherwise drop labels from the prompt for safety fields, and keep no separate bands.
2. **`_flatten` is first-name-wins across nested dicts.** Navigation facts carry both top-level fields and `feasibility.*`, and `minutes_remaining` vs `o2_minutes_remaining` can collide in future tools. A slot could then be filled with the wrong field's value, and the verifier would still pass because the number is in facts. Slot names should be unique paths, or the build should raise on a collision.
3. **The deadline isn't real yet.** Retry stops on summed `result.latency_ms` as reported by the generator. FillerGenerator reports 0, and a real adapter that forgets to report it would never hit the deadline. Use wall-clock time in `run_generation`.
4. **In `digits` mode, grounded prose is checked only by the global verifier.** A passage number can attach to the wrong claim or unit, which the research brief already flagged.
5. **`risk_terms` don't gate `call_llm`.** A grounded question about oxygen or pressure goes to the model with whatever numbers the document has.

## Recommendations on the open decisions
| decision | recommendation |
|---|---|
| Passage-number provenance | Keep `digits` as the shipped default (it's current behaviour and the user leans that way), but add **field-scoped exclusion**: if the question or passage hits `risk_terms` (O2, pressure, dose, etc.), force `off` for that turn. Grounded numbers must be spoken with attribution ("According to the manual…"). Build `quote_span` later, once real documents exist. This gives the user their passage numbers for non-safety topics and keeps the manager's caveat for safety numbers. It needs user sign-off. |
| `risk_terms` gating `call_llm` | Yes, gate it, but with the per-turn `off` mode above rather than a full diversion, so manual lookups still work. |
| `uses_llm` / `spoken` keys | Keep `uses_llm` (the loader already enforces it for preempting tiers). Implement `spoken: false` for warnings, meaning they are logged or shown but not voiced, or delete the key. A key that does nothing is a trap. Human decision. |
| Re-ask on changed situation | Re-ask when the outcome class changes (feasible ↔ not feasible). Otherwise proceed. This is cheap and avoids acting on a stale yes. Human decision. |
| Directive-verb rejection of grounded answers | Keep rejecting. Procedures ("return to…") are exactly what must come from the safety loop or templates. If manuals need procedures read out, add a separate, explicitly attributed "read passage verbatim" mode later. |

## Build next, in order
1. Replace `levels` with tier labels derived from `sim_fields`, or remove safety labels from the prompt. Make slot names collision-safe. Use a wall-clock deadline.
2. Add the per-turn `off` override for passage numbers on risk-term turns, plus attribution.
3. Write a Qwen `Generator` adapter (llama.cpp with the GBNF from `output_gbnf`, thinking off, temperature 0). Measure time to first token, tokens per second and memory on the M5 Pro. This also tells you whether the model can follow the slot protocol at all; small models often write the number anyway, so measure the retry and fallback rates.
4. Build the ~50-question grounded eval set (wrong unit, two similar numbers, an injected passage, no answer) with metrics: unsupported numbers spoken, false refusals, p95 latency.
5. Real retriever and documents, then streaming plus alert barge-in.

## Risks you may be missing
- **Fallback rate.** If Qwen often fails the slot protocol, chitchat collapses to the neutral line. That is safe but annoying; measure before tuning the prompts.
- **Text slots** (`place_name`, `status_word`, `urgency`) are filled verbatim from tools. Once real tools exist, a tool string carrying digits or instructions is spoken. The verifier catches digits but not wording.
- **Number-word gap.** The grammar bans digits only. Number words rely entirely on `NUMBER_WORDS` coverage, which misses "a few", "several" and ordinals.
- **Policy sprawl.** About 20 new PLACEHOLDER keys are awaiting sign-off. Group them for review, putting the safety-relevant ones (labels, deny-lists, passage mode) first.

## Need from main agent
- User sign-off on: removing or replacing `generation.levels` (drift from human-owned thresholds); the passage-number policy (`digits` + risk-term `off` + attribution); `spoken: false` semantics; and re-asking on an outcome-class change.

## Suggested execution prompt (only after the user agrees on item 1)
"In /home/isaach/Nova/router (don't commit): replace `generation.levels` with labels derived from `sim_fields` (`critical`/`warning`/`normal`, no new thresholds); remove the `levels` key; make `derive_slots`/`_flatten` raise on a slot-name collision; make `run_generation` enforce `deadline_ms` with `time.monotonic()`. Update the tests and the demo; `uv run pytest` green with 0 skips."