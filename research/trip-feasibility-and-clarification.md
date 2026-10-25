# Trip feasibility recognition and clarification carry-over

Research/design assessment — 2026-10-24. **No implementation, dataset edits, training, policy changes, live model requests, server changes, commits or pushes.** Examined the existing dirty `feat/live-llm` working tree, including the latest speaking-scope guard. All safety thresholds, coordinates and feasibility assumptions below are demo placeholders, not validated EVA rules.

## Recommendation and direct answer

ModernBERT is largely recognizing what its current training labels taught it: destination feasibility belongs to `call_safety`. But the safety tool has no destination argument, and the latest turn guard deliberately prevents it from answering a trip with a generic “go.” This is a **label/tool-contract mismatch**, not something to fix by lowering confidence thresholds or adding a keyword router override.

Recommend **A first: keep generic safety, consolidate destination feasibility and route/distance requests under the existing navigation action class**, with destination-aware wording and deterministic oxygen assessment on every successful navigation response. Update the labeling contract across **safety, navigation and oxygen**, not just a few “can I make it” examples. Preserve the independent critical-alert path. Implement bounded clarification state separately: a selected option must retain the original destination, not be classified as an isolated word and sent to chitchat.

This is minimally disruptive in class count/tool schema, **not zero work**: it requires a new versioned dataset, retraining and new calibration/evaluation before rollout. Keep the present destination-safety restriction until the replacement is accepted. Do not silently remap a confident `call_safety` to navigation. B is a viable later alternative if the human wants destination feasibility to remain conceptually a safety action. C is not justified merely because the current generic tool is narrow.

**Decisions requiring human approval before implementation:**

1. A vs B vs C and exact semantics: does “can I make it to X” request an assessment only, or start guidance? Recommended A should speak assessment-only wording for feasibility, distance-only wording for distance, and guidance wording only for guidance; these are response modes within one BERT-selected action, not alternate tool selection. Unsupported terrain/hazard questions must not become an oxygen-only “safe.”
2. Clarification authority: permit deterministic selection among BERT-proposed candidates, including a non-top candidate, or require a fresh contextual BERT authorization? Under the strict “ModernBERT only decides which tool runs” rule, recommend contextual reclassification as the conservative default until an explicit exception is approved.
3. Human-owned freshness limits, route/home geometry, reserve/speed/burn model and handling of missing data/unsafe margins; these cannot be inferred from demo thresholds.
4. New dataset annotation/schema, held-out acceptance scenarios, calibration splits, per-action risk/coverage targets and release gate. No numerical safety target is approved by this research.
5. Narrow general-explanation prompt/schema consistency change and subsequent offline/live validation, separately from safety recognition.

## 1. What actually runs today

### Generic safety is not empty, but is not trip feasibility

`router/tools.py:107-109,136-149,158-160`:

- Registry schema is `safety({})`: no destination, duration, route or return-home input.
- It obtains a snapshot, then returns **only** `status_word`: “go” iff `o2_pct >= 16.0` **and** `20.0 <= suit_pressure_kpa <= 40.0`; otherwise “no go.” Equality is permitted.
- It does **not** use `minutes_remaining`, distance, ETA, reserve, destination dwell time, terrain, radiation, temperature or pulse in that calculation. It exposes no checked-field explanation in the result. Saying it “checks nothing” is inaccurate; saying it cannot assess a trip is correct.
- SimState supplies fabricated constants (`tools.py:83-92`). Datasource safety calls `datasource.snapshot()` again, rather than consuming the turn's earlier snapshot. Thus alert polling and tool assessment can observe distinct samples (`turn.py:209-218`; `tools.py:158-160`). A coherent single assessment envelope would be a proposed improvement, not current behavior.
- The public words “Safety check: go” (`turn_policy.json:templates.safety`) can imply more assurance than these two predicates support. Proposed wording should name the limited scope; do not expand this to a certified operational go/no-go.

### Independent critical-alert path is not the safety tool

`turn.py:102-114,206-234` polls before routing. `turn_policy.json:sim_fields` (lines 14-38) configures oxygen, suit pressure, suit temperature, radiation rate and pulse. Critical values speak deterministic alert templates, clear pending confirmation and bypass **router, tools and LLM**. Warning values are returned in `alerts` but do not preempt and are not independently voiced here. These broader checks can stop a normal turn even though the safety tool itself checks only oxygen percentage and pressure. This wrapper runs on turns; it is **not evidence of a continuously scheduled background monitor or alert barge-in**.

Removing the generic tool must never remove this ownership/bypass. Conversely, keeping the tool does not make destination assessment complete. Its code comment about the safety loop owning go/no-go is architectural intent; the current tool predicate is narrow and the wrapper is turn-driven.

### Navigation: where oxygen is checked, and where it is not

Registry navigation requires `place`; map also requires `place` (`tools.py:142-149`). Both registry functions return location distance/ETA, **not** oxygen feasibility (`tools.py:111-116,162-168`). Datasource ETA is `ceil(distance_m / policy_speed)`; SimState uses fabricated distance/ETA pairs. Therefore callers bypassing `TurnHandler` do not receive an oxygen assessment.

Within `TurnHandler`, every **successful navigation** result goes through `_navigation_feasibility`, including recognized “how far”/“distance to/from” utterances (`turn.py:395-446`). Map does **not** receive this suffix. Missing/ambiguous destinations clarify without calling a tool (`pipeline.py:110-126`). Failed navigation tools speak an error, not an assessment.

`assess_feasibility` (`turn.py:117-147`) computes:

- outbound = ceil(current-to-destination distance / configured walking speed);
- return = zero for canonical `homebase`, otherwise **the same outbound duration**;
- reserve = configured 10 minutes; required = outbound + return + reserve;
- oxygen time = floor(snapshot minutes_remaining), else floor(o2_pct × placeholder 2.4); neither available => `None`;
- outcomes: `cannot_reach` if remaining < outbound; `cannot_return` if remaining < outbound + return; `tight` if remaining < required; otherwise `comfortable`.

All derived numbers are added to the result before verification. This explains the transcript's home 0 m/0 min, 10-minute reserve and 666 minutes: it is a consistent fixture computation, **not a field measurement**. It also explains why navigation appears to do safety's trip-budget work.

**“All navigation safety-checks” is only partially true:** successful navigation in this wrapper attempts an oxygen-budget calculation. It does not call generic safety (nor should it add a second call), does not guarantee available oxygen data, and does not assess all hazards. The route is explicitly still spoken even for `cannot_reach` (`turn.py:20-23`; `tests/test_turn.py:470-472`). Missing oxygen currently yields a route with **no feasibility claim or warning** (`turn.py:430-435`; test at `tests/test_turn.py:499-503`). If the requirement means mandatory valid assessment, explicitly report unavailable assessment rather than implying clearance; whether to withhold guidance is a human policy decision.

**Meaningful gaps:**

- Non-home return distance is not destination-to-home geometry: it mirrors current-to-destination. Home classification is only `homebase`; aliases such as habitat may resolve to a different canonical ID even when the datasource maps them to the same HAB location. Validate home identity by approved semantics, not spelling alone.
- Euclidean x/z distance ignores route obstacles, slopes and hazards; position reads have no age check and are separate from telemetry/map reads (`datasource.py:48-54,135-137,281-291`). No coherent route/position timestamp or mission fingerprint is returned to the turn handler.
- Speed is policy 60 m/min, not mission `walk_speed_mps` (seed 0.8 m/s = 48 m/min). Minutes remaining uses **current activity/scenario** burn, not necessarily walking/destination activity; mission percentage reserve is unused. Dwell time is absent. `simulator_rules.py:50-93` already has a route oxygen estimator with walking/dwell activity and percentage reserve, but the router does not call it. Do not silently swap models: reserve units and output meaning differ.
- Refill yes/no recomputes using the **new turn's oxygen snapshot** but uses stored `distance_m` (`turn.py:337-351`). Movement/map/mission changes can therefore stale the geometry. Missing oxygen and “refill no longer needed” currently share the same no-send branch and wording. Propose separate unavailable-data wording and refreshing geometry before an external message, not treating lack of evidence as improved margin.
- Alert snapshot freshness is checked at acquisition in the datasource, not rechecked after routing/tool latency; no assessment timestamp accompanies spoken output. No documented maximum end-to-end age contract exists.
- Broad “hazards near X” and “safe to cross X” are in safety training, but neither tool implements those hazard analyses. They require explicit unsupported/clarification behavior, not confident “oxygen sufficient => safe.”

### Live vs fixture and freshness

`datasource.py:92-127` requires active mission, two finite telemetry rows, advancing simulation seconds, nondecreasing radiation dose and usable burn. Snapshot age outside ±`max_age_s` (policy 10 s) raises. Radiation = dose delta / simulated hours × 1000; remaining minutes = floor(oxygen percentage / live mission activity/scenario burn). `PostgresDatasource.snapshot` reads mission and telemetry in one joined statement and rejects mixed mission IDs (`datasource.py:267-279`); it does not atomically include position/map.

`FixtureDatasource` starts at (0,0) with oxygen 100%, pressure 29.6 kPa, pulse 75 and suit temperature 21 C, then advances one 30-s tick (`datasource.py:147-187,205-227`). Normal walking burn is 0.15 percentage points/minute; after that tick oxygen is 99.925%, yielding floor(99.925/0.15) = **666 min**. HAB/home is at the origin, hence 0 m. Normal seed radiation is about 27.9 µSv/h (above warning 10, below critical 50), so “normal” fixture scenario does not mean no configured warning.

`demo_live.py:69-73,261-267,310-311` freezes its fixture clock and defaults `--advance` to zero (line 101). Values do not automatically consume oxygen while a 37-second model response runs. `/advance`, `/set`, `/pos`, or `--advance` deliberately change the fixture. The ordinary non-frozen FixtureDatasource can become stale; the live demo intentionally avoids that. No claim that the supplied transcript used live Postgres is warranted.

## 2. Why recognition and clarification fail

`labels.json:1-13` has eight ordered class indices, five routed tool labels. `pipeline.py:84-108` uses current text only: confidence ≥ .90 accepts a tool; .50–.90 requires an arbiter agreeing with the **top** label or clarifies; < .50 does not execute. Non-tool labels route separately. Top-3 candidate tool labels become clarification options (`pipeline.py:128-134`). Existing “safety or navigation” wording is therefore a candidate menu, not a saved continuation.

`TurnHandler` stores `pending_confirmation` for refill messages but **no pending clarification**, and `_clarify_turn` merely speaks (`turn.py:171-192,215-229,383-393`). Thus “navigation” is classified as a fresh isolated utterance, its original home destination is lost, and a NO_TOOL outcome can reach chitchat. LLM history is not a repair: it defaults to zero and only stores spoken LLM turns (`turn.py:268-272,313-315`; `generation.py:393-403`).

Latest safety restriction (`turn.py:398-402`) prevents a selected generic safety tool from answering destination/travel requests; it does not authorize navigation or store a choice. The .944 safety prediction being blocked and the .886 prediction clarifying are expected under current contracts. Raising confidence alone cannot make an inappropriate tool schema correct.

## 3. Options and migration cost

| Option | Intent/tool contract and one call/turn | Model/data/schema migration | Calibration / HCW | Authority and safety-loop ownership |
|---|---|---|---|---|
| **A. Keep generic safety; destination feasibility -> navigation (recommended first)** | Generic health query remains safety({}); “make it X,” route and distance use navigation(place), assessment computed internally/wrapper, not a second safety call. Distinguish assessment vs guidance wording. | Existing class count/order and schemas can stay. New dataset must move destination-budget examples from safety **and o2** to navigation; retain generic safety/o2 readouts, preserve unsupported hazard negatives. Retrain; version semantic response mode metadata if adopted. | Existing T and gate results are not portable even without renaming labels. Refit calibration and evaluate each action including navigation's expanded scope, false activation and wrong tool. | BERT still selects tool. Deterministic destination resolution/feasibility/wording cannot select another tool. Critical alerts remain independent, preemptive and non-LLM. |
| **B. Enrich safety with destination and shared feasibility engine** | safety accepts optional place/explicit scope; generic request uses health checks, destination request uses shared deterministic engine. Navigation uses the same engine internally; never safety then navigation as two registry calls. | Closest to existing safety intent examples, but explicit scope, optional argument/schema validation, place resolution/missing-slot logic, result schemas/templates all change. Destination o2 examples still need reconciliation. Retrain for contextual/slot coverage; schema/version old clients. Avoid implicit default scope confusion. | Recalibrate even if labels unchanged: execution semantics/slots changed. Measure generic-vs-destination scope errors as well as tool HCW; safety/nav confusion must not hide wrong response modes. | Router selects safety vs navigation; shared engine is not an LLM arbiter or extra tool. Health/risk ownership and critical bypass remain deterministic; no trip result may override critical alerts. |
| **C. Remove generic safety tool/label** | Useful oxygen/pressure/health checks move into deterministic loop/status/navigation. Must define how a generic “safety check” gets a supported response via a BERT-selected existing status action, or explicitly signed-off deterministic status path. No tool chain. | Highest burden: retire label/index, retrain changed output head, update config, registry count, tests, schema/docs/templates/action reports and checkpoint compatibility. Existing data only has data/o2 rather than a full health-status tool. Consolidation scope must be approved. | Previous HCW measurements invalid: action set, denominator interpretation and confusion costs change. Refit T/gates, recompute action reports. Merging away a wrong-tool distinction can improve apparent metrics without improving safety. | Monitoring must still run without user asking. Removing tool cannot remove preemption. A new deterministic generic-request dispatcher would broaden router-only authority and needs sign-off. |

All options retain registry schema checks and `calls_this_turn <= 1` (`tools.py:41-80`; `turn.py:369-380`). All need clarification handling; none makes isolated “second” magically contextual. A avoids removing an existing useful health check while addressing the stronger immediate mismatch. B may be preferable if the product insists “safety” must remain the explicit destination-feasibility action. No option licenses autonomous physical navigation or certified hazard clearance.

## 4. Proposed bounded clarification protocol

This is a design, **not implemented**. It applies to intent choices, destination ambiguity and missing destination, with distinct state kinds; do not reuse refill confirmation state.

### State and lifecycle

Store per-session pending clarification with: clarification ID/kind, original utterance, ordered offered choices (`choice_id`, displayed text, original BERT label/tool ID, score/rank, required schema), original place resolution/candidates and excluded origins/negations, creation turn/time, expiry turn/time, config/label/checkpoint version. Keep unresolved slots explicitly unknown. Store no reusable telemetry/oxygen budget in this state.

- Offer only BERT-proposed tool candidates, not invented keyword choices. If a candidate is unsuitable (generic safety for a destination under A), clarify its limited scope rather than let it silently clear the trip. Choice labels must accurately describe available capability.
- Proposed bound: next two user turns plus short wall-clock TTL (value needs approval); expiry on either limit, session/mission change or cancel. Do not extend forever through repeated ambiguous answers; allow a bounded re-ask and then clear. No cross-session carry-over.
- Poll critical alerts first, clear clarification and confirmation on critical preemption; do not resume old actions after the alert. A clear new intent interrupts/clears pending state and routes normally; cancellation calls nothing. If interruption-vs-answer is uncertain, clarify, not execute or chitchat.
- Missing-slot followups fill only the requested slot for the retained authorized intent; validate current gazetteer/schema before execution. Preserve original destination for intent selection (“navigation” after “can I make it home”), and only replace it when the user clearly corrects the destination. Unknown/ambiguous places remain questions, not a fallback to home.
- Choice resolution accepts unambiguous displayed name/ID or ordinal (“first,” “second”) **within the active menu**. “Yes” to two choices, “both,” mixed selections, an unoffered tool, or partial ambiguous names => no tool, bounded re-ask. “Both” must not run a two-tool chain. Without pending state, “second” must not pick an old tool.
- Separate clarification from side-effect confirmation: selecting navigation/safety is not consent to send a refill message. Clear incompatible pending confirmation when entering a new clarification; do not let “first/yes” resolve both. Newly needed refill contact requires its own offer.
- On resolved execution, acquire valid current telemetry/position/map and assess; original text/place is conversational context, **not permission to reuse old sensor data**. Record choice ID, original router candidate, authorization route, fresh assessment provenance and exact tool call in audit results.

### Does deterministic selection obey router-only authority?

Not automatically. Mapping “second” to a BERT-proposed second candidate is not an arbitrary keyword override, but **it authorizes a class that the current pipeline would not execute**: the arbiter is only allowed to confirm the original top label (`pipeline.py:100-108`). A candidate list is not an execution grant. Deterministic user-choice dispatch is a reasonable explicitly bounded rule, yet it broadens the literal “ModernBERT only decides which tool runs” contract. **Request human sign-off before enabling it.** The existing deterministic refill confirmation is a separate approved path, not blanket precedent for tool-choice overrides.

Conservative alternative: parse the answer only as a menu selection, then reclassify a structured contextual input containing original utterance, offered choices, selected choice and retained destination with **ModernBERT**. Add this input format to its training/calibration data. Execute only if the fresh top tool label equals the selected offered label, passes approved calibrated gates, and slots are valid. Ordinal resolution itself selects no tool. Disagreement, non-tool output or low confidence => clarify, **never chitchat while a clarification answer remains unresolved**. The LLM cannot choose a tool. Selecting the second offered tool does not warrant execution merely because it was offered.

Tradeoff: deterministic dispatch is faster and more predictable for exact selections, but requires a documented authority exception and its own misresolution safety evaluation. Contextual BERT maintains strict final authorization but adds inference and can repeatedly disagree or favor the original intent (especially safety vs navigation); exact-choice success and clarification-loop rate must be tested. Neither option should manufacture a high-confidence score, lower thresholds, or silently force the selected label.

## 5. Data audit: small counts, important semantic conflict

Read all seven existing JSONL files with Python; no data edited. Rows have only `text` and `label`, no slots, pending menu, response mode or expected continuation (`gen_synthetic.py:1-25`). Counts below are literal regex diagnostics, **not exhaustive semantic annotation**:

| File | Rows | `can i make it` matches (all safety) | Broader travel-feasibility regex (all safety) | `how far` / `distance to/from`: navigation / LLM |
|---|---:|---:|---:|---:|
| data/train.jsonl | 1600 | 3 | 24 | 4 / 6 |
| data/dev.jsonl | 320 | 0 | 2 | 0 / 0 |
| data/test.jsonl | 320 | 0 | 6 | 0 / 0 |
| data/v2/train.jsonl | 1600 | 5 | 22 | 4 / 7 |
| data/v2/dev.jsonl | 320 | 0 | 3 | 0 / 0 |
| data/v2/test.jsonl | 320 | 0 | 4 | 0 / 0 |
| data/paraphrase_test.jsonl | 64 | 1 | 1 | 0 / 0 |

Broad feasibility regex: `can i make|will i make|enough margin|can i safely|can i get.*(?:back|home)|can.*return.*safe`. Distance regex: `how far|distance (?:to|from)`. No bare `navigation[.!?]?` rows matched, and no lowercased/whitespace-normalized exact text had conflicting labels within or across these files. This does **not** mean the semantic labeling is coherent.

Examples with physical line references:

- `data/v2/train.jsonl:270`: “can i make it two the supply depot and back” -> safety; line 695: “like can i make it to the home base and back” -> safety; line 1339: “Can I make it toward the supply depot and back?” -> safety.
- `data/paraphrase_test.jsonl` contains “can I make it home safely” -> safety. This existing regression is not an independent new held-out example.
- `data/v2/train.jsonl:233`: “so how far is hab” -> navigation; line 1235: “how far is east outcrop” -> navigation; line 3: “How far away is the sun?” -> LLM. These are valid target distinctions, not exact-label conflicts. Local distance is very sparsely represented and absent from v2 dev/test under this regex.
- **Additional overlap:** `gen_synthetic.py:280-305` assigns “enough oxygen to get X,” “oxygen for the trip X,” and “will oxygen last X” to `call_o2`, while safety templates assign reaching/returning/margin to safety. A diagnostic of v2 train o2 rows matching `get |trip |reach |before i|enough .* (?:go|head|travel|move)` finds **25** rows (heuristic, not a complete count). Examples: `data/v2/train.jsonl:76` “Will my O2 last until I get toward the supply depot?” and line 193 “Is my oxygen enough to go back to the homebase?” Generic o2 tool cannot assess those routes either.

The generator explicitly places “can I make it ... and back,” safe travel, hazards and return feasibility in safety (`gen_synthetic.py:154-190`); navigation contains directions/where/how-far (`192-220`). It also intentionally partitions template/noise families but shares core places (`1-25,39-81`), so family splits are useful yet do not establish real-speech coverage. The right fix is a versioned annotation contract covering all overlapping classes, not adding five examples and leaving contrary templates in place.

### Proposed dedicated dataset/schema (new artifacts only, after approval)

Use a new version directory, not mutation of v1/v2 or existing regression files. A scenario record should contain `scenario_id`, `family_id`, `split`, `turns`, and for each turn: text, original intent context/pending offered choice IDs, expected action label, response mode (generic health / distance / guidance / destination feasibility / unsupported hazard), expected destination ID or unresolved/ambiguous status, expected tool args or no call, cancel/interrupt outcome, and telemetry/geometry fixture ID. Export text-label inputs for the chosen ModernBERT format separately from deterministic state/slot/assessment assertions. Include schema/version and author/source metadata; model input must not leak expected labels or telemetry-derived outcomes.

Label contract for A: destination oxygen-budget/reachability/return assessment -> navigation; generic health -> safety; current oxygen amount/time alone -> o2; generic explanations -> call_llm; map display remains map. Unsupported terrain/hazard assurance is not automatically a positive feasibility example. Annotators should adjudicate ambiguous requests and distinguish negation, hypothetical questions and operational requests.

Create **human-authored held-out scenario families** before synthesis/training. Split by speaker/session, semantic frame and alias/number family, not random paraphrases; use disjoint train, model-selection dev, calibration, threshold-selection and locked test sets. Synthetic expansion must not copy held-out wording. Existing demo/paraphrase cases stay regression only.

Required families:

- “can I make it home/base 2/the tower,” “can we reach X,” enough air for X, there-and-back vs home-bound vs outbound-only; both broad health and destination assessment in matched pairs.
- Known aliases, held-out aliases, numbered destinations (`base 2`, `base2`, `second base`, `base two`, unseen `base 3`, `crater 1`), unknown places, missing place, two candidates, origin/destination contrasts. Current number guard deliberately rejects base two/base 3 rather than collapsing to home (`places.py:14-22,69-89`). Generalization is tested, not assumed.
- Negative/hypothetical requests: “don't navigate home,” “not base 2,” “can I make it as a singer,” “explain what trip feasibility means,” scientific distance vs destination distance. Place names must not alone determine action.
- Fresh comfortable/tight/cannot-return/cannot-reach oxygen, critical O2/pressure preemption, low-but-noncritical oxygen, missing/stale/nonfinite telemetry, movement or mission change between turns and before refill confirmation; map vs distance behavior and exact reserve boundaries.
- Clarification selecting either choice, especially **non-top/wrong-for-the-original-task offered tool**. Choosing generic safety must not clear a retained destination trip or erase that destination; explain scope or re-ask. Contextual BERT disagreement never executes. Test first/second, aliases, ambiguous/yes/both/unoffered answers, expiry, cancel, new intent, critical interruption, no stale state, zero or one call/turn.
- Grey-sky “Mars” and typo **“mar”**: router intent and generated answer quality separately; typo must not become live telemetry/unsafe guidance. These do not belong in the trip class simply because mission terms appear.

### Retraining, calibration and release evidence

Retrain ModernBERT only after the semantic contract is approved. `core.py:112-144` returns temperature-scaled confidence; `train.py:134-141` currently chooses a checkpoint and fits temperature on the same dev set. Separate those roles in a future approved experiment. Refit temperature on independent calibration data; inspect reliability and classwise errors for each action. Choose action-class gates against human-approved risk/coverage targets on a separate threshold-selection set. Per-action thresholds would require explicit pipeline/config work: today's pipeline has one global hi/lo pair, not existing class-specific gates. Do not lower thresholds as a shortcut.

Report both classifier and end-to-end metrics: intent confusion, destination slot correctness, false activation, wrong tool, missed feasibility, unsupported clearance, unsafe/available-data claims, accepted coverage, clarification completion/loop rate, side-effect consent errors and state-expiry errors. HCW in `report_gate.py:11-30` means accepted wrong action / **all utterances**, with accepted-action precision error and Wilson upper bound alongside it. Include both, plus risk-coverage curves and subgroup denominators; zero observed errors on small synthetic sets is not demonstrated safety. Its arbiter-agree-wrong region is currently unmeasured. Contextual selection and deterministic-choice dispatch need their own executed-wrong accounting, not only the old high-confidence single-turn report.

**Class merging/removal or changed semantics invalidate previous HCW measurements and calibration claims.** Historical v2 numbers may be archived, not reused as proof for A/B/C. Compare on a common human-authored task taxonomy as well as new labels to prevent apparent gains from simply deleting a distinction. No accuracy, latency or safety improvement from these proposals has been validated.

## 6. General-knowledge refusal is a separate issue

For no-document explanatory call_llm turns, `turn.py:251-258` already permits `general_explanation` only outside risk terms. `generation.py:386-403` sets no passages/facts/numeric provenance and attaches `general_prompt`. `render_system` **already replaces** the usual “If the context does not answer ... refuse” with the general-knowledge prompt for this kind (`generation.py:412-429`). `render_context` adds KIND, no slots/actions, refusal codes and utterance; it contains **no additional grounded-only instruction** (`433-456`). Therefore the inspected current code does not show a surviving explicit grounded-only contradiction for general_explanation. Empty context plus generic context-oriented wording may still bias refusal; that is a hypothesis, not a proven cause of this model response.

The supplied `{refusal:'no_info', text:'The sky appears grey ...'}` is intentionally treated as a refusal: `postprocess` honors a known non-null code before reading prose (`generation.py:679-685`). **Do not ignore the refusal to salvage its text.** The answer may be plausible but is not proof the model intended an accepted answer or that the atmospheric explanation was specific enough for Mars.

Proposed narrow prompt improvement: explicitly describe `KIND: general_explanation` / general-knowledge provenance in context, say empty PASSAGES is expected, answer permitted general knowledge with `refusal:null`, and use a refusal code plus empty text only when uncertain/out of scope. Add consistent success/refusal examples and optionally constrain a non-null refusal branch to empty text in the output schema. Preserve number/live-reading/action/directive gates and existing risk/retrieval boundaries; no general knowledge for suit/repair operational advice. Apply no policy/code edits in this task.

Validation proposal: rendered-prompt assertions for each kind; mixed refusal+text regression proving refusal still wins; schema parity tests; fake-generator gate regressions; later approved real-Qwen tests of harmless explanations including sky/Mars/mar, unknown facts and risky questions, measuring accepted answers, refusal/malformed/fallback rates and latency. Current tests use injected generators (`tests/test_speaking_scope.py:30-55`) and do not validate real Qwen answer selection. The 37.7 s and 16.8 s transcript timings do not establish a router/state diagnosis or justify deadline/server changes; no live requests were made here.

## 7. Primary web evidence and its limits

Fetched with Python stdlib HTTPS GETs, each request ≤25 s; **accessed 2026-10-24 UTC**. These sources support design patterns, not a measured Nova fix or a requirement to adopt their frameworks.

| Primary source (HTTP result) | Evidence read | Application/inference for Nova |
|---|---|---|
| [Rasa forms](https://rasa.com/docs/rasa/forms/) → legacy-docs-oss.rasa.com/docs/rasa/forms/ (200) | Active form loop, required slots, automatic deactivation, unhappy paths/early breakout and slot-mapping conditions. | Retain original intent/required destination through followups; explicitly handle interruption/cancel rather than route every fragment afresh. Nova need not install Rasa. |
| [Rasa slot validation](https://rasa.com/docs/rasa/slot-validation-actions/) → legacy-docs-oss.rasa.com/docs/rasa/slot-validation-actions/ (200) | Custom extraction/validation inside/outside forms; validation actions return slot events, separating them from unrelated action events. | Separate selection/slot validation from final tool execution and side-effect consent. Evidence for a separation pattern, not authority to bypass BERT. |
| [Dialogflow ES input/output contexts](https://cloud.google.com/dialogflow/es/docs/contexts-input-output) → docs.cloud.google.com/dialogflow/es/docs/contexts-input-output (200) | Context constrains followup intent matching; active contexts have turn lifespans and time expiration (documented defaults five normal turns, two followup turns, 20-minute expiry). | Bounded pending context is an established mechanism for elliptical choices. Nova should choose its own short TTL, **not copy those defaults as safety rules**. |
| [Wu et al., Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems (TRADE), ACL 2019](https://aclanthology.org/P19-1078/) (200) | Dialogue state as domain/slot/value triplets, copy mechanism and challenges tracking unseen values. | Supports explicit destination state and held-out unseen-value tests. Does not show a single-turn ModernBERT can infer retained state without receiving it; no published benchmark is used as a Nova performance forecast. |
| [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) (200) | Calibration concerns probabilities representing correctness; temperature scaling studied as a post-processing calibration method. | Refit and validate confidence for the new task distribution; confidence is not trip correctness. Paper does not certify Nova's action thresholds under changed labels. |
| [Geifman & El-Yaniv, SelectiveNet, ICML 2019](https://proceedings.mlr.press/v97/geifman19a.html) (200) | Selective prediction/reject option and risk-coverage tradeoffs; contrasts thresholding pretrained confidence with integrated rejection. | Report errors jointly with coverage/abstention, not accuracy alone. No recommendation to replace ModernBERT with SelectiveNet here. |

Research trace: an initially guessed PMLR URL `https://proceedings.mlr.press/v80/geifman18a.html` returned **404** and is not evidence; the 2019 URL above was fetched successfully. An initial ACL page `2020.sigdial-1.1` returned 200 but was not used for the DST claim; the identified TRADE primary page was fetched instead. Claims about label changes invalidating old Nova HCW are an inference from changed event definitions/probabilities and the repo's label-index contract, not a quoted theorem from those papers.

## 8. Exact staged implementation proposals (all gated; none executed)

**Stage 0 — approve semantics/evidence plan.** Sign off A/B/C; assessment-only vs guidance wording; generic safety scope/unsupported hazards; strict contextual BERT vs bounded deterministic-choice exception; TTL/expiry; mandatory assessment-unavailable behavior; source/model/freshness assumptions and evaluation targets. Preserve current guard and checkpoints meanwhile.

**Stage 1 — bounded conversation state only.** Proposed seams: `turn.py` pending clarification lifecycle/ordering, `pipeline.py` structured offered IDs and contextual-input contract, explicitly approved policy/config fields, and source-named tests. Under strict authority, no selected tool executes until contextual BERT supports it. Missing destinations remain questions. No dataset mutation/retraining implicit in this stage; if contextual format is not yet trained/calibrated, fail closed rather than claim recognized selections work.

**Stage 2 — new data/ModernBERT experiment.** New dataset/schema/version + annotation guide, human-authored frozen scenarios and separate calibration/threshold sets. Adjust generation in a new version rather than rewrite old files; train a new external checkpoint, compare end-to-end scenarios, calibrate and measure HCW by action. Release only after explicit review. Do not change tau merely to make demo phrases pass.

**Stage 3 — deterministic feasibility integrity under approved model.** Keep one registry call; produce a structured current assessment envelope with telemetry/position/mission/time provenance, valid home/return geometry, approved speed/burn/reserve model, explicit unavailable outcome and refreshed confirmation geometry. Share deterministic assessment code where useful, but do not silently replace minute reserve with mission percentage reserve. Add route-vs-map coverage per approved “all navigation” meaning. Preserve critical bypass and no unsafe clearance. This may be split into small approved changes; it is not permission to integrate live DB/server operations now.

**Stage 4 — separate general-explanation contract cleanup.** Narrow prompt/refusal/schema tests above, followed by human-approved live model quality/latency assessment; no retraining of Qwen is currently indicated by this evidence.

## 9. Validation performed and limits

Executed offline inspection, count scripts and:

```text
router/.venv/bin/python -m pytest router/tests/test_speaking_scope.py router/tests/test_turn.py router/tests/test_datasource.py router/tests/test_pipeline.py router/tests/test_data_coverage.py -q -p no:cacheprovider
376 passed, 1 failed in 11.22s
```

Failure: `tests/test_turn.py:249`, `test_each_tool_class_runs_exactly_one_fake_tool[call_safety]`, sends “do it at the airlock,” expects a tool, receives new `destination_safety`. This is a **pre-existing dirty-tree test/guard expectation mismatch**, not a fix made or hidden by this research. Existing targeted tests cover feasibility boundaries, missing oxygen, fresh oxygen on refill confirmation, critical bypass and scope restrictions (`tests/test_turn.py:434-565,622-626`; `tests/test_datasource.py:142-149,254-266`; `tests/test_speaking_scope.py:97-110`). They do not implement or prove contextual clarification resolution.

No tests added/updated; no full-suite or model evaluation, live Postgres, Qwen request, training or safety validation. No .env read or secret output. Intended file writes: this report and a 2-line append to NOTES.md only; branch and unrelated dirty files retained, staging untouched. Independent reviewer gate is still required.
