# LLM-only vs ModernBERT router: rescope decision (2026-10-24)

Status: design analysis. No code, policy, data or model was changed or run. Nothing here was measured against a real Qwen. Branch `feat/live-llm`, uncommitted.

## 0. The question
Replace the ModernBERT router and the deterministic extractor with one tool-calling Qwen that picks the tool and arguments (templates and number verifier kept)? And does the existing training transfer ("we wouldn't forgo all training")?

## 1. Bottom line
**Recommendation: do not go LLM-only (B). Move toward the hybrid (C) in stages, and decide with one cheap offline experiment first.** Keep the encoder as the only component that decides *which tool*; use the Qwen you already have wired for what it is good at (wording, and later argument extraction). "Slimmer" is true on a diagram and false on this repo's actual cost drivers (section 3).

- **Training is not forgone, but it transfers partially.** The labelled data is `(text, label)` only. It carries the tool name, not the arguments. It is also template-generated (`data/gen_synthetic.py`), so every accuracy number below is synthetic-text accuracy. LoRA on it would teach tool *selection* well, and teach *arguments* only after the generator is changed to emit them (section 5).
- **The pain you hit is not a router problem.** Section 3 classifies each failure. None of the three would have been avoided by an LLM tool caller.
- **What an LLM-only design loses is a validated measurement, not measurability in principle.** Under an enum-constrained tool name, the logprobs over the 8 names form a softmax, so temperature scaling and `report_gate.py` would apply unchanged. What is unverified is runtime logprob support (llama-server likely, mlx_lm.server unknown, thinking must be off) and Qwen's calibration. The genuinely unmeasured part is *argument* HCW (wrong place, wrong recipient), which has no analogue in the current gate.

## 2. What exists today (repo evidence)
**Router** (`router/`): ModernBERT-base fine-tuned as an 8-class sequence classifier. Labels: `call_safety, call_navigation, call_map, call_data, call_o2, call_llm, NO_TOOL, OOD`. Data: 1600 train / 320 dev / 320 test, 200 and 40 per class, template-generated with held-out frames per split. Temperature scaling T=1.359 (v2), fitted on dev.

| Metric (v2) | Value | Note |
|---|---|---|
| test | 306/320 (95.6%) | synthetic text |
| paraphrase | 63/64 | hand-written, 64 items |
| demo_47 | 45/47 | v1 got 47/47 |
| HCW @ tau_hi .90, 5 action tools | dev 4/320, test 0/320 | test Wilson 95% upper 1.19%, over the 0.5% target |
| HCW @ .98 (dev-chosen) | dev 1/320, test 0/320 | correct-action recall only 57-58% |
| Latency | p50 105 ms, p95 132 ms | Linux CPU, not Mac/MPS |

Caveats the README already states: dev is used three times (epochs, T, tau_hi), so the gate numbers are optimistic. The mid-band arbiter is unwired, so HCW is a lower bound on the real executed-wrong rate. There is no real push-to-talk data. Gate: `tau_hi=0.90` executes, `0.5-0.9` goes to an arbiter that can only confirm the router's class (unwired, so it clarifies), `<0.5` goes to LLM clarify.

**Deterministic layers** (these are what make a turn safe and correct):
- `places.py` place resolver: gazetteer aliases, fuzzy matching, destination-span extraction, negation, and (new) the numbered-suffix guard.
- Slot checks against a fixed tool schema (`tools.py`): missing place becomes a slot question, never a guess. Two candidates become a "which place" question.
- Confidence gate and HCW measurement (`report_gate.py`, `reports/v2_gate.json`).
- Alert poll runs before the router; critical alerts speak a fixed template and bypass everything.
- Templates + number verifier decide every spoken line, including LLM output.

**Wired LLM** (`generation.py`, `llm_generator.py`, `demo_live.py`): Qwen is only used after the decision, for chitchat, `call_llm` paraphrase, and low-confidence fallback. It has never run against a real server.

## 3. The failures you hit, by cause
| Failure | Where the fault was | Would LLM-only have fixed it? |
|---|---|---|
| Thin spoken outputs | Generation/policy. Tool turns speak fixed templates by design (`distance_templates`, `docs/post-tool-generation.md`). | No. The LLM is *already* the proposed fix, as a speaker after a verified tool result. That is option C's job, not a router replacement. |
| "base 2" resolved to homebase | Data/policy. The gazetteer had no `base_2`, so the fuzzy resolver matched the nearest alias. | Partly, differently. An unconstrained LLM would likely emit "base 2", which an allowlist rejects, so the turn fails closed (better than homebase). An enum-constrained LLM with `base_2` absent from the enum is forced to pick a listed value, likely homebase, which reproduces the bug. Both still need the gazetteer and DB rows. Note: if the v1 to v2 checkpoint switch was forced by v1 misrouting, that one cost is router-attributable; I did not check. |
| "how far is base 2" needed a new template | Tool/policy. There is no distance tool, only navigation and a template that reads its result. | No. It also needed the new place and the policy numbers (ETA at 60 m/min vs DB 0.8 m/s). The router classified it correctly (nav, 0.965). |

All three costs come from a closed tool/place/wording surface that humans own, not from classification. An LLM-only tool caller keeps that surface (allowlist, templates, verifier) and *adds* a new thing to evaluate.

## 4. External research (researcher, web-only; weak evidence)
The researcher completed one search round and could not fetch most pages. Treat all of this as snippet-level. **No source measured the question that matters here** (calibrated high-confidence-wrong rate for a small Qwen on ~6 tools), and no Qwen3.5 0.8B-9B function-calling, Apple Silicon TTFT, M5 LoRA time, or ModernBERT-vs-LLM calibration number was found.

| Topic | What the sources support | Strength |
|---|---|---|
| Reliability | Qwen3-4B 53.8 and 1.7B 40.9 on BFCL *Live* v3 (aggregator, unclear provenance). Large open-ended tool sets, so a poor proxy for 6 fixed tools. | low |
| Latency | A blog/LangGraph issue put LLM triage/tool-call at ~0.2-3 s versus sub-10 ms embedding routers. Hardware unspecified. | low |
| Constrained decoding | XGrammar reports up to 3.5x (JSON schema) and 10x (CFG) lower overhead than llama.cpp's grammar engine (Llama-3-8B, 4090/H100). MLX/Apple overhead not verified. A grammar guarantees valid JSON, **not** the right tool. Enum on `tool_name` removes invented tools, not wrong picks. | medium (snippet) |
| Confidence | Papers report LLM stated confidence is poorly aligned with actions (arXiv 2511.13240, MICE NAACL 2025). No Qwen-small numbers. | medium |
| Tool/abstain decisions | Small models (Gemma-3-4B, Llama-3.2-3B) struggle with call vs ask vs abstain. Student report, non-Qwen. | low |
| LoRA | `mlx_lm.lora` supports QLoRA. One community Qwen3.5-9B adapter used 552 examples (rank 16). Hundreds is a plausible floor; thousands with ASR-noise is safer. No M5 or Linux GPU timing found. | low |
| Voice latency | One LLM call plus speech is two decode phases. Inference only: for safety actions a template ack can start speech without waiting on the LLM. | unsourced |
| Researcher's own bottom line | Indirect evidence favours an encoder-first cascade, with the LLM for fallback/arguments/speech, not as sole decision-maker. "Not a head-to-head measurement." | inference |

**Gaps to close before believing any of this:** Qwen3.5 tool-call accuracy and TTFT on the target Mac, ASR-noise robustness, place-grounding errors in LLM argument extraction, logprob calibration, and prompt-injection guidance (OWASP LLM Top 10).

## 5. Does the labelled data transfer to LoRA?
| Asset | Transfers? |
|---|---|
| `train/dev/test.jsonl` (`text, label`) | **Tool selection: yes**, by mapping label to tool name (and `call_llm`/`NO_TOOL`/`OOD` to "no tool" outputs). **Arguments: no.** The files hold no place/slot. |
| `gen_synthetic.py` | **Yes, with a change.** It already picks the place per template, so emitting `{"tool":..,"place":..}` alongside the text is a small generator change. Labels would then come from the generator, not from the resolver (which would be circular). |
| `paraphrase_test.jsonl`, `demo_calls_47.json` | Yes as held-out eval only. They are small (64, 47). |
| `labels.json`, held-out-frame split discipline, `report_gate.py` HCW method | Yes. The split design and the HCW metric are model-agnostic and are the most valuable thing to keep. |
| ModernBERT checkpoints v1/v2 | No. |
| Real speech | None exists. A fine-tuned LLM trained on template text has the same synthetic-to-real gap as the router. |

Practical consequence: "training isn't forgone" holds for the **data and the evaluation harness**, not for the model weights, and not for arguments without regenerating data.

## 6. Options against the hard rules
| | A. Keep current | B. LLM-only tool caller, templates + verifier kept | C. Hybrid (encoder decides tool; LLM for args/ambiguity/speech) |
|---|---|---|---|
| Safety loop owns go/no-go | Yes | Yes, if LLM output is validated before any tool runs | Yes |
| Critical alerts bypass LLM | Yes (pre-router) | Yes if the alert poll stays pre-LLM; must be kept | Yes |
| 1 call/turn, max 8 tools | Yes | Yes, but multiple-call output must be rejected by validator | Yes |
| "LLM never overrides the router" | Holds by construction | **Rule has no referent.** Equivalent: allowlist + schema validator + per-tool risk class + confirmation for high-risk tools. This weakens the *independence* the rule gives (router and LLM were two models; B has one) | Holds: encoder picks the tool, LLM may only fill/confirm arguments or ask |
| Number verifier | Unchanged | Unchanged | Unchanged |
| Calibrated HCW measurable | Yes, already (lower bound) | Possible in principle (enum-constrained logprobs, same temperature-scaling and gate report), but runtime logprob support and Qwen calibration are unverified, and arguments need their own error metric | Yes for tool choice (encoder). Args need their own error metric |
| `send_message` (free-text recipient/urgency/message; reaches crew; injection surface via retrieved passages) | Confirmation-gated, never router-chosen | Most LLM-shaped and riskiest tool. Must stay confirmation-gated and never LLM-initiated | Same as A |
| Multi-turn referents ("how far is it", "take me there") | Not handled (single utterance) | Natural strength; also adding a tool is a schema line, not retrain | Bounded answer: LLM proposes args, gazetteer validates, disagreement means ask |
| Latency (voice) | ~105 ms CPU classifier + template, ~0 LLM on ~70% of turns | LLM prefill+decode on **every** turn. Unmeasured; plausibly several hundred ms to >1 s on the Mac | Same as A on tool turns. LLM only when it adds value |
| Descriptive speech | Weak (templates) | Only if you add the speaker LLM anyway | LLM speaker after verified tool result |
| Components to maintain | Router, resolver, templates, verifier, LLM | LLM, validator, allowlist, templates, verifier, LoRA pipeline | A plus one optional LLM role |
| Build cost | Zero | High: new training, new eval, new calibration | Low-moderate (reuse `HTTPChatGenerator`) |

**Is the "never overrides the router" rule a human decision?** Yes, for B. The rule exists to keep an independently calibrated, measurable component in control. Dropping it is a safety-architecture change, not an engineering detail, and Option B should not proceed without a human explicitly retiring or rewording it.

## 7. Recommendation and migration plan
**Go with C, staged. Do not choose B yet; revisit it only if stage 1 shows the encoder is the bottleneck.**

Stage 0 (no code): get a real Qwen running via `demo_live.py` as already planned. Confirm TTFT on the Mac (target a few hundred ms to first token), thinking off, and constraint field accepted. This is a prerequisite for *every* option and is currently unverified.

Stage 1, speech (addresses "thin outputs", no router change): speak the template immediately and treat an LLM elaboration of the *verified* tool_result as optional, added only if it arrives inside the deadline; otherwise the latency cost is a full LLM pass on tool turns and the "~0 LLM on 70% of turns" claim in the table stops being true. Number verifier and template fallback intact. Excluded from paraphrase: `call_safety` and all alerts (safety loop owns go/no-go), feasibility outcomes (tight, cannot_return, cannot_reach), the refill offer, o2 below normal tier, the always-on radiation warning, `send_message`. Which turns may be paraphrased is human-owned wording policy: it needs a PLACEHOLDER in `turn_policy.json` and sign-off. Measure fallback rate.

Stage 2, offline **screen** (needs your confirmation, not run; it cannot decide on its own). Two parts:
- 2a, Qwen as the unwired mid-band arbiter: score Qwen3.5 (2B/4B/9B), enum-constrained and thinking off, on the v2 items whose router confidence is 0.5-0.9. This closes the README's "arbiter-agree-wrong unmeasured" gap, needs no rule change, and is C's natural first routing role.
- 2b, zero-shot Qwen as a full classifier on v2 test (320), paraphrase (64), demo_47, using the logprob softmax as confidence and `report_gate.py`.
Confounds, stated up front: v2 test is template text from the encoder's own generator family and `gen_synthetic.py` shares `PLACES` across splits (home-field advantage for the encoder); `call_llm`/`NO_TOOL`/`OOD` are project conventions ("how far away is the sun?" is `call_llm`), so the prompt needs written label definitions, which are themselves a tunable; the LLM threshold needs its own calibration split, not dev. Power: with 0 errors the Wilson 95% upper bound is ~5.6% on 64 items and ~7.6% on 47, so neither can be compared to a 0.5% target. A result can rule an option out, not certify one in. Real push-to-talk data is the only decisive test.

Stage 3, only if stage 2 is competitive: argument extraction by LLM (place only, with the resolver/gazetteer as validator and disagreement meaning "ask"), also the bounded answer to anaphora ("take me there"). Note an enum without `base_2` forces a wrong pick, so the enum must come from the same gazetteer/DB rows. Plus a generator change to emit args for LoRA.

Stage 4, only if stages 2 and 3 clear the bar: LoRA Qwen run alongside the encoder, requiring agreement. That is the existing arbiter rule generalised and needs no rule change. Only a true replacement (B) needs a human to retire the rule.

**What would change this recommendation:** zero-shot or LoRA Qwen at <=4B matches the encoder's measured HCW and recall *and* meets TTFT, or real push-to-talk data shows the encoder failing on speech it can't be fixed on.

## 8. Oracle challenge (addressed)
The oracle agreed with C over B and with doing Stage 0 first, and raised these, all now incorporated above:
1. Stage 2 could not decide anything as first drafted (home-field confound, label semantics, no power, no calibration split). Rewritten as a screen with 2a (Qwen as mid-band arbiter) added.
2. "LLM loses measurability" was overstated. Reworded in sections 1 and 6; the real gap is argument HCW and runtime logprob support.
3. The "base 2" row overclaimed "No". Corrected to "partly, differently", including that an enum constraint can reproduce the bug.
4. Stage 1 contradicted the latency claim and the paraphrase scope was unbounded. Fixed with speak-template-first, an exclusion list, and a human sign-off item.
5. Strongest case for B was missing: multi-turn context and cheap tool addition. Added to the table; the hybrid's answer is in Stage 3.
6. `send_message` was missing. Row added.
7. Stage 4 reworded as the existing arbiter rule; section 8 filled.

## 8b. Human decisions needed
- Whether "the LLM never overrides the router" may be retired. Required only for a true B; not for C or Stage 4.
- Which tool-turn outcomes the LLM may paraphrase (new PLACEHOLDER wording policy).
- Confirmation to run Stage 2 offline scoring (needs a working Qwen server first).

## 9. Unmeasured
- Any Qwen tool-calling accuracy on this task, any Qwen latency on the Mac.
- Calibration of an LLM's tool choice (no validated confidence signal).
- The real encoder HCW (mid-band arbiter unwired, dev reused three times, no real speech).
- LoRA data size and training time on M5 / Linux (no sourced numbers).
- ASR-noise robustness of either approach.
