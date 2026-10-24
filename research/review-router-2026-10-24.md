# ModernBERT router review — 2026-10-24

## Review

Reviewed the supplied Git status/diff artifacts and router source/tests read-only. Verification commands were run through the supervisor; no training or base-model/cache inspection was performed.

### Prioritized findings

1. **High / P1 — Pipeline bypasses Architecture A’s middle-confidence arbitration.**  
   **Location:** `router/pipeline.py:51–66`, `router/pipeline_config.json:3`.  
   Every tool prediction at confidence **≥0.5** becomes a tool route. There is no `tau_hi`, top-three agreement gate, or deterministic disagreement clarification. Architecture A §2 requires these before execution. This matters now: checkpoint evaluation misclassifies **“okay im home please” → navigation at 0.91** and **“Get me the coolant level.” → navigation at approximately 0.90**. Temperature scaling does not make an arbitrary execution threshold safe.  
   **Smallest fix:** Keep this explicitly a non-executing stub; before wiring tools, implement the two-threshold decision table and agreement/disagreement tests. Select thresholds from action-specific dev risk, not overall accuracy.

2. **Med / P1 — Place selection can turn an origin into the destination.**  
   **Location:** `router/places.py:68–80`, `router/pipeline.py:57–61`.  
   Resolver reproduction: **“go from the rover to the airlock” returns `rover`**, because equal exact matches prefer the earlier occurrence. Pipeline then places that result directly in `args.place`. The tie rule is deterministic but semantically wrong for navigation.  
   **Smallest fix:** Extract destination-bearing spans before fuzzy resolution. If multiple candidates remain unresolved, return ambiguity and ask a deterministic slot question rather than choosing by position. Add origin/destination, negation, and multiple-place tests.

3. **Med / P2 — Literal template pools are disjoint; expanded linguistic frames are not.**  
   **Location:** `router/data/gen_synthetic.py:113–117,171,184,191`; `router/tests/test_training.py:50–54`.  
   Train’s `"{take} {dest}"` expands to both **“walk me …”** and **“get me …”**, while dev explicitly contains `"walk me {dest}"` and test `"get me {dest}"`. The test checks literal template strings, so it passes despite this shared frame. Actual normalized-text comparisons found **zero train/dev/test overlap**, so this is not demonstrated verbatim leakage; it weakens the claimed template-held-out evaluation.  
   **Smallest fix:** Group equivalent expanded frames under a family ID and split by family. Add family/near-duplicate checks, or describe these splits as utterance-disjoint rather than genuinely frame-held-out.

4. **Med / P2 — Paraphrase/demo scores are development-set results, not independent validation.**  
   **Location:** `router/README.md:80–81,99–104`; `router/eval_sets.py:1–5`.  
   README honestly acknowledges that v0 failures were known while building v1, but also says these sets are used “only for evaluation”; `eval_sets.py` says they may not inform template writing. The generator specifically expands the failed home-navigation constructions. Excluding exact evaluation strings prevents direct memorization, **not tuning to their error patterns**.  
   **Smallest fix:** Call these regression/development sets consistently. Freeze v1 and collect an untouched speaker/PTT evaluation set before claiming generalization. Preserve the current caveat; the reported counts themselves reproduced correctly.

5. **Med / P2 — Calibration report cannot establish the required action-tool error gate.**  
   **Location:** `router/evaluate.py:102–126`.  
   The sweep reports only aggregate coverage and error across all eight labels. Architecture A requires action-tool high-confidence-wrong measurement; aggregate ECE and aggregate accuracy can hide false tool activations. The reproduced high-confidence navigation errors demonstrate why that distinction matters.  
   **Smallest fix:** Add per-tool/class accepted counts, wrong counts, coverage, false activations from NO_TOOL/OOD, and explicit denominators. Fit/select on dev and report separately on untouched test data.

6. **Low / P2 — Generated artifacts remain tracked; resolver documentation contains a false example.**  
   **Location:** tracked `router/__pycache__/*.pyc`, `router/nova_router.egg-info/*`; `router/README.md:113`.  
   Supplied Git status shows modified tracked bytecode and package metadata. `router/.gitignore:5–7` cannot untrack them. Separately, README claims `creator` matches `crater` at 0.92; reproduction of **“take me to the creator” returns `None`** at the default threshold.  
   **Smallest fix:** Remove generated artifacts from Git’s index while retaining ignore rules; correct or remove the resolver example.

### What is correct / verified

- **Results reproduced:** v1 test **306/320**, paraphrase **64/64**, demo **15/15**, expanded demo **47/47**; temperature **1.501**.
- **67 tests passed, two model-loading tests deselected** in the targeted verification. The complete claimed 78-test suite was not independently rerun.
- `core.py:77–82` rejects metadata label-order mismatch and restores saved temperature/max length.
- `train.py:127–140` selects the checkpoint on dev, fits temperature on dev logits, and scores test afterward. No test-based checkpoint selection is present in that loop.
- ECE and threshold-sweep formulas are correct for valid probability inputs. Metric tests include independently specified confusion/count examples, not merely shape assertions.
- The pipeline returns at most one tool route and never executes a class proposed by an LLM. Resolving **“fix the wifi at home” → homebase** is not itself a pipeline false activation: resolution is gated behind navigation/map classification.

### Remaining integration risks

Critical-alert preemption and numerical grounding are **not implemented here**. The placeholder emits no tool-derived answer, so this is not evidence of current fabricated telemetry; however, injected `llm_fn` output is returned unchecked. Real integration needs the separate alert path, number verifier, missing-slot handling, and timeout/error contracts.

Seeds are set, but cross-device deterministic execution is not established. MPS and ONNX INT8 remain unverified: measure target-Mac batch/single parity, confidence drift, memory and p95 latency, then revalidate calibration after export/quantization.

## Verdict

**v1 is credible as a narrow, synthetic text-demo baseline—not a validated speech router or Architecture A implementation.** The reported accuracy counts are honest and reproducible, but 47/47 does not establish the confidence gate: minimum demo confidence is **0.584**, including the safety request “should I turn back or keep going.”

**Merge verdict: OK with notes for an explicitly non-executing demo stub. Block real tool execution until threshold arbitration and destination ambiguity handling are implemented.**