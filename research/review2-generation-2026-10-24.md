## Review

**Scope:** Uncommitted working-tree changes and the named generation, verifier, turn, policy, tests, and documentation files. Read-only review; no files changed.

### Finding: P1 — Unicode compatibility characters bypass the new prose guards

**Locations:** `router/generation.py:654–665`; `router/verifier.py:19,33–39`; final speech gate at `router/turn.py:416–422`.

The placeholder check recognizes only ASCII `[` and `]`. The digit check and verifier both use `\d`, which recognizes Unicode decimal digits but **not** superscript or circled digits. Neither layer normalizes compatibility characters before checking.

Consequently, these scripted replies pass the shipped policy’s off-mode postprocessing and the final default verifier:

```json
{"refusal": null, "text": "The cycle takes ［number］ minutes."}
{"refusal": null, "text": "The cycle takes ² minutes."}
{"refusal": null, "text": "The cycle takes ⑫ minutes."}
```

**Source proof:**
- Fullwidth brackets do not satisfy the ASCII bracket condition at `generation.py:655`.
- Superscript `²` and circled `⑫` do not match `\d`; neither contributes an English number word.
- The remaining length, slot, and deny-list checks accept these sentences.
- `verify_numbers` finds neither a numeric value nor a vague-quantity word, so returns `ok=True`. `_speak` therefore does not stop them.

This leaves the new “no prose numbers” and “no bracketed placeholder echoes” protections bypassable, including when a risk term forces passage numbers off. These are static, code-traced reproductions, **not executed probes**.

**Smallest fix:** Apply Unicode compatibility normalization consistently before number/placeholder checks, and ensure the verifier checks the same normalized representation. For the generator boundary, rejecting changed numeric/bracket compatibility characters is also reasonable. Add regression cases for fullwidth brackets, superscript/circled digits, and ordinary fullwidth/Arabic-Indic decimal digits in both off and digits modes.

### Correct: Prior fixes substantially verified

- **ASCII mask echoes:** `[number]`, `[ number ]`, stray brackets, and nested ASCII bracket placeholders are rejected by `router/generation.py:655–656`. Regression tests cover case/spacing and unmatched brackets at `router/tests/test_generation.py:1006–1035`. Unicode variants remain the finding above.
- **Scale and vague quantities:** `million`, `billion`, and `trillion` are recognized scales; `twice`, `several`, plural scales, and `dozens` are fail-closed vague words (`router/verifier.py:19–28,98–108`). Tests cover scale composition and vague quantities at `router/tests/test_generation.py:1038–1050`.
- **GBNF:** Sentence-ending punctuation now exists, slots are explicit alternatives, and the sentence/piece product cannot exceed the template-word budget (`router/generation.py:454–491`). Structural regex tests check filler compatibility and boundary budgets (`router/tests/test_generation.py:1071–1146`). Actual runtime grammar execution remains unverified; the documentation correctly calls it illustrative.
- **Injected verifier:** `postprocess` invokes the injected verifier, and `TurnHandler` passes it through `run_generation` (`router/generation.py:697`; `router/turn.py:258–259`). Tests exercise rejection followed by a successful retry.
- **Attribution budget and duplication:** Prefix words count before acceptance (`router/generation.py:668–670`); existing attribution leads avoid a second prefix (`router/generation.py:332–346`). Boundary and integration tests appear at `router/tests/test_generation.py:1170–1202`.
- **JSON serialization:** Live request objects are kept on `last_generation_request`; result dictionaries contain request summaries instead. Tests serialize generated, template-only, and fallback results and check per-turn reset (`router/tests/test_generation.py:1205–1231`).
- **Final gate:** Successful, refused, error, and fallback generation paths submit their selected line through `_speak`. Recording-verifier tests specifically cover replacement lines (`router/tests/test_generation.py:691–718`).
- **Critical isolation:** Critical alerts return before routing or request construction (`router/turn.py:205–210`).
- **Registry cap:** Limits above eight and registrations beyond the configured cap remain rejected (`router/tools.py:38–49`).
- **Policy preservation:** The policy diff adds the generation section and removes only the authorized dead `severity_tiers.*.spoken` keys. Existing policy values are otherwise unchanged.
- **Tests:** Meaningful assertions cover the requested fixes; no skip/xfail markers were found in `router/tests/test_generation.py`.
- **Secrets:** No secrets observed in the inspected changes.

### Adversarial assessment

| Input | Code-traced outcome |
|---|---|
| `[ number ]`, `[[number]]` | Rejected as placeholder prose |
| `［number］` | **Accepted incorrectly** |
| `a million` | Rejected without numeric provenance |
| `millions`, `dozens`, `twice`, `several` | Rejected without matching source words |
| Fullwidth/Arabic-Indic decimal digits | Detected by `\d`; prohibited in off-mode prose |
| Superscript/circled digits | **Accepted incorrectly** |
| Nested curly placeholders | Rejected by malformed/unknown-slot checks |

### Validation limits

No shell/execution or write-capable tool is available in this review session. I could not independently verify the documented **623 passed, zero skipped**, or write `/home/isaach/Nova/research/review2-generation-2026-10-24.md`. This response is the complete findings artifact.

Required supervisor test command:

```bash
cd router && NOVA_MODELS_DIR=/home/isaach/models/Nova/router uv run pytest -q
```

### Merge verdict: BLOCK

The requested second-round fixes are substantially present and meaningfully tested, but Unicode compatibility characters still bypass the newly introduced number/placeholder guards. Fix that boundary and run the full suite before merging.