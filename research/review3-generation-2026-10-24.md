## Review

**Scope:** Uncommitted Unicode-folding changes in `router/verifier.py`, generation postprocessing, regression tests, and the documentation’s “Second review fixes.” Read-only review; no files changed.

### Finding: P1 — Invisible combining characters still bypass number and directive guards

**Locations:** `router/verifier.py:38–40,60–65,81`; `router/generation.py:591–592,635–640,678–683`.

`strip_format_chars` removes only category **Cf** characters. Combining Grapheme Joiner (`U+034F`) and variation selectors such as `U+FE0F` are category **Mn**, survive NFKC, and split the ASCII regex tokens used by both generation checks and the verifier.

Code-traced examples:

```text
The cycle takes fi\u034fve minutes.
The cycle takes fi\uFE0Fve minutes.
Please re\u034fturn to base.
```

The first two contain no recognized digit or complete English number token after folding; the third contains no recognized `return` directive token. These short, slot-free replies consequently pass the relevant guards and default final verifier.

Mixed-script homoglyphs have the same remaining boundary weakness: `The cycle takes fіve minutes.` with Cyrillic `і` (`U+0456`) is neither folded into `five` nor rejected.

**Smallest fix:** Extend the generator boundary to reject or safely normalize unsupported invisible/default-ignorable characters, including Mn variation selectors and CGJ. Establish an explicit handling rule for mixed-script confusables rather than assuming NFKC converts them. Apply equivalent numeric-token handling in the verifier. Add off-mode, chitchat, directive, and final-handler regressions for these examples.

These are **static reproductions**, not executed probes.

### Finding: P1 — NFKC turns vulgar fractions into unrelated allowed numbers

**Locations:** `router/verifier.py:60–65,73–75,108–116`; `router/tests/test_generation.py:1291–1297`.

NFKC transforms `½` into `1⁄2`, which the digit scanner interprets as **two independent numbers: 1 and 2**, not 0.5. The new test explicitly expects `[1.0, 2.0]`, pinning the incorrect interpretation.

Because source strings use the same folding, this change also introduces false provenance:

```python
verify_numbers("The cycle takes 2 minutes.",
               ["The cycle takes ½ minute."])
```

The source now supplies allowed values `[1, 2]`, so the unsupported two-minute claim passes. Before this change, that source contained no recognized digits and did not authorize `2`.

Rejecting non-ASCII numerals in generated prose does **not** prevent this case: the generated answer uses ordinary ASCII digits, while the fraction occurs in the retrieved source.

**Smallest fix:** Recognize vulgar fractions by numeric value before compatibility decomposition, or fail closed on fractional source tokens until fraction parsing is supported. Do not add their numerator and denominator independently to allowed provenance. Replace the `[1, 2]` expectation and add an assertion that a `½` passage cannot authorize `2`.

### Correct

- **Previously reported inputs are blocked at the generator boundary:** fullwidth brackets, superscript/circled digits, `½`, Arabic-Indic/fullwidth digits, zero-width-space splits, and soft-hyphen splits have explicit regression coverage at `router/tests/test_generation.py:1234–1285`. Normalized text reaches the checks at `router/generation.py:635–640`.
- **`blocked_text` normalization is intentional and tested:** handler regressions compare it with `fold_text(text)` at `router/tests/test_generation.py:1263–1277`. Original generator output remains available separately as `raw`.
- **Existing ASCII signed-number behavior is preserved by the folding step:** ASCII `-20.0`, decimals, and `minus twenty` are unchanged. Existing assertions cover negative values, Unicode minus, ranges, identifier hyphens, and negative-temperature alerts at `router/tests/test_turn.py:315–356`.
- **ASCII false positives:** No additional folding-induced change was identified for ordinary ASCII hyphens or ordinals. Existing strict number/vague-word behavior remains; this is not evidence of general ordinal support.
- **Format controls:** Bidi/RTL controls, tag characters, and soft hyphen are covered by category-Cf removal (`router/verifier.py:38–40`). Variation selectors are not, as noted above.
- **Hard rules remain intact:** critical alerts return before routing/request construction (`router/turn.py:205–210`); generated speech still goes through `_speak` (`router/turn.py:416–422`); registry configuration and registration enforce the eight-tool cap (`router/tools.py:11,38–49`).
- **Policy/secrets/skips:** The policy diff preserves existing values apart from the previously authorized removal of dead `spoken` keys. No secrets observed in the inspected changes. No skip/xfail markers found in the generation tests.

### Validation limits

No execution or write-capable tool is available in this session. I could neither independently verify the documented **681 passed, zero skipped** nor write `research/review3-generation-2026-10-24.md`.

Required validation:

```bash
cd router && NOVA_MODELS_DIR=/home/isaach/models/Nova/router uv run pytest -q
```

### Merge verdict: BLOCK

The originally named Unicode examples are now rejected by generation postprocessing, but invisible Mn characters still bypass the guards, and fraction folding introduces incorrect numeric provenance. Address both findings and run the full suite before merging.