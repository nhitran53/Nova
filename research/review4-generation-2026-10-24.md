## Review

**Summary:** The two original P1 examples are structurally fixed, but the new allowlist configuration can re-enable homoglyph bypasses, and ASCII-only number evasions still pass both checks. **Merge verdict: BLOCK.**

Read-only review of the current uncommitted files. Reproductions below are **code-traced, not executed**; this session exposes no command-execution tool.

### Findings, ranked

#### P1 — `allowed_extra_chars` accepts letters, reopening the homoglyph bypass
**Location:** `router/generation.py:211–213,647–649`

The validator promises “visible non-ASCII punctuation,” but rejects only Unicode categories beginning with `C`, `M`, `Z`, or `N`. **Letters (`L`) and symbols (`S`) are accepted.**

Concrete reproduction:
1. Set `generation.allowed_extra_chars` to Cyrillic `і` (`U+0456`, category `Ll`).
2. Policy validation succeeds.
3. Submit `The cycle takes fіve minutes.` in off mode or chitchat.
4. The allowlist accepts `і`; NFKC leaves it unchanged; neither prose scanning nor `verify_numbers` recognizes `five`. The unsupported quantity therefore passes.

The shipped configuration does not contain that letter, but the newly supported configuration path contradicts its validation contract and reopens the exact class of bypass being fixed.

**Smallest fix:** Require punctuation categories explicitly, or restrict this setting to a fixed approved punctuation set. Add negative policy-validation tests for Cyrillic/Greek letters, fullwidth letters, and symbols.

#### P1 — ASCII-separated number words and spelled ordinals pass as number-free
**Location:** `router/generation.py:675–684`; `router/verifier.py:103–130,157–173`

The printable-ASCII allowlist is not sufficient to enforce the documented off-mode rule that prose contains **no number at all**. Both checks depend on complete recognized number-word tokens.

These short, slot-free replies pass the current off-mode/chitchat checks and the final verifier with an empty source:

| Model text | Code-traced result |
|---|---|
| `The cycle takes f i v e minutes.` | Accepted; no recognized numeric token |
| `The cycle takes fi-ve minutes.` | Accepted; tokens `fi` and `ve` are unrecognized |
| `The fifth cycle is ready.` | Accepted; `fifth` is absent from both word sets |

The same boundary also permits split directives such as `Please re-turn to base.`: `_tokens` produces `re`, `turn`, not the denied token `return`.

**Smallest fix:** Add conservative rejection or a separate detection-only normalization for split number/directive spellings, and explicitly fail closed on spelled ordinals until supported. Preserve the original spoken text rather than globally deleting ordinary spaces or hyphens. Add handler regressions proving retry/fallback and final-gate behavior.

**Other requested probes:** `f1ve` is rejected in off mode/chitchat because it contains a digit; `a couple` is rejected by the existing vague-quantity check. Those are not additional findings.

### Correct

- **CGJ and variation-selector examples are blocked:** the pre-fold allowlist rejects them at `router/generation.py:647–649,669–670`; verifier folding also removes the designated invisible Mn characters (`router/verifier.py:42,50–52,79–87`). Regression coverage includes off mode, chitchat, grounded digits mode, and handler retry/fallback (`router/tests/test_generation.py:1345–1435`).
- **Default-policy homoglyph examples are blocked:** Cyrillic/Greek letters and fullwidth letters fail the pre-NFKC allowlist. The configuration loophole above is distinct.
- **Vulgar-fraction false provenance is fixed:** fractions become an unsupported marker instead of independent numerator/denominator numbers (`router/verifier.py:79–87,163–164`). Tests explicitly deny authorization of `1`, `2`, and `0.5` from `½`, including a handler regression (`router/tests/test_generation.py:1298–1314,1444–1451`).
- **Tool-supplied non-ASCII text is not subjected to the model-prose allowlist:** tool templates still reach `_speak` directly (`router/turn.py:373–392,416–422`). Ordinary accented place names such as `Café` or `São Paulo` introduce no number and remain acceptable by code trace. Tool text containing a fraction is deliberately rejected by the new fail-closed verifier. I found no added regression test specifically exercising accented tool-supplied place names through the handler.
- **Critical alerts still bypass generation:** the critical branch returns before routing/request construction (`router/turn.py:207–209`); tests assert no generator calls across router labels (`router/tests/test_generation.py:646–652`).
- **Final gate and registry cap remain:** `_speak` invokes the verifier (`router/turn.py:416–422`); registry construction and registration enforce the eight-tool maximum (`router/tools.py:11,38–49`).
- **Existing policy values are preserved**, except the previously authorized removal of dead `severity_tiers.*.spoken` keys.
- No secrets observed in the inspected changes. No skip/xfail markers found in `test_generation.py`; other test files contain pre-existing conditional skips.

### Validation limit

The documentation reports **761 passed, zero skipped**, but I could not independently execute or confirm that result. Required validation:

```bash
cd router && NOVA_MODELS_DIR=/home/isaach/models/Nova/router uv run pytest -q
```

**Merge verdict: BLOCK** — fix the punctuation-validation loophole and the demonstrated ASCII guard bypasses, add regressions, and rerun the full suite.