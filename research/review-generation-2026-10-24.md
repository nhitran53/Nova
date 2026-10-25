# Review: post-tool-call generation

Intended artifact: `/home/isaach/Nova/research/review-generation-2026-10-24.md`

Reviewed the uncommitted diff and all four named new files. **Two P1 findings and one P2 finding. No files changed.**

## Findings, ranked by severity

### P1 — Generation fallback bypasses the required final `_speak` gate

**Location:** `router/turn.py:258–268`, `router/turn.py:402–405`

When generation exhausts its attempts, `_llm_reply` calls `_block(..., replacement=outcome.text)` directly. `_block` assigns that replacement to `spoken` without calling `_speak` or verifying the replacement. `handle` subsequently sends `result["spoken"]` to TTS (`router/turn.py:184–187`).

The verifier invoked at line 263 checks the **rejected model text**, not the replacement that will actually be spoken. This makes the newly introduced generation-fallback path bypass the explicitly required final gate.

The shipped neutral fallback is number-free, and the policy loader checks generation fixed lines, so this is **not evidence of an unsupported number escaping under the shipped configuration**. It is nevertheless a concrete gate-contract violation, and the documentation’s claim that generation always ends at `_speak` is inaccurate (`router/docs/post-tool-generation.md:13,22`).

**Smallest fix:** Verify the replacement through `_speak` before returning it, while preserving the original rejected text and generation diagnostics separately.

**Regression test:** Use a recording verifier and force generation fallback. Assert that the final spoken replacement—not just rejected model prose—is submitted to the final verifier.

### P1 — Deeply nested JSON can crash the turn instead of producing retry/fallback

**Location:** `router/generation.py:528–531`, `router/generation.py:637`

`postprocess` catches `TypeError` and `ValueError` from `json.loads`, but deeply nested JSON can raise `RecursionError`. The call to `postprocess` in `run_generation` is outside the exception handler surrounding `generator.generate`, so this exception propagates out of `TurnHandler.handle`.

**Static reproduction:** Supply a scripted generator response consisting of several thousand opening brackets followed by matching closing brackets. At normal Python recursion limits, JSON decoding raises `RecursionError` before the reply can be classified as a bad shape. This reproduction was not executed in this review environment.

That contradicts `postprocess`’s “never raises on model output” contract (`router/generation.py:524–525`). The current odd-output test covers only a handful of shallow values (`router/tests/test_generation.py:307–309`).

**Smallest fix:** Catch `RecursionError` at the JSON parsing boundary and classify it as `parse_error`. Consider an explicit raw-output size limit as additional protection.

**Regression test:** Exercise nested JSON through both `postprocess` and `TurnHandler`, asserting bounded retry/fallback and no propagated exception.

### P2 — Enabling bounded prompt history retains unbounded session history

**Location:** `router/turn.py:243–246,270–271`

`history_turns` bounds only the slice placed into the next request. Every accepted LLM turn is appended to `self.history`, and old entries are never discarded.

With `history_turns=1`, a long-lived handler still retains every accepted user utterance and assistant response. This creates unnecessary memory growth and retains raw utterances beyond the useful history window. The shipped default of zero avoids the issue until the optional feature is enabled.

**Smallest fix:** Trim stored history to `history_turns` after appending.

**Regression test:** Run more accepted turns than the configured window and assert both stored-history length and request-history contents.

## Correct / verified

- **Critical alerts preempt generation:** the alert branch returns before pipeline routing or request construction (`router/turn.py:199–209`). Tests check generator non-invocation for four router labels (`router/tests/test_generation.py:610–617`).
- **Generator output does not choose router routes or execute tools:** `_llm_reply` fixes the route to `"llm"`; tool turns retain template speech.
- **Existing policy values are unchanged:** the inspected `turn_policy.json` diff contains additions only.
- **Tool cap remains enforced:** `ToolRegistry` rejects limits above eight and refuses registration beyond its configured cap (`router/tools.py:38–49`); six tools are registered (`router/tools.py:122–130`).
- **Number safeguards are substantive:** prose scanning precedes slot filling, filled output receives a number backstop, and accepted generated replies reach `_speak` (`router/generation.py:557–605`, `router/turn.py:269`). History is not added as numeric provenance.
- **Escaping is correctly limited to delimiter protection:** angle brackets and ampersands are escaped. This does not establish general resistance to natural-language prompt injection.
- **Tests are meaningful overall:** they cover critical-alert isolation, slot filling, grounding modes, retry/fallback, refusals, and integration. No skip or xfail markers were found in the new test file. The “every violation exercised” test compares a hard-coded set rather than measuring exercised behavior (`router/tests/test_generation.py:245–249`).
- **No secrets observed** in the inspected changes and named new files.

## Report accuracy and validation limits

The report documents the important `digits` provenance exception and missing semantic/unit binding. Its blanket “Numbers never come from the model” statement (`router/docs/post-tool-generation.md:18`) should be qualified: grounded `digits` mode intentionally permits model-written numbers backed by passages.

**Requested pytest run was not executed.** This reviewer has no execution tool and is prohibited from using shell commands. The reported **535 passed, zero skipped** result therefore remains independently unverified. The required command is:

`cd router && NOVA_MODELS_DIR=/home/isaach/models/Nova/router uv run pytest -q`

No write-capable tool is available either, so this response contains the complete findings artifact rather than claiming the file was written.

## Merge verdict

**BLOCK** pending correction of the final-gate bypass and JSON parsing failure path, followed by the requested test run.