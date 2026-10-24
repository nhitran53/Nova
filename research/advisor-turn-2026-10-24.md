# Advisor verdict: turn-layer review (2026-10-24)

I checked every finding in `research/review-turn-2026-10-24.md` against the code by reading and grepping it. I did not run the tests. I had no write tool, so the runtime has to save this response to `research/advisor-turn-2026-10-24.md`; the file may not exist yet.

**Verdict:** the review's BLOCK stands. Five of its findings are real and cheap to fix, and two of those are safety bugs: negative temperature (#2) and negation (#5). Two findings ask for decisions the user owns, so they should not be fixed silently: #3 (do document numbers count as a tool_result?) and D1 (removing keys from the policy file).

## Verdicts

| # | Finding | Verdict | Evidence / note |
|---|---|---|---|
| B1 | `SpeechError` has an empty body, so `speech.py` won't import | **Confirm** | `speech.py:20`. The fix is **only** to add `pass`. Do **not** restore the docstring: the user deleted it on purpose, so the review's "restore docstring" option is rejected. |
| B2 | Negative temperature silently suppresses the critical alert | **Confirm, High** | `temp_c.critical_below = -10.0`. The alert line formats `-20.0`, but the `DIGITS` pattern drops the sign, so the verifier sees 20.0, which isn't in the snapshot (-20.0). The alert line is blocked and the critical instruction is never spoken. This breaks "critical alerts bypass the LLM / always speak". |
| B3 | Numbers from retrieved passages are accepted as provenance | **Downgrade → DEFER (user decision)** | This is a deliberate design from the LLM-scope follow-up, and a test (`test_turn.py:595`) asserts it. Whether document text counts as a "tool_result" is the user's call. There are no documents yet, so it can't happen at runtime today. Don't change it silently. |
| B4 | The verifier doesn't fail closed on fractions, signs and spoken decimals | **Downgrade → SHOULD** | Real, but the LLM is a stub. Fixing B2 covers signs. Cheap partial fix: treat a short explicit list of numeric words as unsupported unless backed by a source (`half`, `quarter`, `third`, `dozen`, `couple`, `minus`). Don't add `point` or `few`: they would block ordinary speech. |
| B5 | "do not go / never go to the crater" navigates to the forbidden place | **Confirm, High** | `places.py:120`. The negation carry-over only fires when the previous word is `dont`. After "not go", the verb `go` resets the role to destination. |
| B6 | The API key could leak through the error message or a client-construction exception | **Downgrade → SHOULD** | Speculative: ElevenLabs doesn't normally echo the key back. Still worth a cheap defensive fix. |
| B7 | Confirming the refill call sends stale O2 numbers and stale urgency | **Confirm, Med → MUST** | Urgency must be decided by deterministic code on *current* state, and the user explicitly owns that rule. `_confirmation_turn` uses only `pending["feasibility"]`. |
| B8 | Tool and LLM exceptions skip the spoken fallback | **Confirm, Med → SHOULD** | The tools are fake today. Retrieval validation is the cheapest part. Keep the existing behaviour where an alert-source failure propagates. |
| B9 | `ToolRegistry(max_tools=9)` gets past the hard limit of 8 | **Confirm → MUST** | `tools.py:38`. Trivial. |
| B10 | `turn_policy.json` is missing from package data | **Confirm → MUST** | `pyproject.toml:29`. Trivial. A full clean-wheel test is a nice-to-have. |
| R1 | `_tool_turn` mixes too many concerns | **Confirm → SHOULD** | A pure extract, no behaviour change. Tests cover it well. |
| R2 | Nested conditional in `demo_turn.py:95–98` | **Confirm → SHOULD** | Trivial. |
| D1 | `uses_llm` and `spoken` in `severity_tiers` are never read | **Confirm, but alter the fix** | Don't remove keys from the user-owned policy file. Instead, make the loader refuse any tier with `preempts_router: true` and `uses_llm: true`. That turns the key into a guard and changes no values. Removing the keys, or wiring `spoken` up so warnings get spoken, is **DEFER** to the user. |
| D2 | Unused `topk` parameter on `Pipeline._tool` | **Confirm → SHOULD** | Only used at `pipeline.py:94,97,110`. No test calls `_tool` directly. |
| D3 | Duplicated placeholder and registry-size assertions (`test_turn.py:625–630`) | **Confirm → SHOULD** | Keep the check that the four section names exist. Drop the repeated assertions. |
| D4 | No-op monkeypatch and wrong comment (`test_speech.py:121`) | **Confirm → SHOULD** | Remove the line; the reload is what actually restores `_play`. |

## Fix list for the worker

### MUST fix now
1. **`speech.py`:** put `pass` in the body of `class SpeechError(RuntimeError):`. Change nothing else in that edit and restore no docstrings.
2. **Signed numbers in the verifier (B2):**
   - Let a leading `-` or `−` (Unicode minus) count as a sign only at a token start, i.e. not preceded by a word character, digit or `.`. This keeps a range like "5-10" from becoming -10.
   - Apply the sign the same way to spoken text and to digit runs inside source strings.
   - Spell the sign out when it's a word: "minus twenty" means -20.
   - Tests:
     - `verify_numbers("-5", {"v": 5})` fails.
     - `verify_numbers("-20.0", {"t": -20.0})` passes.
     - End-to-end: a TurnHandler snapshot with `temp_c=-20` speaks the temperature alert unblocked.
     - Two critical alerts at once (O2 plus negative temp) both get spoken.
3. **Negation carry-over (B5):** at `places.py:120`, change `tokens[i - 1] == "dont"` to `tokens[i - 1] in {"dont", "not", "never"}`.
   - Do **not** use the whole `NEGATION_WORDS` set. Words like `no`, `instead` and `rather` start an affirmative command: "no, take me to the airlock" and "rather go to the airlock".
   - Add place-resolver and TurnHandler tests for "do not go to the crater" and "never go to the crater", where the crater must not be the destination.
   - Add a regression test that "no, take me to the airlock" still resolves to the airlock.
4. **Fresh state on confirmation (B7):**
   - Store the navigation data (distance, place, `is_home`) in `pending`.
   - On "yes", rerun `assess_feasibility` against *this turn's* snapshot. Pass the snapshot into `_confirmation_turn`.
   - Build the message and urgency from the recomputed feasibility.
   - If the recomputed outcome no longer appears in `urgency_by_outcome`, i.e. no refill is needed any more, send nothing and speak a fixed line. Mark that line PLACEHOLDER in `turn_policy.json` under `confirmation`; this adds wording and changes no existing value.
   - Test: offer at 55 minutes, confirm at 20 minutes. The message carries the 20-minute figure and the urgency mapped from the new outcome.
5. **Hard cap (B9):** in `ToolRegistry.__init__`, raise `ValueError` unless `1 <= max_tools <= MAX_TOOLS`. Add a test for `ToolRegistry(max_tools=9)`.
6. **Package data (B10):** add `"turn_policy.json"` to package data in `pyproject.toml`. Optionally add a test that `importlib.resources` finds it.

### SHOULD fix (robustness, readability, dead code)
7. **Numeric words (B4):** treat `half`, `quarter`, `third`, `dozen`, `couple` as unsupported unless the source backs them. Add tests.
8. **Key redaction (B6):** redact the API key value from every `SpeechError` message. Wrap the client factory call in `init()` so a failure raises a sanitized `SpeechError`. Test with the secret placed in `body.detail.message` and in a factory exception.
9. **Spoken fallback on failures (B8):**
   - Catch exceptions from `registry.call` and `llm_fn` at their boundaries and speak `templates.tool_error` or a fixed LLM-error line. Never retry the tool.
   - Inside the existing retrieval `try`, validate the retriever's output: materialize it with `list(...)`, require a numeric `score`, and treat `None` as "no document".
   - Alert-source exceptions must still propagate, as they do now.
10. **D1 guard:** make the loader refuse a preempting tier that has `uses_llm: true`. Leave the JSON as it is.
11. **R1:** extract the navigation-enrichment helper so it returns `(tool_result, suffix, offer)`. Keep verification and the commit of pending state visible in `_tool_turn`.
12. **R2:** replace the nested conditional with explicit branches.
13. **D2:** remove `topk` from `_tool` and from both callers.
14. **D3:** dedupe the placeholder and registry assertions.
15. **D4:** remove the no-op monkeypatch and fix the comment.

### DEFER (the user decides or it's out of scope)
- **B3:** do retrieved document numbers count as a "tool_result"? Options are (a) keep the current behaviour, (b) block every number in LLM output, or (c) make retrieval a registry tool that counts against the 1-call limit.
- **D1 removal:** whether to delete `uses_llm`/`spoken` from `turn_policy.json`, or make `spoken` control whether warnings are spoken.
- **B7 follow-up:** whether a changed outcome should ask the user to confirm again instead of sending with the updated figures. MUST #4 sends with the fresh numbers, or skips the send when no refill is needed.
- **Clean-wheel install test (B10):** nice to have; needs a build environment.

## Policy and behaviour the user owns
- **No threshold, tier or wording values change in `turn_policy.json`.** MUST #4 *adds* one PLACEHOLDER line under `confirmation`. Flag it for sign-off.
- **B7** changes which urgency gets sent. The mapping itself is untouched; only the input is fresher. This brings the code in line with the user's rule rather than overriding it.
- **B5** only changes how negation is parsed, which is a safety fix. The choice to leave out `no`, `instead` and `rather` is deliberate.

## Risks to tests
- **B1** must land first. Until `speech.py` imports again, the speech, TTS and demo tests can't even be collected, so the baseline count of 369 isn't reproducible right now.
- **B2 sign parsing** could break existing tests whose sources contain hyphenated strings or ranges. The lookbehind rule above avoids that. Run `test_turn.py` and the verifier tests.
- **B5:** existing negation tests (`dont go`, "avoid the crater") must still pass. Run `test_places.py` and the destination-aware place tests.
- **B7:** the existing confirmation tests in `test_turn.py` (~474–540) probably assume the stale feasibility is sent unchanged. Update their expectations; don't delete them.
- **D2:** `test_pipeline.py` doesn't call `_tool` directly, so this is low risk.
- **D3/D4:** these edit tests only, so the total count will drop slightly. Report the new count.

## Suggested prompt for the worker
> In `/home/isaach/Nova/router`, on branch `feat/router`: no commits, no branch switches, no retraining, no real LLM or ElevenLabs calls. Don't edit any values in `turn_policy.json`, `demo_calls*.json` or `data/paraphrase_test.jsonl`.
>
> 1. Implement MUST items 1–6, then SHOULD items 7–15, from `research/advisor-turn-2026-10-24.md`. For `speech.py`, only add `pass` to `SpeechError`; restore no docstrings.
> 2. Add the tests listed for each item.
> 3. Done when `uv run pytest` is green with 0 skips. Report the before and after counts, the files changed, and the one new PLACEHOLDER line added under `confirmation` in `turn_policy.json` (needs user sign-off).
> 4. Append a 2–3 line, 2026-10-24 entry to `NOTES.md`.
> 5. List the three DEFER items as decisions still open for the user.