# Turn-layer code review

**Scope/limitations:** Read-only review of the named files and relevant tests in the current checkout. The available diff tool exposes only working-tree changes, not commits `0d8e0ca` and `ec2c679`; their exact committed diffs were therefore **not verified**. The checkout additionally contains an uncommitted `speech.py` change, identified separately below. Tests were inspected, not executed. Paths below are relative to `router/`.

## Bugs/Risks

1. **High — Current working tree cannot import `speech.py`.**  
   **Location:** `speech.py:20–23`  
   `class SpeechError(RuntimeError):` has no body. The working-tree diff shows removal of its sole docstring without adding `pass`, producing an `IndentationError`. This prevents collection of speech/demo tests and importing `demo_turn`, even without `--speak`. **This is an uncommitted regression, not attributed to the two commits.**  
   **Fix:** Restore the class docstring or add `pass`.

2. **High — Negative temperatures can suppress the entire critical alert.**  
   **Location:** `verifier.py:14,23–24,69–71`; `turn.py:178–179`  
   `DIGITS` excludes signs. For a snapshot containing `temp_c=-20.0`, the critical template says `-20.0`, but the verifier extracts positive `20.0` and compares it against negative `-20.0`. With no matching positive value elsewhere, `_block` replaces the critical instruction with the generic fallback. Joining multiple alerts means this can suppress other critical instructions too.  
   **Fix:** Parse signed numeric values consistently, including Unicode minus if accepted; add end-to-end tests for negative-temperature critical alerts and simultaneous alerts.

3. **High — Retrieved text is accepted as numeric provenance despite the tool-result hard rule.**  
   **Location:** `turn.py:210–211`; `tests/test_turn.py:595–599`  
   Grounded LLM output is verified against retrieved passage strings, not a `tool_result`. The test explicitly requires speaking `384400` obtained only from `PASSAGE`. This contradicts the requested rule that every spoken number traces to a tool result.  
   **Fix:** Represent retrieval as an explicit, provenance-bearing tool result and account for it under the one-call limit, or block numerical LLM output unless backed by the permitted tool result. Update the test to enforce that contract.

4. **High — Number verification is not fail-closed for unsupported numeric expressions.**  
   **Location:** `verifier.py:14–20,27–48,69–73`  
   An LLM reply such as “You have half a minute left” contains no recognized number words, so verification against `{}` succeeds. Signs are also discarded: `verify_numbers("-5", {"value": 5})` succeeds despite changing the value’s sign. The documented parser limitations do not satisfy the universal spoken-number rule.  
   **Fix:** Support the numeric forms permitted in speech and reject unsupported numeric constructions before TTS. Add adversarial tests for fractions, signed values and spoken decimals; alternatively constrain LLM speech to a tightly validated nonnumeric format.

5. **High — Negation is lost when followed by an ordinary navigation verb.**  
   **Location:** `places.py:120–123`  
   Only the immediately adjacent `dont go` pattern preserves negation. In “do not go to the crater” or “never go to the crater”, `not`/`never` sets the negated role, but `go` resets it to destination. A confident navigation classification consequently executes navigation to the expressly forbidden place. Existing negation tests cover `dont`, not these forms.  
   **Fix:** Preserve command negation across its verb and destination span; reset it only at an explicit new affirmative clause/contrast. Add resolver and TurnHandler regressions for both examples.

6. **High — Speech errors can still expose the ElevenLabs key.**  
   **Location:** `speech.py:43,55–67`; `turn.py:163–164`; `tests/test_speech.py:99–110`  
   The request exception handler copies `body["detail"]["message"]` verbatim into `SpeechError`; an error body containing the key therefore reaches CLI stderr or `result["tts_error"]`. Client construction also happens outside that handler, allowing a factory exception to propagate unsanitized. The secrecy test puts the secret only in `str(exc)` and uses a benign body, leaving both paths untested.  
   **Fix:** Do not expose arbitrary SDK/server error text. Emit allowlisted error categories and validated status codes; sanitize client-construction failures too. Test secrets in the response body and factory exceptions, and assert their absence from errors and turn results.

7. **Med — Confirmation sends stale oxygen readings and potentially stale urgency.**  
   **Location:** `turn.py:181–184,224–233`  
   The confirmation turn polls fresh telemetry but `_confirmation_turn` uses only the previous offer’s feasibility data. Example: offer at 55 minutes (`normal`), then confirm at 20 minutes with otherwise noncritical telemetry. The message still claims 55 minutes and retains `normal`, although the current outcome is `cannot_reach`/`urgent`.  
   **Fix:** Pass the new snapshot into confirmation handling and recompute deterministic feasibility/urgency before sending. Reconfirm if the change invalidates the original offer. Test telemetry changes between offer and acceptance.

8. **Med — Operational failures bypass the promised spoken fallback.**  
   **Location:** `turn.py:205,210,215–222,236,260`  
   Tool and LLM exceptions escape the turn entirely. Retrieval catches invocation failures, but not invalid/lazy results: `None`, a generator that raises during iteration, or a passage lacking `score` fails outside the `try`. Consequently `handle` never reaches TTS or returns its normal result structure. Existing retrieval-error coverage tests only a retriever that raises immediately.  
   **Fix:** Catch operational tool/LLM failures at their boundaries and produce deterministic error replies without retrying a tool. Validate/materialize retrieval results inside its error boundary. Preserve the intentionally propagating alert-source failure contract.

9. **Med — Eight-tool ceiling is configurable beyond the hard limit.**  
   **Location:** `tools.py:38–47`  
   `ToolRegistry(max_tools=9)` permits registration of nine tools; the constructor never caps or rejects values greater than `MAX_TOOLS`. The test exercises only the default constructor.  
   **Fix:** Reject limits outside `1..MAX_TOOLS` or enforce the absolute ceiling independently. Add an explicit oversized-constructor test.

10. **Med — Distribution configuration omits the new required policy file.**  
    **Location:** `pyproject.toml:28–29`; `turn.py:48–54`  
    Package data lists `labels.json`, `places.json` and `pipeline_config.json`, but not `turn_policy.json`. A clean build whose included files follow this explicit list lacks the policy required by default `TurnHandler` construction. `tests/test_package.py:30–36` checks only labels from the editable package, not an installed wheel.  
    **Fix:** Add `turn_policy.json` to package data and test `load_policy()` from a clean wheel installation outside the source checkout.

## Readability

1. **Low — Turn orchestration mixes several independently changing concerns.**  
   **Location:** `turn.py:254–283`  
   `_tool_turn` executes the tool, formats its reply, computes feasibility, rewrites result data, selects outcome wording and constructs confirmation state inside one block. The long inline offer condition obscures when state becomes pending.  
   **Fix:** Extract a small navigation-enrichment helper returning the augmented result, speech suffix and optional offer; keep verification and pending-state commitment visible in `_tool_turn`.

2. **Low — Demo route formatting is unnecessarily difficult to follow.**  
   **Location:** `demo_turn.py:95–98`  
   A nested conditional expression combines alert/confirmation explanations and pipeline formatting across a long statement.  
   **Fix:** Replace it with an explicit `if pipe is None` branch and a simple alert-versus-confirmation branch.

## Duplicates/Dead code

1. **Low — Severity configuration advertises controls that are never read.**  
   **Location:** `turn_policy.json:7–13`; `turn.py:174–179`  
   Scoped searches found `uses_llm` only in the JSON; severity-tier `spoken` is likewise not consumed. Runtime behavior reads only `preempts_router`: critical alerts are always spoken without the LLM and warnings are never spoken. Changing the other keys has no effect.  
   **Fix:** Remove the inert keys or label them explicitly as descriptive metadata. Do not make critical-alert LLM bypass configurable.

2. **Low — `_tool` receives an unused `topk` parameter.**  
   **Location:** `pipeline.py:94,97,110–126`  
   Both callers pass `topk`, but the method never uses it; scoped search confirms its uses belong to arbitration and intent clarification instead.  
   **Fix:** Remove the parameter and corresponding keyword arguments.

3. **Low — New-section placeholder test repeats existing assertions.**  
   **Location:** `tests/test_turn.py:625–630`  
   Its per-section status assertions repeat `test_every_policy_section_is_labelled_placeholder` at lines 331–336. Its registry-size assertion repeats the stronger exact-name/size checks at lines 212–215.  
   **Fix:** Retain only an assertion that the four specifically required section names exist, or fold that into the earlier policy test; remove the repeated registry check.

4. **Low — Player test contains a no-op monkeypatch and misleading explanation.**  
   **Location:** `tests/test_speech.py:120–122`  
   `monkeypatch.setattr(speech, "_play", speech.__dict__["_play"])` assigns `_play` to itself—the fixture’s fake—not “the real one” as the comment claims. The following module reload is what actually restores the implementation.  
   **Fix:** Remove the no-op assignment and accurately document the reload, or restructure the fixture to avoid replacing `_play` for player-specific tests.

## Merge verdict

**BLOCK.** Resolve the high-severity safety/provenance/secrecy findings and the current working-tree syntax error before merging.

**Verification to run after fixes:** from the repository parent, using the project environment, run the focused turn, speech, TTS, demo, pipeline, places, report-gate and package tests; additionally build/install a wheel in a clean environment and instantiate `TurnHandler` there. No live ElevenLabs request is needed for these checks.