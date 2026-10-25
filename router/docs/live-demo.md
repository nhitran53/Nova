# Live LLM demo (router/llm_generator.py, router/demo_live.py)

Status: built and tested against a FAKE OpenAI-compatible server only. It has **never been run against a real Qwen**. Every value in
`turn_policy.json["live_llm"]` is PLACEHOLDER. Runtime research: `research/live-llm-runtime.md`.

## Start a model server (default: llama-server + Qwen3.5-4B Q4)

```
llama-server -hf unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL -ngl 99 -c 4096 --chat-template-kwargs '{"enable_thinking":false}' --port 8080
```

Other runtimes (pick with `--runtime`, `--llm-model`, `--base-url`):

| runtime | launch | flags |
|---|---|---|
| `llama-server` | the command above | default; sends `chat_template_kwargs` thinking-off and `json_schema` (or `grammar` with `--constrain gbnf`) |
| `mlx` | `mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit --chat-template-args '{"enable_thinking":false}'` | `--runtime mlx --llm-model mlx-community/Qwen3.5-4B-4bit`; no constraint, no seed, thinking-off is launch-time only |
| `ollama` | `ollama run qwen3.5:4b` | `--runtime ollama --llm-model qwen3.5:4b --base-url http://localhost:11434/v1`; `response_format` + `reasoning_effort:"none"` (unverified; known `format`+`think:false` bug on some models) |
| `openai` | any hosted/OpenAI-compatible endpoint | `--runtime openai --base-url ... --llm-model ...`; key from env `NOVA_LLM_API_KEY` only |

If the server rejects the schema field, try `--constrain none` (postprocess still validates everything).

## Run the demo

```
cd router
uv run python -m router.demo_live                               # interactive; real router checkpoint ($NOVA_MODELS_DIR/v2)
uv run python -m router.demo_live --fake-router                 # keyword router, no checkpoint needed
uv run python -m router.demo_live --fake-router --no-llm        # fully offline (FillerGenerator)
uv run python -m router.demo_live --script turns.txt            # or: printf 'hi\n/quit\n' | uv run python -m router.demo_live
```

Useful flags: `--advance SECONDS` (sim seconds before every turn), `--deadline-ms N` (override `generation.deadline_ms`, 1500 by
default; a CPU-only box will probably need more), `--constrain none|json_schema|gbnf`. Env overrides: `NOVA_LLM_RUNTIME`,
`NOVA_LLM_BASE_URL`, `NOVA_LLM_MODEL`, `NOVA_LLM_CONSTRAIN`, `NOVA_LLM_API_KEY`.

Commands in the prompt: `/advance N`, `/pos X Z` (move the astronaut, meters; starts at 0 0), `/set o2_pct=14` (also `bpm`, `temp_c`, `suit_pressure_kpa`, `scenario=o2_leak`, `activity=...`),
`/sim`, `/help`, `/quit`.

If no server answers, one line says so (`LLM: no LLM server reachable at URL (...); falling back to the offline FillerGenerator`) and the
demo continues offline. If the server fails mid-session the turn speaks `templates.tool_error` and prints `llm_error`.

Each turn prints: router label/confidence/route, tool call, whether the LLM was called, request kind, the raw model output per attempt,
the final spoken line, verifier verdict + `blocked_text`, violations and latency (generator-reported ms and wall ms). Tool turns stay
template-only and critical alerts bypass the LLM. The adapter returns raw text only; `postprocess` does all checking and number
verification.

## Router checkpoint and demo places
- `demo_live` defaults to `$NOVA_MODELS_DIR/v2` (one message and a fall back to `v1` if it is missing); `--model PATH` picks another,
  e.g. `--model /home/isaach/models/Nova/router/v1`. v1 scores short questions such as "how far is base 2" at 0.87 (< `tau_hi` 0.9, so
  it asks you to repeat); v2 scores 0.965. `demo_turn` still defaults to v1.
- Fixture places added for the demo (PLACEHOLDER `turn_policy.json["demo_places"]`, fixture only; a real DB needs the rows): `base 2`
  (aliases "base2", "second base", "base b"; 1000 m) and the `tower` / `comms tower` (500 m), measured from the astronaut position.
  "base two" / "base 3" deliberately resolve to nothing (a number that is not part of the alias never falls back to homebase).
- "how far is X" / "distance to X" speaks `distance_templates.navigation_distance` (PLACEHOLDER) from the same navigation tool_result;
  "take me to X" still says "Heading to ...". Use `/pos 500 0` then "how far is home" to see a non-zero homebase distance.

## Gotchas
- The request timeout is `min(live_llm.request_timeout_cap_s, remaining generation.deadline_ms)`; a first request that loads the
  model will time out at 1500 ms. Warm the server first (`curl localhost:8080/health`) or raise `--deadline-ms`.
- `<think>` blocks are stripped and `reasoning_content` is ignored, but check the raw line for leaked reasoning on a real model.
- Proxy environment variables are ignored on purpose (localhost only).

### Speaking scope after the local transcript

On already-authorized `NO_TOOL` turns, `hi` and exact identity questions use the user's requested navigation/observation introduction, not an astronaut persona. Other conversation still uses generation. `call_llm` first prefers retrieved passages; source-free explanatory forms (`why`, `what is/are`, `explain`) may use explicit `general_knowledge` provenance, with passage numbers **off**, no facts/slots, and the existing directive/action/live-reading postchecks. Sky-grey, Mars and typo `mar` forms are covered with fake generators, not fresh live-Qwen validation.

Repair/procedural and consequential scope terms conservatively offer clarification/contact via policy; no manual availability or sent message is asserted, no confirmation action is armed, and this does not add a manual reader. Existing retrieved informational answers remain supported with directive rejection and risk-term passage numbers off. Risky questions without sources also offer clarification/contact. New scope, prompts and offers remain **PLACEHOLDER pending human sign-off**; the introduction is exact user-requested wording. Lexical guards are not semantic safety proofs: missed paraphrases and false positives (including harmless science mentioning risk terms) remain possible.

A selected generic safety tool is now restricted before execution when a destination resolves or a travel-feasibility phrase matches (including unresolved `base 3` and `can I get there`). It asks for navigation rather than emitting generic `Safety check: go.` for that trip; generic safety wording/thresholds are unchanged. ModernBERT still selects every tool; OOD, low-confidence and clarification gates are not bypassed, and alerts still preempt everything. Distance paraphrase clarifications in the supplied transcript remain router calibration/data limitations, not fixed with threshold changes. Navigation distance/ETA and return feasibility are **not route optimization**; `best route` does not establish an optimized path. Ctrl-C now exits the demo without a traceback.
