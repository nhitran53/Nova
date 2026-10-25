# Research: Local Qwen LLM runtime for space-helmet voice assistant demo (today 2026-10-24)

## Summary
Use **llama-server (llama.cpp) + Qwen3.5-4B Q4 GGUF** on both machines as the single default: one binary, identical `/v1/chat/completions` on `http://localhost:8080/v1`, per-request `chat_template_kwargs` for thinking off, and the only runtime with first-class GBNF/JSON-schema constraints. On the Mac, **mlx_lm.server + `mlx-community/Qwen3.5-4B-4bit`** is the faster alternative (~1.3-1.4x llama.cpp, per one third-party bench), but it has no grammar support. Ollama is the easiest install, but it is slowest and has known Qwen3.5 quirks (GGUF/mmproj, `think:false` plus `format` bug).

Caveat: I confirmed Qwen3.5 (small: 0.8B/2B/4B/9B, released ~Mar 2026) as the current small Qwen line. I did not search for a newer generation such as Qwen3.6 or Qwen4. Check once before the demo.

## Findings

1. **Runtime comparison.**
   - **llama-server**
     - Setup: `brew install llama.cpp` on Mac, or build from source with `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` on Arch. Arch packages exist but I did not verify names.
     - Default port is 8080. Base URL is `http://localhost:8080/v1`. Endpoints: `/v1/chat/completions`, `/v1/models`, `/health`.
     - Model loading: `-hf repo:quant` auto-downloads.
     - Platforms: Metal on Mac, CUDA, Vulkan or ROCm on Linux, and CPU.
   - **Ollama**
     - Setup: one installer. Port 11434. Base URL is `http://localhost:11434/v1/`, with a dummy api_key. Native API at `/api/chat`.
     - Platforms: Mac, Linux (CUDA/ROCm), CPU.
     - Unsloth warns that Qwen3.5 GGUFs from outside the Ollama library don't work in Ollama because of separate mmproj vision files; use the official `qwen3.5:*` tags.
   - **mlx_lm.server**
     - Setup: `pip install mlx-lm`, then `mlx_lm.server --model <hf-repo>`. Port 8080. Base URL is `http://localhost:8080/v1`.
     - Platforms: Apple Silicon only (a CUDA backend may exist, not verified). Not for Linux AMD/CPU.
     - Docs say it is "not recommended for production" and implements only basic security checks.
   - Throughput on an M5 Max, Q4_K_M, gen-512 (BAEM1N/llm-bench, third-party, single source): 9B model MLX 102 tok/s, llama.cpp 75, Ollama 52. So MLX leads llama.cpp, which leads Ollama.
   - **Sources:** [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/HEAD/tools/server/README.md); [Ollama OpenAI compat](https://docs.ollama.com/api/openai-compatibility); [mlx-lm SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md); [BAEM1N/llm-bench](https://github.com/baem1n/llm-bench); [Unsloth Qwen3.5](https://unsloth.ai/docs/models/qwen3.5).
   - **Support:** Ports and URLs for Ollama and MLX are direct evidence. The llama-server default port 8080 and the brew and Arch install notes are from my own knowledge, not fetched. The benchmark is third-party. **Confidence:** medium-high.

2. **Model choice: Qwen3.5-4B at Q4 (fallback 2B, upgrade 9B).**
   - Qwen3.5 small sizes are 0.8B, 2B, 4B and 9B. They are hybrid thinking, multimodal, with 256K context. Ollama tags are `qwen3.5:0.8b`, `:2b`, `:4b`, `:9b`. Sizes on disk are 1.2 GB (0.8b), 2.7 GB (2b), 3.3-4.0 GB (4b).
   - Artificial Analysis notes that the small models use very many output tokens when reasoning. For a voice assistant, thinking must be off.
   - Per-runtime identifiers:
     - **Ollama:** `qwen3.5:4b`. Alternatives are `qwen3.5:2b` and `qwen3.5:9b`.
     - **llama-server:** `-hf unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL`. Q4_K_M also exists. Other repos: `unsloth/Qwen3.5-2B-GGUF`, `unsloth/Qwen3.5-9B-GGUF`.
     - **MLX:** `mlx-community/Qwen3.5-4B-4bit` (3.03 GB weights). Also `mlx-community/Qwen3.5-4B-MLX-4bit` and `mlx-community/Qwen3.5-4B-OptiQ-4bit`. The 2B and 9B names are probably `mlx-community/Qwen3.5-{2B,9B}-4bit`; I did not verify them.
   - Memory (Unsloth table): 4B at 4-bit needs ~5.5 GB total. 9B at 4-bit needs ~6.5 GB. 2B at 4-bit needs ~3.5 GB. All fit easily in 24 GB.
   - Speed (data points only; **no M5 Pro measurement found**):
     - Qwen3.5-4B-4bit on an M6 12-core, 32 GB (oMLX, 4k context): 1,832 tok/s prefill, 47.4 tok/s generation, TTFT 2236 ms at 4k tokens of prompt, 3.8 GB peak.
     - A 9B model on an M5 Max reaches 102 tok/s (MLX) and 75 tok/s (llama.cpp).
     - **Inference (mine):** a short prompt of ~100-300 tokens gives TTFT well under 300 ms. A 25-40 token reply takes ~0.5-1 s total on a Mac. The M5 Pro has less memory bandwidth than the Max, so expect roughly 50-70 tok/s on 4B. Measure on the demo machine.
   - Thinking mode:
     - Unsloth says Qwen3.5 0.8B/2B/4B/9B have reasoning **disabled by default**.
     - This is disputed. An Unsloth HF discussion says the 9B template defaults to thinking on, and a Canfield model card says "Qwen3.5 thinks by default".
     - **Always send the disable flag explicitly.**
     - llama-server: launch flag `--chat-template-kwargs '{"enable_thinking":false}'`, or per request `"chat_template_kwargs":{"enable_thinking":false}`.
     - llama.cpp issue #20409 reports the flag being ignored on some Windows builds. Verify the output has no `<think>` or `reasoning_content`.
     - Ollama: native `"think": false`. On the OpenAI-compatible endpoint, `reasoning_effort:"none"` is likely the equivalent; **not verified**. Test it.
     - MLX: the server gates per-request `chat_template_kwargs` behind `--trust-client-kwargs` (PR #829, merged 2026-02-03). A launch-time `--chat-template-args '{"enable_thinking":false}'` is more robust. Issue #1803 says per-request kwargs bypass the prompt cache on 0.31.3.
     - `/no_think` is the Qwen3 soft switch; I did not verify it for Qwen3.5. Don't rely on it.
   - **Sources:** [Ollama tags](https://ollama.com/library/qwen3.5/tags); [Unsloth guide](https://unsloth.ai/docs/models/qwen3.5); [HF Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B); [mlx-community/Qwen3.5-4B-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit); [Unsloth HF discussion](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/discussions/2); [Canfield card](https://huggingface.co/Canfield/Qwen3.5-4B-Q4_K_M-GGUF/commit/4bc3afea43e5c81d17b8df3ef09c8bfdb9ff56f2); [llama.cpp #20409](https://github.com/ggml-org/llama.cpp/issues/20409); [mlx-lm PR 829](https://github.com/ml-explore/mlx-lm/pull/829); [mlx-lm #1803](https://github.com/ml-explore/mlx-lm/issues/1803); [oMLX benchmark](https://omlx.ai/benchmarks/performance/xaz9surw); [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models).
   - **Confidence:** medium (no M5 Pro numbers; default thinking behaviour disputed).

3. **Constrained / structured output.** Examples assume the demo schema is `{"reply": string}`.
   - **llama-server**
     - Fields: `grammar` (GBNF string), `json_schema` (top-level), and OpenAI-style `response_format`: `{"type":"json_object"}` or `{"type":"json_schema","json_schema":{"schema":{...}}}`. The README/tour confirms `grammar`, `json_schema` and `json_object`; the exact nested `json_schema` shape is from my knowledge.
     - ```
       curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
        "messages":[{"role":"user","content":"Status report?"}],
        "chat_template_kwargs":{"enable_thinking":false},
        "temperature":0,"seed":1,"max_tokens":60,
        "grammar":"root ::= \"{\\\"reply\\\": \\\"\" [^\"\\n]{1,200} \"\\\"}\""}'
       ```
       For a schema instead, use `"json_schema":{"type":"object","properties":{"reply":{"type":"string","maxLength":200}},"required":["reply"]}`.
   - **Ollama**
     - Native `/api/chat`: `"format": <JSON schema object>` (or `"json"`).
     - OpenAI endpoint: `response_format` with `type` + `json_schema.schema` (the `ResponseFormat`/`JsonSchema{Schema}` structs in Ollama's source confirm this).
     - ```
       curl localhost:11434/api/chat -d '{"model":"qwen3.5:4b","stream":false,"think":false,
        "messages":[{"role":"user","content":"Status report?"}],
        "format":{"type":"object","properties":{"reply":{"type":"string"}},"required":["reply"]},
        "options":{"temperature":0,"seed":1,"num_predict":60}}'
       ```
     - **Known bug:** with `qwen3.5:27b`, `think:false` plus `format` returned invalid JSON (Ollama issue #14850, closed as duplicate, with a +1 as late as 2026-07-17). Test 4B, or avoid constraining on Ollama. There is no GBNF.
   - **MLX (mlx_lm.server)**
     - SERVER.md lists no `response_format`, `grammar` or `json_schema` request fields. There is **no built-in constrained decoding**.
     - Options: a third-party library (outlines or lm-format-enforcer, via a custom Python loop and not the server; I did not verify their MLX integration), or prompt for JSON and validate/retry client-side. `logit_bias` is the only logits control the server exposes.
     - ```
       curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
        "messages":[{"role":"user","content":"Status report?"}],"temperature":0,"max_tokens":60}'
       ```
   - **Sources:** [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/HEAD/tools/server/README.md); [llama-server tour](https://mvysny.github.io/llama-server-endpoints/); [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs); [Ollama openai.go](https://github.com/ollama/ollama/blob/main/openai/openai.go); [Ollama #14850](https://github.com/ollama/ollama/issues/14850); [mlx-lm SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md).
   - **Confidence:** medium-high. The curl examples were not executed.

4. **Recommendation.** The model is Qwen3.5-4B everywhere.

   | Platform | Default | Fallback |
   |---|---|---|
   | Mac M5 Pro 24 GB | llama-server `-hf unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL -ngl 99 -c 4096 --chat-template-kwargs '{"enable_thinking":false}' --port 8080` | `mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit --chat-template-args '{"enable_thinking":false}'` if it needs more speed and no grammar. Last resort: Ollama `qwen3.5:4b`. |
   | Linux, NVIDIA | llama-server built with CUDA, same model, `-ngl 99` | Ollama `qwen3.5:4b` |
   | Linux, AMD | llama-server built with Vulkan or ROCm, same model | Ollama (ROCm) |
   | Linux, CPU-only | llama-server with `qwen3.5:2b` or the 4B at Q4. Expect a few to ~15 tok/s depending on CPU (my estimate). | Qwen3.5-0.8B |

   - Request parameters for short, near-deterministic replies:
     ```
     {"temperature":0,"max_tokens":60,"seed":42,"stop":["\n\n"],"stream":true,
      "chat_template_kwargs":{"enable_thinking":false}}
     ```
   - Use `stream:true` and start TTS on the first sentence for the lowest perceived latency.
   - Unsloth's non-thinking sampling is temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5. Use that only if greedy decoding loops or sounds flat.
   - A system prompt like "Reply in at most 2 short sentences, no markdown, no emoji" does most of the length control; `max_tokens` is only a safety cap.
   - Ollama: `max_tokens` maps from the OpenAI field (the `MaxTokens` field is in its source). `seed` is under `options` on the native API.
   - MLX server: SERVER.md documents `stop`, `max_tokens` (default 512) and `temperature` (default 0.0), but **no `seed` field**. Rely on `temperature:0`.
   - Pre-warm the model with one request before the demo (ping `/health` on llama-server).
   - Pre-download the models and use `--offline` or `LLAMA_CACHE` so the demo doesn't depend on the network.
   - **Support:** researcher recommendation (inference from findings 1-3). **Confidence:** medium.

## Contradictions
- Qwen3.5 small-model thinking default: Unsloth docs say disabled by default; an Unsloth HF discussion says 9B defaults on and 4B/9B templates differ; a Canfield card says Qwen3.5 thinks by default. Resolution: always pass the disable flag.
- llama.cpp issue #20409 reports `--chat-template-kwargs` ignored on some Windows builds, which contradicts the Unsloth docs. Not confirmed on Linux or Mac; verify.

## Missing evidence
- No M5 Pro measurements of TTFT or tok/s. The numbers are from M6, M5 Max and other chips.
- Whether a newer Qwen generation (Qwen3.6 or later) exists as of Oct 2026 was not checked.
- Ollama's OpenAI-endpoint switch to disable thinking (`reasoning_effort`) was not verified.
- The exact `response_format` nested JSON for llama-server and Ollama, and the Arch package names, were not tested.
- MLX constrained decoding (outlines-mlx, lm-format-enforcer) compatibility with Qwen3.5 was not researched.
- Whether the Qwen3.5 vision/mmproj parts matter for text-only llama-server use (the mmproj is optional, so likely not) was not tested.

## Sources
- Kept:
  - llama.cpp server README (https://github.com/ggml-org/llama.cpp/blob/HEAD/tools/server/README.md): official, covers grammar and json_schema.
  - mlx-lm SERVER.md (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md): official, covers port, fields and defaults.
  - mlx-lm PR 829 (https://github.com/ml-explore/mlx-lm/pull/829) and issue 1803 (https://github.com/ml-explore/mlx-lm/issues/1803): chat_template_kwargs behaviour.
  - Ollama OpenAI compatibility (https://docs.ollama.com/api/openai-compatibility), structured outputs (https://docs.ollama.com/capabilities/structured-outputs) and issue 14850 (https://github.com/ollama/ollama/issues/14850).
  - Ollama qwen3.5 tags (https://ollama.com/library/qwen3.5/tags): model sizes.
  - Unsloth Qwen3.5 guide (https://unsloth.ai/docs/models/qwen3.5): memory, sampling, thinking flags.
  - Hugging Face: mlx-community/Qwen3.5-4B-4bit (https://huggingface.co/mlx-community/Qwen3.5-4B-4bit).
  - Benchmarks (third-party): https://github.com/baem1n/llm-bench and https://omlx.ai/benchmarks/performance/xaz9surw.
  - Artificial Analysis (https://artificialanalysis.ai/articles/qwen3-5-small-models): dated 2026-03-05.
- Rejected/deprioritized: siliconscore.com and llamabench.ai (estimated or aggregated numbers); the vllm-mlx benchmark (16 GB M5 serving, off-topic).

## Next steps
- On the demo Mac, run a 10-request timing script (TTFT and tok/s) for llama-server and mlx_lm.server on `Qwen3.5-4B`.
- Confirm that no reasoning text leaks into the reply, and that `chat_template_kwargs` works on the installed build.
- Check Hugging Face and Ollama once for a newer Qwen generation.

Note: the task text asked for /home/isaach/Nova/research/live-llm-runtime.md, but the authoritative output path overrides it, so only this file was written.

---

## Repo integration points (scoped by the main agent, 2026-10-24)

- `router/generation.py`: `Generator.generate(req) -> GenerationResult(raw, latency_ms, ttft_ms)` (a bare `str` is also accepted). A raising generator becomes `status="error"` (no retry, tool_error line spoken). `supports_retry` attribute optional. `run_generation` does retry/fallback/verification, so the adapter must NOT validate; it returns the raw JSON text `{"refusal": null|code, "text": "...{slot}..."}`.
- Prompt: `render_prompt(req)` returns chat messages (system+context+user); `req.corrective` is already folded in on retry. Constraints: `output_json_schema(req)` / `output_gbnf(req)` (illustrative, currently unused). `req.style.max_tokens`, `req.seed`, `req.deadline_ms` (use as HTTP timeout).
- `router/turn.py`: `TurnHandler(router, alert_source, registry, generator=...)`; `last_generation_request`; result keys `generation`, `llm_called`, `llm_error`, `blocked_text`, `spoken`, `verifier`. Alerts and tool turns never reach a generator.
- `router/demo_turn.py`: only scripted; uses `llm_fn=demo_llm`. New `demo_live` builds its own handler with `FixtureDatasource` (`ds.snapshot`, `default_registry(datasource=ds)`, `ds.advance(s)`).
- Config: new PLACEHOLDER section `live_llm` in `turn_policy.json` (runtime, base_url, model, timeouts, extra request params). Existing values untouched.
- Design: one OpenAI-compatible adapter (`/v1/chat/completions`) for llama-server, MLX and Ollama; runtime only changes the constraint/thinking fields: llama-server `chat_template_kwargs`+`json_schema`/`grammar`; Ollama `response_format` json_schema + `reasoning_effort:"none"` (unverified); MLX none (prompt-only, validated by postprocess).
- Recommendation summary: Qwen3.5-4B Q4 on llama-server for both Mac and Linux; MLX 4bit as faster Mac alternative (no grammar); Ollama last resort. Verify no newer Qwen generation and that thinking is really off (never run against a real model in this work).
