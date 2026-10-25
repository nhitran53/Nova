# Local Qwen3.5-4B setup and smoke results (2026-10-24)

## Outcome and scope

**Actual local inference works on this Linux computer.** Qwen3.5-4B UD-Q4_K_XL also produced first-attempt, schema-valid, postprocess/verifier-accepted chitchat and grounded fixture answers through the existing `router/demo_live.py`, including a run with the real ModernBERT v2 router. Navigation stayed deterministic/template-only. This is a text/fixture integration smoke test, **not verified speech, real telemetry, safety certification, training, or a latency benchmark**. No TTS or hosted API was used. Existing code, policy, templates, data, checkpoints and dirty work were left unchanged.

The default demo settings are **not sufficient on this CPU**: even `--deadline-ms 120000` still hits the adapter's 10-second HTTP cap. A supervisor-approved external helper overrides that cap to 180 seconds **in memory only**, paired with CLI `--deadline-ms 180000`. These are CPU diagnostic settings, not production approval.

## Host and resource inspection (before downloads)

- Linux x86_64, AMD Ryzen 7 PRO 4750U, 8 cores / 16 threads, AVX2.
- RAM 14 GiB total, initially 6.3 GiB available; swap initially 2.5/4 GiB used. Disk 510 GiB free.
- AMD integrated Renoir GPU; `vulkaninfo --summary` successfully identified RADV RENOIR, Mesa 26.2.5. No `nvidia-smi`, CUDA compiler, or preinstalled llama-server found on PATH. CPU fallback selected to keep setup small and avoid shared GPU memory pressure; Vulkan acceleration was **not built or benchmarked**.
- Existing model directory 1.7 GiB and HF cache 2.6 GiB inspected; no reusable Qwen GGUF found. Existing `router/.venv` already imported torch/transformers successfully; no Python install needed.
- llama-server RSS after smoke: 4,369,936 KiB (~4.17 GiB), with ~5.1 GiB RAM available. Swap grew to 3.5 GiB, later 3.7 GiB (other host processes also exist; attribution is not proven). **Server was stopped** to release resources. Final available RAM 6.9 GiB. Do not treat this as a low-memory/long-duration stability test.

## Exact artifacts

All runtime files are outside git:

- Runtime root: `/home/isaach/.local/share/nova-runtime` (`R` below).
- llama.cpp source: `$R/llama.cpp`, immutable commit `78a8ae14be559b45f5f13ddee82c188b683d5d8d`.
- Binary: `$R/llama.cpp/build/bin/llama-server`, version `0.7.0-dev` build 1, GNU 16.2.1 Linux x86_64.
- Model repo: `unsloth/Qwen3.5-4B-GGUF`, immutable revision `e87f176479d0855a907a41277aca2f8ee7a09523`.
- Exact model: `/home/isaach/models/Nova/qwen3.5-4b/Qwen3.5-4B-UD-Q4_K_XL.gguf`, **2,912,109,728 bytes**.
- SHA-256: `b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7`, checked against HF LFS metadata after download (passed).
- Text-only: no vision projector downloaded. Model download ~2.912 GB, source/build tree 328 MB; below the authorized 8 GB download budget. No sudo or system package changes.
- Source clean after build. Build fetched public llama-ui source/assets automatically: pinned `b1` fetch failed and upstream fell back to `latest`, archive checksum verified by build. Thus the embedded UI is **not independently immutable/pinned here**; the text inference code commit and GGUF are pinned. UI was not used.

Build configuration used:

```bash
R=/home/isaach/.local/share/nova-runtime
cmake -S "$R/llama.cpp" -B "$R/llama.cpp/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build "$R/llama.cpp/build" --target llama-server -j 2
```

Build succeeded (227 steps, exit 0); logs in `$R/configure.log` and `$R/build.log`. Download script/log and build/download PID records remain in `$R`. Long tasks ran in background; bounded kernel pidfd readiness waits were used instead of sleep/poll loops.

## Restart and stop (currently stopped)

```bash
R=/home/isaach/.local/share/nova-runtime
# Check the port is free first; never replace an unrelated listener.
ss -ltn '( sport = :8080 )'
nohup "$R/start-server.sh" > "$R/server.log" 2>&1 < /dev/null &
echo $! > "$R/server.pid"
curl --noproxy '*' --max-time 5 http://127.0.0.1:8080/health
# If loading returns 503, check readiness again before starting the demo.
# Stop after confirming PID still belongs to this llama-server:
ps -p "$(cat "$R/server.pid")" -o pid,comm,args
kill -TERM "$(cat "$R/server.pid")"
```

`start-server.sh` executes exactly:

```bash
/home/isaach/.local/share/nova-runtime/llama.cpp/build/bin/llama-server \
  -m /home/isaach/models/Nova/qwen3.5-4b/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --alias qwen3.5-4b --host 127.0.0.1 --port 8080 \
  -ngl 0 -c 2048 -t 4 -tb 4 -np 1 -b 256 -ub 128 \
  --chat-template-kwargs '{"enable_thinking":false}' --reasoning-budget 0 \
  --log-disable
```

Listener verified by `ss`: **127.0.0.1:8080 only**. Last server PID **3310874**, terminated and exit confirmed; no listener remained. PID file is a historical record until restart. Server logging disabled to keep logs bounded; startup warnings note no GPU backend (expected) and deprecation of `enable_thinking` in favor of newer `--reasoning off`. The tested launch retains explicit thinking-off plus zero reasoning budget; each adapter request also sends thinking-off. Do not silently change a known-working invocation without retesting.

## Reproduce smoke tests

From `/home/isaach/Nova`, after server health is OK:

```bash
python /home/isaach/.local/share/nova-runtime/raw-smoke.py

env -u NOVA_LLM_API_KEY -u NOVA_LLM_RUNTIME -u NOVA_LLM_BASE_URL \
  -u NOVA_LLM_MODEL -u NOVA_LLM_CONSTRAIN \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  NOVA_MODELS_DIR=/home/isaach/models/Nova/router OMP_NUM_THREADS=2 \
  router/.venv/bin/python /home/isaach/.local/share/nova-runtime/demo-cpu.py \
  --runtime llama-server --base-url http://127.0.0.1:8080/v1 \
  --llm-model qwen3.5-4b --constrain json_schema --deadline-ms 180000 \
  --script /home/isaach/.local/share/nova-runtime/smoke-turns.txt
```

Add `--fake-router` for the independently exercised keyword-router path. No `.env` was read, environment dump printed, credentials supplied, speech requested, or remote inference used. HF offline flags prevent encoder downloads; `NOVA_MODELS_DIR` points to existing router checkpoints, not Qwen weights.

The external `demo-cpu.py` imports `router.demo_live`, replaces its `load_live_llm_config` reference with a wrapper calling the **existing** config loader with `request_timeout_cap_s=180`, then calls unchanged `demo_live.main`. Only this diagnostic process's config changes. Script contents:

```text
hi
what is the moon
how far is the tower
/quit
```

The grounded question uses the existing **fake keyword retriever**, passage `The Moon is Earth's only natural satellite.` No real manuals/retrieval/DB are claimed.

## Observations and evidence

| Run | Result | Latency |
|---|---|---|
| Raw `/v1/chat/completions`, temp 0, seed 42, max_tokens 48, thinking explicitly false | `Hello there, I'm Nova!`; no `<think>` tags and no nonempty `reasoning_content` | 4.414 s wall; 35 prompt tokens / 8 output tokens; prefill 14.43 tok/s, generation 3.54 tok/s |
| Unmodified demo CLI, keyword router, deadline 120 s, HTTP cap still 10 s | Both LLM calls timed out; fixed `I could not complete that request.`; **not model success**. Tool turn succeeded. | 10.011 / 10.006 s turn wall; tool 5 ms |
| Approved helper, keyword router | Chitchat `Hello astronaut. How is your day going?` and Moon answer both `outcome: ok`, first attempt, no violations, verifier OK | 10.079 / 11.635 s generator; 10.081 / 11.637 s turn wall; tool 4 ms |
| Approved helper, **real v2 router** | Same two accepted Qwen answers; NO_TOOL confidence .975, call_llm .989. Navigation confidence .947; all verifier OK | 8.245 / 6.734 s generator; 8.446 / 7.101 s turn wall; tool 198 ms |

Repeated runs benefit from server prompt/cache warmup; these few observations are not a performance distribution or first-token measurement. JSON-schema accepted without switching to `none`/GBNF. No retries, rejected prose, model refusal, offline FillerGenerator fallback, or thinking leak occurred in the successful runs. The existing adapter strips reasoning; the separate **raw HTTP test** verifies it was absent before adapter stripping for that request. General adversarial thinking/leak resistance was not established.

Deterministic tool turn called `navigation({'place': 'comms_tower'})`, never the LLM, and returned fixture distance 500 m / 9 minutes with feasibility numbers; verifier accepted [500, 9, 28, 10, 666]. These are placeholder fixture/policy values, not real mission facts.

Evidence outside repo: `$R/raw-smoke.json`, `$R/demo-fake-router.log` (default cap failures), `$R/demo-cpu-fake-router.log`, `$R/demo-cpu-v2.log`, plus helper scripts and setup logs. Existing targeted tests: `NOVA_MODELS_DIR=/home/isaach/models/Nova/router HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 router/.venv/bin/python -m pytest router/tests/test_llm_generator.py router/tests/test_demo_live.py -q` -> **94 passed in 7.16s**. No tests added; full suite intentionally not rerun for setup-only task.

## Remaining limitations / next step

- Default generation deadline 1500 ms and HTTP cap 10 s remain untouched and unsuitable for reliable CPU use; any CLI/config product fix requires a separate approved code change.
- Server deliberately stopped due to swap pressure. Restart only when desired; memory/thermal/endurance behavior unmeasured. CPU is slow but functional; Vulkan optimization remains optional future work.
- Full Nova speech/audio, live sensor datasource, real RAG, safety behavior and model quality beyond these fixture turns are **not verified**.
- No training, model substitution, staging, commit, push or branch switch. Only this report and a short NOTES append were written in repo.
