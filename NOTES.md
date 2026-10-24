
- 2026-10-24 (router recovery): the uncommitted router work tree (old `Nova/Nova/router/`) was lost; restored byte-for-byte from
  our subagent transcripts into `/home/isaach/Nova/router/` (now committed: 586bc88 v1, 5b94c2e v2). Gotchas: base model + checkpoints now live
  outside git in `$NOVA_MODELS_DIR` (default `/home/isaach/models/Luna/router`: `ModernBERT-base` @ 8949b909, `v1`); v1 retrained, same recipe/seed => T=1.501, test 306/320, paraphrase 64/64, demo_47 47/47.

- 2026-10-24 (router v2 review): completed confidence/arbiter gate, destination-span resolution and family-split checks; kept v1 reproducible, v2 already trained once (T=1.359), no retraining on resume.
  Linux CPU: 179 tests pass (no skips); v2 test 306/320, paraphrase 63/64, demo_47 45/47; DEV-only tau_hi suggestion .98 gives test action HCW 0/320 (not a real-speech safety claim).
  README documents regressions and unbuilt safeguards; research review preserved. Committed in 5b94c2e.

- 2026-10-24 (router turn wrapper, UNCOMMITTED on feat/router): added `turn.py` (alert poll -> router -> slot check -> 1 fake tool -> template -> number verifier), `tools.py`, `verifier.py`, `turn_policy.json` (ALL values PLACEHOLDER, human must sign off), `demo_turn.py`; pipeline now returns `clarify/missing_slot` (map requires a place now); gate action set = all 5 tools; "dont"/"avoid" added to place negation.
  Next: real number verifier + LLM-output validation, real arbiter, 10-class labels/v3, real push-to-talk data + a calibration split, real tools, MPS/ONNX INT8 on the Mac.
  Gotchas: dev data is used 3x (epochs, T, tau_hi) so gate numbers are optimistic; mid-band arbiter-agree-wrong is unmeasured so HCW rates are lower bounds; `router/reports/` is gitignored.

- 2026-10-24 (turn follow-up, UNCOMMITTED): turn.py now adds return-to-home feasibility on navigation (numbers in the tool_result), a stateful refill-call offer -> yes/no matcher -> fake `send_message` (6 tools), and an LLM scope policy (risk terms for NO_TOOL, retriever + score threshold for call_llm); 344 tests pass, no skips; every new value is a PLACEHOLDER in `turn_policy.json` (feasibility/confirmation/send_message/llm_scope) awaiting sign-off. Unbuilt: real retriever/documents, real position/home model, real send_message channel, real LLM.
- 2026-10-24: refill-call offer is suppressed when the destination is a home place (home_places in turn_policy.json).

- 2026-10-24 (speech, UNCOMMITTED): `router/speech.py` (ElevenLabs singleton client, voice VmPkIaPdP997cjIfBWRB, model eleven_v4 verified valid by a real TTS call; bogus id -> 400; pcm_24000 -> paplay) + `TurnHandler(tts_fn=...)` (called once per turn with the final verified line, errors -> `result["tts_error"]`) + `demo_turn --speak`; elevenlabs/python-dotenv added to pyproject. 369 tests pass, no skips (models now under `/home/isaach/models/Nova/router`; set `NOVA_MODELS_DIR`, the default dir `Luna` no longer exists).
  Live test: API call OK, 222720 bytes PCM, paplay exit 0, but audible output not verified. Unbuilt: streaming, alert barge-in, Unity/Quest audio, VAD.
