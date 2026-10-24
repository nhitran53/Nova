
- 2026-10-24 (router recovery): the uncommitted router work tree (old `Nova/Nova/router/`) was lost; restored byte-for-byte from
  our subagent transcripts into `/home/isaach/Nova/router/` (still UNCOMMITTED - copy the repo or commit soon; user decides). Gotchas: base model + checkpoints now live
  outside git in `$NOVA_MODELS_DIR` (default `/home/isaach/models/Luna/router`: `ModernBERT-base` @ 8949b909, `v1`); v1 retrained, same recipe/seed => T=1.501, test 306/320, paraphrase 64/64, demo_47 47/47.

- 2026-10-24 (router v2 review): completed confidence/arbiter gate, destination-span resolution and family-split checks; kept v1 reproducible, v2 already trained once (T=1.359), no retraining on resume.
  Linux CPU: 179 tests pass (no skips); v2 test 306/320, paraphrase 63/64, demo_47 45/47; DEV-only tau_hi suggestion .98 gives test action HCW 0/320 (not a real-speech safety claim).
  README documents regressions and unbuilt safeguards; research review preserved. All changes remain UNCOMMITTED; no commit/push performed.
