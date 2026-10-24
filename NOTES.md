
- 2026-10-24 (router recovery): the uncommitted router work tree (old `Nova/Nova/router/`) was lost; restored byte-for-byte from
  our subagent transcripts into `/home/isaach/Nova/router/` (still UNCOMMITTED - copy the repo or commit soon; user decides). Gotchas: base model + checkpoints now live
  outside git in `$NOVA_MODELS_DIR` (default `/home/isaach/models/Luna/router`: `ModernBERT-base` @ 8949b909, `v1`); v1 retrained, same recipe/seed => T=1.501, test 306/320, paraphrase 64/64, demo_47 47/47.
