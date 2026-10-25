# Legacy and provenance ledger

- Authoring source: owner's eight-label taxonomy, construction guidance and
  approved arithmetic relaxation; agent manually wrote 592 templates (512 in initial v4, 80 added in v4.1) in
  `families.json`. No human speech was collected and no model was trained.
- Design reference: `../v3-trip/release.py` exact/normalized/family/template/hash
  design, extended here with frame delexicalization, numeric count gates,
  near-duplicate reporting and a counts-only external-overlap command.
- `seen_during_development.jsonl` contains exactly the 15 v3
  `heldout_scenarios.jsonl` rows and eight `regression.jsonl` controls. Original
  texts and `expected_label` values are preserved. Source provenance fields are
  preserved but supplemented by `reference_source`, explicit
  `evidence_role=seen_during_development_not_independent`, and a non-scored split.
  Historical “heldout” in a filename/provenance field is **not** a v4 independence
  claim. All 23 were seen during development. They are regression references only,
  never training/calibration/threshold input or independent acceptance evidence.
- V3 replay scripts were not copied. No new independent frozen acceptance
  scenarios were invented. The small clarification probe is a non-scored owner
  decision inventory, not a replacement benchmark.
- V3 source files, frozen files and hashes are not changed by this release.
  Source SHA256 values are recorded in `manifest.json`; its normalizer rejects
  reuse of old seed texts in every generated classifier split.
- There is no relabeling or scoring of historical v2/demo/paraphrase material.
  Existing runtime, labels, policies, thresholds and training/packaging tools
  are outside this data change. See the tooling compatibility report for blockers.

Limitations: all five classifier splits share this author, taxonomy and surface
slot generator. Splitting whole families and one whole construction is useful
but not independent natural-language or safety validation. Repeated slot variants
are correlated; word order, short elliptical queries and ASR simulations are only
synthetic approximations. The near-duplicate report deliberately retains review
flags rather than hiding them. Performance, calibration, high-confidence wrong
calls, live clarification behavior and real sensor support are unmeasured.

## Superseded initial v4 release hashes

The owner-approved **v4.1** revision supersedes the initial `v4-trip-1` release
(schema now `v4-trip-1.1`; dataset name still `v4-trip`). The following initial
v4 hashes are retained as historical provenance only, **not current locks**.
Frozen references/probes and v3 source hashes are unchanged. No historical model
result is attributed to the revised data.

```json
{
  "artifact_sha256": {
    "calibration.jsonl": "f97cdaae40fc95a941a28f0d09a5375b5c920b9b5dc75a0392220e8eaff5af1b",
    "dev.jsonl": "f5268aa9c30d85d93207006b37df471288a6327fb6ab7e1810891e22b52b453d",
    "near_duplicate_report.json": "f54fdda29e0357ce84ffddc6c2999e4a523688594b6a6cf91a17b4c3a8053ac6",
    "test.jsonl": "ed327205b8c42b5c3a28cbca5a09763238bfce6615a5466c3842fe203de0d091",
    "threshold.jsonl": "1739b04c71f18420f38cdbf556edc16e43cdbfd33daf5f0e9b1f51b50faab8f3",
    "train.jsonl": "56b844e92c6d28bf41934060919d04ad42cee91a7bd4f18fdd902c285f42bb83"
  },
  "family_sha256": "18d445ecd4b5b6a8a54bc617b5286254d643acfd348625465eb7a363bd66a018",
  "generator_sha256": "a2d41676e69e2014484ecb27042e059474aa86fdb4e1a81c0337a00faf497608",
  "manifest_sha256": "e88690b83a5365d3f27fae4f7205bfee2919fef3211ab035dee7679809a8a40a",
  "schema_version": "v4-trip-1",
  "status": "SUPERSEDED by v4.1"
}
```

## SUPERSEDED v4.1 release / v4.2 revision

V4.2 (`v4-trip-1.2`) supersedes v4.1. `SUPERSEDED_V4_1.json` records all
old release source/generator/manifest/artifact/documentation SHA256s and is
embedded in the new manifest. Historical documentation is preserved in the
`V4_1_*.md` files; their statements are historical, not current locks. All 148
old families/templates/splits remain. Sixteen new train families bring this
agent-authored synthetic candidate to 164 families / 656 templates / 2,624 rows.
Broad correction semantics now occur in train; the explicit repair-marker
construction and its eight whole families remain test-only. Frozen files, v3
sources and the original v4 ledger above remain unchanged.
