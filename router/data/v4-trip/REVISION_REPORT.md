# V4.2 revision audit

**Agent-authored synthetic candidate only.** No training or model evaluation.

## Counts

| Split | Families | Rows | call_safety | call_navigation | call_map | call_data | call_o2 | call_llm | NO_TOOL | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | 18 | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| dev | 12 | 192 | 32 | 16 | 16 | 32 | 16 | 32 | 16 | 32 |
| test | 20 | 320 | 32 | 48 | 48 | 32 | 48 | 32 | 48 | 32 |
| threshold | 18 | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| train | 96 | 1536 | 192 | 192 | 192 | 192 | 192 | 192 | 192 | 192 |

164 families, 656 source templates, 2,624 rows. Every original v4.1 family,
template and split remains; 16 train families add 256 rows. At most 16 rows per
family; balanced train labels at 192 rows each. Unchanged evaluation split bytes.

## Held-out scope and trade-off

Explicit repair-marker families (I meant / Correction: / Actually / Let me rephrase:) remain test-only; contrastive scope selection and quoted-mention contrasts now occur in train. Broad correction semantics are NOT held out.

128 test rows use the eight original explicit-repair families. The remaining
192 use nine shared constructions. New train scope-selection contrasts cover
all eight labels (16 rows each). Semantic correction generalization is no longer
a holdout claim; only these explicit repair markers/families/templates remain
test-only. No false-activation improvement is measured in this data-only task.

## Lexical and overlap diagnostics

Near review pairs: 96; template-pair groups: 6; hard pairs: 0; max: 0.818182.
Review >=0.72, hard >=0.985, all review pairs same-label; no gate weakened.
Token presence purity: 16 pure tokens at >=16 rows (v4.1: 14); zero
non-allowlisted pure tokens at >=24 rows. All residual pure tokens:

| Token | Label | Rows | Intent allowlisted |
|---|---|---:|---|
| album | OOD | 16 | False |
| band | OOD | 16 | False |
| buy | OOD | 40 | True |
| centered | call_map | 16 | False |
| condition | call_safety | 20 | False |
| describe | call_llm | 16 | False |
| diffusion | call_llm | 20 | False |
| forget | NO_TOOL | 16 | False |
| heat | call_llm | 20 | False |
| mind | NO_TOOL | 16 | False |
| never | NO_TOOL | 16 | False |
| order | OOD | 20 | False |
| safe | call_safety | 20 | False |
| stand | call_data | 16 | False |
| tickets | OOD | 16 | True |
| vitals | call_data | 24 | True |

## Train construction × label rows

| Construction | call_safety | call_navigation | call_map | call_data | call_o2 | call_llm | NO_TOOL | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ability_modal | 32 | 0 | 0 | 0 | 16 | 16 | 16 | 16 |
| asr_noise | 0 | 0 | 16 | 16 | 16 | 0 | 16 | 16 |
| comparative | 16 | 0 | 0 | 16 | 16 | 16 | 16 | 0 |
| conditional_feasibility | 0 | 0 | 16 | 16 | 16 | 16 | 0 | 16 |
| conditional_projection | 16 | 16 | 16 | 16 | 16 | 0 | 0 | 0 |
| contrastive_scope_selection | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| elliptical | 0 | 16 | 0 | 16 | 0 | 16 | 16 | 16 |
| embedded_indirect | 0 | 16 | 32 | 16 | 0 | 16 | 16 | 16 |
| hypothetical | 16 | 0 | 0 | 0 | 16 | 16 | 16 | 16 |
| imperative_status | 16 | 32 | 16 | 32 | 32 | 32 | 16 | 32 |
| modal_sufficiency | 0 | 32 | 16 | 16 | 0 | 0 | 16 | 16 |
| negated_scope | 0 | 16 | 16 | 0 | 16 | 16 | 0 | 16 |
| polite_request | 32 | 0 | 16 | 0 | 16 | 0 | 16 | 16 |
| see_check_if | 16 | 16 | 16 | 16 | 0 | 16 | 0 | 0 |
| truncated | 16 | 16 | 0 | 16 | 0 | 16 | 16 | 0 |
| wh_quantity | 16 | 16 | 16 | 0 | 16 | 0 | 16 | 0 |

## Current SHA256 evidence

```json
{
  "artifact_sha256": {
    "calibration.jsonl": "77f93009ad7875561a048af4b7e05e7e8e62e2b2578ce84ecadf7390315111c1",
    "dev.jsonl": "8e4ddd3ee1105e13f408d71b794aa8048b84ad931233491271987cf1f7843023",
    "near_duplicate_report.json": "5d3ea954abf8ff236fbe516f289b5245b368e268a5e3b3732fed88109e6c5f33",
    "test.jsonl": "4d1f4fe7d2d2a88ad92260b65d596bdcecd718372ff6d893e1d86d7e289b8a18",
    "threshold.jsonl": "97cbe8f277ca37f769669bcbe230a56c0b0b133624bed31152ce7edd3b443c88",
    "train.jsonl": "17296bfab5b15b73a62783decbcbeae6841914a095ae7b04e5e05bde567305cc"
  },
  "family_sha256": "6892180d976aee2a36aff8a053744f4fe98c851ac7fb0f4a24dfbc8274014e14",
  "generator_sha256": "2ea2ad981867e513d5e9958bccb3e67907454d64221d22a310933459290ee91c",
  "schema_version": "v4-trip-1.2"
}
```

## SUPERSEDED history and consumers

V4.1 hashes are SUPERSEDED in `SUPERSEDED_V4_1.json` (also embedded in manifest).
The complete historical audit remains in `V4_1_REVISION_REPORT.md`; its old
“current” hashes are not current v4.2 evidence. Frozen references/probes and
v3 hashes are unchanged. The initial-v4 ledger remains in `LEGACY_PROVENANCE.md`.

Grep-only consumer finding: `router/training_tools.py:27-32` needs to accept
`v4-trip-1.2`; its tooling tests should cover the new version. Other v4 tools
compare the canonical `v4-trip-1` contract and need no literal schema replacement
if canonicalization is retained. No consumer was edited. See compatibility note.
