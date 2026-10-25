# V4.2 schema compatibility (grep-only inspection)

No consumer or runtime file was edited. Data schema is `v4-trip-1.2`, version
`v4.2`; all rows remain the same text/label/metadata format.

`router/training_tools.py:27-32` explicitly accepts only `v4-trip-1` and
`v4-trip-1.1`, canonicalizing both to contract `v4-trip-1`. **This consumer needs
the new `v4-trip-1.2` schema accepted**, with its error message updated, before
any downstream run. Do not downgrade the data manifest to bypass validation.
`router/tests/test_v4_tooling.py` has 1.1 fixtures/assertions (lines 29, 323);
its schema acceptance coverage should be extended in that separate change.

Grep finds canonical `v4-trip-1` contract comparisons in `train.py`,
`report_gate.py`, `acceptance.py`, `calibrate.py`, `evaluate.py`, and
`preflight.py`. Those are canonical contract names, not explicit 1.1 schema
allowlists: they do not themselves need a literal 1.2 replacement if the shared
loader continues to canonicalize the schema. This inspection does not certify
those tools or run training, acceptance, smoke or runtime behavior.

`V4_1_TOOLING_COMPATIBILITY.md` preserves the older audit as **SUPERSEDED
historical evidence**. Some of its findings no longer describe the dirty
working tree; do not apply them as current blockers without separate review.
