# V4.1 revision audit

Agent-authored synthetic candidate only; no training, promotion or safety validation.

## Split / label counts

| Split | Rows | call_navigation | call_safety | call_map | call_data | call_o2 | call_llm | NO_TOOL | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1280 | 160 | 160 | 160 | 160 | 160 | 160 | 160 | 160 |
| dev | 192 | 16 | 32 | 16 | 32 | 16 | 32 | 16 | 32 |
| calibration | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| threshold | 288 | 32 | 32 | 32 | 32 | 32 | 32 | 48 | 48 |
| test | 320 | 48 | 32 | 48 | 32 | 48 | 32 | 48 | 32 |

## Token purity before / after

Counts are normalized token presence per row. Pure means 100% of containing rows
have one label. Initial v4: **34** pure tokens at >=16 rows; v4.1: **14**.
V4.1 non-allowlisted pure tokens at >=24 rows: **0**. Only `buy` remains pure
at >=24 rows (32 rows); it is an explicitly allowlisted intent word.

| Token | Initial pure label:rows | Revised pure label:rows |
|---|---|---|
| action | NO_TOOL:16 | — |
| album | OOD:20 | OOD:16 |
| around | call_map:16 | — |
| band | OOD:28 | OOD:16 |
| basic | call_safety:44 | — |
| book | OOD:16 | — |
| bpm | call_data:20 | — |
| budget | — | call_navigation:16 |
| buy | OOD:52 | OOD:32 |
| by | OOD:24 | — |
| centered | — | call_map:16 |
| chart | call_map:28 | — |
| condition | call_safety:28 | call_safety:20 |
| crew | call_llm:32 | — |
| describe | — | call_llm:16 |
| diffusion | call_llm:16 | call_llm:20 |
| display | call_map:36 | — |
| example | call_llm:24 | — |
| explain | call_llm:64 | — |
| forget | — | NO_TOOL:16 |
| game | OOD:20 | — |
| generic | call_safety:44 | — |
| going | NO_TOOL:16 | — |
| heat | call_llm:24 | call_llm:20 |
| minutes | call_o2:28 | — |
| okay | call_safety:16 | — |
| order | — | OOD:16 |
| percentage | call_o2:32 | — |
| reaching | call_navigation:16 | — |
| request | NO_TOOL:16 | — |
| restaurant | OOD:16 | — |
| safe | — | call_safety:16 |
| screen | call_map:28 | — |
| stand | — | call_data:16 |
| this | call_navigation:24 | — |
| tickets | OOD:16 | — |
| using | call_llm:24 | — |
| view | call_map:32 | — |
| vitals | call_data:16 | call_data:20 |
| walk | call_navigation:16 | — |
| why | call_llm:24 | — |

A dash means not pure with >=16 rows, not necessarily absent. The former named
incidental cues cross labels; `action` is absent from training. The allowlist and
all residual pure tokens (including non-allowlisted ones below the hard gate)
are recorded in the manifest. No broad exemption for incidental vocabulary.

## Within-train construction × label row counts

| Construction | Labels | call_navigation | call_safety | call_map | call_data | call_o2 | call_llm | NO_TOOL | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ability_modal | 5 | 0 | 32 | 0 | 0 | 16 | 16 | 16 | 16 |
| asr_noise | 5 | 0 | 0 | 16 | 16 | 16 | 0 | 16 | 16 |
| comparative | 5 | 0 | 16 | 0 | 16 | 16 | 16 | 16 | 0 |
| conditional_feasibility | 5 | 0 | 0 | 16 | 16 | 16 | 16 | 0 | 16 |
| conditional_projection | 5 | 16 | 16 | 16 | 16 | 16 | 0 | 0 | 0 |
| elliptical | 5 | 16 | 0 | 0 | 16 | 0 | 16 | 16 | 16 |
| embedded_indirect | 6 | 16 | 0 | 32 | 16 | 0 | 16 | 16 | 16 |
| hypothetical | 5 | 0 | 16 | 0 | 0 | 16 | 16 | 16 | 16 |
| imperative_status | 5 | 16 | 0 | 0 | 16 | 16 | 16 | 0 | 16 |
| modal_sufficiency | 5 | 32 | 0 | 16 | 16 | 0 | 0 | 16 | 16 |
| negated_scope | 5 | 16 | 0 | 16 | 0 | 16 | 16 | 0 | 16 |
| polite_request | 5 | 0 | 32 | 16 | 0 | 16 | 0 | 16 | 16 |
| see_check_if | 5 | 16 | 16 | 16 | 16 | 0 | 16 | 0 | 0 |
| truncated | 5 | 16 | 16 | 0 | 16 | 0 | 16 | 16 | 0 |
| wh_quantity | 5 | 16 | 16 | 16 | 0 | 16 | 0 | 16 | 0 |

All 15 train constructions cover at least five labels; correction remains test-only.

## NO_TOOL self-announcing diagnostic

Phrase set: `just a statement`, `no action`, `no request`, `not a request`,
`was a statement`. Counts are rows containing any phrase, case-insensitive.

| Split | Self-announcing / NO_TOOL | Percent |
|---|---:|---:|
| train | 0/160 | 0.00% |
| dev | 0/16 | 0.00% |
| calibration | 0/48 | 0.00% |
| threshold | 0/48 | 0.00% |
| test | 0/48 | 0.00% |

## Near-duplicate summary

Unchanged thresholds: review >=0.72, hard >=0.985. Initial v4 had 64 review pairs;
v4.1 has **96 review pairs, six template-pair groups, zero hard pairs**, maximum
**0.818182**, all same-label. Review flags are retained, not independence claims.
Exact/normalized v3 heldout/regression overlap is rejected in every split.

## Thirty seeded training samples

Selection seed 41024, sorted row IDs before sampling: five new OOD and five new
NO_TOOL rows, then twenty other train rows. No model inference or cherry-picked scores.

1. `OOD` (`modal_sufficiency--OOD--mission-1:3:0`): Is my singing good enough; can I make it as a singer?
2. `OOD` (`embedded_indirect--OOD--mission-4:3:2`): um, I would like you to display the lyrics of the song Walking on Air.
3. `OOD` (`polite_request--OOD--mission-3:3:0`): Please book a table by the window for the crew tonight.
4. `OOD` (`ability_modal--OOD--mission-2:0:0`): Can you message my sister that I reached base 2?
5. `OOD` (`modal_sufficiency--OOD--mission-1:0:3`): please, Is there enough time for you to book a flight to Tokyo for me?
6. `NO_TOOL` (`polite_request--NO_TOOL--mission-3:3:3`): Please leave it there; we are at homebase now.
7. `NO_TOOL` (`embedded_indirect--NO_TOOL--mission-4:2:1`): All I was saying is that my O2 is fine here at base b.
8. `NO_TOOL` (`ability_modal--NO_TOOL--mission-2:0:0`): I can stay here at second base; never mind the walk.
9. `NO_TOOL` (`modal_sufficiency--NO_TOOL--mission-1:1:3`): My tank is fine; I have enough to sit here at second base, just thinking out loud.
10. `NO_TOOL` (`polite_request--NO_TOOL--mission-3:2:2`): Please cancel the air check at home, my tank is fine.
11. `call_data` (`asr_noise--call_data:2:0`): so um suit pressure here base 2 reading
12. `call_data` (`conditional_projection--call_data:3:1`): If the data are fresh, check my vitals while I stand at base b.
13. `call_data` (`conditional_projection--call_data:1:2`): If you have telemetry, give my present BPM at base 3.
14. `call_data` (`elliptical--call_data:1:1`): Radiation rate at my current position at base too, please.
15. `call_safety` (`polite_request--call_safety--mission-2:3:3`): Please give a generic oxygen safety assessment, using the current readings at homebase.
16. `NO_TOOL` (`modal_sufficiency--NO_TOOL--mission-1:3:2`): This is sufficient; forget it, I am staying put at base2.
17. `call_map` (`conditional_feasibility--call_map:2:0`): With the map screen ready, can we view crater 19?
18. `call_llm` (`conditional_feasibility--call_llm:2:0`): With a sealed suit at crater 19, explain why carbon dioxide must be removed.
19. `call_navigation` (`modal_sufficiency--call_navigation:3:0`): Would this much oxygen get me as far as base 2?
20. `call_map` (`embedded_indirect--call_map:0:1`): What I need displayed is the map of base b, not the O2 gauge.
21. `call_safety` (`ability_modal--call_safety:3:0`): Are you able to assess pressure and oxygen health while I stay at second base?
22. `call_o2` (`negated_scope--call_o2:2:3`): No health judgment, just the current tank amount at crater 19.
23. `OOD` (`asr_noise--OOD:2:2`): so um pizza order shop named base2
24. `call_safety` (`ability_modal--call_safety:0:3`): Can you assess my basic oxygen and pressure health at homebase?
25. `call_map` (`embedded_indirect--call_map:1:0`): The thing I need displayed is the chart of homebase.
26. `OOD` (`polite_request--OOD--mission-3:0:0`): Please send an email to my landlord about the broken heater.
27. `call_o2` (`imperative_status--call_o2:3:3`): Check my tank levels while I stay at homebase.
28. `call_data` (`conditional_feasibility--call_data:3:0`): If my vitals are available, read them now at crater 19.
29. `OOD` (`modal_sufficiency--OOD--mission-1:3:1`): hey, Is my singing good enough; can I make it as a singer?
30. `call_llm` (`elliptical--call_llm:0:1`): Oxygen diffusion explanation for base too, please.

## Current SHA256 hashes

Schema `v4-trip-1.1`, version `v4.1`, seed 41024. Initial hashes are explicitly
superseded in `LEGACY_PROVENANCE.md`. Frozen source locks remain unchanged.

```json
{
  "calibration.jsonl": "77f93009ad7875561a048af4b7e05e7e8e62e2b2578ce84ecadf7390315111c1",
  "dev.jsonl": "8e4ddd3ee1105e13f408d71b794aa8048b84ad931233491271987cf1f7843023",
  "families.json": "6adb585057ce3c34b89c9f2c2951773d58fe07a186b89cd2c499a8da0446bc77",
  "frozen": {
    "clarification_probe.jsonl": "11e54a1dc5f40556a35d36840b13a2329a1afc24e9bc5bad40e04b9e964ed580",
    "seen_during_development.jsonl": "a8f6c0447eecaaf8e2a9b838655fecaca28020e2389b6f896bf12e286bdb82f7"
  },
  "manifest.json": "300fc761c52604f56089506b38cd16a826293877cf118d1e489589cace692575",
  "near_duplicate_report.json": "9d2b72963eb2839773d01679526431cf12bb531ad6fb71c80ed0c3c09f9885fc",
  "release.py": "09e942f6286c4e7b750e176e4a7fe92efda9107aac53a9700adbbc50256c9ec3",
  "test.jsonl": "4d1f4fe7d2d2a88ad92260b65d596bdcecd718372ff6d893e1d86d7e289b8a18",
  "threshold.jsonl": "97cbe8f277ca37f769669bcbe230a56c0b0b133624bed31152ce7edd3b443c88",
  "train.jsonl": "f6aaf9c033c037db856152c39baa2a1781614144bcc822191df8bfdcf6626fb7",
  "v3_sources": {
    "heldout_scenarios.jsonl": "f6198bb2dc12d30e9bac42b7af9b08bf72aae02ad2b6958817e7325c2015f47d",
    "regression.jsonl": "523c2494ce6a2eb0b18214b179a4f9cb3acccc01173b965abda1765cdcfe9a1b"
  }
}
```
