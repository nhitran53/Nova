# Reference data accessors (`router/reference_data.py`)

Read-only access to three REFERENCE datasets from `origin/nhi/database` (`schema.sql`, not merged into this branch).
They are reference/simulated tables, **not live telemetry**. Nothing here is wired into `tools.py`, `turn.py` or
`simulator_rules.py` (`turn_policy.json` only holds a PLACEHOLDER `reference_data` block read by `reference_data.py`): spoken output, tool result keys, the six-tool registry, labels, feasibility math and the safety loop
are unchanged (`test_reference_data.py` asserts the registry and `data`/`o2` key tuples).

| Table | What it is | Accessor |
| --- | --- | --- |
| `crew_o2_co2` | Simulated NASA BVAD-style crew respiration per environment (`Habitat`, `EVA suit`) and activity (`Sleep`, `Exercise (treadmill/ARED)`, `IVA lab / maintenance`, ...): O2 consumed / CO2 produced in L/min, g and kg/day, metabolic rate, ventilation | `o2_consumption(environment, activity)`: mean L/min and kg/day (O2 and CO2) + row count |
| `mars_radiation` | RAD readings: silicon and tissue-plastic dose rate in uGy/day, `data_flag`, `observation_seconds` | `radiation_baseline()`: median and latest uGy/day over `data_flag = 'none'` rows only (raw). `radiation_baseline_usv_h()`: the tissue column converted to uSv/h (see below) |
| `crew_o2_co2` (derived) | same table | `o2_burn_multiplier(activity, environment, reference)` and `o2_burn_pct_per_min(environment, activity, tank_capacity_l)` (see below) |
| `mars_weather` | Gale Crater air temperature (C) and pressure (Pa) by Mars year, season, day/night; NULL temperatures exist | `weather_summary(day_night=None)`: mean C and Pa, NULLs ignored, optional `'Day'`/`'Night'` |

`FixtureReferenceData` holds a few hand-made SYNTHETIC rows (not from the CSVs). `PostgresReferenceData` runs plain
`SELECT`s (psycopg imported lazily, `DATABASE_URL` default `dbname=nova`, injectable `connect` for tests). Both build
results with the same pure `summarize_*` functions. Every result is a plain dict with a `units` map, a `source` marker and a `spoken_qualifier` string (below). Both fail closed with `DataUnavailableError` (empty selection, unknown
environment/activity/`day_night`, NULL/NaN/inf/non-numeric value, all-NULL column, DB error); nothing partial is returned.

`tests/fixtures/nhi_reference_schema.sql` is a verbatim copy of the three tables' DDL; tests check every column the SQL
names against it. If the nhi schema changes, refresh the copy.

## Source markers and spoken qualifiers (decision 2)

Simulated/generated data is acceptable for the demo; the markers stay honest about which kind it is.

| `source` | Used for | `spoken_qualifier` (PLACEHOLDER wording) |
| --- | --- | --- |
| `reference-simulated` | crew respiration (`crew_o2_co2` and the burn multiplier / percent derived from it): a model, not a measurement | `simulated reference value` |
| `reference-observed-extended` | radiation and weather: observation-based reference data that the dataset extends | `extended reference value` |

Neither is ever a reading from the astronaut's suit. The qualifier strings live in `turn_policy.json` -> `reference_data`
-> `spoken_qualifiers`; `reference_data.reference_policy()` / `spoken_qualifier(source)` read them. Nothing speaks them yet.

## `turn_policy.json` `reference_data` block (all PLACEHOLDER)

`status: "PLACEHOLDER - owner must sign off"` (the loader refuses a section without a PLACEHOLDER status). Only
`reference_data.py` reads it; `turn.py` and `tools.py` do not. Contents: `radiation.quality_factor` (3.0),
`o2.tank_capacity_l` (400.0, round demo number), `o2.multiplier_reference` (`EVA suit` / `EVA surface geology`),
`spoken_qualifiers`. A missing or malformed block makes the accessors raise `DataUnavailableError`.

## Radiation: uGy/day -> uSv/h (decision 1)

`radiation_baseline_usv_h(quality_factor=None)` (Q from the policy when `None`):

    dose equivalent rate (uSv/h) = tissue_ugy_per_day * Q / 24

* Dose equivalent = absorbed dose x quality factor Q. `Q = 3.0` is an **approximation** for Mars-surface GCR (MSL RAD surface
  values are roughly 0.21 mGy/day absorbed vs 0.64 mSv/day dose equivalent, Q about 3.05). It is a PLACEHOLDER.
* Only `tissue_ugy_per_day` is converted. The silicon and any uncorrected column are **not** converted (silicon is absent from
  the converted result).
* Each `data_flag = 'none'` row is converted first, then the median is taken; `latest` is the newest `observed_at`.
* Result: `converted: true`, `quality_factor` (the Q used), `conversion` formula string, `tissue_usv_h_median`,
  `latest.tissue_usv_h`, `units` `uSv/h`, `row_count`. Values round to 4 decimals.
* Fail closed: NaN/inf/bool/non-numeric/negative dose rates, a Q that is not a finite number > 0, no `none` rows, bad timestamp.
* `radiation_baseline()` still returns raw uGy/day (also now rejects negative values). The two results must not be mixed.

## O2 burn as a percentage (decision 3)

1. `o2_burn_multiplier(activity, environment, reference=("EVA suit", "EVA surface geology"))` = mean `o2_consumed_l_min` of the
   selected group / mean of the reference group. A pure, deterministic, dimensionless ratio (ratio rounded to 4 decimals,
   means not rounded first). Fails closed if either group is empty, a value is NaN/inf/negative/non-numeric, the reference is
   not an `(environment, activity)` tuple of strings, or the reference mean is not > 0. **Argument order note:** this one
   takes `activity` first (as specified) while `o2_consumption`, `o2_burn_pct_per_min` and `reference` take `environment`
   first; a swapped call fails closed because no environment is named like an activity.
   *Limitation:* the dataset's `EVA suit` group is only ~20 rows of ONE activity (`EVA surface geology`). Habitat activities
   are therefore mapped through this single ratio only. It says nothing about EVA activities the dataset does not contain.
2. `o2_burn_pct_per_min(environment, activity, tank_capacity_l)` = `o2_consumed_l_min / tank_capacity_l * 100` (percent of the
   tank per minute). `tank_capacity_l` is a **required explicit** parameter (litres of O2 at STPD, the dataset's basis); there
   is no default argument. The policy carries an unsigned-off PLACEHOLDER (`o2.tank_capacity_l`, 400.0, a round demo number,
   not a suit specification and not a physiological claim) that a caller may pass explicitly. Non-finite, non-positive,
   bool or missing capacity fails closed.
3. `scaled_sim_base_burn_pct_per_min(mission_base_pct_per_min, multiplier)` is **documentation only** and is called nowhere.
   `simulator_rules.o2_burn_pct_per_min` = `missions.o2_burn_pct_per_min * ACTIVITY_O2[activity] * SCENARIO_O2[scenario]`. A
   dataset multiplier would *replace* the `ACTIVITY_O2` factor: `missions.o2_burn_pct_per_min * multiplier * SCENARIO_O2`.
   Stacking it on top of `ACTIVITY_O2` would count the activity twice. `simulator_rules.py` is not imported by
   `reference_data.py` and neither it nor `ACTIVITY_O2` is changed (tests pin both).

## Decisions 4-5

Battery, water, fuel, food and power remain unanswered (no data); the navigation scope guard that clarifies them and the
runtime safety guard are unchanged. Reference data must not feed them without sign-off.

## Potential tool uses (none built)

* `o2_consumption` -> a baseline O2 draw for an activity, e.g. "how much oxygen does maintenance work use".
* `radiation_baseline_usv_h` -> a "typical background" figure in the same unit as the existing `radiation_usv_h` reading.
* `weather_summary` -> a general "how cold / what pressure outside" answer for the `data` tool family.

## REMAINING owner sign-offs

1. **Quality factor Q** (`reference_data.radiation.quality_factor`, 3.0): the approximation itself, whether tissue is the right column, and whether the per-day -> per-hour step (`/ 24`) is acceptable.
2. **Tank capacity** (`reference_data.o2.tank_capacity_l`, 400.0 L at STPD): replace with the real suit tank figure before any percent is spoken or compared with the sim.
3. **Spoken qualifier wording** (`reference_data.spoken_qualifiers`): `simulated reference value`, `extended reference value`, and whether a qualifier is required at all when a value is spoken.
4. **Whether/how to use the multiplier** in the sim (replace vs. keep `ACTIVITY_O2`), given the one-activity EVA reference group.
5. **Weather caveat:** pressure is Pa (loader converts the file's mbar x 100); means mix seasons and Mars years, so they are not conditions at the astronaut's position. Rows are per crew member and sol; the O2 mean is not a population statistic.
