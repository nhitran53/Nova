"""Read the crew respiration and Mars weather and radiation reference datasets.

These are reference values rather than live suit readings.
Results include source labels and units. Missing or invalid data raises DataUnavailableError.
Tissue radiation converts from uGy/day to uSv/h using the configured quality factor.
Weather pressure is stored in Pa. Oxygen percentages also need a tank size.
"""
import copy
import json
import math
import os
import statistics
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .datasource import DEFAULT_DATABASE_URL, DataUnavailableError, _default_connect, _finite_number

SOURCE_SIMULATED = "reference-simulated"
SOURCE_OBSERVED_EXTENDED = "reference-observed-extended"
RADIATION_USABLE_FLAG = "none"
DAY_NIGHT_VALUES = ("Day", "Night")
ROUND_DIGITS = 4
HOURS_PER_DAY = 24
POLICY_PATH = Path(__file__).with_name("turn_policy.json")
POLICY_SECTION = "reference_data"
DEFAULT_MULTIPLIER_REFERENCE = ("EVA suit", "EVA surface geology")
TANK_CAPACITY_UNITS = "L O2 at STPD"

# SQL only names columns of the nhi schema.sql tables (tests cross-check them against a verbatim DDL copy).
SQL_CREW_O2 = ("SELECT c.o2_consumed_l_min, c.o2_consumed_kg_per_day, c.co2_produced_l_min, c.co2_produced_kg_per_day "
               "FROM crew_o2_co2 c WHERE c.environment = %s AND c.activity = %s")
SQL_RADIATION = ("SELECT r.observed_at, r.silicon_ugy_per_day, r.tissue_ugy_per_day, r.data_flag "
                 "FROM mars_radiation r WHERE r.data_flag = %s ORDER BY r.observed_at")
SQL_WEATHER_ALL = "SELECT w.air_temp_c, w.pressure_pa FROM mars_weather w"
SQL_WEATHER_DAY_NIGHT = "SELECT w.air_temp_c, w.pressure_pa FROM mars_weather w WHERE w.day_night = %s"


class ReferenceData(Protocol):
    """Read-only reference-dataset interface; implementations raise DataUnavailableError instead of guessing."""

    def o2_consumption(self, environment: str, activity: str) -> dict:
        """Mean crew O2 consumption (L/min, kg/day) and CO2 production over the rows of one environment + activity."""

    def radiation_baseline(self) -> dict:
        """Median and latest silicon/tissue dose rate (uGy/day) over RAD rows whose data_flag is 'none'."""

    def radiation_baseline_usv_h(self, quality_factor: float | None = None) -> dict:
        """Median and latest TISSUE dose equivalent rate in uSv/h (= uGy/day * Q / 24), 'none' rows only. Q defaults to the policy."""

    def weather_summary(self, day_night: str | None = None) -> dict:
        """Mean air temperature (C) and pressure (Pa), NULLs ignored, optionally for 'Day' or 'Night' only."""

    def o2_burn_multiplier(self, activity: str, environment: str, reference: tuple = DEFAULT_MULTIPLIER_REFERENCE) -> dict:
        """Mean O2 L/min of (environment, activity) divided by that of `reference` = (environment, activity). Pure ratio."""

    def o2_burn_pct_per_min(self, environment: str, activity: str, tank_capacity_l: float) -> dict:
        """Mean O2 L/min / tank_capacity_l * 100 (percent of the tank per minute); tank_capacity_l is required."""


def _mean(values) -> float:
    return round(math.fsum(values) / len(values), ROUND_DIGITS)


@lru_cache(maxsize=4)
def _read_policy(path):
    try:
        policy = json.loads(Path(path).read_text())
        section = policy[POLICY_SECTION]
        if not str(section["status"]).startswith("PLACEHOLDER"):
            raise ValueError(f"section {POLICY_SECTION!r} needs a status starting with PLACEHOLDER")
        if not _finite_number(section["radiation"]["quality_factor"]) or section["radiation"]["quality_factor"] <= 0:
            raise ValueError("radiation.quality_factor must be a finite number > 0")
        if not _finite_number(section["o2"]["tank_capacity_l"]) or section["o2"]["tank_capacity_l"] <= 0:
            raise ValueError("o2.tank_capacity_l must be a finite number > 0")
        for source in (SOURCE_SIMULATED, SOURCE_OBSERVED_EXTENDED):
            qualifier = section["spoken_qualifiers"][source]
            if not isinstance(qualifier, str) or not qualifier.strip():
                raise ValueError(f"spoken_qualifiers[{source!r}] must be a non-empty string")
        return section
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataUnavailableError(f"reference_data policy unusable: {type(exc).__name__}: {exc}"[:300]) from exc


def reference_policy() -> dict:
    """The turn_policy.json "reference_data" block (PLACEHOLDER constants: Q, tank capacity, spoken qualifiers); a copy.

    Raises DataUnavailableError if the block is missing or malformed. turn.py never reads this block."""
    return copy.deepcopy(_read_policy(str(POLICY_PATH)))


def spoken_qualifier(source) -> str:
    """The short PLACEHOLDER phrase that must accompany a value of this `source` if it were ever spoken."""
    return reference_policy()["spoken_qualifiers"][source]


def _require_numbers(rows, fields, what):
    """Every named field of every row must be a finite number (no None, bool, string, NaN or inf)."""
    for index, row in enumerate(rows):
        for field in fields:
            if not _finite_number(row.get(field)):
                raise DataUnavailableError(f"{what}: row {index} field {field!r} is missing or not a finite number")


def _require_non_negative(rows, fields, what):
    """Dose rates and gas flows cannot be negative: a negative value is bad data, not a reading."""
    for index, row in enumerate(rows):
        for field in fields:
            if row[field] < 0:
                raise DataUnavailableError(f"{what}: row {index} field {field!r} is negative")


def summarize_o2(rows, environment, activity) -> dict:
    """Pure: crew_o2_co2 rows for one environment + activity -> mean L/min and kg/day (O2 and CO2) and the row count."""
    if not rows:
        raise DataUnavailableError(f"no crew O2/CO2 rows for environment={environment!r}, activity={activity!r}")
    fields = ("o2_consumed_l_min", "o2_consumed_kg_per_day", "co2_produced_l_min", "co2_produced_kg_per_day")
    _require_numbers(rows, fields, "crew_o2_co2")
    _require_non_negative(rows, fields, "crew_o2_co2")
    return {
        "source": SOURCE_SIMULATED,
        "spoken_qualifier": spoken_qualifier(SOURCE_SIMULATED),
        "environment": environment,
        "activity": activity,
        "row_count": len(rows),
        "o2_consumed_l_min": _mean([r["o2_consumed_l_min"] for r in rows]),
        "o2_consumed_kg_per_day": _mean([r["o2_consumed_kg_per_day"] for r in rows]),
        "co2_produced_l_min": _mean([r["co2_produced_l_min"] for r in rows]),
        "co2_produced_kg_per_day": _mean([r["co2_produced_kg_per_day"] for r in rows]),
        "units": {"o2_consumed_l_min": "L/min", "o2_consumed_kg_per_day": "kg/day (rate if kept up all day)",
                  "co2_produced_l_min": "L/min", "co2_produced_kg_per_day": "kg/day (rate if kept up all day)"},
    }


def _usable_radiation_rows(rows):
    """data_flag 'none' rows only, each with finite, non-negative silicon/tissue dose rates and a timestamp, else raise."""
    usable = [r for r in rows if r.get("data_flag") == RADIATION_USABLE_FLAG]
    if not usable:
        raise DataUnavailableError(f"no radiation rows with data_flag={RADIATION_USABLE_FLAG!r}")
    fields = ("silicon_ugy_per_day", "tissue_ugy_per_day")
    _require_numbers(usable, fields, "mars_radiation")
    _require_non_negative(usable, fields, "mars_radiation")
    if not all(isinstance(r.get("observed_at"), datetime) for r in usable):
        raise DataUnavailableError("mars_radiation: observed_at is missing or not a timestamp")
    return usable


def summarize_radiation(rows) -> dict:
    """Pure: mars_radiation rows -> median and latest silicon/tissue uGy/day over rows with data_flag 'none' only.

    Rows with any other data_flag are dropped; if none remain this raises. uGy/day is absorbed dose and is NOT converted
    here (see summarize_radiation_usv_h for the dose-equivalent variant)."""
    usable = _usable_radiation_rows(rows)
    latest = max(usable, key=lambda r: r["observed_at"])
    return {
        "source": SOURCE_OBSERVED_EXTENDED,
        "spoken_qualifier": spoken_qualifier(SOURCE_OBSERVED_EXTENDED),
        "row_count": len(usable),
        "silicon_ugy_per_day_median": round(statistics.median(r["silicon_ugy_per_day"] for r in usable), ROUND_DIGITS),
        "tissue_ugy_per_day_median": round(statistics.median(r["tissue_ugy_per_day"] for r in usable), ROUND_DIGITS),
        "latest": {"observed_at": latest["observed_at"].isoformat(),
                   "silicon_ugy_per_day": latest["silicon_ugy_per_day"],
                   "tissue_ugy_per_day": latest["tissue_ugy_per_day"]},
        "units": {"silicon_ugy_per_day_median": "uGy/day", "tissue_ugy_per_day_median": "uGy/day",
                  "latest.silicon_ugy_per_day": "uGy/day", "latest.tissue_ugy_per_day": "uGy/day"},
    }


def ugy_per_day_to_usv_h(ugy_per_day, quality_factor) -> float:
    """Dose equivalent rate (uSv/h) = absorbed dose rate (uGy/day) * Q / 24. Unrounded; rejects NaN/inf/bool/negative values
    and a non-positive Q with DataUnavailableError."""
    if not _finite_number(ugy_per_day) or ugy_per_day < 0:
        raise DataUnavailableError(f"dose rate must be a finite number >= 0 uGy/day, got {ugy_per_day!r}")
    if not _finite_number(quality_factor) or quality_factor <= 0:
        raise DataUnavailableError(f"quality_factor must be a finite number > 0, got {quality_factor!r}")
    return ugy_per_day * quality_factor / HOURS_PER_DAY


def summarize_radiation_usv_h(rows, quality_factor) -> dict:
    """Pure: mars_radiation rows -> median and latest TISSUE dose equivalent rate in uSv/h, 'none' rows only.

    Each row is converted first (uGy/day * Q / 24) and the median is taken over the converted values. Only
    tissue_ugy_per_day is converted; the silicon (and any uncorrected) column is deliberately absent from the result. Q is an
    approximation (policy PLACEHOLDER 3.0, MSL RAD surface ~0.21 mGy/day absorbed vs ~0.64 mSv/day, Q ~ 3.05) and is echoed."""
    usable = _usable_radiation_rows(rows)
    converted = [(r["observed_at"], ugy_per_day_to_usv_h(r["tissue_ugy_per_day"], quality_factor)) for r in usable]
    latest_at, latest_value = max(converted, key=lambda pair: pair[0])
    return {
        "source": SOURCE_OBSERVED_EXTENDED,
        "spoken_qualifier": spoken_qualifier(SOURCE_OBSERVED_EXTENDED),
        "converted": True,
        "quality_factor": quality_factor,
        "conversion": "uSv/h = tissue_ugy_per_day * quality_factor / 24",
        "row_count": len(usable),
        "tissue_usv_h_median": round(statistics.median(value for _, value in converted), ROUND_DIGITS),
        "latest": {"observed_at": latest_at.isoformat(), "tissue_usv_h": round(latest_value, ROUND_DIGITS)},
        "units": {"tissue_usv_h_median": "uSv/h", "latest.tissue_usv_h": "uSv/h"},
    }


def check_day_night(day_night):
    if day_night is not None and day_night not in DAY_NIGHT_VALUES:
        raise DataUnavailableError(f"unknown day_night {day_night!r} (expected one of {DAY_NIGHT_VALUES} or None)")


def summarize_weather(rows, day_night=None) -> dict:
    """Pure: mars_weather rows -> mean air_temp_c and pressure_pa, ignoring NULLs (None), with how many values each used.

    A non-None value that is not a finite number raises. A column with no values at all raises."""
    check_day_night(day_night)
    if not rows:
        raise DataUnavailableError(f"no weather rows for day_night={day_night!r}")
    result = {"source": SOURCE_OBSERVED_EXTENDED, "spoken_qualifier": spoken_qualifier(SOURCE_OBSERVED_EXTENDED),
              "day_night": day_night, "row_count": len(rows)}
    for field, unit in (("air_temp_c", "C"), ("pressure_pa", "Pa")):
        present = [r.get(field) for r in rows if r.get(field) is not None]
        _require_numbers([{field: v} for v in present], (field,), "mars_weather")
        if not present:
            raise DataUnavailableError(f"mars_weather: no non-NULL {field} for day_night={day_night!r}")
        result[field] = _mean(present)
        result[f"{field}_sample_count"] = len(present)
    result["units"] = {"air_temp_c": "C", "pressure_pa": "Pa"}
    return result


def _mean_o2_l_min(rows, environment, activity) -> float:
    """Unrounded mean o2_consumed_l_min of one environment + activity group; raises if the group is empty or has bad values."""
    if not rows:
        raise DataUnavailableError(f"no crew O2/CO2 rows for environment={environment!r}, activity={activity!r}")
    _require_numbers(rows, ("o2_consumed_l_min",), "crew_o2_co2")
    _require_non_negative(rows, ("o2_consumed_l_min",), "crew_o2_co2")
    return math.fsum(r["o2_consumed_l_min"] for r in rows) / len(rows)


def _check_reference(reference):
    if (not isinstance(reference, tuple) or len(reference) != 2
            or not all(isinstance(part, str) and part for part in reference)):
        raise DataUnavailableError(f"reference must be an (environment, activity) tuple of strings, got {reference!r}")


def summarize_o2_burn_multiplier(selected_rows, reference_rows, environment, activity, reference) -> dict:
    """Pure: multiplier = mean(selected o2_consumed_l_min) / mean(reference o2_consumed_l_min), a dimensionless ratio.

    Fails closed if either group is empty, a value is bad, or the reference mean is not > 0 (no divide-by-zero, no default).
    The reference group is the dataset's EVA surface-geology rows: only ~20 rows of ONE activity, so the multiplier says how
    much more or less oxygen an (environment, activity) group uses than that one EVA activity and nothing more."""
    _check_reference(reference)
    selected_mean = _mean_o2_l_min(selected_rows, environment, activity)
    reference_mean = _mean_o2_l_min(reference_rows, *reference)
    if reference_mean <= 0:
        raise DataUnavailableError(f"reference group {reference!r} has a mean O2 of {reference_mean} L/min (must be > 0)")
    return {
        "source": SOURCE_SIMULATED,
        "spoken_qualifier": spoken_qualifier(SOURCE_SIMULATED),
        "environment": environment,
        "activity": activity,
        "reference": {"environment": reference[0], "activity": reference[1], "row_count": len(reference_rows),
                      "o2_consumed_l_min": round(reference_mean, ROUND_DIGITS)},
        "selected": {"row_count": len(selected_rows), "o2_consumed_l_min": round(selected_mean, ROUND_DIGITS)},
        "multiplier": round(selected_mean / reference_mean, ROUND_DIGITS),
        "units": {"multiplier": "ratio (dimensionless, selected / reference)"},
    }


def _check_tank_capacity(tank_capacity_l):
    if not _finite_number(tank_capacity_l) or tank_capacity_l <= 0:
        raise DataUnavailableError(f"tank_capacity_l must be a finite number > 0 ({TANK_CAPACITY_UNITS}), got {tank_capacity_l!r}")


def summarize_o2_burn_pct_per_min(o2_summary, tank_capacity_l) -> dict:
    """Pure: o2_consumption() result + tank_capacity_l -> percent of the tank used per minute = L/min / capacity_L * 100.

    tank_capacity_l is in litres of O2 at STPD, the same basis as the dataset's L/min, and has NO default here. The policy
    block carries an unsigned-off PLACEHOLDER figure (reference_policy()["o2"]["tank_capacity_l"]) a caller may pass."""
    _check_tank_capacity(tank_capacity_l)
    return {
        "source": SOURCE_SIMULATED,
        "spoken_qualifier": spoken_qualifier(SOURCE_SIMULATED),
        "environment": o2_summary["environment"],
        "activity": o2_summary["activity"],
        "row_count": o2_summary["row_count"],
        "o2_consumed_l_min": o2_summary["o2_consumed_l_min"],
        "tank_capacity_l": tank_capacity_l,
        "o2_burn_pct_per_min": round(o2_summary["o2_consumed_l_min"] / tank_capacity_l * 100, ROUND_DIGITS),
        "units": {"o2_burn_pct_per_min": "% of tank_capacity_l per minute", "o2_consumed_l_min": "L/min (O2 at STPD)",
                  "tank_capacity_l": TANK_CAPACITY_UNITS},
    }


def scaled_sim_base_burn_pct_per_min(mission_base_pct_per_min, multiplier) -> float:
    """DOCUMENTATION ONLY, not used anywhere: how o2_burn_multiplier could scale the simulator base burn.

    simulator_rules.o2_burn_pct_per_min(mission, activity, scenario) returns missions.o2_burn_pct_per_min * ACTIVITY_O2[a] *
    SCENARIO_O2[s]. A dataset multiplier would replace the ACTIVITY_O2 factor, i.e. missions.o2_burn_pct_per_min * multiplier
    * SCENARIO_O2[s]; applying it on top of ACTIVITY_O2 would count the activity twice. simulator_rules is not imported,
    ACTIVITY_O2 is not changed, and nothing calls this. Returns the base * multiplier product, rejecting non-finite or
    non-positive inputs."""
    for name, value in (("mission_base_pct_per_min", mission_base_pct_per_min), ("multiplier", multiplier)):
        if not _finite_number(value) or value <= 0:
            raise DataUnavailableError(f"{name} must be a finite number > 0, got {value!r}")
    return mission_base_pct_per_min * multiplier


# Tiny hand-made rows. SYNTHETIC: these numbers are NOT taken from the nhi CSVs and mean nothing physically.
FIXTURE_CREW_O2_ROWS = (
    {"environment": "Habitat", "activity": "Sleep", "o2_consumed_l_min": 0.20, "o2_consumed_kg_per_day": 0.40,
     "co2_produced_l_min": 0.17, "co2_produced_kg_per_day": 0.50},
    {"environment": "Habitat", "activity": "Sleep", "o2_consumed_l_min": 0.24, "o2_consumed_kg_per_day": 0.48,
     "co2_produced_l_min": 0.21, "co2_produced_kg_per_day": 0.58},
    {"environment": "EVA suit", "activity": "IVA lab / maintenance", "o2_consumed_l_min": 0.50,
     "o2_consumed_kg_per_day": 1.00, "co2_produced_l_min": 0.45, "co2_produced_kg_per_day": 1.20},
    {"environment": "EVA suit", "activity": "EVA surface geology", "o2_consumed_l_min": 0.70,
     "o2_consumed_kg_per_day": 1.40, "co2_produced_l_min": 0.60, "co2_produced_kg_per_day": 1.60},
    {"environment": "EVA suit", "activity": "EVA surface geology", "o2_consumed_l_min": 0.90,
     "o2_consumed_kg_per_day": 1.80, "co2_produced_l_min": 0.80, "co2_produced_kg_per_day": 2.00},
)
FIXTURE_RADIATION_ROWS = (
    {"observed_at": datetime.fromisoformat("2026-01-01T00:06:45+00:00"), "silicon_ugy_per_day": 100.0,
     "tissue_ugy_per_day": 150.0, "data_flag": "none"},
    {"observed_at": datetime.fromisoformat("2026-01-01T00:23:50+00:00"), "silicon_ugy_per_day": 110.0,
     "tissue_ugy_per_day": 160.0, "data_flag": "none"},
    {"observed_at": datetime.fromisoformat("2026-01-01T00:40:55+00:00"), "silicon_ugy_per_day": 9999.0,
     "tissue_ugy_per_day": 9999.0, "data_flag": "flagged"},
)
FIXTURE_WEATHER_ROWS = (
    {"day_night": "Day", "air_temp_c": -20.0, "pressure_pa": 700.0},
    {"day_night": "Day", "air_temp_c": -10.0, "pressure_pa": None},
    {"day_night": "Night", "air_temp_c": -60.0, "pressure_pa": 800.0},
    {"day_night": "Night", "air_temp_c": None, "pressure_pa": 820.0},
)


class FixtureReferenceData:
    """In-memory SYNTHETIC reference rows; no Postgres, no files. Pass `crew_rows` / `radiation_rows` / `weather_rows` to test
    other shapes (rows are dicts with the nhi column names)."""

    def __init__(self, crew_rows=None, radiation_rows=None, weather_rows=None):
        self.crew_rows = FIXTURE_CREW_O2_ROWS if crew_rows is None else crew_rows
        self.radiation_rows = FIXTURE_RADIATION_ROWS if radiation_rows is None else radiation_rows
        self.weather_rows = FIXTURE_WEATHER_ROWS if weather_rows is None else weather_rows

    def o2_consumption(self, environment, activity):
        rows = [r for r in self.crew_rows if r.get("environment") == environment and r.get("activity") == activity]
        return summarize_o2(rows, environment, activity)

    def radiation_baseline(self):
        return summarize_radiation([r for r in self.radiation_rows if r.get("data_flag") == RADIATION_USABLE_FLAG])

    def radiation_baseline_usv_h(self, quality_factor=None):
        q = reference_policy()["radiation"]["quality_factor"] if quality_factor is None else quality_factor
        return summarize_radiation_usv_h(
            [r for r in self.radiation_rows if r.get("data_flag") == RADIATION_USABLE_FLAG], q)

    def _crew_group(self, environment, activity):
        return [r for r in self.crew_rows if r.get("environment") == environment and r.get("activity") == activity]

    def o2_burn_multiplier(self, activity, environment, reference=DEFAULT_MULTIPLIER_REFERENCE):
        _check_reference(reference)
        return summarize_o2_burn_multiplier(self._crew_group(environment, activity), self._crew_group(*reference),
                                            environment, activity, reference)

    def o2_burn_pct_per_min(self, environment, activity, tank_capacity_l):
        _check_tank_capacity(tank_capacity_l)
        return summarize_o2_burn_pct_per_min(self.o2_consumption(environment, activity), tank_capacity_l)

    def weather_summary(self, day_night=None):
        check_day_night(day_night)
        rows = [r for r in self.weather_rows if day_night is None or r.get("day_night") == day_night]
        return summarize_weather(rows, day_night)


class PostgresReferenceData:
    """Reads the nhi reference tables (DATABASE_URL, default "dbname=nova"). psycopg is imported on first use only.

    `connect(database_url)` is injectable (tests pass a fake). Any DB error raises DataUnavailableError. Read-only: only
    SELECT statements are issued."""

    def __init__(self, database_url=None, connect=None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        self._connect = connect or _default_connect

    def _rows(self, sql, params=()):
        try:
            with self._connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except ImportError:
            raise
        except Exception as exc:
            raise DataUnavailableError(f"database read failed: {type(exc).__name__}: {exc}"[:300]) from exc

    def o2_consumption(self, environment, activity):
        return summarize_o2(self._rows(SQL_CREW_O2, (environment, activity)), environment, activity)

    def radiation_baseline(self):
        return summarize_radiation(self._rows(SQL_RADIATION, (RADIATION_USABLE_FLAG,)))

    def radiation_baseline_usv_h(self, quality_factor=None):
        q = reference_policy()["radiation"]["quality_factor"] if quality_factor is None else quality_factor
        return summarize_radiation_usv_h(self._rows(SQL_RADIATION, (RADIATION_USABLE_FLAG,)), q)

    def o2_burn_multiplier(self, activity, environment, reference=DEFAULT_MULTIPLIER_REFERENCE):
        _check_reference(reference)
        return summarize_o2_burn_multiplier(self._rows(SQL_CREW_O2, (environment, activity)),
                                            self._rows(SQL_CREW_O2, tuple(reference)), environment, activity, reference)

    def o2_burn_pct_per_min(self, environment, activity, tank_capacity_l):
        _check_tank_capacity(tank_capacity_l)
        return summarize_o2_burn_pct_per_min(self.o2_consumption(environment, activity), tank_capacity_l)

    def weather_summary(self, day_night=None):
        check_day_night(day_night)
        rows = self._rows(SQL_WEATHER_ALL) if day_night is None else self._rows(SQL_WEATHER_DAY_NIGHT, (day_night,))
        return summarize_weather(rows, day_night)
