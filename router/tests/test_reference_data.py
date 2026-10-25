"""router/reference_data.py: read-only accessors for the nhi reference datasets (crew O2/CO2, Mars radiation, Mars weather).

No network, no Postgres, no model. The SQL is never executed against a real database; a verbatim copy of the nhi DDL
(fixtures/nhi_reference_schema.sql) is parsed to cross-check every column the SQL constants name."""
import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from router import reference_data
from router.datasource import DataUnavailableError, SNAPSHOT_KEYS
from router.reference_data import (FixtureReferenceData, PostgresReferenceData, summarize_o2, summarize_radiation,
                                   summarize_weather)
from router.tools import default_registry

DDL_PATH = Path(__file__).resolve().parent / "fixtures" / "nhi_reference_schema.sql"
T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def radiation_row(minute=0, silicon=100.0, tissue=150.0, flag="none"):
    return {"observed_at": T.replace(minute=minute), "silicon_ugy_per_day": silicon, "tissue_ugy_per_day": tissue,
            "data_flag": flag}


# ---- module hygiene -------------------------------------------------------------------------------------------------

def test_importing_reference_data_does_not_import_psycopg():
    code = "import sys, router.reference_data; print('psycopg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_existing_tool_registry_and_result_keys_are_unchanged():
    registry = default_registry()
    assert registry.names() == ["safety", "navigation", "map", "data", "o2", "send_message", "queue_print"]
    assert SNAPSHOT_KEYS == ("o2_pct", "minutes_remaining", "bpm", "radiation_usv_h", "temp_c", "suit_pressure_kpa")
    registry.begin_turn()
    assert tuple(registry.call("data", {})["data"]) == ("bpm", "radiation_usv_h", "temp_c")
    registry.begin_turn()
    assert tuple(registry.call("o2", {})["data"]) == ("o2_pct", "minutes_remaining")


def test_reference_data_is_not_wired_into_tools_turn_or_simulator_rules():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "tools.py", root / "turn.py", root.parent / "simulator_rules.py"):
        text = path.read_text()
        for token in ("reference_data", "ReferenceData", "o2_burn_multiplier", "quality_factor", "tank_capacity_l"):
            assert token not in text, f"{path.name} mentions {token}"
    # the sim's own activity table is untouched by the O2 burn percentage work
    assert (root.parent / "simulator_rules.py").read_text().count(
        'ACTIVITY_O2 = {"resting": 0.6, "walking": 1.0, "working": 1.3, "running": 1.8}') == 1


def test_only_the_fixture_datasource_reads_the_reference_radiation_and_postgres_keeps_the_live_delta():
    from router.datasource import FixtureDatasource, PostgresDatasource
    assert FixtureDatasource().snapshot()["radiation_usv_h"] == FixtureReferenceData().radiation_baseline_usv_h()["tissue_usv_h_median"]
    assert not hasattr(PostgresDatasource(connect=lambda url: None), "reference")


def test_reference_data_module_imports_neither_tools_turn_nor_simulator_rules():
    tree = ast.parse(Path(reference_data.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    assert not {name for name in imported if name.split(".")[-1] in {"tools", "turn", "simulator_rules"}}, imported


def test_policy_block_is_a_placeholder_and_the_loader_still_accepts_the_policy():
    from router.turn import load_policy
    policy = load_policy()
    block = policy["reference_data"]
    assert block["status"] == "PLACEHOLDER - owner must sign off"
    assert block["radiation"]["quality_factor"] == 3.0 and block["o2"]["tank_capacity_l"] == 400.0
    assert "APPROXIMATION" in block["radiation"]["quality_factor_note"]
    assert "UNSIGNED-OFF" in block["o2"]["tank_capacity_note"]
    assert block["o2"]["multiplier_reference"] == ["EVA suit", "EVA surface geology"]
    assert block["spoken_qualifiers"] == {"reference-observed-extended": "extended reference value",
                                          "reference-simulated": "simulated reference value"}
    assert reference_data.reference_policy() == block


def test_unusable_policy_block_fails_closed(tmp_path, monkeypatch):
    good = json.loads(reference_data.POLICY_PATH.read_text())["reference_data"]
    cases = {"missing": None, "no_placeholder": {**good, "status": "approved"},
             "bad_q": {**good, "radiation": {**good["radiation"], "quality_factor": 0}},
             "bad_tank": {**good, "o2": {**good["o2"], "tank_capacity_l": float("nan")}},
             "empty_qualifier": {**good, "spoken_qualifiers": {**good["spoken_qualifiers"], "reference-simulated": " "}}}
    for name, block in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({} if block is None else {"reference_data": block}))
        monkeypatch.setattr(reference_data, "POLICY_PATH", path)
        with pytest.raises(DataUnavailableError, match="policy unusable"):
            FixtureReferenceData().o2_consumption("Habitat", "Sleep")
    monkeypatch.setattr(reference_data, "POLICY_PATH", tmp_path / "absent.json")
    with pytest.raises(DataUnavailableError):
        reference_data.spoken_qualifier("reference-simulated")


# ---- fixture accessors ----------------------------------------------------------------------------------------------

def test_fixture_o2_consumption_is_a_deterministic_labelled_mean():
    result = FixtureReferenceData().o2_consumption("Habitat", "Sleep")
    assert result["source"] == "reference-simulated" and result["spoken_qualifier"] == "simulated reference value"
    assert (result["environment"], result["activity"], result["row_count"]) == ("Habitat", "Sleep", 2)
    assert result["o2_consumed_l_min"] == 0.22 and result["o2_consumed_kg_per_day"] == 0.44
    assert result["units"]["o2_consumed_l_min"] == "L/min" and result["units"]["o2_consumed_kg_per_day"].startswith("kg/day")
    assert result == FixtureReferenceData().o2_consumption("Habitat", "Sleep")


@pytest.mark.parametrize("environment, activity", [("Habitat", "Skydiving"), ("Moon base", "Sleep"),
                                                   ("EVA suit", "Sleep"), ("habitat", "Sleep"), ("", ""), (None, None)])
def test_fixture_o2_unknown_environment_or_activity_fails_closed(environment, activity):
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData().o2_consumption(environment, activity)


def test_o2_empty_dataset_and_bad_values_fail_closed():
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(crew_rows=[]).o2_consumption("Habitat", "Sleep")
    good = {"environment": "Habitat", "activity": "Sleep", "o2_consumed_l_min": 0.2, "o2_consumed_kg_per_day": 0.4,
            "co2_produced_l_min": 0.2, "co2_produced_kg_per_day": 0.5}
    for field in ("o2_consumed_l_min", "o2_consumed_kg_per_day", "co2_produced_l_min", "co2_produced_kg_per_day"):
        for bad in (None, float("nan"), float("inf"), "0.2", True):
            with pytest.raises(DataUnavailableError):
                FixtureReferenceData(crew_rows=[good, {**good, field: bad}]).o2_consumption("Habitat", "Sleep")


def test_fixture_radiation_baseline_uses_only_data_flag_none_and_labels_units():
    result = FixtureReferenceData().radiation_baseline()
    assert result["source"] == "reference-observed-extended" and result["row_count"] == 2
    assert result["spoken_qualifier"] == "extended reference value" and "converted" not in result
    assert result["silicon_ugy_per_day_median"] == 105.0 and result["tissue_ugy_per_day_median"] == 155.0
    assert result["latest"] == {"observed_at": "2026-01-01T00:23:50+00:00", "silicon_ugy_per_day": 110.0,
                                "tissue_ugy_per_day": 160.0}
    assert set(result["units"].values()) == {"uGy/day"}
    assert not any("usv" in key.lower() for key in result)


def test_radiation_latest_is_by_observed_at_not_row_order():
    rows = [radiation_row(30, 120.0, 170.0), radiation_row(10, 100.0, 150.0), radiation_row(20, 110.0, 160.0)]
    result = summarize_radiation(rows)
    assert result["latest"]["silicon_ugy_per_day"] == 120.0 and result["silicon_ugy_per_day_median"] == 110.0


def test_radiation_fails_closed_on_empty_all_flagged_nulls_nan_and_bad_timestamp():
    for rows in ([], [radiation_row(flag="saturated")], [radiation_row(flag="")]):
        with pytest.raises(DataUnavailableError):
            summarize_radiation(rows)
    for bad in (None, float("nan"), float("inf"), "100"):
        for field in ("silicon", "tissue"):
            with pytest.raises(DataUnavailableError):
                summarize_radiation([radiation_row(1), radiation_row(2, **{field: bad})])
    with pytest.raises(DataUnavailableError):
        summarize_radiation([{**radiation_row(), "observed_at": "2026-01-01"}])
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(radiation_rows=[]).radiation_baseline()


def test_radiation_ignores_nan_in_flagged_rows_only():
    rows = [radiation_row(1), radiation_row(2, silicon=float("nan"), flag="bad")]
    assert summarize_radiation(rows)["row_count"] == 1


def test_fixture_weather_ignores_nulls_per_column():
    ds = FixtureReferenceData()
    everything = ds.weather_summary()
    assert everything["source"] == "reference-observed-extended" and everything["day_night"] is None
    assert everything["spoken_qualifier"] == "extended reference value"
    assert everything["day_night"] is None and everything["row_count"] == 4
    assert (everything["air_temp_c"], everything["air_temp_c_sample_count"]) == (-30.0, 3)
    assert (everything["pressure_pa"], everything["pressure_pa_sample_count"]) == (773.3333, 3)
    day = ds.weather_summary("Day")
    assert (day["air_temp_c"], day["pressure_pa"], day["pressure_pa_sample_count"]) == (-15.0, 700.0, 1)
    assert day["units"] == {"air_temp_c": "C", "pressure_pa": "Pa"}
    assert ds.weather_summary("Night")["air_temp_c"] == -60.0


def test_weather_fails_closed_on_unknown_day_night_empty_all_null_and_nan():
    ds = FixtureReferenceData()
    for bad in ("day", "Dusk", ""):
        with pytest.raises(DataUnavailableError):
            ds.weather_summary(bad)
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(weather_rows=[]).weather_summary()
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(weather_rows=[{"day_night": "Day", "air_temp_c": None, "pressure_pa": 700.0}]).weather_summary()
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(weather_rows=[{"day_night": "Day", "air_temp_c": 1.0, "pressure_pa": None}]).weather_summary()
    for bad in (float("nan"), float("-inf"), "5"):
        with pytest.raises(DataUnavailableError):
            summarize_weather([{"air_temp_c": -20.0, "pressure_pa": 700.0}, {"air_temp_c": bad, "pressure_pa": 700.0}])
    with pytest.raises(DataUnavailableError):  # a day_night with no rows
        FixtureReferenceData(weather_rows=[{"day_night": "Day", "air_temp_c": 1.0, "pressure_pa": 2.0}]).weather_summary("Night")


# ---- uGy/day -> uSv/h conversion --------------------------------------------------------------------------------------

def test_ugy_per_day_to_usv_h_exact_math():
    convert = reference_data.ugy_per_day_to_usv_h
    assert convert(24.0, 1.0) == 1.0
    assert convert(240.0, 3.0) == 30.0
    assert convert(150.0, 3.0) == 18.75
    assert convert(0, 3.0) == 0.0
    assert convert(210.0, 3.05) == pytest.approx(210.0 * 3.05 / 24)  # MSL-like 0.21 mGy/day -> about 26.7 uSv/h


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.1, -150.0, None, "150", True])
def test_conversion_rejects_bad_dose_rates(bad):
    with pytest.raises(DataUnavailableError):
        reference_data.ugy_per_day_to_usv_h(bad, 3.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, 0.0, -3.0, None, "3", True])
def test_conversion_rejects_bad_quality_factors(bad):
    with pytest.raises(DataUnavailableError):
        reference_data.ugy_per_day_to_usv_h(150.0, bad)
    if bad is not None:  # None means "use the policy Q"
        with pytest.raises(DataUnavailableError):
            FixtureReferenceData().radiation_baseline_usv_h(bad)


def test_fixture_radiation_baseline_usv_h_uses_tissue_only_none_rows_and_the_policy_q():
    result = FixtureReferenceData().radiation_baseline_usv_h()
    assert result["converted"] is True and result["quality_factor"] == 3.0
    assert result["source"] == "reference-observed-extended" and result["spoken_qualifier"] == "extended reference value"
    assert result["row_count"] == 2  # the 'flagged' 9999 row is excluded
    assert result["tissue_usv_h_median"] == 19.375  # median(150*3/24, 160*3/24) = median(18.75, 20.0)
    assert result["latest"] == {"observed_at": "2026-01-01T00:23:50+00:00", "tissue_usv_h": 20.0}
    assert result["units"] == {"tissue_usv_h_median": "uSv/h", "latest.tissue_usv_h": "uSv/h"}
    assert "silicon" not in json.dumps({k: v for k, v in result.items() if k != "conversion"})
    assert result == FixtureReferenceData().radiation_baseline_usv_h()
    assert FixtureReferenceData().radiation_baseline_usv_h(quality_factor=6.0)["tissue_usv_h_median"] == 38.75
    assert FixtureReferenceData().radiation_baseline_usv_h(quality_factor=6.0)["quality_factor"] == 6.0


def test_radiation_usv_h_latest_is_by_observed_at_and_median_is_over_converted_values():
    rows = [radiation_row(30, 1.0, 240.0), radiation_row(10, 1.0, 24.0), radiation_row(20, 1.0, 48.0)]
    result = reference_data.summarize_radiation_usv_h(rows, 1.0)
    assert result["tissue_usv_h_median"] == 2.0 and result["latest"]["tissue_usv_h"] == 10.0
    assert result["latest"]["observed_at"] == T.replace(minute=30).isoformat()


def test_radiation_usv_h_keeps_the_raw_accessor_and_does_not_mix_units():
    raw = FixtureReferenceData().radiation_baseline()
    assert set(raw["units"].values()) == {"uGy/day"} and raw["tissue_ugy_per_day_median"] == 155.0


def test_radiation_fails_closed_on_negative_nan_and_empty_in_both_paths():
    for convert in (summarize_radiation, lambda rows: reference_data.summarize_radiation_usv_h(rows, 3.0)):
        for field in ("silicon", "tissue"):
            for bad in (-1.0, float("nan"), float("inf"), None):
                with pytest.raises(DataUnavailableError):
                    convert([radiation_row(1), radiation_row(2, **{field: bad})])
        for rows in ([], [radiation_row(flag="x")], [{**radiation_row(), "observed_at": None}]):
            with pytest.raises(DataUnavailableError):
                convert(rows)
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(radiation_rows=[]).radiation_baseline_usv_h()


def test_postgres_radiation_baseline_usv_h_issues_one_select_and_converts():
    ds, log = fake_postgres([radiation_row(1, tissue=240.0), radiation_row(2, tissue=480.0)])
    result = ds.radiation_baseline_usv_h()
    assert result["tissue_usv_h_median"] == 45.0 and result["quality_factor"] == 3.0
    assert len(log) == 1 and log[0][1] == ("none",) and log[0][0].startswith("SELECT ")
    ds, _ = fake_postgres([])
    with pytest.raises(DataUnavailableError):
        ds.radiation_baseline_usv_h()


# ---- O2 burn multiplier and percent-of-tank -----------------------------------------------------------------------------

def crew_row(environment, activity, l_min):
    return {"environment": environment, "activity": activity, "o2_consumed_l_min": l_min, "o2_consumed_kg_per_day": 1.0,
            "co2_produced_l_min": 0.5, "co2_produced_kg_per_day": 1.0}


EVA_REFERENCE = ("EVA suit", "EVA surface geology")


def test_o2_burn_multiplier_is_an_exact_deterministic_ratio_of_group_means():
    result = FixtureReferenceData().o2_burn_multiplier("Sleep", "Habitat")
    assert result["multiplier"] == 0.275  # mean(0.20, 0.24) / mean(0.70, 0.90) = 0.22 / 0.80
    assert result["selected"] == {"row_count": 2, "o2_consumed_l_min": 0.22}
    assert result["reference"] == {"environment": "EVA suit", "activity": "EVA surface geology", "row_count": 2,
                                   "o2_consumed_l_min": 0.8}
    assert result["source"] == "reference-simulated" and result["spoken_qualifier"] == "simulated reference value"
    assert "dimensionless" in result["units"]["multiplier"]
    assert result == FixtureReferenceData().o2_burn_multiplier("Sleep", "Habitat")
    assert FixtureReferenceData().o2_burn_multiplier("EVA surface geology", "EVA suit")["multiplier"] == 1.0
    assert FixtureReferenceData().o2_burn_multiplier("Sleep", "Habitat", reference=("EVA suit", "IVA lab / maintenance"))[
        "multiplier"] == 0.44


def test_o2_burn_multiplier_rounds_the_ratio_not_the_inputs():
    rows = [crew_row("Habitat", "A", 1.0), crew_row("EVA suit", "EVA surface geology", 3.0)]
    assert FixtureReferenceData(crew_rows=rows).o2_burn_multiplier("A", "Habitat")["multiplier"] == 0.3333


def test_o2_burn_multiplier_fails_closed():
    ds = FixtureReferenceData()
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(crew_rows=[r for r in reference_data.FIXTURE_CREW_O2_ROWS
                                        if r["activity"] != "EVA surface geology"]).o2_burn_multiplier("Sleep", "Habitat")
    with pytest.raises(DataUnavailableError):  # empty reference group
        ds.o2_burn_multiplier("Sleep", "Habitat", reference=("EVA suit", "Skydiving"))
    with pytest.raises(DataUnavailableError):  # empty selected group
        ds.o2_burn_multiplier("Skydiving", "Habitat")
    with pytest.raises(DataUnavailableError):  # swapped (environment, activity) order
        ds.o2_burn_multiplier("Habitat", "Sleep")
    zero = [crew_row("Habitat", "Sleep", 0.2), crew_row("EVA suit", "EVA surface geology", 0.0)]
    with pytest.raises(DataUnavailableError, match="must be > 0"):
        FixtureReferenceData(crew_rows=zero).o2_burn_multiplier("Sleep", "Habitat")
    for bad in (float("nan"), float("inf"), -0.5, None, "1"):
        for index in (0, 1):
            rows = [crew_row("Habitat", "Sleep", 0.2), crew_row(*EVA_REFERENCE, 0.8)]
            rows[index] = {**rows[index], "o2_consumed_l_min": bad}
            with pytest.raises(DataUnavailableError):
                FixtureReferenceData(crew_rows=rows).o2_burn_multiplier("Sleep", "Habitat")
    for bad_reference in (None, "EVA suit", ("EVA suit",), ["EVA suit", "EVA surface geology"], ("EVA suit", ""), (1, 2)):
        with pytest.raises(DataUnavailableError):
            ds.o2_burn_multiplier("Sleep", "Habitat", reference=bad_reference)


def test_o2_burn_pct_per_min_is_litres_per_minute_over_tank_litres_times_100():
    result = FixtureReferenceData().o2_burn_pct_per_min("Habitat", "Sleep", 400.0)
    assert result["o2_burn_pct_per_min"] == 0.055  # 0.22 / 400 * 100
    assert result["o2_consumed_l_min"] == 0.22 and result["tank_capacity_l"] == 400.0
    assert result["source"] == "reference-simulated" and result["spoken_qualifier"] == "simulated reference value"
    assert result["units"]["tank_capacity_l"] == "L O2 at STPD" and "% of tank_capacity_l" in result["units"]["o2_burn_pct_per_min"]
    assert FixtureReferenceData().o2_burn_pct_per_min("Habitat", "Sleep", 22.0)["o2_burn_pct_per_min"] == 1.0
    policy_tank = reference_data.reference_policy()["o2"]["tank_capacity_l"]
    assert FixtureReferenceData().o2_burn_pct_per_min("Habitat", "Sleep", policy_tank)["o2_burn_pct_per_min"] == 0.055


def test_o2_burn_pct_per_min_requires_an_explicit_valid_tank_capacity():
    ds = FixtureReferenceData()
    with pytest.raises(TypeError):
        ds.o2_burn_pct_per_min("Habitat", "Sleep")
    for bad in (0, 0.0, -400.0, float("nan"), float("inf"), None, "400", True):
        with pytest.raises(DataUnavailableError):
            ds.o2_burn_pct_per_min("Habitat", "Sleep", bad)
    for environment, activity in (("Habitat", "Skydiving"), ("Moon base", "Sleep"), (None, None)):
        with pytest.raises(DataUnavailableError):
            ds.o2_burn_pct_per_min(environment, activity, 400.0)
    with pytest.raises(DataUnavailableError):
        FixtureReferenceData(crew_rows=[]).o2_burn_pct_per_min("Habitat", "Sleep", 400.0)


def test_postgres_o2_burn_helpers_use_the_shared_pure_functions():
    rows = [crew_row("Habitat", "Sleep", 0.2), crew_row("Habitat", "Sleep", 0.4)]
    ds, log = fake_postgres(rows)
    assert ds.o2_burn_pct_per_min("Habitat", "Sleep", 100.0)["o2_burn_pct_per_min"] == 0.3
    assert ds.o2_burn_multiplier("Sleep", "Habitat")["multiplier"] == 1.0  # fake returns the same rows for both selects
    assert [params for _, params in log] == [("Habitat", "Sleep"), ("Habitat", "Sleep"), ("EVA suit", "EVA surface geology")]
    assert all(sql.startswith("SELECT ") for sql, _ in log)
    ds, log = fake_postgres([])
    with pytest.raises(DataUnavailableError):
        ds.o2_burn_multiplier("Sleep", "Habitat")
    with pytest.raises(DataUnavailableError):
        ds.o2_burn_pct_per_min("Habitat", "Sleep", float("nan"))


def test_scaled_sim_base_burn_is_plain_multiplication_with_validation_and_does_not_touch_the_sim():
    scale = reference_data.scaled_sim_base_burn_pct_per_min
    assert scale(0.15, 2.0) == 0.3 and scale(0.15, 1.0) == 0.15
    for bad in (0, -1.0, float("nan"), float("inf"), None, True):
        with pytest.raises(DataUnavailableError):
            scale(bad, 1.0)
        with pytest.raises(DataUnavailableError):
            scale(0.15, bad)
    assert "simulator_rules" not in vars(reference_data)


# ---- source markers ---------------------------------------------------------------------------------------------------

def test_source_markers_distinguish_simulated_from_observed_extended():
    ds = FixtureReferenceData()
    simulated = {ds.o2_consumption("Habitat", "Sleep")["source"], ds.o2_burn_multiplier("Sleep", "Habitat")["source"],
                 ds.o2_burn_pct_per_min("Habitat", "Sleep", 400.0)["source"]}
    observed = {ds.radiation_baseline()["source"], ds.radiation_baseline_usv_h()["source"], ds.weather_summary()["source"],
                ds.weather_summary("Day")["source"]}
    assert simulated == {"reference-simulated"} and observed == {"reference-observed-extended"}
    qualifiers = reference_data.reference_policy()["spoken_qualifiers"]
    assert set(qualifiers) == simulated | observed and len(set(qualifiers.values())) == 2
    assert ds.weather_summary()["spoken_qualifier"] == qualifiers["reference-observed-extended"]


# ---- Postgres accessor (fake connection) ----------------------------------------------------------------------------

class FakeConnection:
    def __init__(self, rows, log, error=None):
        self.rows, self.log, self.error = rows, log, error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        if self.error:
            raise self.error
        self.log.append((sql, params))

    def fetchall(self):
        return self.rows


def fake_postgres(rows, error=None):
    log = []
    return PostgresReferenceData(database_url="dbname=fake", connect=lambda url: FakeConnection(rows, log, error)), log


def test_postgres_accessors_issue_only_select_and_use_the_shared_summaries():
    ds, log = fake_postgres([{"o2_consumed_l_min": 0.2, "o2_consumed_kg_per_day": 0.4, "co2_produced_l_min": 0.1,
                              "co2_produced_kg_per_day": 0.3}])
    assert ds.o2_consumption("Habitat", "Sleep")["o2_consumed_l_min"] == 0.2
    ds, log_r = fake_postgres([radiation_row(1), radiation_row(2, flag="bad")])
    assert ds.radiation_baseline()["row_count"] == 1
    ds, log_w = fake_postgres([{"air_temp_c": -1.0, "pressure_pa": None}, {"air_temp_c": None, "pressure_pa": 5.0}])
    assert ds.weather_summary("Day")["pressure_pa"] == 5.0
    assert log[0][1] == ("Habitat", "Sleep") and log_r[0][1] == ("none",) and log_w[0][1] == ("Day",)
    for sql, _ in log + log_r + log_w:
        assert sql.startswith("SELECT ")


def test_postgres_fails_closed_on_empty_database_error_and_bad_day_night():
    ds, _ = fake_postgres([])
    for call in (lambda: ds.o2_consumption("Habitat", "Sleep"), ds.radiation_baseline, ds.weather_summary):
        with pytest.raises(DataUnavailableError):
            call()
    ds, _ = fake_postgres([], error=RuntimeError("connection refused"))
    with pytest.raises(DataUnavailableError, match="database read failed"):
        ds.weather_summary()
    ds, log = fake_postgres([])
    with pytest.raises(DataUnavailableError):
        ds.weather_summary("dusk")
    assert log == []  # rejected before any query


def test_postgres_missing_psycopg_is_an_import_error_not_a_fake_reading(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ImportError):
        PostgresReferenceData().radiation_baseline()


# ---- nhi DDL cross-check --------------------------------------------------------------------------------------------

def ddl_columns():
    text = DDL_PATH.read_text()
    return {name: {m.group(1) for m in re.finditer(r"^    ([a-z_0-9]+)\s+[A-Z]", body, re.M)}
            for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", text, re.S)}


def test_ddl_parser_finds_the_three_tables_and_known_columns():
    tables = ddl_columns()
    assert set(tables) == {"mars_weather", "crew_o2_co2", "mars_radiation"}
    assert {"air_temp_c", "pressure_pa", "day_night"} <= tables["mars_weather"]
    assert {"environment", "activity", "o2_consumed_l_min", "o2_consumed_kg_per_day"} <= tables["crew_o2_co2"]
    assert {"observed_at", "silicon_ugy_per_day", "tissue_ugy_per_day", "data_flag"} <= tables["mars_radiation"]


@pytest.mark.parametrize("name", ["SQL_CREW_O2", "SQL_RADIATION", "SQL_WEATHER_ALL", "SQL_WEATHER_DAY_NIGHT"])
def test_select_sql_columns_exist_in_the_nhi_ddl(name):
    sql, tables = getattr(reference_data, name), ddl_columns()
    aliases = {alias: table for table, alias in re.findall(r"(?:FROM|JOIN) (\w+) (\w+)", sql)}
    assert aliases and all(table in tables for table in aliases.values()), sql
    used = re.findall(r"\b([a-z])\.(\w+)", sql)
    assert used
    for alias, column in used:
        assert column in tables[aliases[alias]], f"{name}: {aliases[alias]}.{column} not in nhi DDL"


def test_summaries_only_read_columns_the_sql_selects():
    selected = {name: set(re.findall(r"\b[a-z]\.(\w+)", getattr(reference_data, name)))
                for name in ("SQL_CREW_O2", "SQL_RADIATION", "SQL_WEATHER_ALL")}
    source = Path(reference_data.__file__).read_text()
    used_o2 = set(re.findall(r'r\["(\w+)"\]', source[source.index("def summarize_o2"):source.index("def summarize_radiation")]))
    assert used_o2 <= selected["SQL_CREW_O2"]
    used_rad = set(re.findall(r'r\["(\w+)"\]', source[source.index("def summarize_radiation"):source.index("def check_day_night")]))
    assert used_rad <= selected["SQL_RADIATION"]
    assert {"air_temp_c", "pressure_pa"} <= selected["SQL_WEATHER_ALL"]


def test_ddl_fixture_is_a_verbatim_nhi_extract_with_the_expected_constraints():
    text = DDL_PATH.read_text()
    assert "CREATE INDEX IF NOT EXISTS crew_o2_co2_activity ON crew_o2_co2 (environment, activity);" in text
    assert "CHECK (day_night IN ('Day', 'Night'))" in text and "data_flag                   TEXT NOT NULL DEFAULT 'none'" in text
