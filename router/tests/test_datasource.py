"""router/datasource.py: schema-seeded fixture, pure snapshot mapping, fake-connection Postgres, DB-backed tool registry.

No network, no Postgres, no model, no LLM. The SQL is never executed here, so a test parses schema.sql and checks every
column the SQL constants name."""
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from router import datasource
from router.datasource import (DataUnavailableError, FixtureDatasource, PostgresDatasource, distance_m, message_row,
                               snapshot_from_rows)
from router.tools import OneCallPerTurnError, SimState, default_registry
from router.turn import TurnHandler, load_policy

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)
MISSION = {"id": 1, "scenario": "normal", "activity": "walking", "o2_burn_pct_per_min": 0.15, "walk_speed_mps": 0.8,
           "o2_reserve_pct": 20.0}
POLICY = json.loads((ROOT / "router" / "turn_policy.json").read_text())


def row(sim_seconds_value=30.0, ts=T0, **fields):
    base = {"ts": ts, "sim_seconds": sim_seconds_value, "o2_percent": 87.5, "heart_rate": 72.0, "suit_pressure_kpa": 29.6,
            "suit_temp_c": 21.5, "radiation_msv": 0.5}
    return {**base, **fields}


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class FakeRouter:
    def __init__(self, label="call_o2", confidence=0.99):
        self.label, self.confidence = label, confidence

    def predict(self, text):
        return {"label": self.label, "confidence": self.confidence,
                "topk": [{"label": self.label, "p": self.confidence}, {"label": "NO_TOOL", "p": 1 - self.confidence}]}


def handler_for(ds, label="call_o2"):
    return TurnHandler(FakeRouter(label), ds.snapshot, default_registry(datasource=ds))


# ---- module hygiene -------------------------------------------------------------------------------------------------

def test_importing_datasource_does_not_import_psycopg():
    code = "import sys, router.datasource; print('psycopg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT / "router"))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_simulator_rules_is_imported_from_the_repo_root_not_copied():
    assert Path(datasource.simulator_rules.__file__).resolve() == ROOT / "simulator_rules.py"
    assert datasource.REPO_ROOT == ROOT


def test_default_postgres_connect_without_psycopg_raises_a_clear_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ImportError, match="psycopg"):
        PostgresDatasource().snapshot()


def test_data_source_policy_section_is_placeholder_and_loads():
    policy = load_policy()
    assert policy["data_source"]["status"].startswith("PLACEHOLDER")
    assert set(policy["data_source"]) >= {"max_age_s", "place_aliases", "message", "definitions"}


# ---- field mapping (D1, D2) -----------------------------------------------------------------------------------------

def test_each_snapshot_field_maps_from_its_telemetry_column():
    latest = row(30.0, o2_percent=60.0, heart_rate=101.0, suit_pressure_kpa=28.2, suit_temp_c=23.4, radiation_msv=1.0,
                 ext_temp_c=-77.0, co2_mmhg=9.0)
    snap = snapshot_from_rows(MISSION, [latest, row(0.0, radiation_msv=0.9)], T0, 10)
    assert set(snap) == {"o2_pct", "minutes_remaining", "bpm", "radiation_usv_h", "temp_c", "suit_pressure_kpa"}
    assert (snap["o2_pct"], snap["bpm"], snap["temp_c"], snap["suit_pressure_kpa"]) == (60.0, 101.0, 23.4, 28.2)  # temp_c is suit_temp_c, not ext_temp_c


def test_radiation_rate_is_the_dose_delta_over_sim_hours_in_microsieverts():
    snap = snapshot_from_rows(MISSION, [row(3630.0, radiation_msv=1.5), row(30.0, radiation_msv=1.0)], T0, 10)
    assert snap["radiation_usv_h"] == pytest.approx(0.5 / (3600 / 3600) * 1000)  # 0.5 mSv in 1 sim hour = 500 uSv/h


def test_nominal_mars_dose_is_about_27_9_usv_h_and_radiation_event_is_forty_times_that():
    normal, event = FixtureDatasource(), FixtureDatasource(scenario="radiation_event")
    assert normal.snapshot()["radiation_usv_h"] == pytest.approx(0.67 / 24 * 1000)
    assert event.snapshot()["radiation_usv_h"] == pytest.approx(0.67 / 24 * 1000 * 40)


@pytest.mark.parametrize("rows", [[row()], [row(30.0), row(30.0)], [row(30.0), row(60.0)]])
def test_radiation_rate_needs_two_rows_with_advancing_sim_time(rows):
    with pytest.raises(DataUnavailableError):
        snapshot_from_rows(MISSION, rows, T0, 10)


def test_a_decreasing_cumulative_dose_raises():
    with pytest.raises(DataUnavailableError, match="dose"):
        snapshot_from_rows(MISSION, [row(30.0, radiation_msv=0.4), row(0.0, radiation_msv=0.5)], T0, 10)


@pytest.mark.parametrize("o2, expected", [(100.0, 666), (50.0, 333), (45.0, 300), (0.0, 0)])
def test_minutes_remaining_is_floor_of_o2_over_burn_to_an_empty_tank(o2, expected):
    snap = snapshot_from_rows(MISSION, [row(30.0, o2_percent=o2), row(0.0)], T0, 10)
    assert snap["minutes_remaining"] == expected  # no o2_reserve_pct (20) subtracted: feasibility adds its own reserve


def test_minutes_remaining_uses_the_live_scenario_and_activity():
    leak = {**MISSION, "scenario": "o2_leak"}  # burn 0.15 * 6 = 0.9 %/min
    assert snapshot_from_rows(leak, [row(30.0, o2_percent=50.0), row(0.0)], T0, 10)["minutes_remaining"] == 55
    working = {**MISSION, "activity": "working"}  # 0.15 * 1.3 = 0.195 %/min
    assert snapshot_from_rows(working, [row(30.0, o2_percent=50.0), row(0.0)], T0, 10)["minutes_remaining"] == 256


@pytest.mark.parametrize("field", ["o2_percent", "heart_rate", "suit_pressure_kpa", "suit_temp_c", "radiation_msv", "sim_seconds"])
def test_a_missing_or_non_finite_field_raises_instead_of_returning_a_partial_snapshot(field):
    for bad in (None, float("nan")):
        with pytest.raises(DataUnavailableError, match=field):
            snapshot_from_rows(MISSION, [row(30.0, **{field: bad}), row(0.0)], T0, 10)


def test_no_mission_and_no_telemetry_raise():
    with pytest.raises(DataUnavailableError, match="mission"):
        snapshot_from_rows(None, [row()], T0, 10)
    with pytest.raises(DataUnavailableError, match="telemetry"):
        snapshot_from_rows(MISSION, [], T0, 10)


def test_staleness_raises_past_max_age_and_not_at_it():
    rows = [row(30.0), row(0.0)]
    assert snapshot_from_rows(MISSION, rows, T0 + timedelta(seconds=10), 10)["o2_pct"] == 87.5
    with pytest.raises(DataUnavailableError, match="stale|simulator stopped"):
        snapshot_from_rows(MISSION, rows, T0 + timedelta(seconds=10.5), 10)


def test_fixture_goes_stale_when_the_clock_runs_on_and_recovers_with_a_new_row():
    clock = Clock()
    ds = FixtureDatasource(clock=clock)
    assert ds.snapshot()["o2_pct"] < 100.0
    clock.advance(11)
    with pytest.raises(DataUnavailableError):
        ds.snapshot()
    ds.advance(30)
    assert ds.snapshot()["o2_pct"] < 100.0


def test_max_age_comes_from_the_placeholder_policy():
    assert POLICY["data_source"]["max_age_s"] == 10


# ---- places, distance, eta -------------------------------------------------------------------------------------------

def test_alias_hit_maps_to_the_db_location_and_a_miss_is_none():
    ds = FixtureDatasource()
    assert ds.place("rover")["code"] == "ROVER" and ds.place("homebase")["code"] == "HAB"
    assert ds.place("crater_rim")["code"] == "CRATR" and ds.place("lander") is None


def test_distance_is_euclidean_from_the_latest_position_rounded():
    ds = FixtureDatasource()
    assert ds.place("rover")["distance_m"] == 126  # hypot(120, 40) = 126.49
    assert ds.place("homebase")["distance_m"] == 0
    ds.set_position(120, 40)
    assert ds.place("rover")["distance_m"] == 0 and ds.place("homebase")["distance_m"] == 126
    assert distance_m({"x_m": 0, "z_m": 0}, {"x_m": 3, "z_m": 4}) == 5


def test_navigation_eta_uses_the_feasibility_walking_speed_not_the_db_walk_speed():
    registry = default_registry(datasource=FixtureDatasource())
    registry.begin_turn()
    nav = registry.call("navigation", {"place": "crater_rim"})
    speed = POLICY["feasibility"]["walking_speed_m_per_min"]
    assert nav["data"]["distance_m"] == 966  # hypot(900, 350) = 965.7
    assert nav["data"]["eta_min"] == -(-966 // speed) == 17  # ceil(966 / 60); 0.8 m/s (48 m/min) would say 21


def test_unmapped_place_returns_the_existing_unknown_place_result_for_navigation_and_map():
    registry = default_registry(datasource=FixtureDatasource())
    for tool in ("navigation", "map"):
        registry.begin_turn()
        assert registry.call(tool, {"place": "lander"}) == {"tool": tool, "ok": False, "data": {"error": "unknown place"}}


# ---- send_message ---------------------------------------------------------------------------------------------------

def test_send_message_persists_the_formatted_body_with_policy_delays_and_keeps_the_result_shape():
    ds = FixtureDatasource()
    registry = default_registry(datasource=ds)
    registry.begin_turn()
    result = registry.call("send_message", {"recipient": "crater_rim", "urgency": "urgent", "message": "Need oxygen."})
    assert result == {"tool": "send_message", "ok": True,
                      "data": {"recipient": "crater_rim", "recipient_name": "crater rim", "urgency": "urgent", "status": "queued"}}
    assert len(ds.messages) == 1
    sent = ds.messages[0]
    assert sent["body"] == "[urgent] to crater rim: Need oxygen." and sent["direction"] == "outbound"
    assert (sent["real_delay_s"], sent["demo_delay_s"]) == (POLICY["data_source"]["message"]["real_delay_s"],
                                                           POLICY["data_source"]["message"]["demo_delay_s"])


def test_a_failed_message_write_is_ok_false_not_an_exception():
    class Failing(FixtureDatasource):
        def queue_message(self, recipient, urgency, message):
            return False

    registry = default_registry(datasource=Failing())
    registry.begin_turn()
    result = registry.call("send_message", {"recipient": "rover", "urgency": "normal", "message": "hi"})
    assert result["ok"] is False and result["tool"] == "send_message"


# ---- registry contract (D5) -----------------------------------------------------------------------------------------

def test_datasource_registry_has_the_same_six_tools_schemas_and_result_keys_as_the_default():
    fake, real = default_registry(), default_registry(datasource=FixtureDatasource())
    assert real.names() == fake.names() == ["safety", "navigation", "map", "data", "o2", "send_message"]
    calls = {"safety": {}, "navigation": {"place": "rover"}, "map": {"place": "rover"}, "data": {}, "o2": {},
             "send_message": {"recipient": "rover", "urgency": "normal", "message": "hi"}}
    for name, args in calls.items():
        assert real.schema(name) == fake.schema(name)
        fake.begin_turn(), real.begin_turn()
        a, b = fake.call(name, args), real.call(name, args)
        assert a.keys() == b.keys() and a["data"].keys() == b["data"].keys(), name


def test_datasource_registry_enforces_one_call_per_turn():
    registry = default_registry(datasource=FixtureDatasource())
    registry.begin_turn()
    registry.call("o2", {})
    with pytest.raises(OneCallPerTurnError):
        registry.call("data", {})


def test_default_registry_still_uses_simstate_and_rejects_sim_plus_datasource():
    registry = default_registry()
    registry.begin_turn()
    assert registry.call("o2", {})["data"] == {"o2_pct": 87.5, "minutes_remaining": 212}
    with pytest.raises(ValueError):
        default_registry(sim=SimState(), datasource=FixtureDatasource())


def test_tools_return_snapshot_values_for_safety_data_and_o2():
    ds = FixtureDatasource()
    snap = ds.snapshot()
    registry = default_registry(datasource=ds)
    registry.begin_turn()
    assert registry.call("o2", {})["data"] == {"o2_pct": snap["o2_pct"], "minutes_remaining": snap["minutes_remaining"]}
    registry.begin_turn()
    assert registry.call("data", {})["data"] == {k: snap[k] for k in ("bpm", "radiation_usv_h", "temp_c")}
    registry.begin_turn()
    assert registry.call("safety", {})["data"] == {"status_word": "go"}


def test_a_stale_datasource_makes_the_turn_raise_instead_of_speaking():
    clock = Clock()
    ds = FixtureDatasource(clock=clock)
    clock.advance(60)
    with pytest.raises(DataUnavailableError):
        handler_for(ds).handle("how much oxygen do I have left")


# ---- end to end: TurnHandler over FixtureDatasource -------------------------------------------------------------------

def test_o2_turn_speaks_numbers_that_pass_the_verifier():
    ds = FixtureDatasource()
    out = handler_for(ds).handle("how much oxygen do I have left")
    data = out["tool_result"]["data"]
    assert out["route"] == "tool" and out["verifier"]["ok"] and out["verifier"]["unsupported"] == []
    assert data["minutes_remaining"] == 666 and f"{data['o2_pct']:.1f}" in out["spoken"] and "666 minutes" in out["spoken"]


def test_data_turn_speaks_the_derived_radiation_rate():
    out = handler_for(FixtureDatasource(), "call_data").handle("what's my heart rate and temperature")
    assert out["verifier"]["ok"] and "Radiation 27.92 microsieverts per hour" in out["spoken"]
    assert "Temperature 21.0 degrees Celsius" in out["spoken"]


@pytest.mark.parametrize("utterance, place, distance, eta", [("take me to the crater rim", "crater_rim", 966, 17),
                                                             ("take me to the rover", "rover", 126, 3)])
def test_navigation_turn_has_db_distance_one_consistent_eta_and_feasibility(utterance, place, distance, eta):
    out = handler_for(FixtureDatasource(), "call_navigation").handle(utterance)
    data = out["tool_result"]["data"]
    assert (data["place"], data["distance_m"], data["eta_min"]) == (place, distance, eta)
    feasibility = data["feasibility"]
    assert feasibility["outbound_min"] == eta and feasibility["o2_minutes_remaining"] == 666  # one minute figure per turn
    assert feasibility["outcome"] == "comfortable" and out["verifier"]["ok"], out["spoken"]


def test_navigation_to_an_unmapped_place_speaks_the_tool_error_line():
    out = handler_for(FixtureDatasource(), "call_navigation").handle("take me to the lander")
    assert out["tool_result"]["ok"] is False
    assert out["tool_result"]["data"] == {"error": "unknown place"}


def test_o2_leak_reaches_critical_oxygen_and_the_alert_preempts_the_router():
    ds = FixtureDatasource(scenario="o2_leak")
    assert handler_for(ds).handle("how much oxygen do I have left")["route"] == "tool"
    ds.advance(6000)  # 100 sim min x 0.9 %/min = 90 points
    out = handler_for(ds).handle("how much oxygen do I have left")
    o2 = ds.snapshot()["o2_pct"]
    assert o2 < 16.0 and out["route"] == "alert" and out["pipeline"] is None and out["tool_call"] is None
    assert out["spoken"].startswith(f"Critical alert. Oxygen at {o2:.1f} percent") and out["verifier"]["ok"]
    assert {"field": "o2_pct", "severity": "critical", "value": o2} in out["alerts"]


def test_radiation_event_preempts_with_a_critical_radiation_alert():
    out = handler_for(FixtureDatasource(scenario="radiation_event")).handle("how much oxygen do I have left")
    assert out["route"] == "alert" and "Radiation 1116.7 microsieverts per hour" in out["spoken"]


def test_pinned_current_behavior_nominal_radiation_is_a_warning_on_every_normal_turn():
    """SIGN-OFF flag: 27.9 uSv/h > radiation_usv_h.warning_above 10, so a normal scenario always carries this warning."""
    ds = FixtureDatasource()
    for label in ("call_o2", "call_data", "call_safety"):
        out = handler_for(ds, label).handle("status")
        assert out["route"] == "tool"  # warnings do not preempt
        assert [a for a in out["alerts"] if a["severity"] == "warning"] == [
            {"field": "radiation_usv_h", "severity": "warning", "value": pytest.approx(27.9167, abs=1e-3)}]


def test_end_to_end_low_oxygen_navigation_offers_a_refill_call_and_confirming_persists_the_message():
    ds = FixtureDatasource(scenario="o2_leak")
    ds.advance(5000)  # 83 sim min x 0.9 %/min: o2 about 25 percent, about 27 minutes left, above the 16 percent critical line
    handler = handler_for(ds, "call_navigation")
    out = handler.handle("take me to the crater rim")
    assert out["tool_result"]["data"]["feasibility"]["outcome"] == "cannot_return" and out["verifier"]["ok"]
    assert out["pending_confirmation"] is not None
    handler.router = FakeRouter("call_o2")
    reply = handler.handle("yes please")
    assert reply["route"] == "confirmed_yes" and reply["tool_call"]["name"] == "send_message"
    assert len(ds.messages) == 1 and ds.messages[0]["body"].startswith("[urgent] to crater rim: Nova relay")


# ---- PostgresDatasource over a fake connection ----------------------------------------------------------------------

class FakeCursor:
    def __init__(self, conn):
        self.conn, self.rowcount = conn, 0
        self._result = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.conn.executed.append((sql, params))
        if self.conn.fail_on and self.conn.fail_on in sql:
            raise RuntimeError("connection lost")
        if sql == datasource.SQL_LOCATION:
            location = self.conn.locations.get(params[0])
            self._result = [location] if location else []
        else:
            self._result = self.conn.results.get(sql, [])
        self.rowcount = 1 if sql.startswith("INSERT") else 0

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class FakeConnection:
    """Stands in for a psycopg dict_row connection: returns canned rows per SQL constant."""

    def __init__(self, results, locations, fail_on=None):
        self.results, self.locations, self.fail_on, self.executed = results, locations, fail_on, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self)


def fake_postgres(rows, mission=MISSION, locations=None, fail_on=None):
    locations = {"ROVER": {"code": "ROVER", "name": "Rover", "x_m": 120.0, "z_m": 40.0}} if locations is None else locations
    joined = {**{k: v for k, v in (mission or {}).items() if k != "id"}, "mission_id": (mission or {}).get("id")}
    results = {datasource.SQL_RECENT_TELEMETRY: [{**r, **joined, "db_now": T0} for r in rows] if mission else [],
               datasource.SQL_LATEST_POSITION: [{"x_m": 0.0, "z_m": 0.0}]}
    conn = FakeConnection(results, locations, fail_on)
    seen = []

    def connect(url):
        seen.append(url)
        return conn

    return PostgresDatasource("dbname=test", connect=connect), conn, seen


def test_postgres_snapshot_equals_the_fixture_snapshot_for_the_same_rows():
    fixture = FixtureDatasource(clock=Clock())  # rows stamped T0, same as the fake DB clock
    rows = list(reversed(fixture.telemetry[-2:]))
    ds, conn, seen = fake_postgres(rows, mission=fixture.mission)
    assert ds.snapshot() == fixture.snapshot() and seen == ["dbname=test"]
    assert [sql for sql, _ in conn.executed] == [datasource.SQL_RECENT_TELEMETRY]


def test_postgres_snapshot_fails_closed_without_mission_rows_or_with_old_rows():
    rows = [row(30.0), row(0.0)]
    with pytest.raises(DataUnavailableError, match="no active mission"):
        fake_postgres(rows, mission=None)[0].snapshot()
    with pytest.raises(DataUnavailableError, match="no active mission"):
        fake_postgres([])[0].snapshot()
    stale = [row(30.0, ts=T0 - timedelta(seconds=11)), row(0.0, ts=T0 - timedelta(seconds=41))]
    with pytest.raises(DataUnavailableError):
        fake_postgres(stale)[0].snapshot()


def test_postgres_read_errors_become_data_unavailable():
    ds = fake_postgres([row(30.0), row(0.0)], fail_on="FROM telemetry")[0]
    with pytest.raises(DataUnavailableError, match="connection lost"):
        ds.snapshot()


def test_postgres_place_and_distance():
    ds = fake_postgres([row(30.0), row(0.0)])[0]
    assert ds.place("rover") == {"code": "ROVER", "name": "Rover", "distance_m": 126}
    assert ds.place("lander") is None  # no alias: the DB is never asked
    assert fake_postgres([], locations={})[0].place("rover") is None


def test_postgres_queue_message_inserts_body_and_delays_and_returns_false_on_error():
    ds, conn, _ = fake_postgres([row(30.0), row(0.0)])
    assert ds.queue_message("rover", "normal", "Need oxygen.") is True
    sql, params = conn.executed[-1]
    assert sql == datasource.SQL_INSERT_MESSAGE
    assert params == ("[normal] to rover: Need oxygen.", 600.0, 5.0)
    failing = fake_postgres([row(30.0), row(0.0)], fail_on="INSERT")[0]
    assert failing.queue_message("rover", "normal", "x") is False


def test_postgres_database_url_defaults_to_dbname_nova(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert PostgresDatasource(connect=lambda url: None).database_url == "dbname=nova"
    monkeypatch.setenv("DATABASE_URL", "dbname=other")
    assert PostgresDatasource(connect=lambda url: None).database_url == "dbname=other"


def test_message_row_formats_body_from_policy():
    assert message_row(POLICY["data_source"], "crater_rim", "urgent", "go") == ("[urgent] to crater rim: go", 600.0, 5.0)


# ---- schema.sql cross-check -----------------------------------------------------------------------------------------

def schema_columns():
    text = (ROOT / "schema.sql").read_text()
    tables = {}
    for name, body in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", text, re.S):
        tables[name] = {m.group(1) for m in re.finditer(r"^    ([a-z_0-9]+) [A-Z]", body, re.M)}
    tables["latest_position"] = tables["astronaut_positions"]  # views are SELECT DISTINCT ON ... p.*
    tables["latest_telemetry"] = tables["telemetry"]
    return tables


def test_schema_parser_finds_the_known_columns():
    tables = schema_columns()
    assert {"id", "ts", "radiation_msv", "sim_seconds"} <= tables["telemetry"]
    assert {"code", "x_m", "z_m"} <= tables["map_locations"] and "active" in tables["missions"]


@pytest.mark.parametrize("name", ["SQL_RECENT_TELEMETRY", "SQL_LATEST_POSITION", "SQL_LOCATION"])
def test_select_sql_columns_exist_in_schema_sql(name):
    sql, tables = getattr(datasource, name), schema_columns()
    aliases = {alias: table for table, alias in re.findall(r"(?:FROM|JOIN) (\w+) (\w+)", sql)}
    assert aliases and all(table in tables for table in aliases.values()), sql
    used = re.findall(r"\b([a-z])\.(\w+)", sql)
    assert used
    for alias, column in used:
        assert column in tables[aliases[alias]], f"{name}: {aliases[alias]}.{column} not in schema.sql"


def test_insert_message_sql_columns_exist_in_schema_sql():
    tables = schema_columns()
    match = re.match(r"INSERT INTO (\w+) \(([^)]*)\)", datasource.SQL_INSERT_MESSAGE)
    columns = {c.strip() for c in match.group(2).split(",")}
    assert match.group(1) == "messages" and columns <= tables["messages"]
    assert {"mission_id", "direction", "body", "real_delay_s", "demo_delay_s"} == columns
    assert "FROM missions m WHERE m.active" in datasource.SQL_INSERT_MESSAGE
    assert re.search(r"CHECK \(direction IN \('outbound', 'inbound'\)\)", (ROOT / "schema.sql").read_text())


def test_seed_values_match_schema_sql():
    text = (ROOT / "schema.sql").read_text()
    for loc in datasource.SEED_LOCATIONS:
        assert re.search(rf"'{loc['code']}', '{loc['name']}', '\w+',\s+{loc['x_m']:.0f}, {loc['z_m']:.0f},", text), loc
    assert datasource.SEED_MISSION["o2_burn_pct_per_min"] == 0.15 and "o2_burn_pct_per_min REAL NOT NULL DEFAULT 0.15" in text


def test_fake_demo_ends_with_the_offline_db_backed_section(capsys):
    from router.demo_turn import main
    assert main(["--fake"]) == 0
    out = capsys.readouterr().out
    section = out[out.index("--- DB-backed demo"):]
    assert "Oxygen at 99.9 percent, about 666 minutes remaining." in section
    assert "Radiation 27.92 microsieverts per hour" in section and "walking time 17 minutes." in section
    assert "you cannot get back home without a refill" in section and "marked urgent" in section
    assert "Critical alert. Oxygen at 15.9 percent" in section


def test_postgres_reads_mission_and_telemetry_in_one_statement_and_rejects_mixed_missions():
    ds, conn, _ = fake_postgres([row(30.0), row(0.0)])
    ds.snapshot()
    assert len(conn.executed) == 1
    mixed = conn.results[datasource.SQL_RECENT_TELEMETRY]
    mixed[1] = {**mixed[1], "mission_id": 2, "scenario": "o2_leak"}
    with pytest.raises(DataUnavailableError, match="more than one mission"):
        ds.snapshot()
