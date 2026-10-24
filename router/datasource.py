"""Opt-in data layer between the staging Postgres schema (schema.sql, simulator.py) and the router's tools.

    registry = default_registry(datasource=FixtureDatasource())          # offline fixture, no Postgres
    registry = default_registry(datasource=PostgresDatasource())         # needs psycopg + a running simulator
    handler = TurnHandler(router, datasource.snapshot, registry)         # the same snapshot is the alert source

`snapshot()` returns exactly the six keys the router already uses (o2_pct, minutes_remaining, bpm, radiation_usv_h,
temp_c, suit_pressure_kpa). It FAILS CLOSED: no active mission, no telemetry, a row older than `max_age_s`, fewer than two
rows (the radiation rate needs two), or a missing/non-finite value raises `DataUnavailableError`. It never returns a
partial dict and never falls back to a nominal value, because `evaluate_alerts` skips fields that are missing and a
silent gap would hide an alert. A stopped simulator therefore makes every turn raise (the alert poll propagates by design).

Both datasources build the snapshot through the one pure function `snapshot_from_rows`, so the fixture and Postgres paths
cannot drift apart. Every definition below is a PLACEHOLDER recorded in turn_policy.json["data_source"]:
 * temp_c = telemetry.suit_temp_c (ext_temp_c would trip the -10 C critical immediately); bpm = heart_rate; the rest direct.
 * radiation_usv_h = (radiation_msv[t] - radiation_msv[t-1]) / ((sim_seconds[t] - sim_seconds[t-1]) / 3600) * 1000.
 * minutes_remaining = floor(o2_percent / o2_burn_pct_per_min(mission)), counted to an EMPTY tank at the live activity and
   scenario. The reserve is NOT subtracted here: turn.assess_feasibility already adds reserve_margin_min (no double count).
   missions.o2_reserve_pct is not used by the router.
The burn formula is imported from the repo-root simulator_rules.py (its one definition); psycopg is imported lazily.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
POLICY_PATH = HERE / "turn_policy.json"

if str(REPO_ROOT) not in sys.path:  # simulator_rules.py lives at the repo root, outside the router package
    sys.path.append(str(REPO_ROOT))
try:
    import simulator_rules
except ImportError as exc:  # pragma: no cover - only when the router is installed without the repo checkout
    raise ImportError(f"router.datasource needs simulator_rules.py from the repo root ({REPO_ROOT})") from exc

SNAPSHOT_KEYS = ("o2_pct", "minutes_remaining", "bpm", "radiation_usv_h", "temp_c", "suit_pressure_kpa")
TELEMETRY_FIELDS = ("sim_seconds", "o2_percent", "heart_rate", "suit_pressure_kpa", "suit_temp_c", "radiation_msv")
MISSION_FIELDS = ("id", "scenario", "activity", "o2_burn_pct_per_min", "walk_speed_mps", "o2_reserve_pct")
DEFAULT_DATABASE_URL = "dbname=nova"

# SQL lives in constants and only names schema.sql columns (tests cross-check them). latest_telemetry is ONE row, so the
# radiation delta reads the last two telemetry rows of the active mission. `now()` is the DB clock (no router/DB skew).
SQL_RECENT_TELEMETRY = ("SELECT t.ts, t.sim_seconds, t.o2_percent, t.heart_rate, t.suit_pressure_kpa, t.suit_temp_c, "
                        "t.radiation_msv, now() AS db_now, m.id AS mission_id, m.scenario, m.activity, "
                        "m.o2_burn_pct_per_min, m.walk_speed_mps, m.o2_reserve_pct "
                        "FROM telemetry t JOIN missions m ON m.id = t.mission_id "
                        "WHERE m.active ORDER BY t.ts DESC, t.id DESC LIMIT 2")
SQL_LATEST_POSITION = "SELECT p.x_m, p.z_m FROM latest_position p LIMIT 1"
SQL_LOCATION = "SELECT l.code, l.name, l.x_m, l.z_m FROM map_locations l WHERE l.code = %s"
SQL_INSERT_MESSAGE = ("INSERT INTO messages (mission_id, direction, body, real_delay_s, demo_delay_s) "
                      "SELECT m.id, 'outbound', %s, %s, %s FROM missions m WHERE m.active")


class DataUnavailableError(Exception):
    """The datasource cannot give a complete, fresh snapshot. Callers must not substitute values."""


class Datasource(Protocol):
    def snapshot(self) -> dict:
        """The six router snapshot keys, or raise DataUnavailableError."""

    def place(self, name: str) -> dict | None:
        """{"code", "name", "distance_m": int} for a router place name, or None when it has no map location."""

    def queue_message(self, recipient: str, urgency: str, message: str) -> bool:
        """Persist an outbound message; False when the write failed."""


def load_data_source_policy(path=POLICY_PATH) -> dict:
    return json.loads(Path(path).read_text())["data_source"]


def load_walking_speed_m_per_min(path=POLICY_PATH) -> float:
    """The feasibility walking speed; eta_min uses it so a navigation turn speaks one consistent minute figure."""
    return json.loads(Path(path).read_text())["feasibility"]["walking_speed_m_per_min"]


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def snapshot_from_rows(mission, rows, now, max_age_s) -> dict:
    """Pure: active `mission` row (or None) + telemetry `rows` newest first + `now` (aware datetime) -> snapshot dict.

    Raises DataUnavailableError instead of returning anything partial."""
    if not mission:
        raise DataUnavailableError("no active mission")
    if not rows:
        raise DataUnavailableError("no telemetry for the active mission")
    latest = rows[0]
    age_s = (now - latest["ts"]).total_seconds()
    if age_s > max_age_s or age_s < -max_age_s:
        raise DataUnavailableError(f"telemetry is {age_s:.1f} s from now (limit {max_age_s} s): simulator stopped?")
    if len(rows) < 2:
        raise DataUnavailableError("radiation rate needs two telemetry rows")
    previous = rows[1]
    for field in TELEMETRY_FIELDS:
        if not _finite_number(latest.get(field)) or not _finite_number(previous.get(field)):
            raise DataUnavailableError(f"telemetry field {field!r} is missing or not a finite number")
    elapsed_s = latest["sim_seconds"] - previous["sim_seconds"]
    if elapsed_s <= 0:
        raise DataUnavailableError("telemetry sim_seconds does not advance, so no radiation rate")
    dose_msv = latest["radiation_msv"] - previous["radiation_msv"]
    if dose_msv < 0:
        raise DataUnavailableError("cumulative radiation dose went down")
    try:
        burn_pct_per_min = simulator_rules.o2_burn_pct_per_min(mission)
    except (KeyError, TypeError, ValueError) as exc:
        raise DataUnavailableError(f"mission o2 burn rate unusable: {exc}") from exc
    return {
        "o2_pct": latest["o2_percent"],
        "minutes_remaining": math.floor(latest["o2_percent"] / burn_pct_per_min + 1e-9),
        "bpm": latest["heart_rate"],
        "radiation_usv_h": dose_msv / (elapsed_s / 3600.0) * 1000.0,
        "temp_c": latest["suit_temp_c"],
        "suit_pressure_kpa": latest["suit_pressure_kpa"],
    }


def resolve_place_code(name, place_aliases):
    """Router place name -> map_locations.code via the PLACEHOLDER alias table, or None."""
    return place_aliases.get(name)


def distance_m(position, location) -> int:
    """Euclidean distance on the x/z plane in meters (ignores terrain and hazards), rounded to a whole meter."""
    return round(math.hypot(location["x_m"] - position["x_m"], location["z_m"] - position["z_m"]))


def message_row(policy, recipient, urgency, message):
    """-> (body, real_delay_s, demo_delay_s) for an outbound message; formats and delays are PLACEHOLDER policy."""
    cfg = policy["message"]
    body = cfg["body_format"].format(urgency=urgency, recipient_name=recipient.replace("_", " "), message=message)
    return body, cfg["real_delay_s"], cfg["demo_delay_s"]


# schema.sql seed values (placeholder map coordinates there too)
SEED_MISSION = {"id": 1, "name": "NOVA-1", "planet_id": "mars", "scenario": "normal", "activity": "walking",
                "mission_day_start": 12.0, "time_scale": 30.0, "walk_speed_mps": 0.8, "o2_burn_pct_per_min": 0.15,
                "o2_reserve_pct": 20.0, "active": True}
SEED_PLANET = {"id": "mars", "radiation_msv_per_day": 0.67, "gravity_g": 0.38, "surface_pressure_kpa": 0.6}
SEED_LOCATIONS = (
    {"code": "HAB", "name": "Habitat", "x_m": 0.0, "z_m": 0.0},
    {"code": "ROVER", "name": "Rover", "x_m": 120.0, "z_m": 40.0},
    {"code": "SITEA", "name": "Sample Site A", "x_m": 450.0, "z_m": -200.0},
    {"code": "CRATR", "name": "Crater Rim", "x_m": 900.0, "z_m": 350.0},
    {"code": "SLOPE", "name": "Unstable Slope", "x_m": 600.0, "z_m": 500.0},
    {"code": "RIDGE", "name": "Ridge Overlook", "x_m": 1500.0, "z_m": -100.0},
)
SUIT_PRESSURE_NOMINAL_KPA, SUIT_PRESSURE_O2_LEAK_KPA, SUIT_PRESSURE_TAU_S = 29.6, 24.0, 300.0  # simulator.py step()
RADIATION_EVENT_DOSE_FACTOR = 40.0
FIXTURE_TICK_SIM_S = 30.0  # time_scale 30 x one real second


def _utc_now():
    return datetime.now(timezone.utc)


class FixtureDatasource:
    """In-memory mission, planet, telemetry, position and map rows seeded from schema.sql; no Postgres, no network.

    Starts as the simulator would one tick after launch (two rows, sim_seconds 0 and 30). `advance(sim_seconds)` appends one
    row using the same O2 burn (simulator_rules), dose and suit-pressure drift rules as simulator.step; other vitals hold.
    `clock` returns an aware datetime; rows are stamped with it, so a clock that stops advancing makes the data stale."""

    def __init__(self, scenario="normal", activity="walking", clock=_utc_now, policy=None):
        self.clock = clock
        self.policy = load_data_source_policy() if policy is None else policy
        self.mission = {**SEED_MISSION, "scenario": scenario, "activity": activity}
        self.planet = dict(SEED_PLANET)
        self.locations = {row["code"]: dict(row) for row in SEED_LOCATIONS}
        self.positions = [{"ts": clock(), "x_m": 0.0, "z_m": 0.0, "heading_deg": 0.0}]
        self.messages = []
        self.telemetry = [{"ts": clock(), "sim_seconds": 0.0, "o2_percent": 100.0, "heart_rate": 75.0,
                           "suit_pressure_kpa": SUIT_PRESSURE_NOMINAL_KPA, "suit_temp_c": 21.0, "radiation_msv": 0.0}]
        self.advance(FIXTURE_TICK_SIM_S)

    def advance(self, sim_seconds):
        """Append one telemetry row `sim_seconds` after the latest, stamped with the clock now."""
        last = self.telemetry[-1]
        o2 = min(100.0, max(0.0, last["o2_percent"] - simulator_rules.o2_used_pct(self.mission, sim_seconds)))
        factor = RADIATION_EVENT_DOSE_FACTOR if self.mission["scenario"] == "radiation_event" else 1.0
        dose = last["radiation_msv"] + self.planet["radiation_msv_per_day"] / 86400.0 * sim_seconds * factor
        target = SUIT_PRESSURE_O2_LEAK_KPA if self.mission["scenario"] == "o2_leak" else SUIT_PRESSURE_NOMINAL_KPA
        pressure = target + (last["suit_pressure_kpa"] - target) * math.exp(-sim_seconds / SUIT_PRESSURE_TAU_S)
        self.telemetry.append({**last, "ts": self.clock(), "sim_seconds": last["sim_seconds"] + sim_seconds,
                               "o2_percent": o2, "radiation_msv": dose, "suit_pressure_kpa": pressure})

    def set_position(self, x_m, z_m):
        self.positions.append({"ts": self.clock(), "x_m": float(x_m), "z_m": float(z_m), "heading_deg": 0.0})

    def snapshot(self):
        return snapshot_from_rows(self.mission, list(reversed(self.telemetry[-2:])), self.clock(), self.policy["max_age_s"])

    def place(self, name):
        code = resolve_place_code(name, self.policy["place_aliases"])
        location = self.locations.get(code)
        if location is None:
            return None
        return {"code": code, "name": location["name"], "distance_m": distance_m(self.positions[-1], location)}

    def queue_message(self, recipient, urgency, message):
        body, real_delay_s, demo_delay_s = message_row(self.policy, recipient, urgency, message)
        self.messages.append({"direction": "outbound", "body": body, "real_delay_s": real_delay_s,
                              "demo_delay_s": demo_delay_s, "sent_at": self.clock()})
        return True


def _default_connect(database_url):
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ImportError("PostgresDatasource needs psycopg (pip install 'psycopg[binary]'); it is not a router dependency. "
                          "Use FixtureDatasource for offline runs.") from exc
    return psycopg.connect(database_url, autocommit=True, row_factory=dict_row, connect_timeout=5)


class PostgresDatasource:
    """Reads the live staging DB (DATABASE_URL, default "dbname=nova"). psycopg is imported on first use only.

    `connect(database_url)` is injectable (tests pass a fake). Any DB error while reading raises DataUnavailableError;
    a failed message write returns False."""

    def __init__(self, database_url=None, connect=None, policy=None):
        self.database_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        self._connect = connect or _default_connect
        self.policy = load_data_source_policy() if policy is None else policy

    def _query(self, sql, params=(), many=False):
        try:
            with self._connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall() if many else cur.fetchone()
        except ImportError:
            raise
        except Exception as exc:
            raise DataUnavailableError(f"database read failed: {type(exc).__name__}: {exc}"[:300]) from exc

    def snapshot(self):
        # ONE statement: mission settings and telemetry come from the same DB snapshot, so a mission/scenario change
        # between reads cannot pair normal-scenario burn with leak telemetry.
        rows = self._query(SQL_RECENT_TELEMETRY, many=True)
        mission = None
        if rows:
            if len({r["mission_id"] for r in rows}) != 1:
                raise DataUnavailableError("telemetry rows span more than one mission")
            mission = {**{k: rows[0][k] for k in MISSION_FIELDS if k != "id"}, "id": rows[0]["mission_id"]}
        now = rows[0]["db_now"] if rows else None
        if rows and not isinstance(now, datetime):
            raise DataUnavailableError("database did not return its clock")
        return snapshot_from_rows(mission, rows, now, self.policy["max_age_s"])

    def place(self, name):
        code = resolve_place_code(name, self.policy["place_aliases"])
        if code is None:
            return None
        location = self._query(SQL_LOCATION, (code,))
        if location is None:
            return None
        position = self._query(SQL_LATEST_POSITION)
        if position is None:
            raise DataUnavailableError("no astronaut position for the active mission")
        return {"code": location["code"], "name": location["name"], "distance_m": distance_m(position, location)}

    def queue_message(self, recipient, urgency, message):
        try:
            body, real_delay_s, demo_delay_s = message_row(self.policy, recipient, urgency, message)
            with self._connect(self.database_url) as conn, conn.cursor() as cur:
                cur.execute(SQL_INSERT_MESSAGE, (body, real_delay_s, demo_delay_s))
                return cur.rowcount == 1
        except ImportError:
            raise
        except Exception:
            return False
