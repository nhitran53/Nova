"""Simulate suit telemetry for the NOVA demo.

Run python simulator.py to resume the active mission.
Use --ticks N for a bounded run or --new-mission to start another mission.
Only one simulator can write at a time. Oxygen demo controls share its write lock.
"""

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from simulator_rules import ACTIVITY_O2, SCENARIO_O2, o2_used_pct
from router.db_config import password_options

load_dotenv(Path(__file__).resolve().parent / ".env")
DATABASE_URL = os.getenv("DATABASE_URL", "dbname=nova")
TICK_SECONDS = 1.0
MAX_GAP_S = 10.0              # longer gap between ticks = pause (sleep/outage)
MAX_CONSECUTIVE_ERRORS = 10   # then exit with a clear error
LOCK_KEY = 4242_0001  # advisory lock id: "the NOVA simulator"
CONTROL_LOCK_KEY = LOCK_KEY + 1
MARS_SOL_HOURS = 24.66

ACTIVITY_HEART = {"resting": 70, "walking": 95, "working": 110, "running": 145}
ACTIVITY_CO2 = {"resting": 2.0, "walking": 3.0, "working": 4.0, "running": 5.5}   # mmHg in suit
ACTIVITY_SUIT_TEMP = {"resting": 21.0, "walking": 22.0, "working": 23.0, "running": 25.0}
BATTERY_DRAIN = {"resting": 0.03, "walking": 0.04, "working": 0.05, "running": 0.06}  # %/sim min

SUIT_PRESSURE_NOMINAL = 29.6      # kPa (~4.3 psi)
SUIT_PRESSURE_LEAK = 24.0         # where pressure drifts during o2_leak
RADIATION_EVENT_MULT = 40.0       # dose-rate multiplier during a solar particle event

@dataclass
class Reading:
    sim_seconds: float
    o2_percent: float
    co2_mmhg: float
    heart_rate: float
    spo2: float
    suit_pressure_kpa: float
    battery_percent: float
    suit_temp_c: float
    radiation_msv: float
    ext_temp_c: float
    wind_mps: float
    visibility_km: float
    weather: str
    local_time_hours: float


def starting_reading(mission) -> Reading:
    return Reading(
        sim_seconds=0.0, o2_percent=100.0, co2_mmhg=2.5, heart_rate=75.0, spo2=98.0,
        suit_pressure_kpa=SUIT_PRESSURE_NOMINAL, battery_percent=100.0, suit_temp_c=21.0,
        radiation_msv=0.0, ext_temp_c=-40.0, wind_mps=5.0, visibility_km=30.0,
        weather="clear", local_time_hours=float(mission["mission_day_start"]),
    )


def drift(current: float, target: float, sim_dt: float, tau_s: float) -> float:
    """Move smoothly toward target. tau_s = simulated seconds to get ~63% of the way."""
    return current + (target - current) * (1 - math.exp(-sim_dt / tau_s))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def outside_temp(local_hour: float, planet) -> float:
    """Daily cycle: coldest before dawn, warmest early afternoon.
    Typical equatorial range, not the planet's record extremes."""
    high = float(planet["temp_max_c"]) - 15        # ~5 C
    low = max(float(planet["temp_min_c"]), -85.0)   # ~-85 C
    mid, amp = (high + low) / 2, (high - low) / 2
    return mid + amp * math.cos(2 * math.pi * (local_hour - 14.0) / MARS_SOL_HOURS)


def step(prev: Reading, mission, planet, sim_dt: float) -> Reading:
    activity, scenario = mission["activity"], mission["scenario"]
    minutes = sim_dt / 60.0
    r = Reading(**asdict(prev))

    r.sim_seconds = prev.sim_seconds + sim_dt
    r.local_time_hours = (prev.local_time_hours + sim_dt / 3600.0) % MARS_SOL_HOURS

    # Oxygen tank: shared rule (same function the route check uses)
    r.o2_percent = clamp(prev.o2_percent - o2_used_pct(mission, sim_dt), 0, 100)

    # Suit pressure: scripted drop during the leak scenario
    p_target = SUIT_PRESSURE_LEAK if scenario == "o2_leak" else SUIT_PRESSURE_NOMINAL
    r.suit_pressure_kpa = drift(prev.suit_pressure_kpa, p_target, sim_dt, 300)

    # Heart rate follows activity (+ scripted stress response when tank is low)
    hr_target = ACTIVITY_HEART[activity] + (20 if r.o2_percent < 15 else 0)
    r.heart_rate = clamp(drift(prev.heart_rate, hr_target, sim_dt, 60) + random.uniform(-1, 1), 40, 220)

    # SpO2: scripted demo drop when the tank is nearly empty or suit pressure is low
    spo2_target = 98.0
    if r.o2_percent < 10:
        spo2_target -= (10 - r.o2_percent) * 1.2
    if r.suit_pressure_kpa < 26:
        spo2_target -= (26 - r.suit_pressure_kpa) * 1.5
    r.spo2 = clamp(drift(prev.spo2, spo2_target, sim_dt, 90), 60, 100)

    # CO2 in suit follows effort only (scrubber is independent of the O2 tank)
    r.co2_mmhg = clamp(drift(prev.co2_mmhg, ACTIVITY_CO2[activity], sim_dt, 120)
                       + random.uniform(-0.05, 0.05), 0, 60)

    r.suit_temp_c = drift(prev.suit_temp_c, ACTIVITY_SUIT_TEMP[activity], sim_dt, 300)
    r.battery_percent = clamp(prev.battery_percent - BATTERY_DRAIN[activity] * minutes, 0, 100)

    # Radiation dose accumulates
    dose_rate = float(planet["radiation_msv_per_day"]) / 86400.0
    if scenario == "radiation_event":
        dose_rate *= RADIATION_EVENT_MULT
    r.radiation_msv = prev.radiation_msv + dose_rate * sim_dt

    # Weather
    if scenario == "dust_storm":
        r.weather, wind_t, vis_t = "dust_storm", 25.0, 0.5
    elif scenario == "radiation_event":
        r.weather, wind_t, vis_t = "solar_particle_event", 6.0, 30.0
    else:
        r.weather = "clear" if prev.visibility_km > 10 else "dusty"
        wind_t, vis_t = 5.0, 30.0
    r.wind_mps = clamp(drift(prev.wind_mps, wind_t, sim_dt, 180) + random.uniform(-0.3, 0.3), 0, 60)
    r.visibility_km = clamp(drift(prev.visibility_km, vis_t, sim_dt, 240), 0.05, 50)
    r.ext_temp_c = drift(prev.ext_temp_c, outside_temp(r.local_time_hours, planet), sim_dt, 600)
    return r



# Alerts
@dataclass
class Rule:
    key: str
    channel: str
    field: str
    direction: str    # "low" or "high"
    caution: float
    critical: float
    margin: float     # must recover this far past a line to step down (no flicker)
    label: str
    unit: str


RULES = [
    Rule("low_oxygen",    "o2",     "o2_percent",        "low",  30, 15, 2,    "Oxygen tank",   "%"),
    Rule("high_co2",      "co2",    "co2_mmhg",          "high", 7.6, 15, 0.6, "Suit CO2",      "mmHg"),
    Rule("heart_rate",    "vitals", "heart_rate",        "high", 150, 175, 5,  "Heart rate",    "bpm"),
    Rule("low_spo2",      "vitals", "spo2",              "low",  92, 88, 1,    "Blood oxygen",  "%"),
    Rule("suit_pressure", "suit",   "suit_pressure_kpa", "low",  27, 25, 0.5,  "Suit pressure", "kPa"),
    Rule("low_battery",   "power",  "battery_percent",   "low",  25, 10, 2,    "Suit battery",  "%"),
]

LEVEL_RANK = {None: 0, "info": 0, "caution": 1, "critical": 2}
SPOKEN_UNIT = {"%": "percent", "kPa": "kilopascals", "mmHg": "millimeters of mercury",
               "bpm": "beats per minute"}


def level_for(rule: Rule, value: float, open_level: Optional[str]) -> Optional[str]:
    low = rule.direction == "low"

    def past(line: float, margin: float = 0.0) -> bool:
        return value <= line - margin if low else value >= line + margin

    raw = "critical" if past(rule.critical) else "caution" if past(rule.caution) else None
    # Escalate immediately; step down only once safely past the line + margin
    if LEVEL_RANK[open_level] > LEVEL_RANK[raw]:
        line = rule.critical if open_level == "critical" else rule.caution
        if past(line, -rule.margin):
            return open_level
    return raw


def voice_for(rule: Rule, level: str, value: float) -> str:
    v = f"{value:.0f}" if abs(value) >= 10 else f"{value:.1f}"
    unit = SPOKEN_UNIT.get(rule.unit, rule.unit)
    if rule.key == "low_oxygen":
        if level == "critical":
            return f"Warning. Oxygen at {v} percent. Return to the habitat or rover now."
        return f"Caution. Oxygen is at {v} percent. Plan your return."
    word = "Warning" if level == "critical" else "Caution"
    return f"{word}. {rule.label} is at {v} {unit}."


# One open row per (mission, alert_key). Same row for its whole life:
# level changes are UPDATEs, so recovery (critical -> caution) never creates
# a new id. Escalation clears acknowledged_at so the popup comes back.
UPSERT_ALERT = """
INSERT INTO alerts (mission_id, channel, level, message, voice_text, alert_key)
VALUES (%(mission_id)s, %(channel)s, %(level)s, %(message)s, %(voice)s, %(key)s)
ON CONFLICT (mission_id, alert_key) WHERE resolved_at IS NULL AND alert_key IS NOT NULL
DO UPDATE SET
    level      = EXCLUDED.level,
    channel    = EXCLUDED.channel,
    message    = EXCLUDED.message,
    voice_text = EXCLUDED.voice_text,
    acknowledged_at = CASE
        WHEN (CASE EXCLUDED.level WHEN 'critical' THEN 2 WHEN 'caution' THEN 1 ELSE 0 END)
           > (CASE alerts.level   WHEN 'critical' THEN 2 WHEN 'caution' THEN 1 ELSE 0 END)
        THEN NULL ELSE alerts.acknowledged_at END
"""

RESOLVE_ALERT = """
UPDATE alerts SET resolved_at = now()
WHERE mission_id = %s AND alert_key = %s AND resolved_at IS NULL
"""


def set_alert(cur, mission_id, key, channel, level, message, voice):
    if level is None:
        cur.execute(RESOLVE_ALERT, (mission_id, key))
    else:
        cur.execute(UPSERT_ALERT, dict(mission_id=mission_id, key=key, channel=channel,
                                       level=level, message=message, voice=voice))


def check_alerts(cur, mission, r: Reading):
    mid = mission["id"]
    cur.execute("SELECT alert_key, level FROM alerts WHERE mission_id=%s AND resolved_at IS NULL", (mid,))
    open_levels = {row["alert_key"]: row["level"] for row in cur.fetchall()}

    for rule in RULES:
        value = getattr(r, rule.field)
        level = level_for(rule, value, open_levels.get(rule.key))
        msg = f"{level.upper()}: {rule.label} {value:.1f} {rule.unit}" if level else ""
        set_alert(cur, mid, rule.key, rule.channel, level, msg,
                  voice_for(rule, level, value) if level else "")

    storm = r.weather == "dust_storm"
    set_alert(cur, mid, "dust_storm", "weather", "caution" if storm else None,
              f"CAUTION: Dust storm, visibility {r.visibility_km:.1f} km",
              "Caution. Dust storm in your area. Visibility is dropping. Stay close to the habitat.")
    sep = r.weather == "solar_particle_event"
    set_alert(cur, mid, "solar_event", "radiation", "critical" if sep else None,
              f"CRITICAL: Solar particle event, dose {r.radiation_msv:.2f} mSv",
              "Warning. Solar particle event detected. Take shelter in the habitat.")



# Database


def load_active_mission(cur):
    cur.execute("""SELECT m.*, p.temp_min_c, p.temp_max_c, p.radiation_msv_per_day
                   FROM missions m JOIN planets p ON p.id = m.planet_id
                   WHERE m.active""")
    return cur.fetchone()


def load_last(cur, mission_id):
    cur.execute("SELECT * FROM telemetry WHERE mission_id=%s ORDER BY ts DESC, id DESC LIMIT 1",
                (mission_id,))
    return cur.fetchone()


def insert_reading(cur, mission_id: int, r: Reading):
    d = asdict(r)
    cols = ", ".join(d)
    vals = ", ".join(f"%({k})s" for k in d)
    cur.execute(f"INSERT INTO telemetry (mission_id, {cols}) VALUES (%(mission_id)s, {vals})",
                {"mission_id": mission_id, **d})


def deliver_messages(cur, mission_id: int):
    cur.execute("""UPDATE messages SET delivered_at = now()
                   WHERE mission_id=%s AND delivered_at IS NULL
                     AND sent_at + make_interval(secs => demo_delay_s) <= now()""", (mission_id,))


def tick(conn, real_dt: float) -> Optional[Reading]:
    """One transaction. real_dt = real seconds to simulate (0 = resume, no time passes)."""
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (CONTROL_LOCK_KEY,))
        mission = load_active_mission(cur)
        if not mission:
            print("No active mission. Waiting...")
            return None

        last = load_last(cur, mission["id"])
        if last is None:
            r = starting_reading(mission)
            insert_reading(cur, mission["id"], r)
        else:
            prev = Reading(**{k: last[k] for k in Reading.__dataclass_fields__})
            sim_dt = real_dt * float(mission["time_scale"])
            if sim_dt > 0:
                r = step(prev, mission, planet=mission, sim_dt=sim_dt)  # row also has planet cols
                insert_reading(cur, mission["id"], r)
            else:
                r = prev   # resume tick: nothing changed, no duplicate row

        check_alerts(cur, mission, r)
        deliver_messages(cur, mission["id"])
        return r


def set_demo_oxygen(conn, value: float, relative: bool = False) -> float:
    if not math.isfinite(value) or (not relative and not 0 <= value <= 100):
        raise ValueError("Use an oxygen percentage from 0 to 100.")
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (CONTROL_LOCK_KEY,))
        cur.execute("""SELECT t.id, t.o2_percent FROM telemetry t JOIN missions m ON m.id=t.mission_id
                       WHERE m.active ORDER BY t.ts DESC, t.id DESC LIMIT 1 FOR UPDATE OF t""")
        latest = cur.fetchone()
        if latest is None:
            raise ValueError("Start simulator.py first.")
        oxygen = clamp(latest["o2_percent"] + value if relative else value, 0, 100)
        cur.execute("UPDATE telemetry SET o2_percent=%s WHERE id=%s", (oxygen, latest["id"]))
        return oxygen


def apply_settings(conn, args):
    """CLI changes, all in one transaction."""
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        if args.new_mission:
            cur.execute("SELECT * FROM missions WHERE active FOR UPDATE")
            old = cur.fetchone()
            if old is None:
                sys.exit("No active mission to copy settings from.")
            cur.execute("UPDATE missions SET active = FALSE WHERE id = %s", (old["id"],))
            cur.execute("""
                INSERT INTO missions (name, astronaut, planet_id, scenario, activity,
                    mission_day_start, time_scale, walk_speed_mps, o2_burn_pct_per_min, o2_reserve_pct)
                VALUES (%(name)s, %(astronaut)s, %(planet_id)s, 'normal', 'walking',
                    %(mission_day_start)s, %(time_scale)s, %(walk_speed_mps)s,
                    %(o2_burn_pct_per_min)s, %(o2_reserve_pct)s)
                RETURNING id""", old)
            new_id = cur.fetchone()["id"]
            cur.execute("""INSERT INTO astronaut_positions (mission_id, x_m, z_m, heading_deg)
                           SELECT %s, x_m, z_m, 0 FROM map_locations WHERE code = 'HAB'""", (new_id,))
            print(f"Started mission {new_id} (mission {old['id']} kept as history).")
        for col, val in (("scenario", args.scenario), ("activity", args.activity),
                         ("time_scale", args.time_scale)):
            if val is not None:
                cur.execute(f"UPDATE missions SET {col} = %s WHERE active", (val,))


def connect_and_lock():
    """Connect (retrying while the DB is unreachable) and take the single-simulator lock."""
    delay = 1.0
    while True:
        try:
            conn = psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=5,
                                   **password_options(DATABASE_URL))
            break
        except psycopg.OperationalError as e:
            message = str(e).lower()
            reason = "connection failed"
            for detail in ("no password supplied", "password authentication failed", "connection refused", "timeout expired"):
                if detail in message:
                    reason = detail
                    break
            print(f"Cannot reach database ({reason}). Retrying in {delay:.0f}s...")
            time.sleep(delay)
            delay = min(delay * 2, 15)
    got = conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,)).fetchone()[0]
    if not got:
        conn.close()
        sys.exit("Another NOVA simulator is already running. Stop it first (only one may write telemetry).")
    return conn


def main():
    ap = argparse.ArgumentParser(description="NOVA telemetry simulator")
    ap.add_argument("--scenario", choices=list(SCENARIO_O2))
    ap.add_argument("--activity", choices=list(ACTIVITY_O2))
    ap.add_argument("--time-scale", type=float)
    ap.add_argument("--new-mission", action="store_true",
                    help="end the active mission and start a fresh one (full tank, at habitat)")
    ap.add_argument("--ticks", type=int, default=0, help="stop after N ticks (0 = forever)")
    args = ap.parse_args()
    # Postgres accepts NaN/Infinity in REAL columns and NaN passes "> 0" checks,
    # so validate here before anything reaches the database.
    if args.time_scale is not None:
        if not math.isfinite(args.time_scale) or args.time_scale <= 0:
            ap.error("--time-scale must be a finite number greater than zero.")
    if args.ticks < 0:
        ap.error("--ticks must be zero or greater.")

    conn = connect_and_lock()
    apply_settings(conn, args)

    baseline: Optional[float] = None   # None = next tick is a resume tick
    errors, n = 0, 0
    print("NOVA simulator running. Ctrl+C to stop.")
    try:
        while args.ticks == 0 or n < args.ticks:
            started = time.monotonic()

            if baseline is None:
                real_dt = 0.0
            else:
                real_dt = started - baseline
                if real_dt > MAX_GAP_S:
                    print(f"Gap of {real_dt:.0f}s (sleep or outage). Treating it as a pause.")
                    real_dt = 0.0

            try:
                r = tick(conn, real_dt)
                baseline = started          # advance only after a committed tick
                errors = 0
                if r:
                    print(f"t={r.sim_seconds/60:6.1f} min  O2={r.o2_percent:6.2f}%  "
                          f"CO2={r.co2_mmhg:4.1f}  HR={r.heart_rate:5.1f}  "
                          f"P={r.suit_pressure_kpa:4.1f}kPa  bat={r.battery_percent:5.1f}%  {r.weather}")
            except psycopg.Error as e:
                errors += 1
                if conn.closed or conn.broken:
                    print("Lost database connection. Reconnecting...")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = connect_and_lock()
                    # The lost transaction may or may not have committed.
                    # Reload the latest saved reading and treat this as a pause,
                    # so the interval can't be counted twice.
                    baseline = None
                else:
                    print(f"Tick failed (rolled back, time carried to next tick): {e}")
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    sys.exit(f"{errors} failed ticks in a row. Last error: {e}")

            n += 1
            time.sleep(max(0.0, TICK_SECONDS - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("\nStopped. Run again to resume from the last reading.")
    finally:
        conn.close()   # releases the lock


if __name__ == "__main__":
    main()
