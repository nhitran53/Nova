# Staging DB/pseudodata -> router mapping (2026-10-24)

Source: scout recon of origin/staging @ 0fa3dfe. Read-only. Postgres schema.sql + simulator.py; router currently decoupled from DB.

# Code Context: NOVA DB / simulator / router mapping (branch `staging`, read-only)

## Files Retrieved
1. `schema.sql` (1-490) - all tables, views, indexes, seed data.
2. `simulator.py` (1-422) - telemetry generator, alert engine, DB writer, CLI.
3. `simulator_rules.py` (1-92) - shared O2 burn formula (`o2_burn_pct_per_min`, `o2_used_pct`, `estimate_route_o2`).
4. `apistage1.py` (1-39) - FastAPI `/health` only; connects to DB, reads active mission.
5. `router/tools.py` (1-131) - fake tool registry, `SimState` fabricated values, 6 tools.
6. `router/turn.py` (1-346) - turn pipeline: alert poll -> router -> confirmation -> slot check -> tool -> template -> verifier.
7. `router/turn_policy.json` (1-174) - `sim_fields` thresholds, `templates`, `feasibility`, `send_message`. All sections PLACEHOLDER.
8. Not read in depth: `router/pipeline.py`, `router/verifier.py` (referenced by turn.py).

## Key Code: Database objects (schema.sql)

Connection: `DATABASE_URL` env, default `dbname=nova` (simulator.py:28, apistage1.py:11). psycopg3, autocommit. No ORM.

| Table | Key columns (type, constraint) | Notes |
|---|---|---|
| planets | id TEXT PK; display_name TEXT; gravity_g REAL>=0; surface_pressure_kpa REAL>=0; temp_min_c, temp_max_c REAL (min<=max); day_length_hours REAL>0; radiation_msv_per_day REAL>=0; orbit_au REAL>=0; weather_events TEXT[] | Seed row 'mars': gravity 0.38, 0.6 kPa, -125..20 C, 24.66 h, 0.67 mSv/day, 1.524 AU. |
| missions | id SERIAL PK; name TEXT='NOVA-1'; astronaut TEXT='Commander'; planet_id FK planets(id); scenario ENUM-check {normal,o2_leak,dust_storm,radiation_event}; activity ENUM-check {resting,walking,working,running} default walking; mission_day_start REAL [0,24.66) default 12 (Earth hours into sol); time_scale REAL>0 default 30 (sim s per real s); walk_speed_mps REAL>0 default 0.8; o2_burn_pct_per_min REAL>0 default 0.15 (percentage POINTS/sim min, base); o2_reserve_pct REAL [0,100] default 20; started_at TIMESTAMPTZ; active BOOLEAN default TRUE | Unique partial index `one_active_mission` on (active) WHERE active. |
| telemetry | id BIGSERIAL PK; mission_id FK missions ON DELETE CASCADE; ts TIMESTAMPTZ; sim_seconds DOUBLE>=0; o2_percent REAL [0,100]; co2_mmhg REAL>=0; heart_rate REAL [0,250]; spo2 REAL [0,100]; suit_pressure_kpa REAL>=0; battery_percent REAL [0,100]; suit_temp_c REAL; radiation_msv REAL>=0 (CUMULATIVE dose); ext_temp_c REAL; wind_mps REAL>=0; visibility_km REAL>=0; weather ENUM {clear,dusty,dust_devil,dust_storm,solar_particle_event}; local_time_hours REAL [0,24.66) | Index (mission_id, ts DESC, id DESC). |
| map_locations | id SERIAL PK; code TEXT UNIQUE; name TEXT; location_type ENUM {habitat,rover,sample_site,landmark,crater,hazard,waypoint}; x_m, z_m DOUBLE (meters, no bounds); radius_m REAL>0 default 5; hazard_level ENUM {safe,caution,danger} default safe; o2_refill BOOL default F; source ENUM {seed,vision,astronaut} default seed; description TEXT | Seeded 6 rows (see below). |
| astronaut_positions | id BIGSERIAL PK; mission_id FK CASCADE; ts; x_m, z_m DOUBLE; heading_deg REAL [0,360) default 0 | Index (mission_id, ts DESC, id DESC). Seeded one row at HAB (0,0) per active mission. |
| navigation_requests | id BIGSERIAL PK; mission_id FK; destination_id INT NOT NULL FK map_locations; return_location_id INT FK map_locations NULL; requested_at; input_method ENUM {voice,gesture,combined}; status ENUM {pending,approved,blocked,unknown,cancelled} default pending; outbound_distance_m, return_distance_m REAL>=0; time_at_destination_minutes REAL>=0 default 0; assessed_walk_speed_mps REAL>0; assessed_o2_burn_pct_per_min REAL>0; assessed_o2_reserve_pct REAL [0,100]; estimated_o2_required_pct REAL>=0 (may exceed 100); o2_available_pct REAL [0,100]; warning_message TEXT; telemetry_id FK telemetry ON DELETE SET NULL; position_id FK astronaut_positions ON DELETE SET NULL; assessed_at TIMESTAMPTZ | CHECKs: approved/blocked require assessed_at; approved requires return_location_id, all assessed fields, and required<=available; blocked requires non-empty warning_message. |
| alerts | id BIGSERIAL PK; mission_id FK; ts; channel ENUM {o2,co2,vitals,suit,power,radiation,weather,navigation,vision,comms}; level ENUM {info,caution,critical}; message TEXT NOT NULL; voice_text TEXT; acknowledged_at; alert_key TEXT; resolved_at; navigation_request_id FK SET NULL; location_id FK map_locations SET NULL | Partial unique index (mission_id, alert_key) WHERE resolved_at IS NULL AND alert_key IS NOT NULL. Checks: resolved/ack >= ts. |
| messages | id BIGSERIAL PK; mission_id FK; direction ENUM {outbound,inbound}; body TEXT non-empty; sent_at; real_delay_s, demo_delay_s DOUBLE>=0; delivered_at (>= sent_at) | View `message_status` adds demo_due_at and demo_seconds_left (countdown). Nothing sets delivered_at except simulator `deliver_messages`. |
| hud_commands | id BIGSERIAL PK; mission_id FK; ts; action ENUM {focus,clear}; metrics TEXT[] subset of {o2_percent,co2_mmhg,heart_rate,spo2,suit_pressure_kpa,battery_percent,suit_temp_c,radiation_msv,ext_temp_c,wind_mps,visibility_km,weather,local_time_hours,comm_delay,minimap}; seconds REAL>0 default 8; expires_at NULL=never; applied_at | Unity consumes unseen, unexpired, in id order. Nothing in repo writes it. |
| vision_detections | id BIGSERIAL PK; mission_id FK; ts; label TEXT; confidence REAL [0,1] NULL allowed; est_x_m, est_z_m DOUBLE (both or neither); location_id FK map_locations SET NULL; model_name; frame_ref | No writer in repo. |

Views:
- `latest_telemetry`: DISTINCT ON mission_id, newest telemetry row, only where missions.active.
- `latest_position`: same for astronaut_positions.
- `open_alerts`: alerts with resolved_at NULL on active mission (includes acknowledged).
- `message_status`: messages + demo_due_at, demo_seconds_left (0 if delivered).

Seed data (schema.sql 11): planets 'mars'; one active mission (planet mars) if none; map_locations:
- HAB Habitat habitat (0,0) r15 safe o2_refill=T
- ROVER Rover rover (120,40) r6 safe o2_refill=T
- SITEA Sample Site A sample_site (450,-200) r10 safe
- CRATR Crater Rim crater (900,350) r60 caution
- SLOPE Unstable Slope hazard (600,500) r80 danger
- RIDGE Ridge Overlook landmark (1500,-100) r20 safe
Plus astronaut_positions row at HAB for the active mission. Coordinates are marked "placeholder".

## Simulator telemetry generation (simulator.py)

Loop: `main()` (353-419). One tick per real second (TICK_SECONDS=1.0, line 29). `real_dt` = wall gap since last committed tick; gap > MAX_GAP_S=10 s is treated as pause (dt=0, no row). `sim_dt = real_dt * time_scale` (default 30 -> ~30 sim s per row). First tick after start: real_dt=0 (resume, no row unless no prior telemetry, then `starting_reading`). Each tick is one transaction: load active mission+planet (`load_active_mission`, 256), load last reading (`load_last`, 263), `step()`, `insert_reading()` (269), `check_alerts()`, `deliver_messages()`. Single-writer enforced by `pg_try_advisory_lock(4242_0001)` (LOCK_KEY, 32; connect_and_lock 335).

Starting reading (`starting_reading`, 62-68): sim_seconds 0, o2 100, co2 2.5, HR 75, SpO2 98, suit P 29.6, battery 100, suit T 21.0, radiation 0, ext T -40, wind 5, visibility 30, weather clear, local_time = mission_day_start.

Per-step (`step`, 89-140), dt in sim seconds:
- local_time_hours = (prev + dt/3600) mod 24.66.
- o2_percent = clamp(prev - o2_used_pct(mission, dt), 0, 100). Burn = o2_burn_pct_per_min * ACTIVITY_O2[activity] * SCENARIO_O2[scenario] (simulator_rules.py 16-40). ACTIVITY_O2: resting .6, walking 1.0, working 1.3, running 1.8. SCENARIO_O2: normal 1, o2_leak 6, dust_storm 1.1, radiation_event 1.0.
- suit_pressure_kpa: exponential drift (tau 300 s) toward 29.6 (SUIT_PRESSURE_NOMINAL), or 24.0 in o2_leak.
- heart_rate: drift (tau 60 s) toward ACTIVITY_HEART {resting 70, walking 95, working 110, running 145} (+20 if o2<15) plus uniform(-1,1); clamp 40..220. Unit bpm.
- spo2: drift (tau 90 s) toward 98, minus (10-o2)*1.2 if o2<10, minus (26-P)*1.5 if P<26; clamp 60..100. Unit %.
- co2_mmhg: drift (tau 120) toward ACTIVITY_CO2 {2.0,3.0,4.0,5.5} + noise ±0.05; clamp 0..60.
- suit_temp_c: drift (tau 300) toward ACTIVITY_SUIT_TEMP {21,22,23,25}.
- battery_percent: linear drain BATTERY_DRAIN {0.03,0.04,0.05,0.06} %/sim-min; clamp 0..100.
- radiation_msv (cumulative): += planet.radiation_msv_per_day/86400 * dt (x40 in radiation_event). Note: 86400 s/day, not the 88776 s sol.
- Weather: dust_storm -> weather dust_storm, wind target 25, vis target 0.5 km; radiation_event -> solar_particle_event, wind 6, vis 30; else clear if vis>10 else dusty (wind 5, vis 30). Wind clamp 0..60, vis clamp 0.05..50 km. `dust_devil` never produced.
- ext_temp_c: drift (tau 600) toward outside_temp(local_hour): cosine, high = temp_max_c-15 (=5 C), low = max(temp_min_c, -85) (=-85), peak 14:00 local.

Alert rules (`RULES`, simulator.py 158-165): Rule(key, channel, field, direction, caution, critical, margin, label, unit):
- low_oxygen | o2 | o2_percent | low | 30 | 15 | 2 | % 
- high_co2 | co2 | co2_mmhg | high | 7.6 | 15 | 0.6 | mmHg
- heart_rate | vitals | heart_rate | high | 150 | 175 | 5 | bpm
- low_spo2 | vitals | spo2 | low | 92 | 88 | 1 | %
- suit_pressure | suit | suit_pressure_kpa | low | 27 | 25 | 0.5 | kPa
- low_battery | power | battery_percent | low | 25 | 10 | 2 | %
Hysteresis: `level_for` (170-184) escalates immediately, steps down only after recovering past line +/- margin. Plus weather alerts: `dust_storm` (channel weather, caution), `solar_event` (channel radiation, critical) (check_alerts 230-249). Writes use `UPSERT_ALERT` (201-214) keyed on (mission_id, alert_key), `RESOLVE_ALERT` when level None.

Scenario effects summary: o2_leak -> burn x6 and suit P target 24 (triggers low_oxygen/suit_pressure over time); radiation_event -> dose rate x40 (~1.1 mSv/h) and solar_event alert; dust_storm -> vis/wind change, weather alert.

CLI (simulator.py 353-391): `--scenario`, `--activity`, `--time-scale`, `--new-mission` (deactivates old mission, copies settings, new 'walking' + 'normal', seeds astronaut_positions at HAB), `--ticks N`. Also prints per-tick status line.

Messages: `deliver_messages` (277-280) sets delivered_at when sent_at + demo_delay_s <= now(). Simulator never inserts messages.

Navigation / map_locations / astronaut_positions / hud_commands / vision_detections / messages / alerts semantics:
- map_locations: static coordinate set; no writer except seed. Used for route destination, return location, alert location, vision location.
- astronaut_positions: NOT written by simulator.py (only the seed row and `--new-mission`). The simulator comment says it "does not move the astronaut"; Unity/API must write positions. Heading degrees.
- navigation_requests: designed as a decision log; no writer in repo. Schema notes say backend computes `trip_minutes = (out+ret)/walk_speed/60 + time_at_destination`, `estimated_o2 = trip*burn + reserve` (includes reserve). Implemented by `simulator_rules.estimate_route_o2` (rules 50-93) with travel burn at walking; returns `required_pct`. Not yet called by apistage1.py.
- alerts: written only by simulator (`check_alerts`). Unique open row per alert_key; ack cleared on escalation. Router does NOT read this table; it computes its own alerts from a snapshot (turn.py evaluate_alerts).
- messages: outbound/inbound with real_delay_s and demo_delay_s; countdown view; delivery by simulator tick.
- hud_commands: Unity-consumed visor commands; no writer in repo.
- vision_detections: ML dump; no writer in repo.

apistage1.py: FastAPI app "NOVA API — Stage 1". Only `GET /health` (13-40): `psycopg.connect(DATABASE_URL, connect_timeout=5)`, returns `{status, database, database_connected, active_mission_id, active_mission_name}`; 503 on psycopg.Error. No writes. Does not expose telemetry.

## Architecture

- Postgres `nova` DB is the single source. Simulator (sole writer of telemetry & rule alerts & message delivery) runs as a process with an advisory lock; it does not need the API. apistage1 needs Postgres for /health only.
- Requires live Postgres: YES, for simulator (retries forever with backoff 1->15 s, lines 335-350) and for apistage1 `/health` (503 if down). Apply schema.sql first (idempotent: CREATE IF NOT EXISTS, ON CONFLICT DO NOTHING, but note it is wrapped in BEGIN/COMMIT and the "missions" seed inserts only when no active mission). Python deps: psycopg[binary], fastapi (for apistage1). simulator_rules imports only math.
- Sample/seed data: only the schema.sql seed (planets, 1 mission, 6 map_locations, 1 position). No telemetry fixtures in repo (per git ls-files filter: no other .sql or seed files).
- Router (`router/`) is fully decoupled from DB: `TurnHandler(router, alert_source, registry)`; `alert_source()` must return a dict of snapshot keys (o2_pct, minutes_remaining, bpm, radiation_usv_h, temp_c, suit_pressure_kpa). Nothing in router reads Postgres yet.

Router tool registry (`router/tools.py`):
- MAX_TOOLS = 8; one tool call per turn (`begin_turn`, `call` raises OneCallPerTurnError on 2nd). Schema checked: `{"required": [...], "properties": {arg: type}}`.
- `SimState` (79-88) default values: o2_pct 87.5, minutes_remaining 212, bpm 72, radiation_usv_h 0.34, temp_c 21.5, suit_pressure_kpa 29.6.
- `PLACE_DISTANCES_M` (12-18): 21 fabricated names, e.g. homebase (120 m, 2 min), habitat (150, 3), airlock (80, 1), rover (200, 4), lander (900, 15), crater (600, 10), ridge (500, 9), ...

| Tool | Args schema | Result `data` dict shape | Sim data feeding it |
|---|---|---|---|
| safety | none (`{"required": [], "properties": {}}`) | `{"status_word": "go"|"no go"}`; go iff o2_pct>=16.0 and 20<=suit_pressure_kpa<=40 | o2_pct, suit_pressure_kpa |
| navigation | `place: str` (required) | ok: `{"place", "place_name", "distance_m": int, "eta_min": int}`; unknown place -> ok False, `{"error": "unknown place"}` | PLACE_DISTANCES_M only (no sim). Then feasibility adds `data["feasibility"]` (turn.py 316-332) using minutes_remaining (or o2_pct*2.4) |
| map | `place: str` (required) | same shape as navigation | PLACE_DISTANCES_M only |
| data | none | `{"bpm", "radiation_usv_h", "temp_c"}` | bpm, radiation_usv_h, temp_c |
| o2 | none | `{"o2_pct", "minutes_remaining"}` | o2_pct, minutes_remaining |
| send_message | `recipient: str, urgency: str, message: str` (all required) | `{"recipient", "recipient_name", "urgency", "status": "queued"}` (FAKE, nothing sent; no DB write) | none (urgency from policy `send_message.urgency_by_outcome`) |

Templates (turn_policy.json 59-70) format these dicts: safety {status_word}; navigation {place_name, distance_m:.0f, eta_min:.0f}; map {place_name}; data {bpm:.0f, radiation_usv_h:.2f, temp_c:.1f}; o2 {o2_pct:.1f, minutes_remaining:.0f}. Verifier checks every spoken number against source.

Turn order (turn.py `_run_turn` 172-200): (1) alert_source() snapshot -> evaluate_alerts (92? line 77-89) vs `sim_fields`; any `preempts_router` (critical) tier speaks alert_lines and returns. (2) pending refill confirmation (yes/no). (3) Pipeline route. (4) tool -> `_tool_turn` (292-314) with slot check, one `_call_tool`, template, verifier. Navigation additionally runs `assess_feasibility` (92-122).

turn_policy.json `sim_fields` (critical/warning thresholds; field names must match snapshot keys):
- o2_pct: critical_below 16.0, warning_below 30.0
- suit_pressure_kpa: critical_below 20.0, critical_above 40.0
- temp_c: critical_below -10.0, critical_above 40.0, warning_above 32.0
- radiation_usv_h: critical_above 50.0, warning_above 10.0
- bpm: critical_below 35, critical_above 180, warning_above 150
(no thresholds for minutes_remaining, suit_temp_c, etc.)

`feasibility` (placeholder): walking_speed_m_per_min 60; reserve_margin_min 10; o2_minutes_per_pct_fallback 2.4; home_places ["homebase"]. Outcome: cannot_reach | cannot_return | tight | comfortable, all whole minutes (ceil legs, floor remaining).

## Proposed Mapping: sim/DB field -> router snapshot/tool result

Source: latest row from `latest_telemetry` for active mission (plus `missions`, `planets`, `latest_position`).

| Router field (tool result) | Source | Conversion | Notes/gaps |
|---|---|---|---|
| o2_pct (o2) | telemetry.o2_percent | direct, % | Direct. Range 0..100 both sides. |
| minutes_remaining (o2) | derived; NOT stored | `floor(o2_percent / burn)` with burn = `missions.o2_burn_pct_per_min * ACTIVITY_O2[activity] * SCENARIO_O2[scenario]` (simulator_rules.o2_burn_pct_per_min). Option: `floor((o2_percent - o2_reserve_pct)/burn)` if reserve should be excluded | GAP. Feasibility also adds reserve_margin_min (10) so pick one definition to avoid double counting; DB o2_reserve_pct is 20 points (not minutes). Current fabricated 212 is inconsistent with DB default (100% -> ~533 min at walking, 0.15%/min). |
| bpm (data) | telemetry.heart_rate | direct, bpm | Direct. Policy critical_above 180 / warning 150 vs sim alert 175/150; separate threshold sets. |
| radiation_usv_h (data, snapshot) | telemetry.radiation_msv is CUMULATIVE | rate_uSv_h = (radiation_msv[t] - radiation_msv[t-1]) / (sim hours between rows) * 1000 (sim_seconds delta / 3600). Alt (no history): planets.radiation_msv_per_day / 24 * 1000 = 27.9 uSv/h nominal, x40 = 1116 uSv/h in radiation_event | GAP: no rate column. Nominal 27.9 uSv/h exceeds policy warning_above 10 -> warning would fire constantly in normal scenario. Units: DB mSv -> uSv x1000. |
| temp_c (data, snapshot) | telemetry.suit_temp_c (recommended) or ext_temp_c | direct, C | AMBIGUOUS. ext_temp_c starts at -40 and ranges about -85..5, so mapping it to temp_c triggers critical_below -10 right away. Suit temp 21-25 C is in safe range. Pick suit_temp_c and rename or document. |
| suit_pressure_kpa (safety, snapshot) | telemetry.suit_pressure_kpa | direct, kPa | Direct. Policy critical_below 20 vs sim low-suit alert at 27/25. |
| distance_m, eta_min (navigation, map) | NOT in DB as distances; use astronaut_positions (latest_position x_m,z_m) and map_locations (x_m,z_m) | Euclidean distance = sqrt(dx^2+dz^2) in m; eta_min = ceil(distance / walk_speed) where walk_speed_mps (missions, 0.8 m/s = 48 m/min) | GAP. Name mapping needed: homebase/habitat/airlock/greenhouse -> HAB?; rover -> ROVER; crater -> CRATR; ridge -> RIDGE; lander/landing_*/outpost/etc. have no row. Euclidean ignores terrain; SLOPE is danger hazard. |
| feasibility walking speed | missions.walk_speed_mps (0.8 m/s) | x60 -> 48 m/min | MISMATCH: policy uses 60 m/min (1 m/s) placeholder. Use DB value or align. |
| feasibility fallback | missions.o2_burn_pct_per_min | minutes per % = 1/burn (at walking, 6.67 min/% with default 0.15) | MISMATCH: policy uses 2.4 min/%. Fallback should not be used when DB exists. Prefer minutes_remaining derived above. |
| send_message | messages table (direction outbound) | insert row with body=message, real_delay_s/demo_delay_s from policy or config | GAP: fake tool does not write DB; no urgency column (encode in body or add). |
| alert_source (critical lines) | telemetry + rules | snapshot keys above | Router critical thresholds differ from simulator alert rules; decide single source (sim `alerts` table vs policy). |

Staleness:
- Each telemetry row is one committed tick (~1 row/s real time). Use `latest_telemetry.ts` vs now(). Simulator treats gaps >10 s as pauses, so rows older than ~10 s signal sim stopped (no active writer, or lost advisory lock). Suggested: if now()-ts > a placeholder limit (e.g. 10 s), return no snapshot -> tool error rather than stale numbers (feasibility returns None -> no outcome spoken, which is the intended safe behavior when O2 data missing).
- `latest_position` is not updated by simulator; staleness unknown and likely seeded HAB forever unless Unity writes.
- Tool results are not time-stamped; consider adding `ts` / `sim_seconds` to snapshot.

Other unit notes:
- Sim time vs real: telemetry.ts is real wall time; sim_seconds is simulated. time_scale default 30 means minutes_remaining drops 30x real speed.
- `sim_seconds` CHECK >= 0; `local_time_hours` 0..24.66.
- Radiation dose day = 86400 s in sim (not 88776 s sol). Minor (~3% error on dose rate).
- Simulator alerts use 'heart_rate' > 175 critical; policy bpm critical >180 - different.

## Risks / Open Questions
1. Two threshold sets (simulator RULES vs turn_policy sim_fields) can disagree; decide which the router uses.
2. minutes_remaining has no DB column; definition (to-empty vs to-reserve) must be chosen and fed to feasibility consistently.
3. radiation_usv_h requires derivation from cumulative dose; nominal rate exceeds the policy warning threshold.
4. temp_c mapping ambiguity (suit vs ext).
5. Feasibility placeholders (60 m/min, 2.4 min/%) conflict with DB walk_speed 0.8 m/s and burn 0.15 %/min.
6. Distances: fabricated PLACE_DISTANCES_M vs map_locations coordinates; need name->code mapping.
7. `dust_devil` enum unused; `comm_delay` and `minimap` HUD metrics have no sim field.
8. astronaut_positions never updated by the simulator; navigation needs a writer (Unity or API).
9. No writer for navigation_requests, hud_commands, vision_detections; send_message fake writes nothing.
10. apistage1.py only has /health; no endpoints for telemetry or tools yet.
11. Simulator runs even with no clients; `--new-mission` leaves old mission active=false (history retained).

## Start Here
Open `schema.sql` (lines 1-150 for telemetry/missions, 230-270 for views) then `simulator.py` lines 89-140 (`step`) and 158-184 (rules) to understand what the DB holds, then `router/turn.py` 77-122 and `router/tools.py` 79-131 to see what snapshot keys the router expects.


## Proposed data-access layer (design)
- `router/datasource.py`: `Datasource` Protocol with `snapshot() -> Snapshot|None`, `place(code_or_name)`, `position()`, `queue_message(...)`. Impls: `PostgresDatasource` (psycopg, optional dep, DATABASE_URL) and `FixtureDatasource` (in-memory rows built from schema seed values + simulator_rules formulas; used by tests/demo, no network).
- `tools.py` keeps 6 tools, same schemas and result dict shapes; `SimState` replaced by a datasource-backed snapshot. `alert_source()` returns the same snapshot keys (o2_pct, minutes_remaining, bpm, radiation_usv_h, temp_c, suit_pressure_kpa). `sim_fields` untouched.
- Stale snapshot (placeholder limit ~10 s) -> tool error / no number spoken.
- Distances: Euclidean from latest_position to map_locations; router place name -> location code via a PLACEHOLDER alias table in turn_policy.json.

## Decisions needing human sign-off (PLACEHOLDER until then)
1. `temp_c` = suit_temp_c vs ext_temp_c (ext hits critical_below -10 immediately).
2. `radiation_usv_h` derivation (delta of cumulative dose); nominal 27.9 uSv/h exceeds policy warning_above 10 -> constant warning.
3. `minutes_remaining` definition (to empty vs to reserve) and feasibility placeholders (60 m/min, 2.4 min/%) vs DB (0.8 m/s, 0.15 %/min).
4. Router thresholds vs simulator alert RULES (two sets).
5. Name->location-code alias table; places without DB rows (lander, outpost...).
6. send_message persistence (messages table has no urgency column).
