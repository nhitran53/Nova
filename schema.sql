
CREATE TABLE IF NOT EXISTS planets (
    id TEXT PRIMARY KEY,        
    display_name TEXT NOT NULL,
    gravity_g REAL NOT NULL,
    surface_pressure_kpa REAL NOT NULL,
    temp_min_c REAL NOT NULL,
    temp_max_c REAL NOT NULL,
    day_length_hours REAL NOT NULL,
    radiation_msv_per_day REAL NOT NULL,
    orbit_au REAL NOT NULL,          
    weather_events TEXT[] NOT NULL
);

-- simulator reads/writes the active one.
CREATE TABLE IF NOT EXISTS missions (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'EVA-1',
    astronaut TEXT NOT NULL DEFAULT 'Commander',
    planet_id TEXT NOT NULL REFERENCES planets(id),
    scenario TEXT NOT NULL DEFAULT 'normal',   -- normal, scrubber_failure, o2_leak, weather_event, panic, etc...
    activity TEXT NOT NULL DEFAULT 'walking',  -- resting, walking
    mission_day_start REAL NOT NULL DEFAULT 12,
    time_scale  REAL NOT NULL DEFAULT 30,         -- sim seconds per real second
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    active  BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_mission
ON missions (active)
WHERE active = TRUE;
-- Time series: one row per simulator tick (1/s).
CREATE TABLE IF NOT EXISTS telemetry (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    sim_seconds DOUBLE PRECISION NOT NULL,
    o2_percent REAL NOT NULL,
    co2_mmhg REAL NOT NULL,
    heart_rate REAL NOT NULL,
    spo2 REAL NOT NULL,
    suit_pressure_kpa REAL NOT NULL,
    battery_percent   REAL NOT NULL,
    suit_temp_c  REAL NOT NULL,
    radiation_msv  REAL NOT NULL,
    ext_temp_c REAL NOT NULL,
    wind_mps REAL NOT NULL,
    visibility_km   REAL NOT NULL,
    weather   TEXT NOT NULL,
    local_time_hours  REAL NOT NULL,
    alerts  JSONB NOT NULL DEFAULT '{}'::jsonb   -- {"co2":"caution", ...}
);
CREATE INDEX IF NOT EXISTS telemetry_mission_ts ON telemetry (mission_id, ts DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    channel    TEXT NOT NULL,          -- o2 | co2 | bpm | pressure | battery | weather
    level      TEXT NOT NULL,          -- nominal | caution | critical
    message    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_mission_id ON alerts (mission_id, id);

CREATE TABLE IF NOT EXISTS messages (
    id             BIGSERIAL PRIMARY KEY,
    mission_id     INT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    direction      TEXT NOT NULL,      -- outbound (helmet → Earth) | inbound (Earth → helmet)
    body           TEXT NOT NULL,
    sent_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    real_delay_s   DOUBLE PRECISION NOT NULL,   -- true light delay
    demo_delay_s   DOUBLE PRECISION NOT NULL,   -- compressed, used for the progress bar
    delivered_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS messages_pending ON messages (mission_id) WHERE delivered_at IS NULL;

-- Commands for the headset HUD ("show co2 and o2 in the middle").
-- Written by the AI tool or any client; Unity applies the newest one it hasn't seen.
CREATE TABLE IF NOT EXISTS hud_commands (
    id         BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL REFERENCES missions(id) ON DELETE CASCADE,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    action     TEXT NOT NULL,          -- focus | clear
    metrics    TEXT[] NOT NULL DEFAULT '{}',
    seconds    REAL NOT NULL DEFAULT 8
);

-- Seed planet data (rounded public figures).
INSERT INTO planets (id, display_name, gravity_g, surface_pressure_kpa, temp_min_c, temp_max_c, day_length_hours, radiation_msv_per_day, orbit_au, weather_events) VALUES
 ('moon',   'Moon',   0.165, 0,     -173, 127,  708.7, 1.4,   0, ARRAY['Solar particle event','Lunar night']),
 ('mars',   'Mars',   0.38,  0.6,   -125, 20,   24.6,  0.67,  1.524,   ARRAY['Dust storm','Dust devil','Solar particle event']),
 ('europa', 'Europa', 0.134, 0,     -220, -160, 85.2,  5400,  5.203,  ARRAY['Radiation belt surge','Ice plume venting']),
 ('titan',  'Titan',  0.138, 146.7, -180, -178, 382.7, 0.01,  9.537,  ARRAY['Methane rain','Haze thickening'])
ON CONFLICT (id) DO NOTHING;

INSERT INTO missions (planet_id)
SELECT 'mars' WHERE NOT EXISTS (SELECT 1 FROM missions WHERE active);
