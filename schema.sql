
BEGIN;

-- 1. Planet reference information
CREATE TABLE IF NOT EXISTS planets (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    gravity_g REAL NOT NULL CHECK (gravity_g >= 0),
    surface_pressure_kpa REAL NOT NULL CHECK (surface_pressure_kpa >= 0),
    temp_min_c REAL NOT NULL,
    temp_max_c REAL NOT NULL,
    day_length_hours REAL NOT NULL CHECK (day_length_hours > 0),
    radiation_msv_per_day REAL NOT NULL CHECK (radiation_msv_per_day >= 0),
    orbit_au REAL NOT NULL CHECK (orbit_au >= 0),
    weather_events TEXT[] NOT NULL,
    CHECK (temp_min_c <= temp_max_c)
);

-- 2. Mission settings
CREATE TABLE IF NOT EXISTS missions (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'NOVA-1',
    astronaut TEXT NOT NULL DEFAULT 'Commander',
    planet_id TEXT NOT NULL REFERENCES planets(id),

    scenario TEXT NOT NULL DEFAULT 'normal'
        CHECK (
            scenario IN ('normal','o2_leak','dust_storm','radiation_event')
        ),

    activity TEXT NOT NULL DEFAULT 'walking'
        CHECK (activity IN ('resting', 'walking', 'working', 'running')),

    -- Elapsed Earth hours within the simulated Mars sol.
    mission_day_start REAL NOT NULL DEFAULT 12
        CHECK (mission_day_start >= 0 AND mission_day_start < 24.66),

    -- Simulated seconds per real second.
    time_scale REAL NOT NULL DEFAULT 30 CHECK (time_scale > 0),

    -- Demo route-calculation settings.
    walk_speed_mps REAL NOT NULL DEFAULT 0.8
        CHECK (walk_speed_mps > 0),

    -- Percentage POINTS consumed per simulated minute.
    -- Backend must account for changes caused by activity or leaks.
    o2_burn_pct_per_min REAL NOT NULL DEFAULT 0.15
        CHECK (o2_burn_pct_per_min > 0),

    o2_reserve_pct REAL NOT NULL DEFAULT 20
        CHECK (o2_reserve_pct BETWEEN 0 AND 100),

    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    active BOOLEAN NOT NULL DEFAULT TRUE
);

-- One active mission for the shared demo.
CREATE UNIQUE INDEX IF NOT EXISTS one_active_mission
ON missions (active)
WHERE active = TRUE;

-- 3. Suit and environment readings
CREATE TABLE IF NOT EXISTS telemetry (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),

    sim_seconds DOUBLE PRECISION NOT NULL CHECK (sim_seconds >= 0),
    -- Remaining usable tank supply
    o2_percent REAL NOT NULL CHECK (o2_percent BETWEEN 0 AND 100),
    -- CO2 partial pressure inside the suit.
    co2_mmhg REAL NOT NULL CHECK (co2_mmhg >= 0),
    heart_rate REAL NOT NULL CHECK (heart_rate BETWEEN 0 AND 250),
    spo2 REAL NOT NULL CHECK (spo2 BETWEEN 0 AND 100),
    suit_pressure_kpa REAL NOT NULL CHECK (suit_pressure_kpa >= 0),
    battery_percent REAL NOT NULL CHECK (battery_percent BETWEEN 0 AND 100),
    suit_temp_c REAL NOT NULL,

    -- Accumulated simulated radiation dose during this mission.
    radiation_msv REAL NOT NULL CHECK (radiation_msv >= 0),

    ext_temp_c REAL NOT NULL,
    wind_mps REAL NOT NULL CHECK (wind_mps >= 0),
    visibility_km REAL NOT NULL CHECK (visibility_km >= 0),

    weather TEXT NOT NULL
        CHECK (
            weather IN ('clear',
                'dusty',
                'dust_devil',
                'dust_storm',
                'solar_particle_event'
            )
        ),

    local_time_hours REAL NOT NULL
        CHECK (local_time_hours >= 0 AND local_time_hours < 24.66)
);

CREATE INDEX IF NOT EXISTS telemetry_mission_ts
ON telemetry (mission_id, ts DESC, id DESC);

-- Latest reading for the active mission.
-- Empty until the simulator inserts telemetry.
CREATE OR REPLACE VIEW latest_telemetry AS
SELECT DISTINCT ON (t.mission_id) t.*
FROM telemetry t
JOIN missions m ON m.id = t.mission_id
WHERE m.active
ORDER BY t.mission_id, t.ts DESC, t.id DESC;

-- 4. Minimap markers
CREATE TABLE IF NOT EXISTS map_locations (
    id SERIAL PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,

    location_type TEXT NOT NULL
        CHECK (
            location_type IN (
                'habitat',
                'rover',
                'sample_site',
                'landmark',
                'crater',
                'hazard',
                'waypoint'
            )
        ),

    x_m DOUBLE PRECISION NOT NULL,
    z_m DOUBLE PRECISION NOT NULL,

    -- Approximate zone radius in the demo.
    radius_m REAL NOT NULL DEFAULT 5 CHECK (radius_m > 0),

    hazard_level TEXT NOT NULL DEFAULT 'safe'
        CHECK (hazard_level IN ('safe', 'caution', 'danger')),

    -- Whether this location provides oxygen in the simulation.
    o2_refill BOOLEAN NOT NULL DEFAULT FALSE,

    source TEXT NOT NULL DEFAULT 'seed'
        CHECK (source IN ('seed', 'vision', 'astronaut')),

    description TEXT
);

-- 5. Recorded astronaut positions
CREATE TABLE IF NOT EXISTS astronaut_positions (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    x_m DOUBLE PRECISION NOT NULL,
    z_m DOUBLE PRECISION NOT NULL,
    heading_deg REAL NOT NULL DEFAULT 0
        CHECK (heading_deg >= 0 AND heading_deg < 360)
);

CREATE INDEX IF NOT EXISTS positions_mission_ts
ON astronaut_positions (mission_id, ts DESC, id DESC);

CREATE OR REPLACE VIEW latest_position AS
SELECT DISTINCT ON (p.mission_id) p.*
FROM astronaut_positions p
JOIN missions m ON m.id = p.mission_id
WHERE m.active
ORDER BY p.mission_id, p.ts DESC, p.id DESC;

-- 6. Destination requests and route assessments
--
-- Backend calculates:
-- trip_minutes =
--     (outbound_distance_m + return_distance_m)
--     / walk_speed_mps / 60
--     + time_at_destination_minutes
--
-- estimated_o2_required_pct =
--     trip_minutes * effective_burn_rate
--     + reserve
--
-- Required oxygen INCLUDES reserve.
-- Backend must verify fresh readings, matching mission IDs,
-- traversable routes, and an available return location.
CREATE TABLE IF NOT EXISTS navigation_requests (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,

    destination_id INT NOT NULL REFERENCES map_locations(id),
    return_location_id INT REFERENCES map_locations(id),

    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    input_method TEXT NOT NULL
        CHECK (input_method IN ('voice', 'gesture', 'combined')),

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (
            status IN (
                'pending',
                'approved',
                'blocked',
                'unknown',
                'cancelled'
            )
        ),

    outbound_distance_m REAL CHECK (outbound_distance_m >= 0),
    return_distance_m REAL CHECK (return_distance_m >= 0),

    time_at_destination_minutes REAL NOT NULL DEFAULT 0
        CHECK (time_at_destination_minutes >= 0),

    -- Snapshot of the assumptions used for this assessment.
    assessed_walk_speed_mps REAL CHECK (assessed_walk_speed_mps > 0),
    assessed_o2_burn_pct_per_min REAL
        CHECK (assessed_o2_burn_pct_per_min > 0),
    assessed_o2_reserve_pct REAL
        CHECK (assessed_o2_reserve_pct BETWEEN 0 AND 100),

    -- May exceed 100 if a proposed trip is too long.
    estimated_o2_required_pct REAL
        CHECK (estimated_o2_required_pct >= 0),

    o2_available_pct REAL CHECK (o2_available_pct BETWEEN 0 AND 100),
    warning_message TEXT,

    telemetry_id BIGINT REFERENCES telemetry(id) ON DELETE SET NULL,
    position_id BIGINT REFERENCES astronaut_positions(id) ON DELETE SET NULL,
    assessed_at TIMESTAMPTZ,

    CONSTRAINT nav_decision_has_time CHECK (
        status NOT IN ('approved', 'blocked')
        OR assessed_at IS NOT NULL
    ),

    -- Hazard-based blocking does not require an oxygen calculation.
    CONSTRAINT nav_approved_has_assessment CHECK (
        status <> 'approved'
        OR (
            return_location_id IS NOT NULL
            AND outbound_distance_m IS NOT NULL
            AND return_distance_m IS NOT NULL
            AND assessed_walk_speed_mps IS NOT NULL
            AND assessed_o2_burn_pct_per_min IS NOT NULL
            AND assessed_o2_reserve_pct IS NOT NULL
            AND estimated_o2_required_pct IS NOT NULL
            AND o2_available_pct IS NOT NULL
            AND estimated_o2_required_pct <= o2_available_pct
        )
    ),

    CONSTRAINT nav_blocked_has_reason CHECK (
        status <> 'blocked'
        OR (
            warning_message IS NOT NULL
            AND length(trim(warning_message)) > 0
        )
    )
);

CREATE INDEX IF NOT EXISTS nav_mission_time
ON navigation_requests (mission_id, requested_at DESC);

-- 7. Warnings displayed and spoken to the astronaut
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),

    channel TEXT NOT NULL
        CHECK (
            channel IN (
                'o2',
                'co2',
                'vitals',
                'suit',
                'power',
                'radiation',
                'weather',
                'navigation',
                'vision',
                'comms'
            )
        ),

    level TEXT NOT NULL
        CHECK (level IN ('info', 'caution', 'critical')),

    message TEXT NOT NULL,
    voice_text TEXT,

    -- Acknowledgment does not mean the condition has ended.
    acknowledged_at TIMESTAMPTZ,
    alert_key TEXT,
    resolved_at TIMESTAMPTZ,

    navigation_request_id BIGINT
        REFERENCES navigation_requests(id) ON DELETE SET NULL,
    location_id INT
        REFERENCES map_locations(id) ON DELETE SET NULL,

    CHECK (resolved_at IS NULL OR resolved_at >= ts),
    CHECK (acknowledged_at IS NULL OR acknowledged_at >= ts)
);

CREATE INDEX IF NOT EXISTS alerts_mission_id
ON alerts (mission_id, id);

-- Backend uses a stable key, e.g. 'low_oxygen', and updates the
-- existing open alert instead of inserting it every simulator tick.
CREATE UNIQUE INDEX IF NOT EXISTS alerts_one_open_key
ON alerts (mission_id, alert_key)
WHERE resolved_at IS NULL AND alert_key IS NOT NULL;

-- Includes acknowledged but unresolved conditions.
-- Unity must separately avoid repeating voice playback on every poll.
CREATE OR REPLACE VIEW open_alerts AS
SELECT a.*
FROM alerts a
JOIN missions m ON m.id = a.mission_id
WHERE m.active
  AND a.resolved_at IS NULL;

-- 8. Messages between the astronaut and Earth
CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,

    direction TEXT NOT NULL
        CHECK (direction IN ('outbound', 'inbound')),

    body TEXT NOT NULL CHECK (length(trim(body)) > 0),
    sent_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Estimated physical delay versus compressed demo delay.
    real_delay_s DOUBLE PRECISION NOT NULL CHECK (real_delay_s >= 0),
    demo_delay_s DOUBLE PRECISION NOT NULL CHECK (demo_delay_s >= 0),

    delivered_at TIMESTAMPTZ,
    CHECK (delivered_at IS NULL OR delivered_at >= sent_at)
);

CREATE INDEX IF NOT EXISTS messages_pending
ON messages (mission_id)
WHERE delivered_at IS NULL;

-- Countdown only: backend code must update delivered_at.
CREATE OR REPLACE VIEW message_status AS
SELECT
    msg.*,
    msg.sent_at + make_interval(secs => msg.demo_delay_s) AS demo_due_at,
    CASE
        WHEN msg.delivered_at IS NOT NULL THEN 0
        ELSE GREATEST(
            0,
            EXTRACT(
                EPOCH FROM (
                    msg.sent_at
                    + make_interval(secs => msg.demo_delay_s)
                    - statement_timestamp()
                )
            )
        )
    END AS demo_seconds_left
FROM messages msg;

-- 9. Helmet display commands
-- Unity processes unseen, unexpired commands in ID order.
-- NULL expires_at means no expiration.
CREATE TABLE IF NOT EXISTS hud_commands (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),

    action TEXT NOT NULL CHECK (action IN ('focus', 'clear')),

    metrics TEXT[] NOT NULL DEFAULT '{}'
        CHECK (
            metrics <@ ARRAY[
                'o2_percent',
                'co2_mmhg',
                'heart_rate',
                'spo2',
                'suit_pressure_kpa',
                'battery_percent',
                'suit_temp_c',
                'radiation_msv',
                'ext_temp_c',
                'wind_mps',
                'visibility_km',
                'weather',
                'local_time_hours',
                'comm_delay',
                'minimap'
            ]::TEXT[]
        ),

    seconds REAL NOT NULL DEFAULT 8 CHECK (seconds > 0),
    expires_at TIMESTAMPTZ,
    applied_at TIMESTAMPTZ,

    CHECK (action = 'clear' OR cardinality(metrics) > 0),
    CHECK (expires_at IS NULL OR expires_at >= ts),
    CHECK (applied_at IS NULL OR applied_at >= ts)
);

CREATE INDEX IF NOT EXISTS hud_pending
ON hud_commands (mission_id, id)
WHERE applied_at IS NULL;

-- 10. Optional vision detection results
-- Detect terrain/objects in the Mars demo.
-- Leave world coordinates NULL unless a positioning method supplies them.
CREATE TABLE IF NOT EXISTS vision_detections (
    id BIGSERIAL PRIMARY KEY,
    mission_id INT NOT NULL
        REFERENCES missions(id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),

    label TEXT NOT NULL,

    -- NULL if the model does not provide a confidence score.
    confidence REAL CHECK (confidence BETWEEN 0 AND 1),

    est_x_m DOUBLE PRECISION,
    est_z_m DOUBLE PRECISION,

    location_id INT REFERENCES map_locations(id) ON DELETE SET NULL,
    model_name TEXT,
    frame_ref TEXT,

    CHECK ((est_x_m IS NULL) = (est_z_m IS NULL))
);

CREATE INDEX IF NOT EXISTS vision_mission_ts
ON vision_detections (mission_id, ts DESC);

-- 11. Seed Mars reference/demo values
INSERT INTO planets (
    id,
    display_name,
    gravity_g,
    surface_pressure_kpa,
    temp_min_c,
    temp_max_c,
    day_length_hours,
    radiation_msv_per_day,
    orbit_au,
    weather_events
)
VALUES (
    'mars',
    'Mars',
    0.38,
    0.6,
    -125,
    20,
    24.66,
    0.67,
    1.524,
    ARRAY['Dust storm', 'Dust devil', 'Solar particle event']
)
ON CONFLICT (id) DO NOTHING;

-- Create a Mars mission if none is active.
INSERT INTO missions (planet_id)
SELECT 'mars'
WHERE NOT EXISTS (
    SELECT 1 FROM missions WHERE active
);

-- Placeholder map coordinates: replace with your Unity scene coordinates.
INSERT INTO map_locations (
    code,
    name,
    location_type,
    x_m,
    z_m,
    radius_m,
    hazard_level,
    o2_refill,
    description
)
VALUES
    (
        'HAB', 'Habitat', 'habitat',
        0, 0, 15, 'safe', TRUE,
        'Demo base with oxygen refill.'
    ),
    (
        'ROVER', 'Rover', 'rover',
        120, 40, 6, 'safe', TRUE,
        'Demo pressurized rover with spare oxygen.'
    ),
    (
        'SITEA', 'Sample Site A', 'sample_site',
        450, -200, 10, 'safe', FALSE,
        'Layered sediment outcrop.'
    ),
    (
        'CRATR', 'Crater Rim', 'crater',
        900, 350, 60, 'caution', FALSE,
        'Steep edges.'
    ),
    (
        'SLOPE', 'Unstable Slope', 'hazard',
        600, 500, 80, 'danger', FALSE,
        'Loose regolith.'
    ),
    (
        'RIDGE', 'Ridge Overlook', 'landmark',
        1500, -100, 20, 'safe', FALSE,
        'Demo observation point.'
    )
ON CONFLICT (code) DO NOTHING;

-- Seed a starting position for the active mission at the habitat.
INSERT INTO astronaut_positions (
    mission_id,
    x_m,
    z_m,
    heading_deg
)
SELECT
    m.id,
    loc.x_m,
    loc.z_m,
    0
FROM missions m
JOIN map_locations loc ON loc.code = 'HAB'
WHERE m.active
AND NOT EXISTS (
    SELECT 1
    FROM astronaut_positions p
    WHERE p.mission_id = m.id
);

COMMIT;