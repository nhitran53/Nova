-- Verbatim copy of the mars_weather, crew_o2_co2 and mars_radiation DDL (lines 490-544) from
-- origin/nhi/database:schema.sql, used only by router/tests/test_reference_data.py. Not applied to any database.
CREATE TABLE IF NOT EXISTS mars_weather (
    id               BIGSERIAL PRIMARY KEY,
    planet_id        TEXT NOT NULL DEFAULT 'mars' REFERENCES planets(id),
    mars_year        INT  NOT NULL CHECK (mars_year > 0),
    northern_season  TEXT NOT NULL CHECK (northern_season IN ('Spring', 'Summer', 'Fall', 'Winter')),
    day_night        TEXT NOT NULL CHECK (day_night IN ('Day', 'Night')),
    air_temp_c       REAL,                                  -- blank in the file -> NULL
    air_temp_err_c   REAL CHECK (air_temp_err_c >= 0),      -- the ± value
    pressure_pa      REAL CHECK (pressure_pa > 0),
    pressure_err_pa  REAL CHECK (pressure_err_pa >= 0)      -- the ± value
);
ALTER TABLE mars_weather DROP COLUMN IF EXISTS data_source;

CREATE TABLE IF NOT EXISTS crew_o2_co2 (
    record_id                 INT PRIMARY KEY,               -- from the CSV; stops duplicate loads
    sol                       INT  NOT NULL CHECK (sol > 0),
    crew_id                   TEXT NOT NULL,
    sex                       TEXT NOT NULL CHECK (sex IN ('M', 'F')),
    body_mass_kg              REAL NOT NULL CHECK (body_mass_kg > 0),
    environment               TEXT NOT NULL,                 -- 'Habitat' or 'EVA suit'
    activity                  TEXT NOT NULL,                 -- 'Sleep', 'Exercise (treadmill/ARED)', ...
    duration_hr               REAL NOT NULL CHECK (duration_hr > 0),
    metabolic_rate_w          REAL NOT NULL CHECK (metabolic_rate_w > 0),
    respiratory_quotient      REAL NOT NULL CHECK (respiratory_quotient > 0),   -- CO2 produced / O2 used
    ambient_pressure_kpa      REAL NOT NULL CHECK (ambient_pressure_kpa > 0),
    inspired_o2_pct           REAL NOT NULL CHECK (inspired_o2_pct BETWEEN 0 AND 100),
    ambient_pp_co2_kpa        REAL NOT NULL CHECK (ambient_pp_co2_kpa >= 0),
    minute_ventilation_l_min  REAL NOT NULL CHECK (minute_ventilation_l_min >= 0),
    o2_inhaled_l_min          REAL NOT NULL CHECK (o2_inhaled_l_min >= 0),
    o2_exhaled_l_min          REAL NOT NULL CHECK (o2_exhaled_l_min >= 0),
    o2_consumed_l_min         REAL NOT NULL CHECK (o2_consumed_l_min >= 0),
    co2_inhaled_l_min         REAL NOT NULL CHECK (co2_inhaled_l_min >= 0),
    co2_exhaled_l_min         REAL NOT NULL CHECK (co2_exhaled_l_min >= 0),
    co2_produced_l_min        REAL NOT NULL CHECK (co2_produced_l_min >= 0),
    o2_consumed_g             REAL NOT NULL CHECK (o2_consumed_g >= 0),       -- over duration_hr
    co2_produced_g            REAL NOT NULL CHECK (co2_produced_g >= 0),      -- over duration_hr
    o2_consumed_kg_per_day    REAL NOT NULL CHECK (o2_consumed_kg_per_day >= 0),   -- rate if kept up all day
    co2_produced_kg_per_day   REAL NOT NULL CHECK (co2_produced_kg_per_day >= 0)
);
COMMENT ON TABLE crew_o2_co2 IS
    'Simulated crew respiration (NASA BVAD-based model), from mars_astronaut_o2_co2.csv';
 
CREATE INDEX IF NOT EXISTS crew_o2_co2_activity ON crew_o2_co2 (environment, activity);

CREATE TABLE IF NOT EXISTS mars_radiation (
    id                          BIGSERIAL PRIMARY KEY,
    planet_id                   TEXT NOT NULL DEFAULT 'mars' REFERENCES planets(id),
    observed_at                 TIMESTAMPTZ NOT NULL,      
    silicon_ugy_per_day         REAL NOT NULL CHECK (silicon_ugy_per_day >= 0),
    tissue_ugy_per_day          REAL NOT NULL CHECK (tissue_ugy_per_day >= 0), 
    silicon_uncorrected_ugy_per_day REAL NOT NULL CHECK (silicon_uncorrected_ugy_per_day >= 0),
    observation_seconds         REAL NOT NULL CHECK (observation_seconds > 0),
    data_flag                   TEXT NOT NULL DEFAULT 'none', 
    CONSTRAINT mars_radiation_one_reading UNIQUE (planet_id, observed_at)
);
