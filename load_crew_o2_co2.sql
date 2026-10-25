\set ON_ERROR_STOP on
BEGIN;
CREATE TEMP TABLE crew_csv (
    record_id TEXT, sol TEXT, crew_id TEXT, sex TEXT, body_mass_kg TEXT,
    environment TEXT, activity TEXT, duration_hr TEXT, metabolic_rate_w TEXT,
    respiratory_quotient TEXT, ambient_pressure_kpa TEXT, inspired_o2_pct TEXT,
    ambient_pp_co2_kpa TEXT, minute_ventilation_l_min TEXT,
    o2_inhaled_l_min TEXT, o2_exhaled_l_min TEXT, o2_consumed_l_min TEXT,
    co2_inhaled_l_min TEXT, co2_exhaled_l_min TEXT, co2_produced_l_min TEXT,
    o2_consumed_g TEXT, co2_produced_g TEXT,
    o2_consumed_kg_per_day TEXT, co2_produced_kg_per_day TEXT,
    data_type TEXT                        
) ON COMMIT DROP;
\copy crew_csv FROM pstdin WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
 
WITH added AS (
    INSERT INTO crew_o2_co2
    SELECT
        trim(record_id)::int, trim(sol)::int, trim(crew_id), upper(trim(sex)),
        trim(body_mass_kg)::real, trim(environment), trim(activity),
        trim(duration_hr)::real, trim(metabolic_rate_w)::real,
        trim(respiratory_quotient)::real, trim(ambient_pressure_kpa)::real,
        trim(inspired_o2_pct)::real, trim(ambient_pp_co2_kpa)::real,
        trim(minute_ventilation_l_min)::real,
        trim(o2_inhaled_l_min)::real, trim(o2_exhaled_l_min)::real, trim(o2_consumed_l_min)::real,
        trim(co2_inhaled_l_min)::real, trim(co2_exhaled_l_min)::real, trim(co2_produced_l_min)::real,
        trim(o2_consumed_g)::real, trim(co2_produced_g)::real,
        trim(o2_consumed_kg_per_day)::real, trim(co2_produced_kg_per_day)::real
    FROM crew_csv
    WHERE coalesce(trim(record_id), '') <> ''          
    ON CONFLICT (record_id) DO NOTHING
    RETURNING 1
)
SELECT (SELECT count(*) FROM crew_csv) AS rows_in_file,
       (SELECT count(*) FROM added)    AS rows_added;
 
SELECT count(*) AS rows_in_table FROM crew_o2_co2;
 
COMMIT;