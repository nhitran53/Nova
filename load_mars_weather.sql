\set ON_ERROR_STOP on
BEGIN;
 
CREATE TEMP TABLE weather_csv (
    mars_year TEXT, season TEXT, day_night TEXT,
    air_c TEXT, air_err_c TEXT,
    pressure_mbar TEXT, pressure_err_mbar TEXT,
    data_source TEXT      
) ON COMMIT DROP;
\copy weather_csv FROM pstdin WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
 
DELETE FROM mars_weather WHERE planet_id = 'mars';
 
INSERT INTO mars_weather (mars_year, northern_season, day_night,
                          air_temp_c, air_temp_err_c, pressure_pa, pressure_err_pa)
SELECT
    trim(mars_year)::int,
    initcap(trim(season)),
    initcap(trim(day_night)),
    NULLIF(trim(air_c), '')::real,
    NULLIF(trim(air_err_c), '')::real,
    (NULLIF(trim(pressure_mbar), '')::numeric * 100)::real,      
    (NULLIF(trim(pressure_err_mbar), '')::numeric * 100)::real
FROM weather_csv
WHERE coalesce(trim(mars_year), '') <> '';         
 
SELECT count(*) AS rows_loaded FROM mars_weather WHERE planet_id = 'mars';
 
COMMIT;