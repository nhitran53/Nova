\set ON_ERROR_STOP on
BEGIN;
CREATE TEMP TABLE radiation_csv (
    date_utc TEXT, day_of_year TEXT, hour TEXT, minute TEXT, second TEXT,
    silicon TEXT, tissue TEXT, silicon_uncorrected TEXT,
    observation_seconds TEXT, data_flag TEXT
) ON COMMIT DROP;
 
\copy radiation_csv FROM pstdin WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')
DO $$
DECLARE bad RECORD;
BEGIN
    SELECT * INTO bad FROM radiation_csv
    WHERE coalesce(trim(date_utc), '') <> ''
      AND NOT (trim(hour)::int   BETWEEN 0 AND 23
           AND trim(minute)::int BETWEEN 0 AND 59
           AND trim(second)::numeric >= 0 AND trim(second)::numeric < 60)
    LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'Impossible time in CSV: % hour=% min=% sec=%',
            bad.date_utc, bad.hour, bad.minute, bad.second;
    END IF;
END $$;
 
WITH added AS (
    INSERT INTO mars_radiation (observed_at, silicon_ugy_per_day, tissue_ugy_per_day,
                                silicon_uncorrected_ugy_per_day, observation_seconds, data_flag)
    SELECT
        (trim(date_utc)::date
           + make_time(trim(hour)::int, trim(minute)::int, trim(second)::float8)) AT TIME ZONE 'UTC',
        trim(silicon)::real,
        trim(tissue)::real,
        trim(silicon_uncorrected)::real,
        trim(observation_seconds)::real,
        COALESCE(NULLIF(trim(data_flag), ''), 'none')
    FROM radiation_csv
    WHERE coalesce(trim(date_utc), '') <> ''          
    ON CONFLICT ON CONSTRAINT mars_radiation_one_reading DO NOTHING
    RETURNING 1
)
SELECT (SELECT count(*) FROM radiation_csv) AS rows_in_file,
       (SELECT count(*) FROM added)         AS rows_added;
 
SELECT count(*) AS rows_in_table FROM mars_radiation;
SELECT count(*) AS rows_where_day_of_year_disagrees
FROM radiation_csv
WHERE coalesce(trim(date_utc), '') <> ''
  AND EXTRACT(DOY FROM trim(date_utc)::date) <> trim(day_of_year)::int;
 
COMMIT;
 