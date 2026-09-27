CREATE INDEX IF NOT EXISTS idx_temperatures_date_lat_lon
  ON public.temperatures (date, lat, lon);

ALTER TABLE public.temperatures ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS oceanembed_six_month_read ON public.temperatures;
DROP POLICY IF EXISTS oceanembed_six_month_window ON public.temperatures;

CREATE POLICY oceanembed_six_month_read
  ON public.temperatures
  AS PERMISSIVE
  FOR SELECT
  TO anon, authenticated
  USING (date >= '2026-03-01' AND date < '2026-09-01');

CREATE POLICY oceanembed_six_month_window
  ON public.temperatures
  AS RESTRICTIVE
  FOR SELECT
  TO anon, authenticated
  USING (date >= '2026-03-01' AND date < '2026-09-01');

GRANT SELECT ON public.temperatures TO anon, authenticated;

CREATE OR REPLACE FUNCTION public.oceanembed_coverage()
RETURNS jsonb
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  WITH available AS (
    SELECT DISTINCT t.date
    FROM public.temperatures AS t
    WHERE t.date >= '2026-03-01'
      AND t.date < '2026-09-01'
      AND t.date::text ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
  )
  SELECT jsonb_build_object(
    'dates', COALESCE(jsonb_agg(date ORDER BY date), '[]'::jsonb),
    'earliest', min(date),
    'latest', max(date)
  )
  FROM available;
$$;

CREATE OR REPLACE FUNCTION public.oceanembed_map(
  p_date date,
  p_depth integer,
  p_south double precision,
  p_north double precision,
  p_west double precision,
  p_east double precision,
  p_stride integer,
  p_limit integer DEFAULT 1800
)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SET search_path = public
AS $$
DECLARE
  result jsonb;
BEGIN
  IF p_date < DATE '2026-03-01' OR p_date >= DATE '2026-09-01' THEN
    RAISE EXCEPTION 'Date is outside the March-August 2026 data window';
  END IF;
  IF p_depth NOT IN (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000) THEN
    RAISE EXCEPTION 'Unsupported model depth';
  END IF;
  IF p_south < 5 OR p_north > 30 OR p_west < 45 OR p_east > 105
    OR p_south >= p_north OR p_west >= p_east THEN
    RAISE EXCEPTION 'Bounds are outside the model domain';
  END IF;
  IF p_stride < 1 THEN
    RAISE EXCEPTION 'Stride must be a positive integer';
  END IF;

  SELECT jsonb_build_object(
    'stride', p_stride,
    'points', COALESCE(
      jsonb_agg(
        jsonb_build_object(
          'lat', sample.lat,
          'lon', sample.lon,
          'temperature', sample.temperature
        )
        ORDER BY sample.lat, sample.lon
      ),
      '[]'::jsonb
    )
  )
  INTO result
  FROM (
    SELECT
      t.lat::double precision AS lat,
      t.lon::double precision AS lon,
      CASE p_depth
        WHEN 0 THEN t.d0
        WHEN 5 THEN t.d5
        WHEN 10 THEN t.d10
        WHEN 20 THEN t.d20
        WHEN 30 THEN t.d30
        WHEN 50 THEN t.d50
        WHEN 75 THEN t.d75
        WHEN 100 THEN t.d100
        WHEN 125 THEN t.d125
        WHEN 150 THEN t.d150
        WHEN 200 THEN t.d200
        WHEN 300 THEN t.d300
        WHEN 500 THEN t.d500
        WHEN 700 THEN t.d700
        WHEN 1000 THEN t.d1000
      END::double precision AS temperature
    FROM public.temperatures AS t
    WHERE t.date = to_char(p_date, 'YYYY-MM-DD')
      AND t.lat BETWEEN p_south AND p_north
      AND t.lon BETWEEN p_west AND p_east
      AND t.d0 > 0
      AND mod(round((t.lat - 5.0) * 4)::integer, p_stride) = 0
      AND mod(round((t.lon - 45.0) * 4)::integer, p_stride) = 0
    ORDER BY t.lat, t.lon
    LIMIT LEAST(GREATEST(COALESCE(p_limit, 1800), 1), 1800)
  ) AS sample;

  RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION public.oceanembed_profile(
  p_date date,
  p_lat double precision,
  p_lon double precision
)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SET search_path = public
AS $$
DECLARE
  result jsonb;
BEGIN
  IF p_date < DATE '2026-03-01' OR p_date >= DATE '2026-09-01' THEN
    RAISE EXCEPTION 'Date is outside the March-August 2026 data window';
  END IF;
  IF p_lat < 5 OR p_lat > 30 OR p_lon < 45 OR p_lon > 105 THEN
    RAISE EXCEPTION 'Coordinate is outside the model domain';
  END IF;

  SELECT jsonb_build_object(
    'date', t.date,
    'lat', t.lat,
    'lon', t.lon,
    'depths', jsonb_build_object(
      '0', t.d0,
      '5', t.d5,
      '10', t.d10,
      '20', t.d20,
      '30', t.d30,
      '50', t.d50,
      '75', t.d75,
      '100', t.d100,
      '125', t.d125,
      '150', t.d150,
      '200', t.d200,
      '300', t.d300,
      '500', t.d500,
      '700', t.d700,
      '1000', t.d1000
    )
  )
  INTO result
  FROM public.temperatures AS t
  WHERE t.date = to_char(p_date, 'YYYY-MM-DD')
    AND t.lat BETWEEN p_lat - 0.126 AND p_lat + 0.126
    AND t.lon BETWEEN p_lon - 0.126 AND p_lon + 0.126
    AND t.d0 > 0
  ORDER BY abs(t.lat - p_lat) + abs(t.lon - p_lon)
  LIMIT 1;

  RETURN result;
END;
$$;

REVOKE ALL ON FUNCTION public.oceanembed_coverage() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.oceanembed_map(date, integer, double precision, double precision, double precision, double precision, integer, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.oceanembed_profile(date, double precision, double precision) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.oceanembed_coverage() TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.oceanembed_map(date, integer, double precision, double precision, double precision, double precision, integer, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.oceanembed_profile(date, double precision, double precision) TO anon, authenticated;
