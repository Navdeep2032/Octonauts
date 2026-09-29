CREATE INDEX IF NOT EXISTS idx_temperatures_date_lat_lon
  ON public.temperatures (date, lat, lon);

ALTER TABLE public.temperatures ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS oceanembed_six_month_read ON public.temperatures;
DROP POLICY IF EXISTS oceanembed_six_month_window ON public.temperatures;
DROP POLICY IF EXISTS oceanembed_all_dates_read ON public.temperatures;

CREATE POLICY oceanembed_all_dates_read
  ON public.temperatures
  AS PERMISSIVE
  FOR SELECT
  TO anon, authenticated
  USING (true);

GRANT SELECT ON public.temperatures TO anon, authenticated;

CREATE OR REPLACE FUNCTION public.oceanembed_coverage()
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SET search_path = public
AS $$
DECLARE
  earliest date;
  latest date;
BEGIN
  SELECT min(t.date), max(t.date)
  INTO earliest, latest
  FROM public.temperatures AS t;

  RETURN jsonb_build_object('earliest', earliest, 'latest', latest);
END;
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
    WHERE t.date = p_date
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
  WHERE t.date = p_date
    AND t.lat BETWEEN p_lat - 0.126 AND p_lat + 0.126
    AND t.lon BETWEEN p_lon - 0.126 AND p_lon + 0.126
    AND t.d0 > 0
  ORDER BY abs(t.lat - p_lat) + abs(t.lon - p_lon)
  LIMIT 1;

  RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION public.oceanembed_history(
  p_end_date date,
  p_days integer,
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
  IF p_days NOT IN (7, 14, 30) THEN
    RAISE EXCEPTION 'History window must be 7, 14, or 30 days';
  END IF;
  IF p_lat < 5 OR p_lat > 30 OR p_lon < 45 OR p_lon > 105 THEN
    RAISE EXCEPTION 'Coordinate is outside the model domain';
  END IF;

  SELECT COALESCE(
    jsonb_agg(
      jsonb_build_object(
        'date', history.date,
        'depths', jsonb_build_object(
          '0', history.d0,
          '5', history.d5,
          '10', history.d10,
          '20', history.d20,
          '30', history.d30,
          '50', history.d50,
          '75', history.d75,
          '100', history.d100,
          '125', history.d125,
          '150', history.d150,
          '200', history.d200,
          '300', history.d300,
          '500', history.d500,
          '700', history.d700,
          '1000', history.d1000
        )
      )
      ORDER BY history.date
    ),
    '[]'::jsonb
  )
  INTO result
  FROM generate_series(
    p_end_date - (p_days - 1),
    p_end_date,
    interval '1 day'
  ) AS requested(day)
  CROSS JOIN LATERAL (
    SELECT
      t.date,
      t.d0,
      t.d5,
      t.d10,
      t.d20,
      t.d30,
      t.d50,
      t.d75,
      t.d100,
      t.d125,
      t.d150,
      t.d200,
      t.d300,
      t.d500,
      t.d700,
      t.d1000
    FROM public.temperatures AS t
    WHERE t.date = requested.day::date
      AND t.lat BETWEEN p_lat - 0.126 AND p_lat + 0.126
      AND t.lon BETWEEN p_lon - 0.126 AND p_lon + 0.126
      AND t.d0 > 0
    ORDER BY abs(t.lat - p_lat) + abs(t.lon - p_lon)
    LIMIT 1
  ) AS history;

  RETURN result;
END;
$$;

REVOKE ALL ON FUNCTION public.oceanembed_coverage() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.oceanembed_map(date, integer, double precision, double precision, double precision, double precision, integer, integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.oceanembed_profile(date, double precision, double precision) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.oceanembed_history(date, integer, double precision, double precision) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.oceanembed_coverage() TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.oceanembed_map(date, integer, double precision, double precision, double precision, double precision, integer, integer) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.oceanembed_profile(date, double precision, double precision) TO anon, authenticated;
GRANT EXECUTE ON FUNCTION public.oceanembed_history(date, integer, double precision, double precision) TO anon, authenticated;
