CREATE OR REPLACE FUNCTION public.oceanembed_export_page(
  p_start_date date,
  p_end_date date,
  p_after_date date DEFAULT NULL,
  p_after_lat double precision DEFAULT NULL,
  p_after_lon double precision DEFAULT NULL,
  p_limit integer DEFAULT 5000
)
RETURNS jsonb
LANGUAGE plpgsql
STABLE
SET search_path = public
AS $$
DECLARE
  result jsonb;
  page_limit integer := LEAST(GREATEST(COALESCE(p_limit, 5000), 1), 5000);
BEGIN
  IF p_start_date IS NULL OR p_end_date IS NULL OR p_start_date > p_end_date
    OR p_end_date - p_start_date > 93 THEN
    RAISE EXCEPTION 'Choose a valid date range of up to three months';
  END IF;
  IF (p_after_date IS NULL) <> (p_after_lat IS NULL)
    OR (p_after_date IS NULL) <> (p_after_lon IS NULL) THEN
    RAISE EXCEPTION 'Download cursor fields must be provided together';
  END IF;
  IF p_after_date IS NOT NULL
    AND (p_after_date < p_start_date OR p_after_date > p_end_date
      OR p_after_lat < 5 OR p_after_lat > 30
      OR p_after_lon < 45 OR p_after_lon > 105) THEN
    RAISE EXCEPTION 'Download cursor is outside the requested range';
  END IF;

  WITH page AS (
    SELECT t.date, t.lat, t.lon, t.d0, t.d5, t.d10, t.d20, t.d30,
      t.d50, t.d75, t.d100, t.d125, t.d150, t.d200, t.d300, t.d500,
      t.d700, t.d1000
    FROM public.temperatures AS t
    WHERE t.date BETWEEN p_start_date AND p_end_date
      AND t.d0 > 0
      AND (
        p_after_date IS NULL
        OR (t.date, t.lat, t.lon) > (p_after_date, p_after_lat, p_after_lon)
      )
    ORDER BY t.date, t.lat, t.lon
    LIMIT page_limit
  )
  SELECT jsonb_build_object(
    'rows', COALESCE(
      jsonb_agg(
        jsonb_build_object(
          'date', page.date,
          'lat', page.lat,
          'lon', page.lon,
          'd0', page.d0,
          'd5', page.d5,
          'd10', page.d10,
          'd20', page.d20,
          'd30', page.d30,
          'd50', page.d50,
          'd75', page.d75,
          'd100', page.d100,
          'd125', page.d125,
          'd150', page.d150,
          'd200', page.d200,
          'd300', page.d300,
          'd500', page.d500,
          'd700', page.d700,
          'd1000', page.d1000
        )
        ORDER BY page.date, page.lat, page.lon
      ),
      '[]'::jsonb
    )
  )
  INTO result
  FROM page;

  RETURN result;
END;
$$;

REVOKE ALL ON FUNCTION public.oceanembed_export_page(date, date, date, double precision, double precision, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.oceanembed_export_page(date, date, date, double precision, double precision, integer) TO anon, authenticated;
