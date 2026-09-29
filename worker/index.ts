const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] as const;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const MAX_MAP_POINTS = 1800;
const MAX_EXPORT_ROWS = 5000;

interface Env {
  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  ASSETS: {
    fetch(request: Request): Promise<Response>;
  };
}

interface Bounds {
  south: number;
  north: number;
  west: number;
  east: number;
}

class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isDate(value: unknown): value is string {
  if (typeof value !== "string" || !DATE_PATTERN.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00.000Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function isDepth(value: unknown): value is (typeof DEPTHS)[number] {
  return typeof value === "number" && DEPTHS.some((depth) => depth === value);
}

function numberField(value: unknown, label: string): number {
  if (!isFiniteNumber(value)) throw new HttpError(400, `${label} must be a finite number.`);
  return value;
}

function requireDataDate(value: unknown): asserts value is string {
  if (!isDate(value)) {
    throw new HttpError(400, "Select a valid prediction date.");
  }
}

function parseExportDates(body: Record<string, unknown>): { startDate: string; endDate: string } {
  requireDataDate(body.startDate);
  requireDataDate(body.endDate);
  const daySpan = (Date.parse(`${body.endDate}T00:00:00.000Z`) - Date.parse(`${body.startDate}T00:00:00.000Z`)) / 86_400_000;
  if (body.startDate > body.endDate || daySpan > 93) {
    throw new HttpError(400, "Choose a valid date range of up to three months.");
  }
  return { startDate: body.startDate, endDate: body.endDate };
}

function parseBounds(value: unknown): Bounds {
  if (!isRecord(value)) throw new HttpError(400, "Map bounds are required.");

  const bounds = {
    south: numberField(value.south, "south"),
    north: numberField(value.north, "north"),
    west: numberField(value.west, "west"),
    east: numberField(value.east, "east"),
  };

  const clipped = {
    south: Math.max(5, bounds.south),
    north: Math.min(30, bounds.north),
    west: Math.max(45, bounds.west),
    east: Math.min(105, bounds.east),
  };
  if (clipped.south >= clipped.north || clipped.west >= clipped.east) {
    throw new HttpError(400, "The selected map area is outside the North Indian Ocean data domain.");
  }
  return clipped;
}

async function callSupabase(env: Env, rpc: string, parameters: Record<string, unknown>): Promise<unknown> {
  if (!env.SUPABASE_URL || !env.SUPABASE_ANON_KEY) {
    throw new Error("Supabase is not configured. Set SUPABASE_ANON_KEY for the Worker.");
  }

  const baseUrl = env.SUPABASE_URL.replace(/\/+$/, "");
  const response = await fetch(`${baseUrl}/rest/v1/rpc/${rpc}`, {
    method: "POST",
    headers: {
      apikey: env.SUPABASE_ANON_KEY,
      Authorization: `Bearer ${env.SUPABASE_ANON_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(parameters),
  });

  let result: unknown;
  try {
    result = await response.json();
  } catch {
    throw new Error(`Supabase returned an unreadable response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const message = isRecord(result) && typeof result.message === "string"
      ? result.message
      : `Supabase request failed (HTTP ${response.status}).`;
    throw new Error(message);
  }
  return result;
}

function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

async function handleRequest(request: Request, env: Env): Promise<Response> {
  if (request.method === "OPTIONS") return json(null);
  if (request.method !== "POST") return json({ success: false, error: "Use POST for ocean data requests." }, 405);

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return json({ success: false, error: "Request body must be valid JSON." }, 400);
  }
  if (!isRecord(body) || typeof body.action !== "string") {
    return json({ success: false, error: "A supported request action is required." }, 400);
  }

  try {
    if (body.action === "coverage") {
      const coverage = await callSupabase(env, "oceanembed_coverage", {});
      if (
        !isRecord(coverage) ||
        !isDate(coverage.earliest) ||
        !isDate(coverage.latest) ||
        coverage.earliest > coverage.latest
      ) {
        throw new Error("Supabase returned invalid temperature date coverage.");
      }
      return json({ success: true, data: { earliest: coverage.earliest, latest: coverage.latest } });
    }

    if (body.action === "map") {
      requireDataDate(body.date);
      if (!isDepth(body.depth)) throw new HttpError(400, "Select one of the 15 modelled depth levels.");
      const bounds = parseBounds(body.bounds);
      const latCells = Math.ceil((bounds.north - bounds.south) * 4) + 1;
      const lonCells = Math.ceil((bounds.east - bounds.west) * 4) + 1;
      const stride = Math.max(1, Math.ceil(Math.sqrt((latCells * lonCells) / MAX_MAP_POINTS)));
      const map = await callSupabase(env, "oceanembed_map", {
        p_date: body.date,
        p_depth: body.depth,
        p_south: bounds.south,
        p_north: bounds.north,
        p_west: bounds.west,
        p_east: bounds.east,
        p_stride: stride,
        p_limit: MAX_MAP_POINTS,
      });
      if (!isRecord(map) || !Array.isArray(map.points) || !isFiniteNumber(map.stride) || map.stride !== stride) {
        throw new Error("Supabase returned an invalid map response.");
      }
      const points = map.points.map((point) => {
        if (!isRecord(point) || !isFiniteNumber(point.lat) || !isFiniteNumber(point.lon) || !isFiniteNumber(point.temperature)) {
          throw new Error("Supabase returned an invalid temperature grid point.");
        }
        return { lat: point.lat, lon: point.lon, temperature: point.temperature };
      });
      if (points.length > MAX_MAP_POINTS) throw new Error("Supabase returned too many map points.");
      return json({ success: true, data: { points, stride } });
    }

    if (body.action === "profile") {
      requireDataDate(body.date);
      const lat = numberField(body.lat, "Latitude");
      const lon = numberField(body.lon, "Longitude");
      if (lat < 5 || lat > 30 || lon < 45 || lon > 105) {
        throw new HttpError(400, "The selected coordinate is outside the North Indian Ocean data domain.");
      }
      const profile = await callSupabase(env, "oceanembed_profile", {
        p_date: body.date,
        p_lat: lat,
        p_lon: lon,
      });
      if (profile === null) return json({ success: true, data: { profile: null } });
      if (
        !isRecord(profile) ||
        !isRecord(profile.depths) ||
        typeof profile.date !== "string" ||
        !isDate(profile.date) ||
        !isFiniteNumber(profile.lat) ||
        !isFiniteNumber(profile.lon)
      ) {
        throw new Error("Supabase returned an invalid temperature profile.");
      }
      const depths = {} as Record<(typeof DEPTHS)[number], number>;
      for (const depth of DEPTHS) {
        const value = profile.depths[String(depth)];
        if (!isFiniteNumber(value)) throw new Error(`Supabase returned an invalid temperature at ${depth} m.`);
        depths[depth] = value;
      }
      return json({
        success: true,
        data: { profile: { date: profile.date, lat: profile.lat, lon: profile.lon, depths } },
      });
    }

    if (body.action === "export") {
      const { startDate, endDate } = parseExportDates(body);
      let afterDate: string | null = null;
      let afterLat: number | null = null;
      let afterLon: number | null = null;
      if (body.after !== null) {
        if (
          !isRecord(body.after) ||
          !isDate(body.after.date) ||
          body.after.date < startDate ||
          body.after.date > endDate ||
          !isFiniteNumber(body.after.lat) ||
          !isFiniteNumber(body.after.lon) ||
          body.after.lat < 5 ||
          body.after.lat > 30 ||
          body.after.lon < 45 ||
          body.after.lon > 105
        ) {
          throw new HttpError(400, "The download cursor is invalid.");
        }
        afterDate = body.after.date;
        afterLat = body.after.lat;
        afterLon = body.after.lon;
      }
      const page = await callSupabase(env, "oceanembed_export_page", {
        p_start_date: startDate,
        p_end_date: endDate,
        p_after_date: afterDate,
        p_after_lat: afterLat,
        p_after_lon: afterLon,
        p_limit: MAX_EXPORT_ROWS,
      });
      if (!isRecord(page) || !Array.isArray(page.rows) || page.rows.length > MAX_EXPORT_ROWS) {
        throw new Error("Supabase returned an invalid download page.");
      }
      for (const row of page.rows) {
        if (
          !isRecord(row) ||
          !isDate(row.date) ||
          row.date < startDate ||
          row.date > endDate ||
          !isFiniteNumber(row.lat) ||
          row.lat < 5 ||
          row.lat > 30 ||
          !isFiniteNumber(row.lon) ||
          row.lon < 45 ||
          row.lon > 105
        ) {
          throw new Error("Supabase returned an invalid download row.");
        }
        for (const depth of DEPTHS) {
          const temperature = row[`d${depth}`];
          if (temperature !== null && !isFiniteNumber(temperature)) {
            throw new Error(`Supabase returned an invalid temperature at ${depth} m.`);
          }
        }
      }
      return json({ success: true, data: { rows: page.rows } });
    }

    throw new HttpError(400, "Supported actions are coverage, map, profile, and export.");
  } catch (error) {
    const status = error instanceof HttpError ? error.status : 502;
    const message = error instanceof Error ? error.message : "Ocean data request failed.";
    return json({ success: false, error: message }, status);
  }
}

export default {
  fetch(request: Request, env: Env): Promise<Response> | Response {
    const url = new URL(request.url);
    if (url.pathname === "/api/ocean-query") return handleRequest(request, env);
    return env.ASSETS.fetch(request);
  },
};
