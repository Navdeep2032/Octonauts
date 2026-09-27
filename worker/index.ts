const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] as const;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const MAX_MAP_POINTS = 1800;

interface Env {
  DB: D1Database;
  ASSETS: {
    fetch(request: Request): Promise<Response>;
  };
}

interface D1PreparedStatement {
  bind(...values: (string | number)[]): D1PreparedStatement;
  all<T>(): Promise<D1QueryResult<T>>;
}

interface D1Database {
  prepare(query: string): D1PreparedStatement;
}

interface D1QueryResult<T> {
  success: boolean;
  results: T[];
  error?: string;
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

function isDate(value: unknown): value is string {
  if (typeof value !== "string" || !DATE_PATTERN.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00.000Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}

function isDepth(value: unknown): value is (typeof DEPTHS)[number] {
  return typeof value === "number" && DEPTHS.some((depth) => depth === value);
}

function numberField(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new HttpError(400, `${label} must be a finite number.`);
  }
  return value;
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

async function queryD1(env: Env, sql: string, params: (string | number)[]): Promise<Record<string, unknown>[]> {
  const statement = env.DB.prepare(sql);
  const result = params.length
    ? await statement.bind(...params).all<Record<string, unknown>>()
    : await statement.all<Record<string, unknown>>();

  if (!result.success || !Array.isArray(result.results) || !result.results.every(isRecord)) {
    throw new Error(result.error || "Cloudflare D1 returned an invalid query result.");
  }
  return result.results;
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
      const rows = await queryD1(
        env,
        "SELECT DISTINCT date FROM temperatures WHERE date IS NOT NULL ORDER BY date",
        [],
      );
      const dates = rows
        .map((row) => row.date)
        .filter((value): value is string => typeof value === "string" && isDate(value));
      if (dates.length === 0) throw new Error("Cloudflare D1 contains no dated temperature outputs.");
      return json({ success: true, data: { dates, earliest: dates[0], latest: dates[dates.length - 1] } });
    }

    if (body.action === "map") {
      if (!isDate(body.date)) throw new HttpError(400, "Select a valid prediction date in YYYY-MM-DD format.");
      if (!isDepth(body.depth)) throw new HttpError(400, "Select one of the 15 modelled depth levels.");
      const bounds = parseBounds(body.bounds);
      const latCells = Math.ceil((bounds.north - bounds.south) * 4) + 1;
      const lonCells = Math.ceil((bounds.east - bounds.west) * 4) + 1;
      const stride = Math.max(1, Math.ceil(Math.sqrt((latCells * lonCells) / MAX_MAP_POINTS)));
      const depthColumn = `d${body.depth}`;
      const rows = await queryD1(
        env,
        `SELECT lat, lon, ${depthColumn} AS temperature FROM temperatures
         WHERE date = ? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?
           AND d0 > 0
           AND CAST(ROUND((lat - 5.0) * 4) AS INTEGER) % ? = 0
           AND CAST(ROUND((lon - 45.0) * 4) AS INTEGER) % ? = 0
         ORDER BY lat, lon LIMIT ${MAX_MAP_POINTS}`,
        [body.date, bounds.south, bounds.north, bounds.west, bounds.east, stride, stride],
      );
      const points = rows.map((row) => {
        if (
          typeof row.lat !== "number" ||
          typeof row.lon !== "number" ||
          typeof row.temperature !== "number" ||
          !Number.isFinite(row.temperature)
        ) {
          throw new Error("Cloudflare D1 returned an invalid temperature grid point.");
        }
        return { lat: row.lat, lon: row.lon, temperature: row.temperature };
      });
      return json({ success: true, data: { points, stride } });
    }

    if (body.action === "profile") {
      if (!isDate(body.date)) throw new HttpError(400, "Select a valid prediction date in YYYY-MM-DD format.");
      const lat = numberField(body.lat, "Latitude");
      const lon = numberField(body.lon, "Longitude");
      if (lat < 5 || lat > 30 || lon < 45 || lon > 105) {
        throw new HttpError(400, "The selected coordinate is outside the North Indian Ocean data domain.");
      }
      const columns = DEPTHS.map((depth) => `d${depth}`).join(", ");
      const rows = await queryD1(
        env,
        `SELECT date, lat, lon, ${columns} FROM temperatures
         WHERE date = ? AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? AND d0 > 0
         ORDER BY ABS(lat - ?) + ABS(lon - ?) LIMIT 1`,
        [body.date, lat - 0.126, lat + 0.126, lon - 0.126, lon + 0.126, lat, lon],
      );
      if (rows.length === 0) return json({ success: true, data: { profile: null } });

      const row = rows[0];
      const rowDate = row.date;
      if (typeof rowDate !== "string" || !isDate(rowDate) || typeof row.lat !== "number" || typeof row.lon !== "number") {
        throw new Error("Cloudflare D1 returned an invalid temperature profile.");
      }
      const depths = {} as Record<(typeof DEPTHS)[number], number>;
      for (const depth of DEPTHS) {
        const value = row[`d${depth}`];
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new Error(`Cloudflare D1 returned an invalid temperature at ${depth} m.`);
        }
        depths[depth] = value;
      }
      return json({ success: true, data: { profile: { date: rowDate, lat: row.lat, lon: row.lon, depths } } });
    }

    throw new HttpError(400, "Supported actions are coverage, map, and profile.");
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
