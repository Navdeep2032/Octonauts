import { DEPTHS, type Coverage, type Depth, type MapBounds, type MapPoint, type MapResponse, type Profile } from "./types";

const apiUrl = import.meta.env.VITE_OCEAN_API_URL || "/api/ocean-query";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function parseCoverage(value: unknown): Coverage {
  if (
    !isRecord(value) ||
    typeof value.earliest !== "string" ||
    typeof value.latest !== "string" ||
    !/^\d{4}-\d{2}-\d{2}$/.test(value.earliest) ||
    !/^\d{4}-\d{2}-\d{2}$/.test(value.latest) ||
    value.earliest > value.latest
  ) {
    throw new Error("Ocean data service returned invalid coverage information.");
  }
  return { earliest: value.earliest, latest: value.latest };
}

function parseMap(value: unknown): MapResponse {
  if (
    !isRecord(value) ||
    !Array.isArray(value.points) ||
    !isFiniteNumber(value.stride) ||
    !Number.isInteger(value.stride) ||
    value.stride < 1
  ) {
    throw new Error("Ocean data service returned an invalid map response.");
  }
  const points: MapPoint[] = value.points.map((point) => {
    if (
      !isRecord(point) ||
      !isFiniteNumber(point.lat) ||
      !isFiniteNumber(point.lon) ||
      !isFiniteNumber(point.temperature)
    ) {
      throw new Error("Ocean data service returned an invalid map grid point.");
    }
    return { lat: point.lat, lon: point.lon, temperature: point.temperature };
  });
  return { points, stride: value.stride };
}

function parseProfileResponse(value: unknown): { profile: Profile | null } {
  if (!isRecord(value) || !("profile" in value)) {
    throw new Error("Ocean data service returned an invalid profile response.");
  }
  if (value.profile === null) return { profile: null };
  const profile = value.profile;
  if (!isRecord(profile) || !isRecord(profile.depths)) {
    throw new Error("Ocean data service returned an invalid temperature profile.");
  }

  const depthValues = profile.depths;
  if (
    typeof profile.date !== "string" ||
    !isFiniteNumber(profile.lat) ||
    !isFiniteNumber(profile.lon) ||
    !DEPTHS.every((depth) => isFiniteNumber(depthValues[depth]))
  ) {
    throw new Error("Ocean data service returned an incomplete temperature profile.");
  }

  const depths = {} as Record<Depth, number>;
  for (const depth of DEPTHS) depths[depth] = depthValues[depth] as number;
  return {
    profile: { date: profile.date, lat: profile.lat, lon: profile.lon, depths },
  };
}

async function request<T>(
  payload: Record<string, unknown>,
  parse: (data: unknown) => T,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  let result: unknown;
  try {
    result = await response.json();
  } catch {
    throw new Error(`Ocean data service returned an unreadable response (HTTP ${response.status}).`);
  }

  if (!isRecord(result) || typeof result.success !== "boolean") {
    throw new Error(`Ocean data service returned an invalid response (HTTP ${response.status}).`);
  }
  if (!response.ok || !result.success) {
    const error = typeof result.error === "string" ? result.error : `Ocean data service returned HTTP ${response.status}.`;
    throw new Error(error);
  }
  if (!("data" in result)) throw new Error("Ocean data service returned a response without data.");

  return parse(result.data);
}

export function getCoverage(signal?: AbortSignal): Promise<Coverage> {
  return request({ action: "coverage" }, parseCoverage, signal);
}

export function getMapPoints(
  date: string,
  depth: Depth,
  bounds: MapBounds,
  signal?: AbortSignal,
): Promise<MapResponse> {
  return request({ action: "map", date, depth, bounds }, parseMap, signal);
}

export async function getProfile(
  date: string,
  lat: number,
  lon: number,
  signal?: AbortSignal,
): Promise<Profile | null> {
  const result = await request({ action: "profile", date, lat, lon }, parseProfileResponse, signal);
  return result.profile;
}
