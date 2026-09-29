export const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] as const;

export type Depth = (typeof DEPTHS)[number];

export interface Coverage {
  earliest: string;
  latest: string;
}

export interface MapBounds {
  south: number;
  north: number;
  west: number;
  east: number;
}

export interface MapPoint {
  lat: number;
  lon: number;
  temperature: number;
}

export interface Profile {
  date: string;
  lat: number;
  lon: number;
  depths: Record<Depth, number>;
}

export interface HistoryPoint {
  date: string;
  depths: Record<Depth, number>;
}

export interface MapResponse {
  points: MapPoint[];
  stride: number;
}
