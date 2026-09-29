import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import { divIcon } from "leaflet";
import {
  CircleMarker,
  MapContainer,
  Marker,
  Rectangle,
  TileLayer,
  Tooltip as MapTooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import { getCoverage, getMapPoints, getProfile } from "./api";
import {
  DEPTHS,
  type Coverage,
  type Depth,
  type MapBounds,
  type MapPoint,
  type Profile,
} from "./types";

const DOMAIN = { south: 5, north: 30, west: 45, east: 105 } as const;
const INITIAL_BOUNDS: MapBounds = DOMAIN;
const DOMAIN_BOUNDS: [[number, number], [number, number]] = [
  [DOMAIN.south, DOMAIN.west],
  [DOMAIN.north, DOMAIN.east],
];

const MAP_LABELS = [
  { name: "SOMALIA", lat: 9, lon: 47.5, kind: "country" },
  { name: "YEMEN", lat: 15, lon: 48.5, kind: "country" },
  { name: "OMAN", lat: 21.5, lon: 57.5, kind: "country" },
  { name: "PAKISTAN", lat: 27, lon: 67, kind: "country" },
  { name: "INDIA", lat: 21, lon: 78, kind: "country" },
  { name: "SRI LANKA", lat: 7.5, lon: 80.8, kind: "country" },
  { name: "BANGLADESH", lat: 24.2, lon: 90.5, kind: "country" },
  { name: "MYANMAR", lat: 20, lon: 95, kind: "country" },
  { name: "THAILAND", lat: 14, lon: 101.2, kind: "country" },
  { name: "MALAYSIA", lat: 6.7, lon: 101.5, kind: "country" },
  { name: "INDIAN OCEAN", lat: 7.7, lon: 70, kind: "ocean" },
  { name: "ARABIAN SEA", lat: 15.3, lon: 64.5, kind: "ocean" },
  { name: "BAY OF BENGAL", lat: 14, lon: 87.2, kind: "ocean" },
  { name: "ANDAMAN SEA", lat: 10, lon: 96.5, kind: "ocean" },
] as const;

interface GradientRow {
  depth: number;
  gradient: number;
  value: number;
  unit: string;
}

interface ChartTooltipPoint {
  depth: number;
  value: number;
  unit: string;
}

interface KeyMetric {
  label: string;
  value: string;
}

function isChartTooltipPoint(value: unknown): value is ChartTooltipPoint {
  return (
    typeof value === "object" &&
    value !== null &&
    "depth" in value &&
    typeof value.depth === "number" &&
    "value" in value &&
    typeof value.value === "number" &&
    "unit" in value &&
    typeof value.unit === "string"
  );
}

function ChartTooltip({ active, payload }: TooltipContentProps) {
  const point = payload[0]?.payload;
  if (!active || !isChartTooltipPoint(point)) return null;

  return (
    <div className="chart-tooltip">
      <strong>{point.depth.toFixed(1)} m depth</strong>
      <span>{point.value.toFixed(3)} {point.unit}</span>
    </div>
  );
}

function ChartKeyData({ title, metrics, accent }: { title: string; metrics: KeyMetric[]; accent: string }) {
  return (
    <aside className="chart-key-data" style={{ "--metric-accent": accent } as CSSProperties}>
      <h4>KEY DATA</h4>
      <dl>
        {metrics.map(({ label, value }) => (
          <div className="chart-key-metric" key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <span className="chart-key-caption">{title}</span>
    </aside>
  );
}

function temperatureColor(value: number, minimum: number, maximum: number): string {
  const ratio = maximum === minimum ? 0.5 : (value - minimum) / (maximum - minimum);
  const stops = [
    { at: 0, color: [48, 111, 173] },
    { at: 0.5, color: [68, 184, 177] },
    { at: 1, color: [238, 148, 77] },
  ];
  const first = ratio < 0.5 ? stops[0] : stops[1];
  const second = ratio < 0.5 ? stops[1] : stops[2];
  const amount = (ratio - first.at) / (second.at - first.at);
  const color = first.color.map((channel, index) =>
    Math.round(channel + (second.color[index] - channel) * amount),
  );
  return `rgb(${color.join(",")})`;
}

function MapViewport({ onChange }: { onChange: (bounds: MapBounds) => void }) {
  const map = useMap();
  const minZoom = useRef<number | null>(null);

  useEffect(() => {
    const fitDomain = () => {
      map.invalidateSize({ pan: false });
      const nextMinZoom = map.getBoundsZoom(DOMAIN_BOUNDS, false);
      const previousMinZoom = minZoom.current;
      const wasAtMinimum = previousMinZoom === null || map.getZoom() <= previousMinZoom;
      minZoom.current = nextMinZoom;
      map.setMinZoom(nextMinZoom);

      if (wasAtMinimum) {
        map.fitBounds(DOMAIN_BOUNDS, { animate: false });
      } else if (map.getZoom() < nextMinZoom) {
        map.setZoom(nextMinZoom, { animate: false });
      }
    };

    fitDomain();
    const resizeObserver = new ResizeObserver(fitDomain);
    resizeObserver.observe(map.getContainer());
    return () => resizeObserver.disconnect();
  }, [map]);

  useMapEvents({
    moveend() {
      const bounds = map.getBounds();
      onChange({
        south: Math.max(DOMAIN.south, bounds.getSouth()),
        north: Math.min(DOMAIN.north, bounds.getNorth()),
        west: Math.max(DOMAIN.west, bounds.getWest()),
        east: Math.min(DOMAIN.east, bounds.getEast()),
      });
    },
  });

  useEffect(() => {
    const bounds = map.getBounds();
    onChange({
      south: Math.max(DOMAIN.south, bounds.getSouth()),
      north: Math.min(DOMAIN.north, bounds.getNorth()),
      west: Math.max(DOMAIN.west, bounds.getWest()),
      east: Math.min(DOMAIN.east, bounds.getEast()),
    });
  }, [map, onChange]);

  return null;
}

function MapLabels() {
  return (
    <>
      {MAP_LABELS.map(({ name, lat, lon, kind }) => (
        <Marker
          key={name}
          position={[lat, lon]}
          icon={divIcon({
            className: `map-label-icon map-label-${kind}`,
            html: `<span>${name}</span>`,
            iconSize: [0, 0],
          })}
          interactive={false}
          keyboard={false}
        />
      ))}
    </>
  );
}

function TemperatureMap({
  points,
  loading,
  depth,
  onBoundsChange,
  onSelectPoint,
}: {
  points: MapPoint[];
  loading: boolean;
  depth: Depth;
  onBoundsChange: (bounds: MapBounds) => void;
  onSelectPoint: (lat: number, lon: number) => void;
}) {
  const minimum = points.length ? Math.min(...points.map((point) => point.temperature)) : 0;
  const maximum = points.length ? Math.max(...points.map((point) => point.temperature)) : 1;

  return (
    <div className="map-frame">
      <MapContainer
        center={[17.5, 75]}
        zoom={3}
        minZoom={2}
        maxZoom={9}
        maxBounds={DOMAIN_BOUNDS}
        maxBoundsViscosity={1}
        scrollWheelZoom
        className="temperature-map"
      >
        <MapLabels />
        <Rectangle
          bounds={DOMAIN_BOUNDS}
          pathOptions={{ color: "#287f78", weight: 1.5, opacity: 0.8, fill: false, dashArray: "5 4" }}
          interactive={false}
        />
        <TileLayer
          attribution='Tiles &copy; <a href="https://www.esri.com/">Esri</a> &mdash; Sources: GEBCO, NOAA, National Geographic, DeLorme, HERE, and other contributors'
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}"
        />
        <MapViewport onChange={onBoundsChange} />
        {points.map((point) => (
          <CircleMarker
            key={`${point.lat}-${point.lon}`}
            center={[point.lat, point.lon]}
            radius={4}
            pathOptions={{
              color: "rgba(255,255,255,.78)",
              weight: 0.8,
              fillColor: temperatureColor(point.temperature, minimum, maximum),
              fillOpacity: 0.88,
            }}
            eventHandlers={{ click: () => onSelectPoint(point.lat, point.lon) }}
          >
            <MapTooltip direction="top">{point.temperature.toFixed(2)}°C at {depth} m</MapTooltip>
          </CircleMarker>
        ))}
      </MapContainer>
      <div className="map-overlay map-hint">Pan and zoom to explore · select a point for its depth profile</div>
      {(loading || points.length === 0) && (
        <div className={`map-status ${loading ? "" : "map-status-empty"}`}>
          {loading ? "Loading regional temperatures…" : "No ocean data in this view"}
        </div>
      )}
      <div className="map-overlay map-scale" aria-label="Temperature color scale">
        <span>{points.length ? `${minimum.toFixed(1)}°` : "—"}</span>
        <span className="scale-gradient" />
        <span>{points.length ? `${maximum.toFixed(1)}°` : "—"}</span>
      </div>
    </div>
  );
}

function TemperatureChart({ profile }: { profile: Profile }) {
  const temperatureRows = DEPTHS.map((depth) => ({
    depth,
    temperature: profile.depths[depth],
    value: profile.depths[depth],
    unit: "°C",
  }));
  const gradientRows: GradientRow[] = DEPTHS.slice(0, -1).map((depth, index) => {
    const nextDepth = DEPTHS[index + 1];
    const temperatureChange = profile.depths[nextDepth] - profile.depths[depth];
    const gradient = (temperatureChange / (nextDepth - depth)) * 100;
    return {
      depth: (depth + nextDepth) / 2,
      gradient,
      value: gradient,
      unit: "°C / 100 m",
    };
  });
  const temperatures = DEPTHS.map((depth) => profile.depths[depth]);
  const coldest = Math.min(...temperatures);
  const warmest = Math.max(...temperatures);
  const strongestWarming = gradientRows.reduce((strongest, row) =>
    row.gradient > strongest.gradient ? row : strongest,
  );
  const strongestCooling = gradientRows.reduce((strongest, row) =>
    row.gradient < strongest.gradient ? row : strongest,
  );
  const steepestChange = gradientRows.reduce((strongest, row) =>
    Math.abs(row.gradient) > Math.abs(strongest.gradient) ? row : strongest,
  );
  const meanAbsGradient =
    gradientRows.reduce((total, row) => total + Math.abs(row.gradient), 0) / gradientRows.length;

  return (
    <div className="charts-grid">
      <section className="chart-card">
        <div className="chart-title">
          <div>
            <h3>Temperature profile</h3>
            <p>Predicted temperature at all 15 standard depths</p>
          </div>
          <span className="chart-unit">°C</span>
        </div>
        <div className="chart-card-content">
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 12, right: 22, left: 2, bottom: 8 }}>
                <CartesianGrid stroke="#e9efed" strokeDasharray="3 5" />
                <XAxis
                  type="number"
                  dataKey="temperature"
                  domain={["dataMin - 1", "dataMax + 1"]}
                  tick={{ fill: "#788784", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "Temperature (°C)", position: "insideBottom", offset: -2, fill: "#788784", fontSize: 11 }}
                  name="Temperature"
                />
                <YAxis
                  type="number"
                  dataKey="depth"
                  reversed
                  domain={[0, 1000]}
                  ticks={[0, 100, 200, 500, 1000]}
                  tick={{ fill: "#788784", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "Depth (m)", angle: -90, position: "insideLeft", fill: "#788784", fontSize: 11 }}
                  name="Depth"
                />
                <Tooltip content={ChartTooltip} />
                <Scatter
                  data={temperatureRows}
                  dataKey="temperature"
                  fill="#137c78"
                  line={{ stroke: "#137c78", strokeWidth: 2.5 }}
                  lineType="joint"
                  name="Temperature"
                  shape="circle"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <ChartKeyData
            title={profile.date}
            accent="#137c78"
            metrics={[
              { label: "Surface", value: `${profile.depths[0].toFixed(2)} °C` },
              { label: "At 1,000 m", value: `${profile.depths[1000].toFixed(2)} °C` },
              { label: "Column range", value: `${(warmest - coldest).toFixed(2)} °C` },
              { label: "Warmest", value: `${warmest.toFixed(2)} °C` },
              { label: "Coldest", value: `${coldest.toFixed(2)} °C` },
            ]}
          />
        </div>
      </section>

      <section className="chart-card">
        <div className="chart-title">
          <div>
            <h3>Temperature gradient</h3>
            <p>Change between adjacent model depths</p>
          </div>
          <span className="chart-unit">°C / 100 m</span>
        </div>
        <div className="chart-card-content">
          <div className="chart">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 12, right: 22, left: 2, bottom: 8 }}>
                <CartesianGrid stroke="#e9efed" strokeDasharray="3 5" />
                <XAxis
                  type="number"
                  dataKey="gradient"
                  domain={["auto", "auto"]}
                  tick={{ fill: "#788784", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "°C / 100 m", position: "insideBottom", offset: -2, fill: "#788784", fontSize: 11 }}
                  name="Temperature gradient"
                />
                <YAxis
                  type="number"
                  dataKey="depth"
                  reversed
                  domain={[0, 1000]}
                  ticks={[0, 100, 200, 500, 1000]}
                  tick={{ fill: "#788784", fontSize: 11 }}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "Depth (m)", angle: -90, position: "insideLeft", fill: "#788784", fontSize: 11 }}
                  name="Depth"
                />
                <Tooltip content={ChartTooltip} />
                <Scatter
                  data={gradientRows}
                  dataKey="gradient"
                  fill="#db8b50"
                  line={{ stroke: "#db8b50", strokeWidth: 2.5 }}
                  lineType="joint"
                  name="Gradient"
                  shape="circle"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <ChartKeyData
            title={profile.date}
            accent="#db8b50"
            metrics={[
              { label: "Strongest warming", value: `${strongestWarming.gradient.toFixed(3)} °C / 100 m` },
              { label: "Strongest cooling", value: `${strongestCooling.gradient.toFixed(3)} °C / 100 m` },
              { label: "Steepest change", value: `${Math.abs(steepestChange.gradient).toFixed(3)} °C / 100 m` },
              { label: "At depth", value: `${steepestChange.depth} m midpoint` },
              { label: "Mean |gradient|", value: `${meanAbsGradient.toFixed(3)} °C / 100 m` },
            ]}
          />
        </div>
      </section>
    </div>
  );
}

function App() {
  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const [date, setDate] = useState("");
  const [depth, setDepth] = useState<Depth>(0);
  const [bounds, setBounds] = useState<MapBounds>(INITIAL_BOUNDS);
  const [points, setPoints] = useState<MapPoint[]>([]);
  const [mapStride, setMapStride] = useState(1);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loadingCoverage, setLoadingCoverage] = useState(true);
  const [loadingMap, setLoadingMap] = useState(false);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [error, setError] = useState("");
  const [profileError, setProfileError] = useState("");
  const profileRequest = useRef<AbortController | null>(null);
  const hasDate = useMemo(
    () => Boolean(date && coverage && date >= coverage.earliest && date <= coverage.latest),
    [coverage, date],
  );

  useEffect(() => {
    const controller = new AbortController();
    getCoverage(controller.signal)
      .then((data) => {
        setCoverage(data);
        setDate(data.latest);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "Unable to load the available data dates.");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingCoverage(false);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!hasDate) {
      setPoints([]);
      setMapStride(1);
      setLoadingMap(false);
      setError("");
      return;
    }

    const controller = new AbortController();
    setLoadingMap(true);
    setError("");
    const timeout = window.setTimeout(() => {
      getMapPoints(date, depth, bounds, controller.signal)
        .then((result) => {
          setPoints(result.points);
          setMapStride(result.stride);
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) {
            setError(reason instanceof Error ? reason.message : "Unable to load map temperatures.");
            setPoints([]);
            setMapStride(1);
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoadingMap(false);
        });
    }, 250);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [bounds, date, depth, hasDate]);

  const handleBoundsChange = useCallback((next: MapBounds) => {
    setBounds((current) => {
      const unchanged = (Object.keys(next) as (keyof MapBounds)[]).every(
        (key) => Math.abs(current[key] - next[key]) < 0.0001,
      );
      return unchanged ? current : next;
    });
  }, []);

  const handlePointSelect = useCallback(
    (lat: number, lon: number) => {
      if (!hasDate) return;
      const selectedLat = Math.round((lat - DOMAIN.south) * 4) / 4 + DOMAIN.south;
      const selectedLon = Math.round((lon - DOMAIN.west) * 4) / 4 + DOMAIN.west;
      profileRequest.current?.abort();
      const controller = new AbortController();
      profileRequest.current = controller;
      setProfile(null);
      setLoadingProfile(true);
      setProfileError("");
      getProfile(date, selectedLat, selectedLon, controller.signal)
        .then((result) => {
          if (controller.signal.aborted) return;
          if (result) setProfile(result);
          else setProfileError("No depth profile is available at this location for the selected day.");
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) {
            setProfileError(reason instanceof Error ? reason.message : "Unable to load the selected depth profile.");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) {
            profileRequest.current = null;
            setLoadingProfile(false);
          }
        });
    },
    [date, hasDate],
  );

  useEffect(() => {
    profileRequest.current?.abort();
    profileRequest.current = null;
    setLoadingProfile(false);
    setProfile(null);
    setProfileError("");
  }, [date]);

  useEffect(
    () => () => {
      profileRequest.current?.abort();
    },
    [],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="OceanEmbed home">
          <span className="brand-mark"><span /></span>
          <span className="brand-name">ocean<span>embed</span></span>
        </a>
        <div className="topbar-meta">
          <span className="live-dot" />
          Daily model output
          <span className="meta-divider" />
          North Indian Ocean
        </div>
      </header>

      <section className="page-heading">
        <div>
          <div className="eyebrow">OCEAN INTELLIGENCE · MODEL EXPLORER</div>
          <h1>See beneath the surface.</h1>
          <p>Explore all available daily modelled temperatures across the North Indian Ocean water column.</p>
        </div>
        <div className="coverage-badge">
          <span className={`coverage-indicator ${loadingCoverage ? "pending" : coverage ? "" : "offline"}`} />
          <span>
            {loadingCoverage
              ? "Connecting to data"
              : coverage
                ? `${coverage.earliest} — ${coverage.latest}`
                : "Data unavailable"}
          </span>
        </div>
      </section>

      <section className="control-bar" aria-label="Map controls">
        <label className="date-control">
          <span className="control-label">PREDICTION DATE</span>
          <span className="date-input-wrap">
            <span className="control-icon" aria-hidden="true">◷</span>
            <input
              type="date"
              value={date}
              min={coverage?.earliest}
              max={coverage?.latest}
              onChange={(event) => setDate(event.target.value)}
              disabled={loadingCoverage || !coverage}
              aria-label="Prediction date"
            />
          </span>
        </label>
        <span className="control-separator" />
        <label className="depth-control">
          <span className="control-label">MAP LAYER</span>
          <span className="depth-select-wrap">
            <span className="depth-glyph" aria-hidden="true">◉</span>
            <select
              value={depth}
              onChange={(event) => {
                const selected = DEPTHS.find((level) => level === Number(event.target.value));
                if (selected !== undefined) setDepth(selected);
              }}
            >
              {DEPTHS.map((level) => (
                <option value={level} key={level}>{level === 0 ? "Surface (0 m)" : `${level} m depth`}</option>
              ))}
            </select>
            <span className="select-chevron" aria-hidden="true">⌄</span>
          </span>
        </label>
        <div className="control-spacer" />
        <div className="region-label"><span className="region-dot" /> 5°–30°N <span className="region-slash">/</span> 45°–105°E</div>
      </section>

      {error && <div className="error-banner" role="alert"><strong>Data request failed</strong><span>{error}</span></div>}
      {date && coverage && !hasDate && (
        <div className="notice-banner" role="status">Choose a date from {coverage.earliest} through {coverage.latest}.</div>
      )}

      <section className="map-section">
        <div className="section-heading">
          <div>
            <h2>Regional temperature</h2>
            <p>Each dot represents a modelled ocean grid cell · temperatures in °C</p>
          </div>
          <div className="map-point-count">
            <span className="count-pulse" />
            {loadingMap
              ? "Updating map"
              : `${points.length.toLocaleString()} ${mapStride > 1 ? "sampled points" : "grid points"}`}
          </div>
        </div>
        <TemperatureMap
          points={points}
          loading={loadingMap}
          depth={depth}
          onBoundsChange={handleBoundsChange}
          onSelectPoint={handlePointSelect}
        />
      </section>

      <section className="profile-section">
        <div className="section-heading profile-heading">
          <div>
            <div className="eyebrow">POINT PROFILE</div>
            <h2>{profile ? "Temperature through the water column" : "Choose a location on the map"}</h2>
            <p>
              {profile
                ? `${profile.lat.toFixed(2)}°N, ${profile.lon.toFixed(2)}°E · ${profile.date}`
                : "Select any ocean grid point to explore its full-depth model output."}
            </p>
          </div>
          {profile && <div className="profile-location"><span>SELECTED LOCATION</span><strong>{profile.lat.toFixed(2)}°N&nbsp; {profile.lon.toFixed(2)}°E</strong></div>}
        </div>
        {profileError && <div className="error-banner profile-error" role="alert">{profileError}</div>}
        {loadingProfile && <div className="profile-loading" role="status">Loading the selected location’s depth profile…</div>}
        {profile ? (
          <TemperatureChart profile={profile} />
        ) : (
          <div className="empty-profile">
            <div className="empty-profile-mark">↓</div>
            <span>The vertical temperature profile will appear here</span>
          </div>
        )}
      </section>

      <footer className="footer">
        <span>OCEANEMBED <span className="footer-separator">·</span> SUBSURFACE TEMPERATURE EXPLORER</span>
        <span>Model output · 15 standard depths · 0–1,000 m</span>
      </footer>
    </main>
  );
}

export default App;
