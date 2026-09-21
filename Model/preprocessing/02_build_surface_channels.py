"""
Build the 19-channel daily surface input stack, one year at a time.
Memory-safe: everything is opened with dask chunks and processed in monthly
batches, so no single operation tries to hold a full year's worth of data
in memory at once.

Channels (19 total):
  1-7:   SST, SSS, SLA, U-current, V-current, U-wind, V-wind
  8-10:  lat, lon, bathymetry  (CoordConv)
  11-12: DOY sin, DOY cos
  13-19: validity mask for each of channels 1-7
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
GRID_RES = 0.25
YEARS = list(range(2015, 2024))  # matches your downloaded span

RAW_DIR = "./data/raw"
MASK_DIR = "./data/masks"
OUTPUT_DIR = "./data/processed"

TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
TARGET_LONS = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)

SURFACE_VAR_NAMES = ["sst", "sss", "sla", "u_curr", "v_curr", "u_wind", "v_wind"]

# Candidate variable names, in priority order, per logical field. Different
# product versions/releases occasionally rename fields -- trying several
# known aliases here is more robust than assuming one exact name everywhere.
VAR_CANDIDATES = {
    "sst":    ["analysed_sst", "sst", "SST"],
    "sss":    ["sos", "so", "sss", "SSS"],
    "sla":    ["sla", "SLA"],
    "u_curr": ["u", "uo", "U", "eastward_current"],
    "v_curr": ["v", "vo", "V", "northward_current"],
    "u_wind": ["uwnd", "u10", "U10M", "eastward_wind"],
    "v_wind": ["vwnd", "v10", "V10M", "northward_wind"],
}


def standardize_coord_names(ds):
    """
    Handles two different cases seen across sources:
    1. Simple case (CMEMS products): dimension is named 'latitude', and there's
       no separate 'lat' coordinate -- a plain rename works.
    2. OSCAR/CCMP (PO.DAAC) case: dimension is named 'latitude', but a SEPARATE
       'lat' coordinate variable (indexed by that dimension) already holds the
       actual values -- renaming the dimension would collide with the existing
       'lat' variable, so we swap_dims onto the existing coordinate instead.
    """
    dim_rename = {}
    if "latitude" in ds.dims and "lat" not in ds.dims:
        if "lat" in ds.coords:
            ds = ds.swap_dims({"latitude": "lat"})
        else:
            dim_rename["latitude"] = "lat"
    if "longitude" in ds.dims and "lon" not in ds.dims:
        if "lon" in ds.coords:
            ds = ds.swap_dims({"longitude": "lon"})
        else:
            dim_rename["longitude"] = "lon"
    if dim_rename:
        ds = ds.rename(dim_rename)
    return ds


def normalize_time(ds):
    """Collapse to exactly one value per calendar day via a daily mean.

    Plain flooring to midnight is NOT enough on its own: CCMP wind is 6-hourly
    (4 timestamps/day), so flooring alone produces 4 duplicate '00:00' rows per
    day, which breaks xr.align with a 'duplicate index values' error. Resampling
    to a daily mean both fixes the duplicate-index problem AND is the physically
    correct way to combine sub-daily readings into one representative daily value
    (a real daily-mean wind, not an arbitrary pick of one of the 4 readings).
    For already-daily sources this is a safe no-op (mean of one value = itself)."""
    if "time" not in ds.coords:
        return ds

    # Resample first while the source calendar is still intact.  This handles
    # sub-daily CCMP wind data and also works with CFTimeIndex coordinates.
    ds = ds.resample(time="1D").mean()

    # The currents download uses cftime.DatetimeJulian, whereas the other
    # downloads use numpy.datetime64.  xarray does not reliably align those
    # two index types, even when their printed dates are identical.  Convert
    # all calendar objects to the same datetime64[ns] representation.
    normalized = []
    for value in ds.time.values:
        if hasattr(value, "year"):
            normalized.append(
                np.datetime64(
                    f"{value.year:04d}-{value.month:02d}-{value.day:02d}"
                    f"T{value.hour:02d}:{value.minute:02d}:{value.second:02d}",
                    "ns",
                )
            )
        else:
            normalized.append(pd.Timestamp(value).to_datetime64())

    return ds.assign_coords(
        time=np.asarray(normalized, dtype="datetime64[ns]")
    )


def find_variable(ds, field_name):
    """Resolve the actual variable name for a logical field, trying known
    aliases in order. Raises a clear, actionable error (listing what IS
    available) instead of a bare KeyError if nothing matches."""
    candidates = VAR_CANDIDATES[field_name]
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
    raise KeyError(
        f"Could not find a variable for '{field_name}'. Tried: {candidates}. "
        f"Actually available in this file: {list(ds.data_vars)}. "
        f"Add the correct name to VAR_CANDIDATES['{field_name}'] in this script."
    )


def regrid(da):
    # Some products retain a singleton depth dimension (SSS), and the
    # current product stores horizontal dimensions as (lon, lat).  Normalize
    # both cases before stacking the seven surface fields together.
    da = da.squeeze(drop=True)
    da = da.interp(lat=TARGET_LATS, lon=TARGET_LONS, method="linear")
    return da.transpose("time", "lat", "lon")


def process_year(year, land_sea_mask, bathymetry):
    print(f"\n=== Processing {year} ===")

    sst_ds = normalize_time(standardize_coord_names(xr.open_dataset(f"{RAW_DIR}/sst/SST_NIO_{year}.nc", chunks={"time": 30})))
    sss_ds = normalize_time(standardize_coord_names(xr.open_dataset(f"{RAW_DIR}/sss/SSS_NIO_{year}.nc", chunks={"time": 30})))
    ssh_ds = normalize_time(standardize_coord_names(xr.open_dataset(f"{RAW_DIR}/ssh/SSH_NIO_{year}.nc", chunks={"time": 30})))
    curr_ds = normalize_time(standardize_coord_names(xr.open_dataset(f"{RAW_DIR}/currents/CURRENTS_NIO_{year}.nc", chunks={"time": 30})))
    wind_ds = normalize_time(standardize_coord_names(xr.open_dataset(f"{RAW_DIR}/wind/WIND_NIO_{year}.nc", chunks={"time": 30})))

    print(f"  SST variables available:      {list(sst_ds.data_vars)}")
    print(f"  SSS variables available:      {list(sss_ds.data_vars)}")
    print(f"  SSH variables available:      {list(ssh_ds.data_vars)}")
    print(f"  Currents variables available: {list(curr_ds.data_vars)}")
    print(f"  Wind variables available:     {list(wind_ds.data_vars)}")

    sst_raw = find_variable(sst_ds, "sst")
    sss_raw = find_variable(sss_ds, "sss")
    sla_raw = find_variable(ssh_ds, "sla")
    u_curr_raw = find_variable(curr_ds, "u_curr")
    v_curr_raw = find_variable(curr_ds, "v_curr")
    u_wind_raw = find_variable(wind_ds, "u_wind")
    v_wind_raw = find_variable(wind_ds, "v_wind")

    # SST unit check: only convert Kelvin->Celsius if it actually looks like Kelvin
    # (avoids silently mis-converting if a product already delivers Celsius)
    sst_sample_val = float(sst_raw.isel(time=0).mean(skipna=True).values) if "time" in sst_raw.dims else float(sst_raw.mean(skipna=True).values)
    if sst_sample_val > 100:
        print(f"  SST looks like Kelvin (sample mean {sst_sample_val:.1f}) -- converting to Celsius")
        sst_raw = sst_raw - 273.15
    else:
        print(f"  SST looks like Celsius already (sample mean {sst_sample_val:.1f}) -- no conversion applied")

    sst = regrid(sst_raw)
    sss = regrid(sss_raw)
    sla = regrid(sla_raw)
    u_curr = regrid(u_curr_raw)
    v_curr = regrid(v_curr_raw)
    u_wind = regrid(u_wind_raw)
    v_wind = regrid(v_wind_raw)

    surface_vars = [sst, sss, sla, u_curr, v_curr, u_wind, v_wind]

    print(f"  Time range per source (after flooring to date):")
    for name, v in zip(SURFACE_VAR_NAMES, surface_vars):
        t = v.time.values
        print(f"    {name}: {len(t)} days, {t.min()} to {t.max()}")

    # Align all variables to a common time axis (inner join keeps only days
    # present in ALL sources). Now safe since all timestamps are floored to
    # midnight, so a real calendar-day match will actually be found.
    aligned = xr.align(*surface_vars, join="inner")
    sst, sss, sla, u_curr, v_curr, u_wind, v_wind = aligned
    n_days = sst.sizes["time"]
    print(f"  Aligned to {n_days} common days across all 5 sources")

    if n_days == 0:
        raise RuntimeError(
            f"Zero overlapping days for {year} even after time-flooring. "
            f"Check the per-source date ranges printed above for a real gap "
            f"(e.g. one source missing this year entirely, or a timezone offset "
            f"large enough to shift the calendar date, not just the time-of-day)."
        )

    # Build validity masks BEFORE filling NaNs
    validity_masks = [v.notnull().astype(np.float32) for v in [sst, sss, sla, u_curr, v_curr, u_wind, v_wind]]

    # Fill NaN with 0, then zero out land explicitly via the precomputed land-sea mask
    filled_vars = [v.fillna(0) * land_sea_mask for v in [sst, sss, sla, u_curr, v_curr, u_wind, v_wind]]

    # CoordConv channels
    lat_grid, lon_grid = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")

    # Seasonal channels
    time_index = sst.time.to_index()
    doy = time_index.dayofyear.values
    angle = (doy / 365.25) * 2 * np.pi
    doy_sin, doy_cos = np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/processed_{year}.nc"

    all_days_stacked = []
    batch_size = 30
    for start in range(0, n_days, batch_size):
        end = min(start + batch_size, n_days)
        print(f"  Processing days {start}-{end} of {n_days}...")

        batch_vars = [v.isel(time=slice(start, end)).compute() for v in filled_vars]
        batch_masks = [m.isel(time=slice(start, end)).compute() for m in validity_masks]
        batch_len = end - start

        lat_ch = np.broadcast_to(lat_grid, (batch_len,) + lat_grid.shape).astype(np.float32)
        lon_ch = np.broadcast_to(lon_grid, (batch_len,) + lon_grid.shape).astype(np.float32)
        bathy_ch = np.broadcast_to(bathymetry.values, (batch_len,) + bathymetry.shape).astype(np.float32)
        doy_sin_ch = doy_sin[start:end, None, None] * np.ones((batch_len,) + lat_grid.shape, dtype=np.float32)
        doy_cos_ch = doy_cos[start:end, None, None] * np.ones((batch_len,) + lat_grid.shape, dtype=np.float32)

        stacked = np.stack(
            [v.values for v in batch_vars] + [lat_ch, lon_ch, bathy_ch, doy_sin_ch, doy_cos_ch] +
            [m.values for m in batch_masks],
            axis=1
        )  # [batch_len, 19, H, W]
        all_days_stacked.append(stacked)

    full_year_stack = np.concatenate(all_days_stacked, axis=0)  # [n_days, 19, H, W]

    channel_names = SURFACE_VAR_NAMES + ["lat", "lon", "bathy", "doy_sin", "doy_cos"] + \
                     [f"valid_{v}" for v in SURFACE_VAR_NAMES]

    out_ds = xr.Dataset(
        {"channels": (("time", "channel", "lat", "lon"), full_year_stack)},
        coords={"time": sst.time.values, "lat": TARGET_LATS, "lon": TARGET_LONS, "channel": channel_names}
    )
    out_ds.to_netcdf(out_path)
    print(f"  Saved {out_path}  (shape: {full_year_stack.shape})")

    for ds in [sst_ds, sss_ds, ssh_ds, curr_ds, wind_ds]:
        ds.close()


def main():
    print("Loading precomputed land-sea mask and bathymetry (run 01_build_bathymetry_and_masks.py first if missing)...")
    land_sea_mask = xr.open_dataset(f"{MASK_DIR}/land_sea_mask.nc")["mask"]
    bathymetry = xr.open_dataset(f"{MASK_DIR}/bathymetry.nc")["bathymetry"]

    for year in YEARS:
        out_path = f"{OUTPUT_DIR}/processed_{year}.nc"
        if os.path.exists(out_path):
            print(f"Skipping {year}, already processed.")
            continue
        process_year(year, land_sea_mask, bathymetry)

    print("\nAll years processed.")


if __name__ == "__main__":
    main()
