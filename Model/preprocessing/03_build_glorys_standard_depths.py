"""
Regrid GLORYS subsurface temperature onto the common 0.25deg spatial grid AND
interpolate its native (irregular) depth levels onto the 15 standard depths.

Memory-safe: processes in monthly batches, using dask-backed lazy arrays until
each small batch is explicitly computed -- avoids the 21GB allocation error
from loading a full year of multi-level GLORYS data at once.

Only covers 2015-2020 (Stage 2 labeled years) -- 2021-2023 pretraining data
is surface-only by design and was never meant to have GLORYS labels.
"""
import os
import numpy as np
import xarray as xr

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
GRID_RES = 0.25
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
YEARS = list(range(2015, 2021))

RAW_DIR = "./data/raw/glorys"
OUTPUT_DIR = "./data/glorys_processed"

TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
TARGET_LONS = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)

THETAO_CANDIDATES = ["thetao", "temperature", "water_temp", "TEMP"]


def standardize_coord_names(ds):
    """Same robust handling as the surface-channel script -- see that file's
    docstring for why a plain rename isn't always safe."""
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


def find_thetao(ds):
    for name in THETAO_CANDIDATES:
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Could not find a temperature variable. Tried: {THETAO_CANDIDATES}. "
        f"Actually available: {list(ds.data_vars)}."
    )


def process_year(year):
    print(f"\n=== Processing GLORYS {year} ===")
    path = f"{RAW_DIR}/GLORYS_NIO_{year}.nc"
    ds = xr.open_dataset(path, chunks={"time": 10, "depth": -1})  # keep full depth axis together per chunk
    ds = standardize_coord_names(ds)

    thetao_var = find_thetao(ds)
    print(f"  Variables available: {list(ds.data_vars)}  (using '{thetao_var}')")

    thetao = ds[thetao_var]  # dims: [time, depth, lat, lon], lazy
    n_days = thetao.sizes["time"]
    native_depths = ds["depth"].values
    print(f"  Native depth levels: {len(native_depths)}, range {native_depths.min():.1f}m to {native_depths.max():.1f}m")
    if native_depths.max() < max(STANDARD_DEPTHS):
        print(f"  WARNING: native max depth ({native_depths.max():.1f}m) does not exceed "
              f"the deepest standard depth ({max(STANDARD_DEPTHS)}m) -- the 1000m level "
              f"will be EXTRAPOLATED, not interpolated. Re-download with a higher "
              f"maximum_depth if this file predates the 1100m-margin fix.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/GLORYS_STD_DEPTHS_{year}.nc"

    all_batches = []
    batch_size = 10  # small, since depth-resolved data is heavy
    for start in range(0, n_days, batch_size):
        end = min(start + batch_size, n_days)
        print(f"  Processing days {start}-{end} of {n_days}...")

        batch = thetao.isel(time=slice(start, end)).compute()  # [batch_len, n_native_depths, H, W]

        # Spatial regrid first (cheaper on the native, smaller H,W before depth interpolation blow-up)
        batch_regridded_space = batch.interp(lat=TARGET_LATS, lon=TARGET_LONS, method="linear")

        # Depth interpolation: native irregular levels -> 15 standard depths
        batch_regridded_full = batch_regridded_space.interp(
            depth=STANDARD_DEPTHS, method="linear", kwargs={"fill_value": "extrapolate"}
        )
        # NOTE: fill_value="extrapolate" only matters at the very edges (e.g. slightly
        # below the deepest native level) -- this is why 06_download_glorys_target.py
        # requests to 1100m, so 1000m is always safely INTERPOLATED, never extrapolated.

        all_batches.append(batch_regridded_full.values)

    full_year = np.concatenate(all_batches, axis=0)  # [n_days, 15, H, W]

    out_ds = xr.Dataset(
        {"temperature": (("time", "depth", "lat", "lon"), full_year)},
        coords={"time": thetao.time.values[:full_year.shape[0]], "depth": STANDARD_DEPTHS,
                "lat": TARGET_LATS, "lon": TARGET_LONS}
    )
    out_ds.to_netcdf(out_path)
    print(f"  Saved {out_path}  (shape: {full_year.shape})")
    ds.close()


def main():
    for year in YEARS:
        out_path = f"{OUTPUT_DIR}/GLORYS_STD_DEPTHS_{year}.nc"
        if os.path.exists(out_path):
            print(f"Skipping {year}, already processed.")
            continue
        process_year(year)
    print("\nAll GLORYS years regridded to standard depths.")


if __name__ == "__main__":
    main()
