"""
Build the land-sea mask and a bathymetry PROXY from GLORYS itself, rather than
downloading a separate bathymetry product. Rationale: wherever GLORYS's thetao
stops having a valid (non-NaN) value as depth increases at a given pixel, that's
approximately the local seafloor depth -- GLORYS's own vertical grid is built on
a real ocean model bathymetry, so this is a reasonable proxy without adding a
7th download source.

Run this ONCE (not per year) -- bathymetry doesn't change year to year. Uses
a single GLORYS year (2015) as the reference; any year would give the same mask.

Memory-safe: opens with chunks so nothing is fully loaded until the final
reduction, which operates on one time-step at a time.
"""
import numpy as np
import xarray as xr
import os

GLORYS_REFERENCE_FILE = "./data/raw/glorys/GLORYS_NIO_2015.nc"
OUTPUT_DIR = "./data/masks"
TARGET_GRID_RES = 0.25
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + TARGET_GRID_RES, TARGET_GRID_RES)
TARGET_LONS = np.arange(LON_MIN, LON_MAX + TARGET_GRID_RES, TARGET_GRID_RES)


def main():
    print(f"Opening {GLORYS_REFERENCE_FILE} (chunked, not fully loaded)...")
    ds = xr.open_dataset(GLORYS_REFERENCE_FILE, chunks={"time": 1, "depth": 5})

    # Use a single day (day 0) -- bathymetry doesn't change day to day, so no
    # need to touch the full 365-day array at all.
    single_day = ds["thetao"].isel(time=0)  # dims: [depth, lat, lon], still lazy (dask-backed)

    print("Computing per-pixel maximum valid depth (this triggers computation on ONE day only)...")
    is_valid = single_day.notnull()  # [depth, lat, lon], True where thetao has real data

    # For each (lat, lon), find the deepest depth level that is still valid.
    # Where NO depth is valid at all, that pixel is land.
    depth_da = ds["depth"]
    valid_depth_values = is_valid * depth_da  # broadcasts depth values where valid, 0 elsewhere
    max_valid_depth = valid_depth_values.max(dim="depth").compute()  # only now do we materialize -- small 2D result

    any_valid = is_valid.any(dim="depth").compute()  # [lat, lon] -- True if this pixel is ocean at all

    print("Regridding bathymetry proxy and land-sea mask onto the common 0.25deg grid...")
    rename_map = {}
    if "lat" not in max_valid_depth.dims and "latitude" in max_valid_depth.dims:
        rename_map["latitude"] = "lat"
    if "lon" not in max_valid_depth.dims and "longitude" in max_valid_depth.dims:
        rename_map["longitude"] = "lon"

    bathymetry_native = max_valid_depth.rename(rename_map) if rename_map else max_valid_depth
    land_sea_native = (any_valid.rename(rename_map) if rename_map else any_valid).astype(np.float32)

    bathymetry_regridded = bathymetry_native.interp(lat=TARGET_LATS, lon=TARGET_LONS, method="nearest")
    land_sea_regridded = land_sea_native.interp(lat=TARGET_LATS, lon=TARGET_LONS, method="nearest")
    # nearest-neighbor for the mask specifically -- bilinear would blur a hard land/ocean boundary
    # into fractional values, which is wrong for a binary mask

    bathymetry_regridded = bathymetry_regridded.fillna(0)
    land_sea_regridded = (land_sea_regridded > 0.5).astype(np.float32)  # re-binarize after interpolation

    bathymetry_out = xr.Dataset({"bathymetry": (("lat", "lon"), bathymetry_regridded.values)},
                                 coords={"lat": TARGET_LATS, "lon": TARGET_LONS})
    land_sea_out = xr.Dataset({"mask": (("lat", "lon"), land_sea_regridded.values)},
                               coords={"lat": TARGET_LATS, "lon": TARGET_LONS})

    bathymetry_out.to_netcdf(os.path.join(OUTPUT_DIR, "bathymetry.nc"))
    land_sea_out.to_netcdf(os.path.join(OUTPUT_DIR, "land_sea_mask.nc"))

    ocean_frac = float(land_sea_regridded.mean())
    print(f"Saved bathymetry.nc and land_sea_mask.nc")
    print(f"Ocean fraction of grid: {ocean_frac:.1%} (land fraction: {1-ocean_frac:.1%}) "
          f"-- sanity check this looks right for this box (expect roughly 40-55% ocean)")

    ds.close()


if __name__ == "__main__":
    main()
