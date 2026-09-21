"""
Build the monthly climatology: for each of the 15 standard depths and each
grid cell, the average temperature per calendar month, computed ONLY from
the 2015-2018 TRAINING years (never validation/test years -- using those
would leak information the model is later evaluated against).

Output: climatology.nc, shape [depth, lat, lon, month(1-12)]
Used by: ClimatologyAnchoredHead in model.py, and to compute ΔT training
targets (target_delta_t = actual_temperature - climatology_for_that_month).

Memory-safe: processes one (year, month) combination at a time rather than
loading multiple years simultaneously.
"""
import os
import numpy as np
import xarray as xr

TRAIN_YEARS = [2015, 2016, 2017, 2018]  # matches config.py's TRAIN_YEARS -- deliberately
                                         # excludes 2019 (val) and 2020 (test)
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

GLORYS_DIR = "./data/glorys_processed"
OUTPUT_PATH = "./data/climatology.nc"


def main():
    print("Building monthly climatology from training years:", TRAIN_YEARS)

    # Open all training years lazily (dask-backed) -- nothing loaded yet
    datasets = [xr.open_dataset(f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{y}.nc", chunks={"time": 30})
                for y in TRAIN_YEARS]
    combined = xr.concat([ds["temperature"] for ds in datasets], dim="time")
    print(f"Combined training temperature array: {combined.sizes} (still lazy, not loaded)")

    n_depths = len(STANDARD_DEPTHS)
    lat_vals = combined.lat.values
    lon_vals = combined.lon.values
    H, W = len(lat_vals), len(lon_vals)

    climatology = np.zeros((n_depths, H, W, 12), dtype=np.float32)

    for month in range(1, 13):
        print(f"  Processing month {month}/12...")
        # Select all days across all training years that fall in this month,
        # for ALL depths at once -- but only load THIS month's data, not everything
        month_data = combined.sel(time=combined.time.dt.month == month).compute()  # [n_month_days, depth, H, W]
        month_mean = month_data.mean(dim="time", skipna=True)  # [depth, H, W]
        climatology[:, :, :, month - 1] = month_mean.values
        del month_data  # free memory before next month

    out_ds = xr.Dataset(
        {"climatology": (("depth", "lat", "lon", "month"), climatology)},
        coords={"depth": STANDARD_DEPTHS, "lat": lat_vals, "lon": lon_vals, "month": list(range(1, 13))}
    )
    out_ds.to_netcdf(OUTPUT_PATH)
    print(f"\nSaved {OUTPUT_PATH}, shape {climatology.shape}")

    # Sanity check: surface climatology should show a clear seasonal cycle
    surface_by_month = climatology[0].mean(axis=(0, 1))  # average over space, per month
    print(f"Surface (0m) climatology by month (spatial average): "
          + ", ".join(f"{m}:{v:.1f}C" for m, v in zip(range(1, 13), surface_by_month)))
    print("Sanity check: this should show a real seasonal cycle (e.g. warmer pre-monsoon, "
          "cooler during/after winter monsoon), not a flat line across months.")

    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()
