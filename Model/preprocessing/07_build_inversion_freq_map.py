"""
Build the per-pixel inversion-frequency map: for each grid cell, what fraction
of training-year days show a temperature inversion (deeper water warmer than
shallower water) between ADJACENT standard depths.

This map is what makes the adaptive inversion penalty adaptive: locations
where inversions are climatologically common (e.g. northern Bay of Bengal in
winter, driven by the freshwater halocline) get a LOW penalty weight, so the
model isn't punished for correctly predicting real physics there. Locations
where inversions are rare (e.g. the Arabian Sea) keep a HIGH penalty weight,
since an inversion there usually does indicate a genuine problem.

Output: inversion_freq_map.nc, shape [lat, lon] -- one aggregate frequency
per pixel across ALL depth-pairs and ALL training days combined.

Uses ONLY 2015-2018 training years, matching climatology's split discipline.
Memory-safe: processes one training year at a time.
"""
import os
import numpy as np
import xarray as xr

TRAIN_YEARS = [2015, 2016, 2017, 2018]
GLORYS_DIR = "./data/glorys_processed"
OUTPUT_PATH = "./data/inversion_freq_map.nc"


def main():
    print("Building inversion-frequency map from training years:", TRAIN_YEARS)

    inversion_count = None
    total_count = None

    for year in TRAIN_YEARS:
        print(f"  Processing {year}...")
        ds = xr.open_dataset(f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{year}.nc", chunks={"time": 30})
        temp = ds["temperature"]  # [time, depth, lat, lon]

        n_days = temp.sizes["time"]
        batch_size = 30
        for start in range(0, n_days, batch_size):
            end = min(start + batch_size, n_days)
            batch = temp.isel(time=slice(start, end)).compute()  # [batch_len, depth, H, W]

            # An inversion at depth-pair (i, i+1): temperature INCREASES with depth
            # (deeper is warmer), which is the physically unusual direction below the
            # near-surface mixed layer.
            deeper = batch.isel(depth=slice(1, None))
            shallower = batch.isel(depth=slice(0, -1))
            is_inversion = (deeper.values > shallower.values) & ~np.isnan(deeper.values) & ~np.isnan(shallower.values)
            is_valid_pair = ~np.isnan(deeper.values) & ~np.isnan(shallower.values)

            # Collapse across the depth-pair axis and the day axis for this batch:
            # count how many (day, depth-pair) combinations were inversions vs. total valid
            batch_inversion_count = is_inversion.sum(axis=(0, 1))  # [H, W]
            batch_total_count = is_valid_pair.sum(axis=(0, 1))     # [H, W]

            if inversion_count is None:
                inversion_count = batch_inversion_count.astype(np.float64)
                total_count = batch_total_count.astype(np.float64)
            else:
                inversion_count += batch_inversion_count
                total_count += batch_total_count

        ds.close()

    # Avoid divide-by-zero on permanently-invalid (land/too-shallow) pixels
    inversion_freq = np.where(total_count > 0, inversion_count / np.maximum(total_count, 1), 0.0).astype(np.float32)

    lat_vals = xr.open_dataset(f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{TRAIN_YEARS[0]}.nc")["lat"].values
    lon_vals = xr.open_dataset(f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{TRAIN_YEARS[0]}.nc")["lon"].values

    out_ds = xr.Dataset(
        {"inversion_freq": (("lat", "lon"), inversion_freq)},
        coords={"lat": lat_vals, "lon": lon_vals}
    )
    out_ds.to_netcdf(OUTPUT_PATH)
    print(f"\nSaved {OUTPUT_PATH}")
    print(f"Overall mean inversion frequency: {inversion_freq[inversion_freq > 0].mean():.1%} "
          f"(computed over pixels with any valid data)")
    print("Sanity check: expect NOTABLY higher frequency in the northern Bay of Bengal "
          "(freshwater-driven winter inversions) than in the Arabian Sea.")


if __name__ == "__main__":
    main()
