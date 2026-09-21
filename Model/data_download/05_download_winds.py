"""
Download CCMP surface winds (u, v) year-by-year via NASA earthaccess.
Same pattern as OSCAR currents: CCMP is also hosted on PO.DAAC with daily granules,
so we crop+merge locally rather than relying on Harmony server-side subsetting.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import xarray as xr
import earthaccess
from tqdm import tqdm

SHORT_NAME = "CCMP_WINDS_10M6HR_L4_V3.1"  # 6-hourly, 0.25 deg, daily file granules
RAW_TMP_DIR = "./data/raw/wind/_tmp_raw/"
OUTPUT_DIR = "./data/raw/wind/"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
YEARS = list(range(2015, 2024))  # covers train+val+test+pretrain span
BATCH_SIZE = 10

os.makedirs(RAW_TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

auth = earthaccess.login(strategy="netrc")
print(f"Authenticated: {auth.authenticated}")

for year in YEARS:
    output_filename = f"WIND_NIO_{year}.nc"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"Skipping {year}, exists.")
        continue

    print(f"\nSearching granules for {year}...")
    results = earthaccess.search_data(short_name=SHORT_NAME, temporal=(f"{year}-01-01", f"{year}-12-31"))
    print(f"Found {len(results)} daily granules for {year}")

    year_tmp_dir = os.path.join(RAW_TMP_DIR, str(year))
    os.makedirs(year_tmp_dir, exist_ok=True)
    cropped_datasets = []

    batches = list(range(0, len(results), BATCH_SIZE))
    for batch_start in tqdm(batches, desc=f"Downloading {year}", unit="batch"):
        batch = results[batch_start:batch_start + BATCH_SIZE]
        downloaded_batch = earthaccess.download(batch, year_tmp_dir)

        for raw_path in downloaded_batch:
            with xr.open_dataset(raw_path) as ds:
                # check ds.coords the first run — confirm lat/lon names match before trusting this
                mask_lat = (ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX)
                mask_lon = (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX)
                ds_crop = ds.where(mask_lat & mask_lon, drop=True).load()
            cropped_datasets.append(ds_crop)
            os.remove(raw_path)

    print(f"Merging {year}...")
    ds_year = xr.concat(cropped_datasets, dim="time")
    ds_year.to_netcdf(output_path)
    ds_year.close()
    for d in cropped_datasets:
        d.close()
    os.rmdir(year_tmp_dir)
    print(f"Saved {output_filename}\n")

print("Wind download complete.")
