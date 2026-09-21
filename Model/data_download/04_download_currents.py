"""
Download OSCAR surface currents (u, v) year-by-year, cropped locally to the
North Indian Ocean box (Harmony server-side subsetting is unreliable for this
collection, so we download small global daily granules and crop+merge locally).

NOTE: only needed if ugos/vgos from 03_download_ssh.py are not preferred —
OSCAR gives total (geostrophic + Ekman) currents, DUACS gives geostrophic-only.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import xarray as xr
import earthaccess
from tqdm import tqdm

SHORT_NAME = "OSCAR_L4_OC_FINAL_V2.0"
RAW_TMP_DIR = "./data/raw/currents/_tmp_raw/"
OUTPUT_DIR = "./data/raw/currents/"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
YEARS = list(range(2015, 2024))
BATCH_SIZE = 10

os.makedirs(RAW_TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

auth = earthaccess.login(strategy="netrc")
print(f"Authenticated: {auth.authenticated}")

for year in YEARS:
    output_filename = f"CURRENTS_NIO_{year}.nc"
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
                mask_lat = (ds.lat >= LAT_MIN) & (ds.lat <= LAT_MAX)
                mask_lon = (ds.lon >= LON_MIN) & (ds.lon <= LON_MAX)
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

print("Currents download complete.")
