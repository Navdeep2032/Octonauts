"""Download GLORYS12V1 subsurface temperature (target labels) year-by-year, all 15 standard depths."""
import os
import copernicusmarine

PRODUCT_ID = "GLOBAL_MULTIYEAR_PHY_001_030"  # GLORYS12V1 reanalysis
OUTPUT_DIR = "./data/raw/glorys/"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Stage 2 train+val+test only — GLORYS reprocessed record needed just for the labeled years,
# NOT for the 2021-2023 Stage 1 pretraining pool (which is surface-only, unlabeled by design)
YEARS = list(range(2015, 2021))  # 2015-2020

os.makedirs(OUTPUT_DIR, exist_ok=True)

catalogue = copernicusmarine.describe(product_id=PRODUCT_ID, disable_progress_bar=True)
# GLORYS12V1 has multiple datasets (daily/monthly, different variable groups) —
# pick the one with 'thetao' (potential temperature) in its daily dataset
dataset_id = None
for ds in catalogue.products[0].datasets:
    if "daily" in ds.dataset_id.lower() or "P1D" in ds.dataset_id:
        dataset_id = ds.dataset_id
        break
if dataset_id is None:
    dataset_id = catalogue.products[0].datasets[0].dataset_id
print(f"Using dataset: {dataset_id}")

for year in YEARS:
    output_filename = f"GLORYS_NIO_{year}.nc"
    file_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(file_path):
        print(f"Skipping {year}, exists.")
        continue

    print(f"Downloading GLORYS subsurface temperature for {year}...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["thetao"],  # sea water potential temperature
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        minimum_depth=0, maximum_depth=1100,  # requested beyond 1000m with margin —
        # GLORYS's native levels near 1000m (902.3 / 947.5 / 994.8 / 1045.9m) mean a
        # max_depth=1000 request can cut off before the levels needed to interpolate
        # UP TO 1000m; requesting to 1100m guarantees a bracketing level past 1000m exists
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_directory=OUTPUT_DIR,
        output_filename=output_filename,
    )
    print(f"Saved {output_filename}")

print("\nGLORYS download complete.")
print("NOTE: GLORYS reprocessed record has a known reprocessed-vs-interim version boundary "
      "around 2020-2021 — check the last year downloaded for continuity before training.")
