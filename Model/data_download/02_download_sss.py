"""Download multi-observation SSS year-by-year, cropped to the North Indian Ocean box."""
import os
import copernicusmarine

PRODUCT_ID = "MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013"
OUTPUT_DIR = "./data/raw/sss/"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
YEARS = list(range(2015, 2024))

os.makedirs(OUTPUT_DIR, exist_ok=True)

catalogue = copernicusmarine.describe(product_id=PRODUCT_ID, disable_progress_bar=True)
dataset_id = catalogue.products[0].datasets[0].dataset_id
dataset_meta = catalogue.products[0].datasets[0]
available_vars = [v.short_name for v in dataset_meta.versions[0].parts[0].services[0].variables]
SSS_VAR = next((v for v in available_vars if v.lower() in ("so", "sos")), available_vars[0])
print(f"Using salinity variable: {SSS_VAR}")

for year in YEARS:
    output_filename = f"SSS_NIO_{year}.nc"
    file_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(file_path):
        print(f"Skipping {year}, exists.")
        continue
    print(f"Downloading SSS {year}...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=[SSS_VAR],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_directory=OUTPUT_DIR,
        output_filename=output_filename,
    )
print("SSS download complete.")
