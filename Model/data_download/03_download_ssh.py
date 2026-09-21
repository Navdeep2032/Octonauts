"""Download DUACS SSH/SLA (+ geostrophic currents) year-by-year, cropped to the North Indian Ocean box."""
import os
import copernicusmarine

PRODUCT_ID = "SEALEVEL_GLO_PHY_L4_MY_008_047"
OUTPUT_DIR = "./data/raw/ssh/"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
YEARS = list(range(2015, 2024))
# also pulling ugos/vgos here so a separate OSCAR download may not be needed —
# check the merged output for these variables before running 04_download_currents.py
VARIABLES = ["sla", "adt", "ugos", "vgos"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

catalogue = copernicusmarine.describe(product_id=PRODUCT_ID, disable_progress_bar=True)
dataset_id = catalogue.products[0].datasets[0].dataset_id

for year in YEARS:
    output_filename = f"SSH_NIO_{year}.nc"
    file_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(file_path):
        print(f"Skipping {year}, exists.")
        continue
    print(f"Downloading SSH/SLA {year}...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=VARIABLES,
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_directory=OUTPUT_DIR,
        output_filename=output_filename,
    )
print("SSH download complete.")
