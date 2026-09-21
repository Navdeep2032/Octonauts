"""Download OSTIA SST year-by-year, cropped to the North Indian Ocean box."""
import os
import copernicusmarine

PRODUCT_ID = "SST_GLO_SST_L4_REP_OBSERVATIONS_010_011"
OUTPUT_DIR = "./data/raw/sst/"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
YEARS = list(range(2015, 2024))  # 2015-2018 train, 2019 val, 2020 test, 2021-2023 pretrain

os.makedirs(OUTPUT_DIR, exist_ok=True)

catalogue = copernicusmarine.describe(product_id=PRODUCT_ID, disable_progress_bar=True)
dataset_id = catalogue.products[0].datasets[0].dataset_id

for year in YEARS:
    output_filename = f"SST_NIO_{year}.nc"
    file_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(file_path):
        print(f"Skipping {year}, exists.")
        continue
    print(f"Downloading SST {year}...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=["analysed_sst"],
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        output_directory=OUTPUT_DIR,
        output_filename=output_filename,
    )
print("SST download complete.")
