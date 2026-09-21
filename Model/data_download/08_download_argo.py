"""
STEP 2: Download real, independent ARGO float profiles for validation.

Uses argopy (https://argopy.readthedocs.io) -- the standard, actively
maintained Python package for accessing the international ARGO float
network via the official GDAC servers. This gives you RAW profile data
(exact lat/lon/date/depth per float), which is what evaluate_argo.py
(Script 3) needs for the "raw ARGO profiles" validation tier.

pip install argopy "erddapy==2.2.1"

NOTE ON OMNI: INCOIS's OMNI moored-buoy network has NO public API or bulk-
download service as of this writing -- unlike ARGO, it is not part of any
open international data-sharing framework. Getting OMNI data genuinely
requires a direct data request to INCOIS (https://incois.gov.in). This
script cannot automate that step; treat it as a manual follow-up if you
have time before your deadline, not something to script around.
"""
import os
import argopy
import pandas as pd

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
MIN_DEPTH, MAX_DEPTH = 0, 1100  # margin past 1000m, same reasoning as the GLORYS download

# Match your validation/test years -- these are the years your model was
# NEVER trained on, so they're valid for independent checking.
YEARS = [2019, 2020]

OUTPUT_DIR = "./data/argo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

argopy.set_options(mode="standard", src="erddap")  # erddap is the most reliable default source


def download_year(year):
    out_path = f"{OUTPUT_DIR}/ARGO_NIO_{year}.csv"
    if os.path.exists(out_path):
        print(f"Skipping {year}, exists.")
        return

    print(f"Fetching ARGO profiles for {year}...")
    fetcher = argopy.DataFetcher().region(
        [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, MIN_DEPTH, MAX_DEPTH,
         f"{year}-01-01", f"{year}-12-31"]
    )
    ds = fetcher.to_xarray()  # returns a point-based xarray Dataset, one row per (float, cycle, depth level)

    df = ds.to_dataframe().reset_index()
    # Keep only the columns evaluate_argo.py actually needs
    keep_cols = [c for c in ["TIME", "LATITUDE", "LONGITUDE", "PRES", "TEMP", "PLATFORM_NUMBER"] if c in df.columns]
    df = df[keep_cols]
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} profile-depth rows, "
          f"{df['PLATFORM_NUMBER'].nunique() if 'PLATFORM_NUMBER' in df.columns else '?'} unique floats)")


def main():
    for year in YEARS:
        download_year(year)
    print("\nARGO download complete.")
    print("\nREMINDER: INCOIS OMNI mooring data has no public API -- if you want that")
    print("validation tier too, it requires a direct request to INCOIS, which this")
    print("script cannot automate. The ARGO data above is sufficient to build a real,")
    print("independent evaluation on its own.")


if __name__ == "__main__":
    main()
