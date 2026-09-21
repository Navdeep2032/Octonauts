"""
Verify all six OceanEmbed data downloads before moving to preprocessing.

Checks, per file:
  1. File exists and is non-empty
  2. File opens as valid NetCDF (not corrupted/truncated)
  3. Expected variable(s) are present
  4. Time dimension covers the expected number of days for that year (catches partial downloads)
  5. Spatial extent roughly matches the North Indian Ocean box
  6. No suspiciously large fraction of NaN/missing values
  7. (GLORYS only) all 15 standard depth levels are present

Run this after each download script, or all at once at the end.
"""
import os
import numpy as np
import xarray as xr

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

SOURCES = {
    "sst":      {"dir": "./data/raw/sst",      "prefix": "SST_NIO",      "var_hint": "sst",  "years": range(2015, 2024)},
    "sss":      {"dir": "./data/raw/sss",      "prefix": "SSS_NIO",      "var_hint": "so",   "years": range(2015, 2024)},
    "ssh":      {"dir": "./data/raw/ssh",      "prefix": "SSH_NIO",      "var_hint": "sla",  "years": range(2015, 2024)},
    "currents": {"dir": "./data/raw/currents", "prefix": "CURRENTS_NIO","var_hint": "u",    "years": range(2015, 2024)},
    "wind":     {"dir": "./data/raw/wind",     "prefix": "WIND_NIO",    "var_hint": "u",    "years": range(2015, 2024)},
    "glorys":   {"dir": "./data/raw/glorys",   "prefix": "GLORYS_NIO",  "var_hint": "thetao","years": range(2015, 2021)},
}

NAN_WARN_THRESHOLD = 0.65  # box is ~50% land by area for ocean-only variables (SST/SSS/currents) —
                            # raised from 0.5 to avoid flagging expected land NaN as a false positive
MIN_FILE_SIZE_KB = 5       # anything smaller than this is almost certainly a failed/empty download


def check_file(path, var_hint, is_leap_ok=True):
    issues = []

    # 1. Exists and non-trivial size
    if not os.path.exists(path):
        return ["MISSING FILE"]
    size_kb = os.path.getsize(path) / 1024
    if size_kb < MIN_FILE_SIZE_KB:
        issues.append(f"suspiciously small file ({size_kb:.1f} KB) — likely a failed/empty download")
        return issues  # no point checking further if the file is basically empty

    # 2. Opens as valid NetCDF (chunked/lazy — critical for GLORYS, whose full array
    #    can be tens of GB if loaded eagerly; chunks keep peak memory low)
    try:
        ds = xr.open_dataset(path, chunks={"time": 30})
    except Exception as e:
        return [f"FAILED TO OPEN — corrupted or incomplete file ({e})"]

    # 3. Expected variable present
    matching_vars = [v for v in ds.data_vars if var_hint.lower() in v.lower()]
    if not matching_vars:
        issues.append(f"expected variable containing '{var_hint}' not found — has: {list(ds.data_vars)}")

    # 4. Time coverage — rough day-count check
    if "time" in ds.dims:
        n_days = ds.sizes["time"]
        year = int(os.path.basename(path).split("_")[-1].split(".")[0])
        expected_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        if n_days < expected_days * 0.95:  # allow small tolerance
            issues.append(f"only {n_days} time steps found, expected ~{expected_days} — likely a partial download")
    else:
        issues.append("no 'time' dimension found — check this file manually")

    # 5. Spatial extent check
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if lat_name and lon_name:
        lat_vals = ds[lat_name].values
        lon_vals = ds[lon_name].values
        if lat_vals.min() > LAT_MIN + 1 or lat_vals.max() < LAT_MAX - 1:
            issues.append(f"latitude range ({lat_vals.min():.1f} to {lat_vals.max():.1f}) doesn't cover expected box")
        lon_check_min, lon_check_max = lon_vals.min(), lon_vals.max()
        # allow for 0-360 vs -180-180 longitude convention differences
        if not (LON_MIN - 1 <= lon_check_min <= LON_MIN + 1 or 0 <= lon_check_min <= 5):
            issues.append(f"longitude range starts at {lon_check_min:.1f}, check convention (0-360 vs -180/180)")
    else:
        issues.append("could not find lat/lon coordinates to verify spatial extent")

    # 6. NaN fraction check — SAMPLE a small subset rather than loading the full array.
    #    A full year of GLORYS at multiple depth levels can be tens of GB as float64 once
    #    decoded (scale/offset decoding upcasts from the compact on-disk int16), which can
    #    exceed available RAM. A handful of time steps is enough to catch a real problem.
    if matching_vars:
        var = ds[matching_vars[0]]
        if "time" in var.dims:
            n_sample = min(5, var.sizes["time"])
            sample_idx = np.linspace(0, var.sizes["time"] - 1, n_sample, dtype=int)
            sample = var.isel(time=sample_idx).load()
        else:
            sample = var.load()
        nan_frac = np.isnan(sample.values).mean()
        if nan_frac > NAN_WARN_THRESHOLD:
            issues.append(f"high NaN fraction ({nan_frac:.1%}, sampled) in '{matching_vars[0]}' — check land masking or download completeness")

    # 7. GLORYS-specific: check native depth coordinate spans the needed range with enough
    #    resolution to interpolate onto the 15 standard depths later (preprocess.py's job) —
    #    GLORYS's native vertical grid does NOT sit at exact meter values like 50, 75, 100...,
    #    so we check coverage/density here, not exact matches.
    if var_hint == "thetao" and "depth" in ds.coords:
        native_depths = np.sort(ds["depth"].values)
        min_d, max_d = native_depths.min(), native_depths.max()
        if min_d > 2:
            issues.append(f"native depth grid starts at {min_d:.1f}m, expected ~0m")
        if max_d < 1000:
            issues.append(f"native depth grid only reaches {max_d:.1f}m — need a level PAST 1000m "
                           f"to safely interpolate up to 1000m without extrapolating; "
                           f"re-download with a higher maximum_depth request")
        elif max_d < 1020:
            issues.append(f"native depth grid reaches {max_d:.1f}m, just barely past 1000m — "
                           f"verify this is a real bracketing level, not a coincidental cutoff")
        n_levels_in_range = np.sum(native_depths <= 1000)
        if n_levels_in_range < 15:
            issues.append(f"only {n_levels_in_range} native levels within 0-1000m — "
                           f"may be too coarse to interpolate all 15 standard depths accurately")

    ds.close()
    return issues


def main():
    print("=" * 70)
    print("OceanEmbed Data Download Verification")
    print("=" * 70)

    total_files = 0
    total_issues = 0

    for source_name, cfg in SOURCES.items():
        print(f"\n--- {source_name.upper()} ---")
        for year in cfg["years"]:
            path = os.path.join(cfg["dir"], f"{cfg['prefix']}_{year}.nc")
            total_files += 1
            issues = check_file(path, cfg["var_hint"])
            if issues:
                total_issues += 1
                print(f"  [{year}] ISSUES FOUND:")
                for issue in issues:
                    print(f"      - {issue}")
            else:
                print(f"  [{year}] OK")

    print("\n" + "=" * 70)
    print(f"Checked {total_files} files, {total_issues} had issues.")
    if total_issues == 0:
        print("All files passed verification. Safe to proceed to preprocessing.")
    else:
        print("Re-run the relevant download script(s) for any year/source flagged above")
        print("before proceeding — issues here will silently propagate through preprocessing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
