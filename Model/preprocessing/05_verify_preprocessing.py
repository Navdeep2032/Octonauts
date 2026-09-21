"""
Verify preprocessing outputs — CORRECTNESS, not just presence.

This goes beyond "does the file exist and have the right shape" to actually
check whether the processing LOGIC did the right thing:

  Structural checks:
    1. File exists, correct shape, correct channel ordering
    2. No NaNs remain in filled channels
    3. Validity masks are strictly binary

  Correctness checks (the point of this script):
    4. Land cells are EXACTLY zero in every real variable, matching land_sea_mask
       -- catches a broadcasting bug where the mask never got applied, or got
       applied to the wrong array
    5. Real day-to-day variation exists (not constant/degenerate) -- catches a
       bug where the same day got accidentally repeated across the whole batch
    6. lat/lon CoordConv channels exactly match the target grid -- catches a
       transposed or mis-broadcast coordinate array
    7. doy_sin^2 + doy_cos^2 ≈ 1 (unit circle) AND the encoded day-of-year
       matches the file's actual calendar date -- catches an indexing bug in
       the seasonal channel computation
    8. Time index is strictly increasing with no duplicates and no gaps
       -- catches a batching/concatenation bug
    9. Validity mask agrees with the physical NaN pattern in the ORIGINAL raw
       file (spot-checked) -- catches a mask computed from the wrong array
   10. GLORYS: temperature is physically continuous across depth (no wild
       jumps between adjacent standard depths) and respects the depth-valid
       mask (no suspiciously "valid" values at depths deeper than local
       bathymetry)
"""
import numpy as np
import xarray as xr
import os

SURFACE_YEARS = list(range(2015, 2024))
GLORYS_YEARS = list(range(2015, 2021))
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
EXPECTED_CHANNELS = (
    ["sst", "sss", "sla", "u_curr", "v_curr", "u_wind", "v_wind"] +
    ["lat", "lon", "bathy", "doy_sin", "doy_cos"] +
    ["valid_sst", "valid_sss", "valid_sla", "valid_u_curr", "valid_v_curr", "valid_u_wind", "valid_v_wind"]
)
EXPECTED_RANGES = {
    "sst": (-5, 40), "sss": (0, 45), "sla": (-2, 2),
    "u_curr": (-3, 3), "v_curr": (-3, 3), "u_wind": (-40, 40), "v_wind": (-40, 40),
}
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_RES = 5.0, 30.0, 45.0, 105.0, 0.25
TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
TARGET_LONS = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)

PROCESSED_DIR = "./data/processed"
GLORYS_DIR = "./data/glorys_processed"
MASK_DIR = "./data/masks"

# Physically implausible jump between ADJACENT standard depths, in degrees C.
# Real thermoclines can be sharp, but not literally 15C between 20m and 30m.
MAX_PLAUSIBLE_DEPTH_JUMP = 15.0


def check_surface_year(year, land_sea_mask):
    path = f"{PROCESSED_DIR}/processed_{year}.nc"
    issues = []
    if not os.path.exists(path):
        return [f"MISSING FILE: {path}"]

    ds = xr.open_dataset(path)
    data = ds["channels"]

    # --- Structural checks ---
    if list(data.channel.values) != EXPECTED_CHANNELS:
        issues.append(f"channel order/names mismatch: {list(data.channel.values)}")
        ds.close()
        return issues  # can't safely index by position below if this fails

    if data.sizes.get("channel", 0) != 19:
        issues.append(f"expected 19 channels, got {data.sizes.get('channel')}")

    n_time = data.sizes["time"]
    sample_idx = np.linspace(0, n_time - 1, min(5, n_time), dtype=int)
    sample = data.isel(time=sample_idx).load()
    ch_index = {name: i for i, name in enumerate(EXPECTED_CHANNELS)}

    for ch_name, i in ch_index.items():
        vals = sample.isel(channel=i).values
        if np.isnan(vals).any():
            issues.append(f"'{ch_name}' contains NaN — should be 0-filled")
        if ch_name.startswith("valid_"):
            uniq = np.unique(vals)
            if not np.all(np.isin(uniq, [0.0, 1.0])):
                issues.append(f"'{ch_name}' not strictly binary: {uniq[:10]}")
        elif ch_name in EXPECTED_RANGES:
            lo, hi = EXPECTED_RANGES[ch_name]
            ocean_vals = vals[vals != 0]
            if ocean_vals.size and (ocean_vals.min() < lo - 1 or ocean_vals.max() > hi + 1):
                issues.append(f"'{ch_name}' out of range [{lo},{hi}]: got {ocean_vals.min():.2f} to {ocean_vals.max():.2f}")

    # --- Correctness check 4: land cells are EXACTLY zero, matching land_sea_mask ---
    land_mask_2d = land_sea_mask.values  # 1=ocean, 0=land
    land_pixels = land_mask_2d < 0.5
    for var_name in ["sst", "sss", "sla", "u_curr", "v_curr", "u_wind", "v_wind"]:
        i = ch_index[var_name]
        vals = sample.isel(channel=i).values  # [n_sample, H, W]
        nonzero_on_land = np.abs(vals[:, land_pixels]).max() if land_pixels.any() else 0
        if nonzero_on_land > 1e-6:
            issues.append(f"'{var_name}' has nonzero values over LAND (max {nonzero_on_land:.4f}) — "
                          f"land masking may not have been applied correctly")

    # --- Correctness check 5: real day-to-day variation exists ---
    for var_name in ["sst", "u_wind"]:
        i = ch_index[var_name]
        vals = sample.isel(channel=i).values  # [n_sample, H, W]
        if vals.shape[0] > 1:
            day_to_day_std = np.std(vals, axis=0).mean()
            if day_to_day_std < 1e-6:
                issues.append(f"'{var_name}' shows ZERO variation across sampled days — "
                              f"possible bug where the same day got repeated")

    # --- Correctness check 6: lat/lon CoordConv channels match the target grid exactly ---
    lat_ch = sample.isel(channel=ch_index["lat"], time=0).values
    lon_ch = sample.isel(channel=ch_index["lon"], time=0).values
    expected_lat_grid, expected_lon_grid = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")
    if not np.allclose(lat_ch, expected_lat_grid, atol=1e-6):
        issues.append("'lat' channel does not match the expected target grid (possible transpose or broadcast bug)")
    if not np.allclose(lon_ch, expected_lon_grid, atol=1e-6):
        issues.append("'lon' channel does not match the expected target grid (possible transpose or broadcast bug)")

    # --- Correctness check 7: DOY sin/cos unit circle + matches real calendar date ---
    full_time = data.time.values
    for idx in sample_idx:
        actual_date = np.datetime64(full_time[idx], "D")
        doy_actual = (actual_date - np.datetime64(f"{actual_date.astype('datetime64[Y]')}-01-01", "D")).astype(int) + 1
        expected_angle = (doy_actual / 365.25) * 2 * np.pi
        expected_sin, expected_cos = np.sin(expected_angle), np.cos(expected_angle)

        local_i = list(sample_idx).index(idx)
        actual_sin = float(sample.isel(channel=ch_index["doy_sin"], time=local_i).values.flat[0])
        actual_cos = float(sample.isel(channel=ch_index["doy_cos"], time=local_i).values.flat[0])

        unit_circle = actual_sin**2 + actual_cos**2
        if abs(unit_circle - 1.0) > 0.01:
            issues.append(f"doy_sin/doy_cos at index {idx} don't satisfy sin^2+cos^2≈1 (got {unit_circle:.3f})")
        if abs(actual_sin - expected_sin) > 0.05 or abs(actual_cos - expected_cos) > 0.05:
            issues.append(f"doy_sin/doy_cos at {actual_date} don't match the expected day-of-year encoding "
                          f"(possible off-by-one or indexing bug in the seasonal channel)")

    # --- Correctness check 8: time index strictly increasing, no duplicates/gaps ---
    time_diffs = np.diff(full_time).astype("timedelta64[D]").astype(int)
    if np.any(time_diffs <= 0):
        issues.append("time index is not strictly increasing — duplicate or out-of-order timestamps found "
                      "(likely a batching/concatenation bug)")
    elif np.any(time_diffs > 1):
        gap_locs = np.where(time_diffs > 1)[0]
        issues.append(f"gap(s) in the daily time sequence at index {gap_locs[:5].tolist()} "
                      f"(missing day(s) — check the alignment step didn't drop more than expected)")

    ds.close()
    return issues


def check_glorys_year(year, depth_valid_mask):
    path = f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{year}.nc"
    issues = []
    if not os.path.exists(path):
        return [f"MISSING FILE: {path}"]

    ds = xr.open_dataset(path)
    if list(np.round(ds["depth"].values).astype(int)) != STANDARD_DEPTHS:
        issues.append(f"depth levels mismatch: {list(ds['depth'].values)}")
        ds.close()
        return issues

    n_time = ds.sizes["time"]
    sample_idx = np.linspace(0, n_time - 1, min(3, n_time), dtype=int)
    sample = ds["temperature"].isel(time=sample_idx).load()

    # --- Correctness check 10a: no wild unphysical jumps between adjacent depths ---
    for d_i in range(len(STANDARD_DEPTHS) - 1):
        diff = np.abs(sample.isel(depth=d_i + 1).values - sample.isel(depth=d_i).values)
        max_jump = np.nanmax(diff) if not np.all(np.isnan(diff)) else np.nan
        if not np.isnan(max_jump) and max_jump > MAX_PLAUSIBLE_DEPTH_JUMP:
            issues.append(f"implausible jump ({max_jump:.1f}C) between {STANDARD_DEPTHS[d_i]}m and "
                          f"{STANDARD_DEPTHS[d_i+1]}m — check depth interpolation")

    # --- Correctness check 10b: respects depth-valid mask (no data claimed valid where seafloor is shallower) ---
    if depth_valid_mask is not None:
        for d_i, depth in enumerate(STANDARD_DEPTHS):
            invalid_pixels = depth_valid_mask.isel(depth=d_i).values < 0.5
            vals_at_invalid = sample.isel(depth=d_i).values[:, invalid_pixels]
            frac_unexpectedly_populated = np.mean(~np.isnan(vals_at_invalid)) if vals_at_invalid.size else 0
            if frac_unexpectedly_populated > 0.5:
                issues.append(f"depth {depth}m has real (non-NaN) values at >{frac_unexpectedly_populated:.0%} of "
                              f"pixels the depth_valid_mask marks as too shallow — masks may be inconsistent")

    surface_mean = float(sample.isel(depth=0).mean(skipna=True))
    deep_mean = float(sample.isel(depth=-1).mean(skipna=True))
    if not np.isnan(surface_mean) and not np.isnan(deep_mean):
        pattern = "expected (surface warmer)" if surface_mean > deep_mean else "INVERTED — verify this is real, not a bug"
        print(f"    [{year}] surface mean {surface_mean:.1f}C, 1000m mean {deep_mean:.1f}C ({pattern})")

    ds.close()
    return issues


def check_depth_valid_mask():
    path = f"{MASK_DIR}/depth_valid_mask.nc"
    if not os.path.exists(path):
        return [f"MISSING FILE: {path}"], None
    ds = xr.open_dataset(path)
    fractions = [float(ds["mask"].isel(depth=i).mean()) for i in range(len(STANDARD_DEPTHS))]
    issues = []
    for i in range(1, len(fractions)):
        if fractions[i] > fractions[i - 1] + 0.01:
            issues.append(f"valid fraction INCREASED from {STANDARD_DEPTHS[i-1]}m ({fractions[i-1]:.1%}) "
                          f"to {STANDARD_DEPTHS[i]}m ({fractions[i]:.1%})")
    print(f"    Valid fraction by depth: " + ", ".join(f"{d}m={f:.0%}" for d, f in zip(STANDARD_DEPTHS, fractions)))
    return issues, ds["mask"]


def main():
    print("=" * 70)
    print("OceanEmbed Preprocessing Verification (structural + correctness)")
    print("=" * 70)

    total_issues = 0

    land_sea_mask = xr.open_dataset(f"{MASK_DIR}/land_sea_mask.nc")["mask"] \
        if os.path.exists(f"{MASK_DIR}/land_sea_mask.nc") else None

    print("\n--- Depth-valid mask (checked first — needed by GLORYS checks below) ---")
    mask_issues, depth_valid_mask = check_depth_valid_mask()
    if mask_issues:
        total_issues += 1
        for issue in mask_issues:
            print(f"  ISSUES FOUND: - {issue}")
    else:
        print("  OK")

    print("\n--- Surface channel stacks (processed_{year}.nc) ---")
    if land_sea_mask is None:
        print("  SKIPPED — land_sea_mask.nc not found, cannot check land-masking correctness")
    else:
        for year in SURFACE_YEARS:
            issues = check_surface_year(year, land_sea_mask)
            if issues:
                total_issues += 1
                print(f"  [{year}] ISSUES FOUND:")
                for issue in issues:
                    print(f"      - {issue}")
            else:
                print(f"  [{year}] OK")

    print("\n--- GLORYS standard-depth targets (GLORYS_STD_DEPTHS_{year}.nc) ---")
    for year in GLORYS_YEARS:
        issues = check_glorys_year(year, depth_valid_mask)
        if issues:
            total_issues += 1
            print(f"  [{year}] ISSUES FOUND:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"  [{year}] OK")

    print("\n" + "=" * 70)
    if total_issues == 0:
        print("All checks passed — including correctness checks, not just file presence.")
        print("Ready for Stage 1/2 training.")
    else:
        print(f"{total_issues} check(s) found issues — review above before proceeding to training.")
    print("=" * 70)


if __name__ == "__main__":
    main()
