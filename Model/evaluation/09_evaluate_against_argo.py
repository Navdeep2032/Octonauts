"""
STEP 3: Evaluate the trained model against REAL, INDEPENDENT ARGO profiles.

This is the validation that actually matters -- unlike the GLORYS validation
loss from training, ARGO floats are genuine physical measurements the model
has never seen in any form (not in Stage 1, not in Stage 2, not in the
climatology or inversion-frequency map, which were built from 2015-2018 only).

For each ARGO profile:
  1. Find the model's prediction at the float's exact date (run the model for
     that day if not already cached).
  2. Bilinearly sample the model's gridded output at the float's exact
     lat/lon (floats are almost never exactly on a 0.25deg grid point).
  3. Linearly interpolate the float's own depth readings onto the 15
     standard depths for a fair comparison.
  4. Compare, and also compare GLORYS's own value at that point/time --
     this separates "error inherited from the reanalysis label" from
     "genuine model error," per the technical approach.
"""
import torch
import numpy as np
import pandas as pd
import xarray as xr
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from config import STANDARD_DEPTHS, CHECKPOINT_DIR, ENCODER_BASE_CH, FOURIER_N_FREQS, DEPTH_MLP_HIDDEN, CLIMATOLOGY_PATH, TIME_WINDOW
from model import OceanEmbedModel
from dataset import SubsurfaceDataset
from torch.utils.data import DataLoader

ARGO_DIR = "./data/argo"
GLORYS_DIR = "./data/glorys_processed"
YEARS = [2019, 2020]


def load_model(device):
    ds = xr.open_dataset(CLIMATOLOGY_PATH)
    climatology = torch.tensor(np.nan_to_num(ds["climatology"].values, nan=0.0), dtype=torch.float32).to(device)
    model = OceanEmbedModel(
        climatology_field=climatology, in_ch=19, t_window=14, base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS, depth_hidden=DEPTH_MLP_HIDDEN, standard_depths=STANDARD_DEPTHS,
    ).to(device)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/stage2_best.pt", map_location=device, weights_only=True))
    model.eval()
    return model


def bilinear_sample(field_2d, lat_vals, lon_vals, target_lat, target_lon):
    """Bilinearly sample a [H, W] numpy array at an arbitrary (lat, lon) point."""
    if target_lat < lat_vals.min() or target_lat > lat_vals.max() or \
       target_lon < lon_vals.min() or target_lon > lon_vals.max():
        return np.nan  # profile falls outside the model's domain

    lat_idx = np.searchsorted(lat_vals, target_lat) - 1
    lon_idx = np.searchsorted(lon_vals, target_lon) - 1
    lat_idx = np.clip(lat_idx, 0, len(lat_vals) - 2)
    lon_idx = np.clip(lon_idx, 0, len(lon_vals) - 2)

    lat_frac = (target_lat - lat_vals[lat_idx]) / (lat_vals[lat_idx + 1] - lat_vals[lat_idx])
    lon_frac = (target_lon - lon_vals[lon_idx]) / (lon_vals[lon_idx + 1] - lon_vals[lon_idx])

    v00 = field_2d[lat_idx, lon_idx]
    v01 = field_2d[lat_idx, lon_idx + 1]
    v10 = field_2d[lat_idx + 1, lon_idx]
    v11 = field_2d[lat_idx + 1, lon_idx + 1]

    return (v00 * (1 - lat_frac) * (1 - lon_frac) + v01 * (1 - lat_frac) * lon_frac +
            v10 * lat_frac * (1 - lon_frac) + v11 * lat_frac * lon_frac)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    print(f"Loaded model from {CHECKPOINT_DIR}/stage2_best.pt")

    results = {depth: {"model_err": [], "glorys_err": []} for depth in STANDARD_DEPTHS}

    for year in YEARS:
        argo_path = f"{ARGO_DIR}/ARGO_NIO_{year}.csv"
        if not os.path.exists(argo_path):
            print(f"Skipping {year} -- {argo_path} not found (run 08_download_argo.py first)")
            continue

        argo_df = pd.read_csv(argo_path, parse_dates=["TIME"])
        val_ds = SubsurfaceDataset(
            years=[year],
            glorys_path_template=f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{{year}}.nc",
            depth_mask_path="./data/masks/depth_valid_mask.nc",
        )
        loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        glorys_ds = xr.open_dataset(f"{GLORYS_DIR}/GLORYS_STD_DEPTHS_{year}.nc")
        lat_vals = glorys_ds.lat.values
        lon_vals = glorys_ds.lon.values

        # Group ARGO readings by unique (float, date) to get one profile at a time
        argo_df["date"] = argo_df["TIME"].dt.date
        profile_groups = argo_df.groupby(["PLATFORM_NUMBER", "date"]) if "PLATFORM_NUMBER" in argo_df.columns \
            else argo_df.groupby("date")

        print(f"\n{year}: {len(loader)} model-prediction days, {profile_groups.ngroups} ARGO profiles")

        # Cache predictions per date so we don't re-run the model for every single profile.
        # IMPORTANT: SubsurfaceDataset.__getitem__(idx) uses target_day = idx + TIME_WINDOW - 1
        # (it needs TIME_WINDOW days of history before its first prediction), so loader batch
        # index 0 corresponds to CALENDAR day index (TIME_WINDOW - 1), not day 0. Cache keyed
        # by the actual calendar day index to match how we'll look profiles up below.
        prediction_cache = {}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["x"].to(device)
                month_idx = batch["month_idx"].to(device)
                pred = model(x, month_idx, depths=STANDARD_DEPTHS)[0].cpu().numpy()  # [15, H, W]
                calendar_day_idx = i + TIME_WINDOW - 1
                prediction_cache[calendar_day_idx] = pred

        for (key, group) in profile_groups:
            profile_date = group["date"].iloc[0]
            profile_lat = group["LATITUDE"].iloc[0]
            profile_lon = group["LONGITUDE"].iloc[0]

            day_idx = (pd.Timestamp(profile_date) - pd.Timestamp(f"{year}-01-01")).days
            if day_idx not in prediction_cache or day_idx < 0:
                continue  # profile date falls outside the model's 14-day-window-adjusted valid range

            pred_field = prediction_cache[day_idx]         # [15, H, W]
            glorys_field = glorys_ds["temperature"].isel(time=min(day_idx, glorys_ds.sizes["time"]-1)).values  # [15, H, W]

            # Interpolate the float's raw depth readings onto the 15 standard depths
            float_depths = group["PRES"].values  # approx depth in dbar ~ meters
            float_temps = group["TEMP"].values
            valid = ~np.isnan(float_depths) & ~np.isnan(float_temps)
            if valid.sum() < 2:
                continue
            interp_temp_at_std_depths = np.interp(
                STANDARD_DEPTHS, float_depths[valid], float_temps[valid],
                left=np.nan, right=np.nan  # do NOT extrapolate beyond the float's own measured range
            )

            for d_i, depth in enumerate(STANDARD_DEPTHS):
                true_val = interp_temp_at_std_depths[d_i]
                if np.isnan(true_val):
                    continue
                model_val = bilinear_sample(pred_field[d_i], lat_vals, lon_vals, profile_lat, profile_lon)
                glorys_val = bilinear_sample(glorys_field[d_i], lat_vals, lon_vals, profile_lat, profile_lon)
                if not np.isnan(model_val):
                    results[depth]["model_err"].append((model_val - true_val) ** 2)
                if not np.isnan(glorys_val):
                    results[depth]["glorys_err"].append((glorys_val - true_val) ** 2)

        glorys_ds.close()

    print(f"\n{'Depth (m)':>10} {'N profiles':>11} {'Model RMSE':>12} {'GLORYS RMSE':>13} {'Diff':>8}")
    print("-" * 58)
    for depth in STANDARD_DEPTHS:
        model_errs = results[depth]["model_err"]
        glorys_errs = results[depth]["glorys_err"]
        if len(model_errs) == 0:
            print(f"{depth:>10} {0:>11} {'N/A':>12} {'N/A':>13}")
            continue
        model_rmse = np.sqrt(np.mean(model_errs))
        glorys_rmse = np.sqrt(np.mean(glorys_errs)) if glorys_errs else float("nan")
        diff = model_rmse - glorys_rmse
        note = " (model error mostly inherited from GLORYS)" if abs(diff) < 0.1 else \
               " (model adds real extra error beyond GLORYS)" if diff > 0 else \
               " (model beats GLORYS's own accuracy here)"
        print(f"{depth:>10} {len(model_errs):>11} {model_rmse:>12.4f} {glorys_rmse:>13.4f} {diff:>+8.4f}{note}")

    print("\nThis is your FIRST genuinely independent accuracy number -- unlike the")
    print("training/validation loss, these ARGO floats were never touched by Stage 1,")
    print("Stage 2, the climatology, or the inversion-frequency map.")


if __name__ == "__main__":
    main()
