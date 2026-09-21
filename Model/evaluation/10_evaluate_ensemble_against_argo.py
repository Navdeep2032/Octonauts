"""
STEP 5: Evaluate 3-Seed Ensemble vs Single Model against REAL, INDEPENDENT ARGO profiles.

This script evaluates both the single best Stage 2 checkpoint (stage2_best.pt)
and the 3-seed ensemble (stage2_seed42.pt, stage2_seed123.pt, stage2_seed2024.pt)
against independent ARGO float observations across 2019 and 2020.

For each ARGO profile:
  1. Retrieves gridded predictions for the single model and each ensemble seed.
  2. Averages the ensemble predictions: P_ensemble = mean(P_seed42, P_seed123, P_seed2024).
  3. Bilinearly samples both single-model and ensemble-averaged 3D temperature fields
     at the float's exact lat/lon coordinates.
  4. Interpolates float observations onto standard depths.
  5. Computes side-by-side RMSE metrics for Single Model, Ensemble, and GLORYS reanalysis.
"""
import torch
import numpy as np
import pandas as pd
import xarray as xr
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from config import (STANDARD_DEPTHS, CHECKPOINT_DIR, ENCODER_BASE_CH,
                    FOURIER_N_FREQS, DEPTH_MLP_HIDDEN, CLIMATOLOGY_PATH, TIME_WINDOW)
from model import OceanEmbedModel
from dataset import SubsurfaceDataset
from torch.utils.data import DataLoader

ARGO_DIR = "./data/argo"
GLORYS_DIR = "./data/glorys_processed"
YEARS = [2019, 2020]
ENSEMBLE_SEEDS = [42, 123, 2024]


def build_model_instance(device, climatology):
    return OceanEmbedModel(
        climatology_field=climatology, in_ch=19, t_window=14, base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS, depth_hidden=DEPTH_MLP_HIDDEN, standard_depths=STANDARD_DEPTHS,
    ).to(device)


def load_all_models(device):
    ds = xr.open_dataset(CLIMATOLOGY_PATH)
    climatology = torch.tensor(np.nan_to_num(ds["climatology"].values, nan=0.0), dtype=torch.float32).to(device)

    # Load single best model
    single_model = build_model_instance(device, climatology)
    single_ckpt = f"{CHECKPOINT_DIR}/stage2_best.pt"
    single_model.load_state_dict(torch.load(single_ckpt, map_location=device, weights_only=True))
    single_model.eval()
    print(f"Loaded single model from {single_ckpt}")

    # Load ensemble seed models
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = f"{CHECKPOINT_DIR}/stage2_seed{seed}.pt"
        if os.path.exists(ckpt_path):
            m = build_model_instance(device, climatology)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            m.eval()
            ensemble_models.append(m)
            print(f"Loaded ensemble seed {seed} model from {ckpt_path}")
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found, skipping seed {seed}")

    if not ensemble_models:
        print("Warning: No ensemble seed checkpoints found! Ensemble fallback to single model.")
        ensemble_models = [single_model]

    return single_model, ensemble_models


def bilinear_sample(field_2d, lat_vals, lon_vals, target_lat, target_lon):
    """Bilinearly sample a [H, W] numpy array at an arbitrary (lat, lon) point."""
    if target_lat < lat_vals.min() or target_lat > lat_vals.max() or \
       target_lon < lon_vals.min() or target_lon > lon_vals.max():
        return np.nan  # Profile falls outside domain

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
    print(f"Using device: {device}")

    single_model, ensemble_models = load_all_models(device)

    results = {
        depth: {
            "single_err": [],
            "ensemble_err": [],
            "glorys_err": []
        } for depth in STANDARD_DEPTHS
    }

    for year in YEARS:
        argo_path = f"{ARGO_DIR}/ARGO_NIO_{year}.csv"
        if not os.path.exists(argo_path):
            print(f"Skipping {year} -- {argo_path} not found")
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

        argo_df["date"] = argo_df["TIME"].dt.date
        profile_groups = argo_df.groupby(["PLATFORM_NUMBER", "date"]) if "PLATFORM_NUMBER" in argo_df.columns \
            else argo_df.groupby("date")

        print(f"\n{year}: {len(loader)} model prediction days, {profile_groups.ngroups} ARGO profiles")

        single_cache = {}
        ensemble_cache = {}

        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["x"].to(device)
                month_idx = batch["month_idx"].to(device)

                # Single model prediction
                single_pred = single_model(x, month_idx, depths=STANDARD_DEPTHS)[0].cpu().numpy()

                # Ensemble prediction (average across seeds)
                ens_preds = [m(x, month_idx, depths=STANDARD_DEPTHS)[0].cpu().numpy() for m in ensemble_models]
                ensemble_pred = np.mean(ens_preds, axis=0)

                calendar_day_idx = i + TIME_WINDOW - 1
                single_cache[calendar_day_idx] = single_pred
                ensemble_cache[calendar_day_idx] = ensemble_pred

        for (key, group) in profile_groups:
            profile_date = group["date"].iloc[0]
            profile_lat = group["LATITUDE"].iloc[0]
            profile_lon = group["LONGITUDE"].iloc[0]

            day_idx = (pd.Timestamp(profile_date) - pd.Timestamp(f"{year}-01-01")).days
            if day_idx not in single_cache or day_idx < 0:
                continue

            single_field = single_cache[day_idx]
            ensemble_field = ensemble_cache[day_idx]
            glorys_field = glorys_ds["temperature"].isel(time=min(day_idx, glorys_ds.sizes["time"] - 1)).values

            float_depths = group["PRES"].values
            float_temps = group["TEMP"].values
            valid = ~np.isnan(float_depths) & ~np.isnan(float_temps)
            if valid.sum() < 2:
                continue

            interp_temp_at_std_depths = np.interp(
                STANDARD_DEPTHS, float_depths[valid], float_temps[valid],
                left=np.nan, right=np.nan
            )

            for d_i, depth in enumerate(STANDARD_DEPTHS):
                true_val = interp_temp_at_std_depths[d_i]
                if np.isnan(true_val):
                    continue

                single_val = bilinear_sample(single_field[d_i], lat_vals, lon_vals, profile_lat, profile_lon)
                ens_val = bilinear_sample(ensemble_field[d_i], lat_vals, lon_vals, profile_lat, profile_lon)
                glorys_val = bilinear_sample(glorys_field[d_i], lat_vals, lon_vals, profile_lat, profile_lon)

                if not np.isnan(single_val):
                    results[depth]["single_err"].append((single_val - true_val) ** 2)
                if not np.isnan(ens_val):
                    results[depth]["ensemble_err"].append((ens_val - true_val) ** 2)
                if not np.isnan(glorys_val):
                    results[depth]["glorys_err"].append((glorys_val - true_val) ** 2)

        glorys_ds.close()

    print(f"\n{'Depth (m)':>10} {'N profiles':>11} {'Single RMSE':>13} {'Ensemble RMSE':>15} {'GLORYS RMSE':>13} {'Ens Gain':>10}")
    print("-" * 76)

    tot_single_err, tot_ens_err, tot_glorys_err, tot_n = 0.0, 0.0, 0.0, 0

    for depth in STANDARD_DEPTHS:
        s_errs = results[depth]["single_err"]
        e_errs = results[depth]["ensemble_err"]
        g_errs = results[depth]["glorys_err"]

        if len(s_errs) == 0:
            print(f"{depth:>10} {0:>11} {'N/A':>13} {'N/A':>15} {'N/A':>13} {'N/A':>10}")
            continue

        s_rmse = np.sqrt(np.mean(s_errs))
        e_rmse = np.sqrt(np.mean(e_errs))
        g_rmse = np.sqrt(np.mean(g_errs)) if g_errs else float("nan")
        ens_gain = s_rmse - e_rmse  # Positive means Ensemble improved over Single Model

        tot_single_err += np.sum(s_errs)
        tot_ens_err += np.sum(e_errs)
        tot_glorys_err += np.sum(g_errs)
        tot_n += len(s_errs)

        note = "  (Ensemble outperforms Single)" if ens_gain > 0.01 else ""
        print(f"{depth:>10} {len(s_errs):>11} {s_rmse:>13.4f} {e_rmse:>15.4f} {g_rmse:>13.4f} {ens_gain:>+10.4f}{note}")

    print("-" * 76)
    overall_s_rmse = np.sqrt(tot_single_err / tot_n)
    overall_e_rmse = np.sqrt(tot_ens_err / tot_n)
    overall_g_rmse = np.sqrt(tot_glorys_err / tot_n)
    overall_gain = overall_s_rmse - overall_e_rmse

    print(f"{'OVERALL':>10} {tot_n:>11} {overall_s_rmse:>13.4f} {overall_e_rmse:>15.4f} {overall_g_rmse:>13.4f} {overall_gain:>+10.4f}")
    print("\nSummary Interpretation:")
    print("  Ens Gain > 0: 3-seed ensemble reduces real-world error compared to single checkpoint.")
    print("  Ens Gain ~ 0: Seed ensembling has minimal impact on observation error.")
    print("  Ens Gain < 0: Single checkpoint outperformed ensemble average.")


if __name__ == "__main__":
    main()

