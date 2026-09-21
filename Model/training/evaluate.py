"""
Three-tier validation: gridded ARGO -> raw ARGO profiles -> INCOIS OMNI moorings.
Reports per-depth x per-season x per-basin RMSE/R2/bias, skill score vs climatology,
and GLORYS-vs-observation error alongside model-vs-observation error.
NEVER used during training — this is the only place ARGO/OMNI data is touched.
"""
import torch
import numpy as np
import xarray as xr
from config import STANDARD_DEPTHS, CHECKPOINT_DIR, TEST_YEARS, ENCODER_BASE_CH, FOURIER_N_FREQS, DEPTH_MLP_HIDDEN
from model import OceanEmbedModel


def rmse(pred, true, mask):
    diff2 = ((pred - true) ** 2) * mask
    return np.sqrt(diff2.sum() / max(mask.sum(), 1))


def r_squared(pred, true, mask):
    m = mask.astype(bool)
    if m.sum() < 2:
        return np.nan
    ss_res = np.sum((true[m] - pred[m]) ** 2)
    ss_tot = np.sum((true[m] - true[m].mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-9)


def bias(pred, true, mask):
    m = mask.astype(bool)
    return (pred[m] - true[m]).mean() if m.sum() > 0 else np.nan


def skill_score(rmse_model, rmse_climatology):
    return 1 - (rmse_model / rmse_climatology) if rmse_climatology > 0 else np.nan


def load_model(climatology, device):
    model = OceanEmbedModel(
        climatology_field=climatology, in_ch=19, t_window=14, base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS, depth_hidden=DEPTH_MLP_HIDDEN, standard_depths=STANDARD_DEPTHS,
    ).to(device)
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/stage2_best.pt", map_location=device))
    model.eval()
    return model


def evaluate_against_observations(model, obs_dataset, glorys_dataset, climatology_rmse_map, device, tier_name):
    """
    obs_dataset: yields (x_window, month_idx, obs_temp, obs_mask, season, basin_id) per sample
                 obs_temp/obs_mask already interpolated to model grid & standard depths
    glorys_dataset: matching GLORYS values at the same points, for the inherited-error comparison
    """
    print(f"\n=== Validation Tier: {tier_name} ===")
    results = {}

    with torch.no_grad():
        for x, month_idx, obs_temp, obs_mask, season, basin, glorys_val in obs_dataset:
            x = x.unsqueeze(0).to(device)
            month_idx_t = torch.tensor([month_idx], device=device)
            pred = model(x, month_idx_t, depths=STANDARD_DEPTHS).cpu().numpy()[0]  # [n_depths, H, W]

            for d_idx, depth in enumerate(STANDARD_DEPTHS):
                key = (depth, season, basin)
                results.setdefault(key, {"pred": [], "obs": [], "glorys": [], "mask": []})
                results[key]["pred"].append(pred[d_idx])
                results[key]["obs"].append(obs_temp[d_idx])
                results[key]["glorys"].append(glorys_val[d_idx])
                results[key]["mask"].append(obs_mask[d_idx])

    print(f"{'Depth':>6} {'Season':>8} {'Basin':>10} {'Model RMSE':>11} {'GLORYS RMSE':>12} {'R2':>6} {'Bias':>7} {'Skill':>7}")
    for (depth, season, basin), vals in sorted(results.items()):
        pred_arr = np.array(vals["pred"])
        obs_arr = np.array(vals["obs"])
        glorys_arr = np.array(vals["glorys"])
        mask_arr = np.array(vals["mask"])

        model_rmse = rmse(pred_arr, obs_arr, mask_arr)
        glorys_rmse = rmse(glorys_arr, obs_arr, mask_arr)  # inherited reanalysis error
        r2 = r_squared(pred_arr, obs_arr, mask_arr)
        b = bias(pred_arr, obs_arr, mask_arr)
        clim_rmse = climatology_rmse_map.get(depth, np.nan)
        skill = skill_score(model_rmse, clim_rmse) if not np.isnan(clim_rmse) else np.nan

        print(f"{depth:>6} {season:>8} {basin:>10} {model_rmse:>11.3f} {glorys_rmse:>12.3f} {r2:>6.3f} {b:>7.3f} {skill:>7.3f}")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Load your climatology, ARGO/OMNI datasets, and GLORYS comparison arrays, "
          "then call evaluate_against_observations() for each of the three validation tiers "
          "(gridded ARGO, raw ARGO profiles, INCOIS OMNI) as described in the technical approach doc.")
