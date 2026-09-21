"""Evaluate the official full-loss three-seed ensemble against GLORYS validation."""
import os
import sys

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CHECKPOINT_DIR,
    CLIMATOLOGY_PATH,
    DEPTH_MLP_HIDDEN,
    ENCODER_BASE_CH,
    FOURIER_N_FREQS,
    STANDARD_DEPTHS,
)
from dataset import SubsurfaceDataset
from model import OceanEmbedModel


SEEDS = [42, 123, 2024]


def build_model(climatology, device):
    return OceanEmbedModel(
        climatology_field=climatology,
        in_ch=19,
        t_window=14,
        base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS,
        depth_hidden=DEPTH_MLP_HIDDEN,
        standard_depths=STANDARD_DEPTHS,
    ).to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with xr.open_dataset(CLIMATOLOGY_PATH) as ds:
        climatology_values = np.nan_to_num(ds["climatology"].values, nan=0.0)
    climatology = torch.tensor(climatology_values, dtype=torch.float32, device=device)

    models = []
    for seed in SEEDS:
        model = build_model(climatology, device)
        path = os.path.join(CHECKPOINT_DIR, f"stage2_seed{seed}.pt")
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)

    dataset = SubsurfaceDataset(
        years=[2019],
        glorys_path_template="./data/glorys_processed/GLORYS_STD_DEPTHS_{year}.nc",
        depth_mask_path="./data/masks/depth_valid_mask.nc",
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    model_sq_err = np.zeros(len(STANDARD_DEPTHS), dtype=np.float64)
    clim_sq_err = np.zeros(len(STANDARD_DEPTHS), dtype=np.float64)
    valid_count = np.zeros(len(STANDARD_DEPTHS), dtype=np.float64)
    clim_by_month = climatology.permute(3, 0, 1, 2)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            month_idx = batch["month_idx"].to(device)
            predictions = [
                model(x, month_idx, depths=STANDARD_DEPTHS) for model in models
            ]
            prediction = torch.stack(predictions, dim=0).mean(dim=0)
            clim_prediction = clim_by_month[month_idx]
            for depth_index in range(len(STANDARD_DEPTHS)):
                valid = mask[:, depth_index]
                model_sq_err[depth_index] += (
                    ((prediction[:, depth_index] - y[:, depth_index]) ** 2) * valid
                ).sum().item()
                clim_sq_err[depth_index] += (
                    ((clim_prediction[:, depth_index] - y[:, depth_index]) ** 2) * valid
                ).sum().item()
                valid_count[depth_index] += valid.sum().item()

    print(f"{'Depth (m)':>10} {'Ensemble RMSE':>15} {'Clim RMSE':>12} {'Skill':>10}")
    print("-" * 52)
    total_model = total_clim = total_count = 0.0
    for index, depth in enumerate(STANDARD_DEPTHS):
        model_rmse = np.sqrt(model_sq_err[index] / valid_count[index])
        clim_rmse = np.sqrt(clim_sq_err[index] / valid_count[index])
        skill = 1.0 - model_rmse / clim_rmse
        print(f"{depth:>10} {model_rmse:>15.4f} {clim_rmse:>12.4f} {skill:>10.4f}")
        total_model += model_sq_err[index]
        total_clim += clim_sq_err[index]
        total_count += valid_count[index]

    model_rmse = np.sqrt(total_model / total_count)
    clim_rmse = np.sqrt(total_clim / total_count)
    print("-" * 52)
    print(f"{'OVERALL':>10} {model_rmse:>15.4f} {clim_rmse:>12.4f} {1.0 - model_rmse / clim_rmse:>10.4f}")


if __name__ == "__main__":
    main()
