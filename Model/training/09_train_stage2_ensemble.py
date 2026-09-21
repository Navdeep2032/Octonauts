"""
STEP 4: Address the overfitting seen in your single Stage 2 run (best val_loss
at epoch 29, then val_loss got WORSE while train_loss kept improving -- the
textbook overfitting signature) by training 3 models with different random
seeds and averaging their predictions at inference.

This does not "fix" overfitting in any one run -- it reduces its impact by
averaging out each individual run's overfit quirks, and the spread across
seeds tells you how much to trust any single number (if 3 seeds give wildly
different val_loss, that's itself useful information about how stable this
setup is on this amount of data).

Run this INSTEAD of your original single train_stage2.py call, not in
addition to it -- it's the same training loop, repeated 3 times with
different seeds and distinct checkpoint filenames.
"""
import torch
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (TRAIN_YEARS, VAL_YEARS, BATCH_SIZE, STAGE2_EPOCHS, LR_STAGE2,
                     ENCODER_BASE_CH, FOURIER_N_FREQS, DEPTH_MLP_HIDDEN, STANDARD_DEPTHS,
                     CHECKPOINT_DIR, LAMBDA_SMOOTH, LAMBDA_GRAD, LAMBDA_THERMO,
                     CLIMATOLOGY_PATH, N_ENSEMBLE)
from model import OceanEmbedModel
from dataset import SubsurfaceDataset
from losses import total_loss
from torch.utils.data import DataLoader
import xarray as xr

SEEDS = [42, 123, 2024][:N_ENSEMBLE]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_climatology(device):
    ds = xr.open_dataset(CLIMATOLOGY_PATH)
    clim_vals = np.nan_to_num(ds["climatology"].values, nan=0.0)
    return torch.tensor(clim_vals, dtype=torch.float32).to(device)


def load_inversion_freq_map(device):
    ds = xr.open_dataset("./data/inversion_freq_map.nc")
    return torch.tensor(ds["inversion_freq"].values, dtype=torch.float32).to(device)


def train_one_seed(seed, device, climatology, inversion_freq_map, depths):
    print(f"\n{'='*60}\nTraining seed {seed}\n{'='*60}")
    set_seed(seed)

    train_ds = SubsurfaceDataset(
        years=TRAIN_YEARS,
        glorys_path_template="./data/glorys_processed/GLORYS_STD_DEPTHS_{year}.nc",
        depth_mask_path="./data/masks/depth_valid_mask.nc",
    )
    val_ds = SubsurfaceDataset(
        years=VAL_YEARS,
        glorys_path_template="./data/glorys_processed/GLORYS_STD_DEPTHS_{year}.nc",
        depth_mask_path="./data/masks/depth_valid_mask.nc",
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = OceanEmbedModel(
        climatology_field=climatology, in_ch=19, t_window=14, base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS, depth_hidden=DEPTH_MLP_HIDDEN, standard_depths=STANDARD_DEPTHS,
    ).to(device)

    ckpt = torch.load(f"{CHECKPOINT_DIR}/stage1_pretrained.pt", map_location=device, weights_only=True)
    model.temporal.load_state_dict(ckpt["temporal"])
    model.hires_branch.load_state_dict(ckpt["hires"])
    model.encoder.load_state_dict(ckpt["encoder"])
    model.bottleneck_attn.load_state_dict(ckpt["bn_attn"])

    opt = torch.optim.Adam(model.parameters(), lr=LR_STAGE2)

    best_val_loss = float("inf")
    ckpt_path = f"{CHECKPOINT_DIR}/stage2_seed{seed}.pt"
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = np.nan_to_num(batch["y"].numpy(), nan=0.0)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            mask = batch["mask"].to(device)
            month_idx = batch["month_idx"].to(device)

            pred = model(x, month_idx, depths=STANDARD_DEPTHS)
            loss, _ = total_loss(pred, y, mask, depths, inversion_freq_map,
                                  lambda_smooth=LAMBDA_SMOOTH, lambda_grad=LAMBDA_GRAD, lambda_thermo=LAMBDA_THERMO)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = np.nan_to_num(batch["y"].numpy(), nan=0.0)
                y = torch.tensor(y, dtype=torch.float32).to(device)
                mask = batch["mask"].to(device)
                month_idx = batch["month_idx"].to(device)
                pred = model(x, month_idx, depths=STANDARD_DEPTHS)
                loss, _ = total_loss(pred, y, mask, depths, inversion_freq_map,
                                      lambda_smooth=LAMBDA_SMOOTH, lambda_grad=LAMBDA_GRAD, lambda_thermo=LAMBDA_THERMO)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        improved = val_loss < best_val_loss
        print(f"  seed {seed} epoch {epoch+1}/{STAGE2_EPOCHS} val_loss: {val_loss:.4f}" +
              ("  -> new best" if improved else ""))
        if improved:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"Seed {seed} done. Best val_loss: {best_val_loss:.4f}, saved to {ckpt_path}")
    return best_val_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depths = torch.tensor(STANDARD_DEPTHS, dtype=torch.float32, device=device)
    climatology = load_climatology(device)
    inversion_freq_map = load_inversion_freq_map(device)

    results = {}
    for seed in SEEDS:
        results[seed] = train_one_seed(seed, device, climatology, inversion_freq_map, depths)

    print(f"\n{'='*60}\nEnsemble training summary\n{'='*60}")
    for seed, val_loss in results.items():
        print(f"  seed {seed}: best val_loss = {val_loss:.4f}")
    vals = list(results.values())
    print(f"\n  mean: {np.mean(vals):.4f}, std: {np.std(vals):.4f}")
    print(f"  (a large std relative to the mean means this setup is sensitive to random")
    print(f"   initialization on the current amount of training data -- worth knowing")
    print(f"   regardless of what the ARGO evaluation shows)")


if __name__ == "__main__":
    main()
