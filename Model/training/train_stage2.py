"""
Stage 2: supervised fine-tuning on GLORYS-labeled days (2015-2018 train / 2019 val / 2020 test).
Loads the Stage 1 pretrained encoder weights, attaches the full decoder + implicit depth head,
and trains with the composite loss. NO ARGO/OMNI data used here — validation only.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
from config import (TRAIN_YEARS, VAL_YEARS, BATCH_SIZE, STAGE2_EPOCHS, LR_STAGE2,
                     ENCODER_BASE_CH, FOURIER_N_FREQS, DEPTH_MLP_HIDDEN, STANDARD_DEPTHS,
                     CHECKPOINT_DIR, LAMBDA_SMOOTH, LAMBDA_GRAD, LAMBDA_THERMO,
                     CLIMATOLOGY_PATH)
from model import OceanEmbedModel
from dataset import SubsurfaceDataset
from losses import total_loss


def load_climatology():
    """Precomputed [n_depths, H, W, 12] monthly climatology from GLORYS training years."""
    import xarray as xr
    ds = xr.open_dataset(CLIMATOLOGY_PATH)
    clim_vals = np.nan_to_num(ds["climatology"].values, nan=0.0)
    return torch.tensor(clim_vals, dtype=torch.float32)


def load_inversion_freq_map():
    """Precomputed [H, W] per-pixel inversion-frequency map from GLORYS training data."""
    import xarray as xr
    ds = xr.open_dataset("./data/inversion_freq_map.nc")
    return torch.tensor(ds["inversion_freq"].values, dtype=torch.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    depths = torch.tensor(STANDARD_DEPTHS, dtype=torch.float32, device=device)

    climatology = load_climatology().to(device)
    inversion_freq_map = load_inversion_freq_map().to(device)

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
        climatology_field=climatology,
        in_ch=19, t_window=14, base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS, depth_hidden=DEPTH_MLP_HIDDEN,
        standard_depths=STANDARD_DEPTHS,
    ).to(device)

    # Load Stage 1 pretrained encoder weights
    ckpt = torch.load(f"{CHECKPOINT_DIR}/stage1_pretrained.pt", map_location=device)
    model.temporal.load_state_dict(ckpt["temporal"])
    model.hires_branch.load_state_dict(ckpt["hires"])
    model.encoder.load_state_dict(ckpt["encoder"])
    model.bottleneck_attn.load_state_dict(ckpt["bn_attn"])
    print("Loaded Stage 1 pretrained weights into encoder.")

    opt = torch.optim.Adam(model.parameters(), lr=LR_STAGE2)

    best_val_loss = float("inf")
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        running = {"total": 0.0, "recon": 0.0, "smooth": 0.0, "grad_preserve": 0.0, "inv_penalty": 0.0}
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            month_idx = batch["month_idx"].to(device)

            pred = model(x, month_idx, depths=STANDARD_DEPTHS)
            loss, parts = total_loss(
                pred, y, mask, depths, inversion_freq_map,
                lambda_smooth=LAMBDA_SMOOTH, lambda_grad=LAMBDA_GRAD, lambda_thermo=LAMBDA_THERMO,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in running:
                running[k] += parts[k]

        n = len(train_loader)
        print(f"[Stage 2] Epoch {epoch+1}/{STAGE2_EPOCHS} - " +
              " ".join(f"{k}:{v/n:.4f}" for k, v in running.items()))

        # quick validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mask = batch["mask"].to(device)
                month_idx = batch["month_idx"].to(device)
                pred = model(x, month_idx, depths=STANDARD_DEPTHS)
                loss, _ = total_loss(pred, y, mask, depths, inversion_freq_map,
                                      lambda_smooth=LAMBDA_SMOOTH, lambda_grad=LAMBDA_GRAD,
                                      lambda_thermo=LAMBDA_THERMO)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))
        print(f"           val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/stage2_best.pt")
            print("           -> saved new best checkpoint")

    print("Stage 2 training complete.")


if __name__ == "__main__":
    main()
