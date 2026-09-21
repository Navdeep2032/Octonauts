"""Verify the documented OceanEmbed tensor contract and gradient flow."""
import os
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CLIMATOLOGY_PATH,
    DEPTH_MLP_HIDDEN,
    ENCODER_BASE_CH,
    FOURIER_N_FREQS,
    STANDARD_DEPTHS,
)
from model import OceanEmbedModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with xr.open_dataset(CLIMATOLOGY_PATH) as ds:
        climatology = torch.tensor(
            np.nan_to_num(ds["climatology"].values, nan=0.0),
            dtype=torch.float32,
            device=device,
        )

    model = OceanEmbedModel(
        climatology_field=climatology,
        in_ch=19,
        t_window=14,
        base=ENCODER_BASE_CH,
        n_freqs=FOURIER_N_FREQS,
        depth_hidden=DEPTH_MLP_HIDDEN,
        standard_depths=STANDARD_DEPTHS,
    ).to(device)
    model.train()

    x = torch.randn(1, 19, 14, 101, 241, device=device)
    month_idx = torch.tensor([5], dtype=torch.long, device=device)
    output = model(x, month_idx, depths=STANDARD_DEPTHS)
    expected_shape = (1, len(STANDARD_DEPTHS), 101, 241)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Expected output {expected_shape}, got {tuple(output.shape)}")

    loss = output.square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not gradients or not any(g is not None and torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("No finite parameter gradients were produced")

    print(f"Architecture verification passed on {device}")
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print("Forward pass: PASS")
    print("Backward pass: PASS")


if __name__ == "__main__":
    main()
