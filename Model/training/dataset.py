"""
PyTorch Dataset classes.
SurfaceOnlyDataset  -> Stage 1 self-supervised pretraining (no subsurface labels)
SubsurfaceDataset   -> Stage 2 supervised fine-tuning (GLORYS-labeled days)
Both slice a 14-day rolling window from the preprocessed per-year NetCDFs.
"""
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from config import TIME_WINDOW, PROCESSED_DIR, STANDARD_DEPTHS


class SurfaceOnlyDataset(Dataset):
    """Stage 1: returns a 14-day surface-channel window + the target = last day's surface frame
    (used for the reconstruction pretext task) + a mixed-layer-depth climatology target."""

    def __init__(self, years, mld_climatology=None):
        self.data = xr.concat(
            [xr.open_dataset(f"{PROCESSED_DIR}/processed_{y}.nc") for y in years],
            dim="time"
        )["channels"]
        self.mld_clim = mld_climatology  # optional [H, W, 12] array
        self.n_days = self.data.shape[0]

    def __len__(self):
        return max(0, self.n_days - TIME_WINDOW)

    def __getitem__(self, idx):
        window = self.data.isel(time=slice(idx, idx + TIME_WINDOW)).values  # [14, 19, H, W]
        window = np.transpose(window, (1, 0, 2, 3))  # -> [19, 14, H, W]
        target_surface = window[:7, -1]  # last day's 7 real surface vars, reconstruction target

        item = {
            "x": torch.tensor(window, dtype=torch.float32),
            "target_surface": torch.tensor(target_surface, dtype=torch.float32),
        }
        if self.mld_clim is not None:
            month_idx = int(self.data.time[idx + TIME_WINDOW - 1].dt.month.item()) - 1
            item["mld_target"] = torch.tensor(self.mld_clim[:, :, month_idx], dtype=torch.float32)
        return item


class SubsurfaceDataset(Dataset):
    """Stage 2: returns a 14-day surface window + GLORYS subsurface temperature label
    at all standard depths + the depth-varying validity mask + month index."""

    def __init__(self, years, glorys_path_template, depth_mask_path=None):
        self.surface = xr.concat(
            [xr.open_dataset(f"{PROCESSED_DIR}/processed_{y}.nc") for y in years],
            dim="time"
        )["channels"]
        self.glorys = xr.concat(
            [xr.open_dataset(glorys_path_template.format(year=y)) for y in years],
            dim="time"
        )["temperature"]  # expected dims: [time, depth, lat, lon]
        self.depth_mask = xr.open_dataset(depth_mask_path)["mask"].values if depth_mask_path else None
        self.n_days = self.surface.shape[0]

    def __len__(self):
        return max(0, self.n_days - TIME_WINDOW)

    def __getitem__(self, idx):
        window = self.surface.isel(time=slice(idx, idx + TIME_WINDOW)).values
        window = np.transpose(window, (1, 0, 2, 3))  # [19, 14, H, W]

        target_day = idx + TIME_WINDOW - 1
        target_temp = np.nan_to_num(self.glorys.isel(time=target_day).values, nan=0.0)  # [n_depths, H, W]
        month_idx = int(self.surface.time[target_day].dt.month.item()) - 1

        mask = self.depth_mask if self.depth_mask is not None else np.ones_like(target_temp)

        return {
            "x": torch.tensor(window, dtype=torch.float32),
            "y": torch.tensor(target_temp, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "month_idx": torch.tensor(month_idx, dtype=torch.long),
        }
