"""
Build the depth-varying validity mask: for each of the 15 standard depths,
mark a grid cell as INVALID (0) if the local bathymetry is shallower than that
depth -- prevents shelf regions from corrupting the loss/metrics at depths
that physically don't exist there (e.g. a 200m-deep shelf cell has no real
'700m' or '1000m' value to learn from).

Run once, after 01_build_bathymetry_and_masks.py.
"""
import numpy as np
import xarray as xr
import os

MASK_DIR = "./data/masks"
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def main():
    bathymetry = xr.open_dataset(f"{MASK_DIR}/bathymetry.nc")["bathymetry"]  # [lat, lon], meters
    land_sea_mask = xr.open_dataset(f"{MASK_DIR}/land_sea_mask.nc")["mask"]  # [lat, lon], 1=ocean

    H, W = bathymetry.shape
    depth_valid = np.zeros((len(STANDARD_DEPTHS), H, W), dtype=np.float32)

    for i, depth in enumerate(STANDARD_DEPTHS):
        # valid where: it's ocean AND the seafloor is at least this deep
        depth_valid[i] = ((land_sea_mask.values > 0.5) & (bathymetry.values >= depth)).astype(np.float32)
        pct_valid = depth_valid[i].mean() * 100
        print(f"Depth {depth}m: {pct_valid:.1f}% of grid cells valid")

    out_ds = xr.Dataset(
        {"mask": (("depth", "lat", "lon"), depth_valid)},
        coords={"depth": STANDARD_DEPTHS, "lat": bathymetry.lat.values, "lon": bathymetry.lon.values}
    )
    out_ds.to_netcdf(f"{MASK_DIR}/depth_valid_mask.nc")
    print(f"\nSaved {MASK_DIR}/depth_valid_mask.nc")
    print("Sanity check: valid fraction should DECREASE as depth increases "
          "(shallower shelf cells drop out at deeper levels).")


if __name__ == "__main__":
    main()
