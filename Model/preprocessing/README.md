# OceanEmbed — Preprocessing, Step by Step

Run these four scripts **in order**. Each is memory-conscious (chunked/batched)
given the RAM constraints your GLORYS download already surfaced.

## Why bathymetry wasn't downloaded separately

The original plan referenced a `bathymetry.nc` file as if it were downloaded,
but no download script for it was ever actually created — a gap caught only
now. Rather than adding a 7th data source and account to manage, **Script 01
derives both bathymetry and the land-sea mask directly from GLORYS's own
depth-availability pattern**: wherever GLORYS's temperature data stops being
valid as depth increases at a given pixel, that's approximately the seafloor
depth there. This is a reasonable proxy (GLORYS's vertical grid is itself
built on a real ocean-model bathymetry) and keeps the pipeline self-contained.

## Run order

```bash
python 01_build_bathymetry_and_masks.py     # once — derives bathymetry.nc + land_sea_mask.nc from GLORYS 2015
python 02_build_surface_channels.py          # per year — builds the 19-channel input stack (2015-2023)
python 03_build_glorys_standard_depths.py    # per year — regrids GLORYS onto 15 standard depths (2015-2020 only)
python 04_build_depth_valid_mask.py          # once — builds the shelf-exclusion mask for the loss function
python 05_verify_preprocessing.py            # verify everything above before moving to training
```

## What each script produces

| Script | Output | Purpose |
|---|---|---|
| 01 | `data/masks/bathymetry.nc`, `data/masks/land_sea_mask.nc` | CoordConv bathymetry channel; land masking everywhere |
| 02 | `data/processed/processed_{year}.nc` (2015-2023) | The 19-channel daily input stack, ready for `dataset.py`'s `SurfaceOnlyDataset`/`SubsurfaceDataset` |
| 03 | `data/glorys_processed/GLORYS_STD_DEPTHS_{year}.nc` (2015-2020) | Training/validation/test labels at exactly the 15 standard depths |
| 04 | `data/masks/depth_valid_mask.nc` | Excludes shelf cells from loss/metrics at depths deeper than the local seafloor |

## Memory safety notes

- **Script 02** processes each year in 30-day batches, computing (materializing)
  only one batch at a time rather than a full year across 5 sources at once.
- **Script 03** processes GLORYS in 10-day batches (smaller than script 02's,
  since depth-resolved data is much heavier per day) and does the **spatial
  regrid before the depth regrid** — shrinking the array early keeps the more
  expensive depth-interpolation step cheaper.
- If you still hit memory errors on your machine, reduce `batch_size` further
  in either script (e.g. from 30→10 or 10→5) — smaller batches trade a bit of
  speed for a smaller peak memory footprint, with no effect on the final output.

## Before running Script 02 for the first time

**Variable-name resolution is now automatic.** Both Script 02 and Script 03
try several known aliases per field (e.g. SST tries `analysed_sst`, `sst`,
`SST` in order) and print every source's actual available variable names as
they run — so a naming mismatch shows up immediately as a clear error message
listing what IS available, rather than a bare `KeyError` partway through a
multi-year loop. You no longer need to manually inspect files beforehand,
though doing so is still fine if you want to double-check ahead of time.

**Time alignment is now robust to time-of-day differences.** An earlier
version of Script 02 aligned sources by exact timestamp, which failed
entirely (0 overlapping days) if one source was stamped at 00:00 and another
at 12:00 for "the same" calendar day. Both scripts now floor every timestamp
to midnight before aligning, so this can't happen again.

**SST unit conversion is now automatic and logged.** Script 02 checks the
actual sample values and only subtracts 273.15 (Kelvin→Celsius) if the data
looks like Kelvin (mean > 100) — printed either way, so you can confirm it
did the right thing rather than assuming.

## Script 05 — verification

Run this after 01–04 finish, before touching the training scripts:
```bash
python 05_verify_preprocessing.py
```
Checks: correct file presence/shape for every year, correct 19-channel
ordering, no leftover NaNs where there shouldn't be any, validity masks are
strictly binary, physical values fall in sane ranges (catches e.g. a missed
Kelvin conversion), GLORYS depth levels match your 15 standards, and the
depth-valid mask's ocean fraction decreases monotonically with depth (a
shallower shelf should never suddenly become "valid" at a deeper level).

## After all four finish

You're ready for the Stage 1 and Stage 2 training scripts (`src/train_stage1.py`,
`src/train_stage2.py`) — those already expect exactly this directory layout.

**Still needed before Stage 2 training specifically** (not part of these four
scripts): the monthly climatology (`climatology.nc`) and the per-pixel
inversion-frequency map (`inversion_freq_map.nc`), both computed FROM the
2015-2018 training portion of Script 03's output. These come next, after
preprocessing is confirmed working — say the word when you're ready for those.
