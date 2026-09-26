# OceanEmbed — Project Handoff Document

**Purpose of this file:** complete technical state of the project, written so another AI agent (or a human picking this up cold) can continue without re-deriving context. Covers what's built, what's verified, every real bug found and fixed (with root cause, so they aren't re-discovered), current results, and the exact remaining task list.

**Environment:** Windows machine (`D:\Octonauts\Model`), Python venv, PyTorch installed via `pip` (not `npm` — this was a real early confusion, see Known Issues). No GPU confirmed yet as of last check-in (`torch.cuda.is_available()` result unknown to this document — verify before assuming CPU-only).

---

## 1. Project Goal

Reconstruct subsurface ocean temperature (0–1000m, 15 standard depths) for the North Indian Ocean (5–30°N, 45–105°E) at 0.25°/daily resolution, using only surface satellite observations (SST, SSS, SSH, currents, winds), via a deep-learning "satellite embedding" framework. Built for a Smart India Hackathon submission under an INCOIS/MoES problem statement. Validated against GLORYS reanalysis (training label) and independent ARGO float observations (ground truth check).

---

## 2. Current Status Summary

| Stage | Status |
|---|---|
| Data download (6 sources) | ✅ Complete, verified |
| Preprocessing (masks, 19-channel stack, GLORYS regrid, climatology, inversion map) | ✅ Complete, verified |
| Stage 1 training (self-supervised pretrain) | ✅ Complete |
| Stage 2 training (single run) | ✅ Complete — overfit after epoch 29 |
| Stage 2 training (official 3-seed ensemble) | ✅ Complete — full documented loss, stable across seeds |
| Architecture verification | ✅ Complete — documented input/output contract and backward pass verified on CUDA |
| Skill score vs. climatology | ✅ Complete — **0.1995 overall** for the official ensemble |
| ARGO data download | ✅ Complete — 2019 and 2020 profiles downloaded |
| ARGO evaluation (real independent check) | ✅ Complete — single model and official ensemble evaluated |
| Ensemble-averaged ARGO evaluation | ✅ Complete — official 3-seed ensemble evaluated |
| INCOIS OMNI validation | ❌ Not started — no public API exists, requires manual data request to INCOIS |
| MC-dropout uncertainty quantification | ❌ Designed, never implemented |
| MLD climatology (Stage 1 auxiliary task) | ⚠️ Skipped by design decision — Stage 1 ran surface-reconstruction-only |
| GLORYS bias correction (reprocessed vs interim) | ❌ Designed, never implemented |
| Final report / slides | ⚠️ Partial — final measured results are now available; plots and presentation updates remain |
| Final plots and tables | ✅ Complete — depth skill, ARGO RMSE, and summary CSV/PNG outputs generated in `plots/` |

---

## 3. Directory Structure (as it stands)

```
Model/
├── data/
│   ├── raw/{sst,sss,ssh,currents,wind,glorys}/*.nc   -- downloaded, verified
│   ├── processed/processed_{2015-2023}.nc             -- 19-channel stacks, verified
│   ├── glorys_processed/GLORYS_STD_DEPTHS_{2015-2020}.nc  -- verified
│   ├── masks/{land_sea_mask,bathymetry,depth_valid_mask}.nc  -- verified
│   ├── climatology.nc                                  -- verified (2015-2018 only)
│   ├── inversion_freq_map.nc                            -- verified (2015-2018 only)
│   └── argo/ARGO_NIO_{2019,2020}.csv                    -- downloaded for independent validation
├── data_download/
│   ├── 01_download_sst.py ... 07_verify_downloads.py    -- all working, verified
│   └── 08_download_argo.py                               -- run successfully
├── preprocessing/
│   ├── 01_build_bathymetry_and_masks.py ... 07_build_inversion_freq_map.py  -- all working, verified
│   └── 05_verify_preprocessing.py                        -- all checks passing
├── training/
│   ├── config.py, model.py, losses.py, dataset.py       -- working
│   ├── train_stage1.py                                    -- run successfully once
│   ├── train_stage2.py                                     -- run successfully once (single-seed)
│   ├── 08_skill_score_check.py                             -- legacy single-checkpoint evaluation
│   ├── 09_train_stage2_ensemble.py                          -- official full-loss 3-seed training
│   ├── 13_verify_architecture.py                            -- input/output and gradient smoke test
│   └── 14_evaluate_official_ensemble.py                     -- official GLORYS ensemble evaluation
├── evaluation/
│   ├── 09_evaluate_against_argo.py                          -- single-model ARGO evaluation
│   └── 10_evaluate_ensemble_against_argo.py                 -- official ensemble ARGO evaluation
└── checkpoints/
    ├── stage1_pretrained.pt
    ├── stage2_best.pt              -- from single run, best at epoch 29, val_loss 0.7748
    ├── stage2_seed42.pt            -- official full-loss run, best val_loss 0.8259
    ├── stage2_seed123.pt           -- official full-loss run, best val_loss 0.8106
    └── stage2_seed2024.pt          -- official full-loss run, best val_loss 0.8227
```

---

## 4. Technical Architecture (as implemented)

**Model:** CNN encoder (with bottleneck self-attention + global-context pooling) → Attention U-Net-style decoder (sub-pixel upsampling + parallel high-resolution branch) → Fourier depth-conditioned implicit output head → climatology-anchored output. The documented tensor contract was verified with a CUDA forward/backward smoke test: `[B,19,14,H,W] → [B,15,H,W]`.

**Inputs:** 19 channels × 14-day window — 7 surface vars (SST, SSS, SLA, U/V current, U/V wind) + 3 CoordConv (lat, lon, bathymetry) + 2 seasonal (DOY sin/cos) + 7 validity masks.

**Target:** Model internally predicts ΔT (anomaly from monthly climatology); a non-trainable head adds climatology back so the shipped output is absolute temperature.

**Loss:** masked MSE + spacing-aware smoothness + gradient-preservation + adaptive (GLORYS-derived, ENSO-modulation-ready but not yet wired to real climate indices) inversion penalty.

**Training:** Stage 1 self-supervised (2021-2023 surface-only) → official Stage 2 supervised fine-tuning (GLORYS 2015-2018 train / 2019 validation), using the full documented composite loss. The official run used seeds 42, 123, and 2024; best validation losses were 0.8259, 0.8106, and 0.8227 (mean 0.8197, standard deviation 0.0066). No ARGO/OMNI data was used in training — both are reserved strictly for validation.

**Full technical rationale, flaw analysis, and design decisions are in the companion documents already produced earlier in this project:** `OceanEmbed_Technical_Approach.md`, `OceanEmbed_Final_Approach_Complete.md`, `OceanEmbed_Model_Architecture_Deep_Dive.md`, `OceanEmbed_Complete_Technical_Solution.md` — this handoff file summarizes status/next-steps; those files hold the full "why" behind every design choice.

---

## 5. Every Real Bug Found and Fixed (so they aren't re-discovered)

This project was built and debugged interactively against real execution — every item below was a genuine failure encountered when code actually ran, not a hypothetical. Listed with root cause so a future agent doesn't waste time re-diagnosing the same class of issue.

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | Decoder channel-count mismatch | `SubPixelUpsample` preserved input channel count instead of reducing it to match the skip connection it concatenates with | Added explicit `out_ch` parameter, sized to match each skip connection |
| 2 | Climatology batch-indexing silently wrong (not a crash) | Indexing `clim[depth_idx,:,:,month_idx]` with a batched `month_idx` tensor produces `[H,W,B]`, not `[B,H,W]` — silently broadcasts wrong values | Permute to `[12,H,W]` first, then fancy-index on the month axis |
| 3 | `scipy` missing | `xarray.interp()` depends on `scipy.interpolate` under the hood, not bundled with `xarray` | `pip install scipy` |
| 4 | Dimension rename collision | Some datasets (OSTIA/CMEMS style) have `latitude` as both dim name and only coordinate; OSCAR/CCMP (PO.DAAC) have `latitude` as a bare dimension with a SEPARATE `lat` coordinate variable — a plain `.rename()` collides in the second case | Check which case applies; use `.swap_dims()` for the PO.DAAC case, `.rename()` otherwise |
| 5 | Zero overlapping days after alignment | `xr.align(join="inner")` requires exact timestamp match; sources had different times-of-day (00:00 vs some other hour) | Resample every source to daily mean (`resample(time="1D").mean()`) before aligning — also fixes CCMP's 6-hourly duplicates |
| 6 | Still zero overlap after resampling | OSCAR uses `cftime.DatetimeJulian`, other sources use `numpy.datetime64` — these don't reliably align even with identical printed dates | Explicitly convert every cftime object to `datetime64[ns]` via its year/month/day/hour/minute/second fields |
| 7 | GLORYS depth grid only reached 902.3m despite requesting max_depth=1000 | Copernicus Marine's depth subset boundary behavior cuts off before including the exact requested value | Request `maximum_depth=1100` (margin past 1000m) so a bracketing native level always exists for safe interpolation, never extrapolation |
| 8 | Verification script OOM'd (21.2 GiB allocation) | Loading a full year of multi-depth-level GLORYS data as float64 for a NaN check | Sample 5 time steps instead of loading the whole array |
| 9 | Land-sea mask rename bug | Bathymetry array had its dims correctly renamed before regridding; land-sea mask array (built right after) never got the same treatment — straightforward copy-paste gap | Apply the same rename/swap_dims logic to both arrays consistently |
| 10 | Singleton depth dim / transposed dimension order (user-caught, not this agent) | Some SSS product variants keep a leftover size-1 `depth` dim on the surface field; OSCAR stores horizontal dims as `(lon, lat)` not `(lat, lon)` — neither raises an error, both silently corrupt the stacked tensor | `.squeeze(drop=True)` + `.transpose("time","lat","lon")` enforced on every source before stacking |
| 11 | Windows `DataLoader` multiprocessing crash (`pickle data was truncated`) | Windows spawns fresh worker processes (doesn't fork like Linux) and must pickle the entire Dataset object; xarray/NetCDF-backed datasets don't survive this reliably | Set `num_workers=0` in every `DataLoader` |
| 12 | Training loss = NaN from epoch 1 | GLORYS target and climatology arrays contain real NaN over land; `0 × NaN = NaN` in floating point, so masking alone doesn't zero these out — one NaN anywhere poisons the entire loss sum, and once weights go NaN via backprop they're permanently dead | `np.nan_to_num(..., nan=0.0)` applied at data-loading time to both the GLORYS target array and the climatology array |
| 13 | `npm i torch` did nothing useful | `npm` is the Node.js package manager, unrelated to Python | `pip install torch` |
| 14 | ARGO evaluation would have silently compared wrong days to each other (caught before running) | `SubsurfaceDataset.__getitem__(idx)` uses `target_day = idx + TIME_WINDOW - 1` (needs 14 days of history before its first prediction) — loader batch index 0 is calendar day 13, not day 0. First draft cached predictions by loader index and compared against calendar-day indices directly | Cache predictions keyed by `loader_index + TIME_WINDOW - 1` (the actual calendar day), not the raw loader index |

**Pattern worth noting for whoever continues this:** most of these bugs (4, 5, 6, 10) stem from the same underlying issue — different satellite data providers structure their NetCDF files differently (dimension names, calendar types, dimension order), and none of these differences raise errors on their own; they silently produce wrong results. Any NEW data source added to this pipeline should be checked against this same category of issue before being trusted.

---

## 6. Current Results (official run)

### Official training

- **Architecture smoke test:** passed on CUDA for input `[1, 19, 14, 101, 241]` and output `[1, 15, 101, 241]`, including backward propagation with finite gradients.
- **Official full-loss three-seed run:** seeds 42, 123, and 2024, initialized from `stage1_pretrained.pt`.
- **Best validation losses:** seed 42 = 0.8259, seed 123 = 0.8106, seed 2024 = 0.8227; mean = 0.8197, standard deviation = 0.0066.
- **Official checkpoints:** `stage2_seed42.pt`, `stage2_seed123.pt`, and `stage2_seed2024.pt`.

### GLORYS validation

The official ensemble evaluation (`training/14_evaluate_official_ensemble.py`) on the 2019 validation year produced:

- **Ensemble RMSE:** 0.8746°C
- **Climatology RMSE:** 1.0926°C
- **Overall skill score:** 0.1995, equivalent to approximately 20% RMSE reduction versus climatology.

Skill was positive at every standard depth. It was strongest around 75–150 m (23.81–25.83%) and small at 500–1000 m (1.94–2.38%).

### Independent ARGO validation

The official ensemble evaluation (`evaluation/10_evaluate_ensemble_against_argo.py`) used independent ARGO profiles from 2019 and 2020:

- **Official ensemble ARGO RMSE:** 1.1859°C
- **Legacy single-checkpoint ARGO RMSE:** 1.1906°C
- **Ensemble improvement:** 0.0047°C
- **GLORYS-versus-ARGO RMSE:** 0.6476°C

The ensemble improvement is positive but small. Model error is substantially larger than GLORYS-versus-ARGO error, especially in the thermocline and deep layers, so the ARGO results should not be described as near-observation accuracy.

---

## 7. Remaining Task List

### Completed validation work

1. Architecture parity and forward/backward smoke test.
2. Official full-loss three-seed training.
3. Per-depth GLORYS skill-score evaluation.
4. Independent ARGO download and evaluation.
5. Official ensemble comparison against the legacy single checkpoint.
6. GLORYS-versus-ARGO error comparison.

### Remaining optional or presentation work

7. **Final plots and tables:** ✅ Completed — CSV and PNG artifacts generated under `plots/`.
8. **Final report/slide deck update:** replace projected values with measured results and state deep-ocean limitations explicitly.
9. **MC-dropout uncertainty quantification:** add stochastic inference and report prediction mean plus uncertainty bands.
10. **INCOIS OMNI validation:** requires a manual data request to INCOIS because no public bulk API is available.
11. **GLORYS reprocessed-vs-interim bias correction:** required only for deployment on a different GLORYS product/version.
12. **MLD climatology auxiliary Stage 1 task:** optional; Stage 1 currently uses surface reconstruction only.
13. **Overfitting mitigation experiments:** optional future work, to be evaluated against ARGO rather than only GLORYS validation loss.

---

## 8. Final Plot / Table Artifacts Generated

The following official artifacts were generated in the project root under `plots/`:

- `plots/depth_skill_score.png` — per-depth skill score vs climatology
- `plots/argo_rmse_by_depth.png` — ARGO RMSE by depth for single model, ensemble, and GLORYS
- `plots/depth_skill_table.csv` — per-depth skill and climatology RMSE table
- `plots/argo_rmse_table.csv` — ARGO RMSE by depth table
- `plots/overall_summary_table.csv` — overall metrics summary
- `plots/final_summary.md` — concise markdown summary of the final results

These were produced from the official Stage 2 ensemble checkpoints and match the measured final validation numbers recorded in this document.

## 9. Environment Notes for Whoever Continues This

- Windows machine, PowerShell, Python venv at `D:\Octonauts\venv`.
- Package manager confusion already occurred once (`npm i torch` instead of `pip install torch`) — if a new agent is instructing the user, be explicit that Python packages need `pip`, not `npm`.
- `DataLoader(num_workers=...)` MUST stay at `0` on this machine — Windows multiprocessing pickling of xarray-backed Datasets fails reliably otherwise (see bug #11 above). Do not "optimize" this back to a positive number without testing.
- GPU execution is confirmed; the architecture smoke test and official training/evaluation ran on CUDA.
- All file paths in the scripts use relative paths (`./data/...`) — scripts must be run with the working directory set correctly (typically from inside `training/`, `data_download/`, or `preprocessing/` respectively, matching how each script's relative paths were written and tested).

---

## 10. How to Resume

The core architecture, final official training, and final figures are complete. The next recommended work is the remaining presentation/report polishing:

```bash
python Model/training/14_evaluate_official_ensemble.py
python Model/evaluation/10_evaluate_ensemble_against_argo.py
python Model/training/15_generate_final_plots.py
```

These commands reproduce the official GLORYS and ARGO validation tables and regenerate the final plots under `plots/`. After that, the optional future work remains limited to uncertainty estimation, OMNI validation, and deeper-depth bias correction.
