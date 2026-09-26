# OceanEmbed

OceanEmbed is a deep-learning pipeline for reconstructing daily subsurface ocean temperature from surface satellite observations over the North Indian Ocean.

The model ingests a multi-source, multi-day surface signal and predicts temperature at 15 standard ocean depths from 0 to 1000 m. It is trained and validated using GLORYS reanalysis and checked against independent ARGO float profiles.

## Main objective

Create a daily subsurface temperature product that can be generated from satellite inputs alone, with the final output suitable for decision support, ocean monitoring, and downstream analysis.

## Core architecture

- Surface feature encoder for 19 channels across a 14-day window
- CNN + attention-based decoder
- Fourier-conditioned depth prediction head
- Climatology-anchored temperature reconstruction
- Output target: 15 standard depths across the water column

## What has been completed

- Data pipeline for raw satellite and reanalysis inputs
- Preprocessing pipeline for masks, climatology, and GLORYS standard-depth targets
- Official architecture verification and CUDA smoke test
- Official three-seed full-loss training
- Official GLORYS evaluation
- Independent ARGO validation
- Final plots and summary tables generated in `plots/`

## What remains

- Final presentation and polishing for open-source release
- Optional uncertainty estimation
- Optional OMNI validation
- Optional deeper-water bias correction and operational refinements

## Current reported metrics

- GLORYS RMSE: 0.8746°C
- Climatology RMSE: 1.0926°C
- Skill score: 0.1995
- ARGO RMSE: 1.1859°C

## Repository layout

- `Model/data_download/` — satellite and validation data download scripts
- `Model/preprocessing/` — data cleaning, mask generation, climatology, and standard-depth preparation
- `Model/training/` — model training, checks, and final plots
- `Model/evaluation/` — validation and ARGO comparison scripts
- `checkpoints/` — trained models
- `data/` — processed inputs and labels
- `plots/` — final figures and summary tables

## How the model connects to satellite data

1. Download satellite and reanalysis fields for the target region.
2. Crop them to the North Indian Ocean window.
3. Align them to a common daily grid and temporal window.
4. Build a 19-channel input tensor containing surface variables, seasonal features, and validity masks.
5. Run the trained OceanEmbed model to predict daily temperature anomalies and absolute temperatures at standard depths.
6. Output a gridded daily subsurface temperature field for the region.

This is the operational path to generate a daily subsurface temperature product from live satellite feeds and model inference.

## Recommended first run

```bash
cd Model
python training/14_evaluate_official_ensemble.py
python evaluation/10_evaluate_ensemble_against_argo.py
python training/15_generate_final_plots.py
```

## Project status

This repository is the main OceanEmbed implementation and is the version intended for open-source release.
