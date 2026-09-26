# OceanEmbed Technical README

## Purpose

OceanEmbed reconstructs subsurface temperature from surface-only satellite inputs. The target is the North Indian Ocean, with fields generated at daily time resolution and 15 standard depth levels: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, and 1000 m.

## Data flow

1. `Model/data_download/` downloads raw satellite and reanalysis products.
2. `Model/preprocessing/` assembles masks, monthly climatology, and GLORYS-standard-depth targets.
3. `Model/training/` trains the model and verifies the architecture contract.
4. `Model/evaluation/` checks performance against GLORYS and ARGO.
5. `plots/` stores final figures and summary tables.

## Input tensor structure

The model expects a tensor of the form:

- `[B, 19, 14, H, W]`

This corresponds to:

- 7 surface variables
- 3 coordinate channels
- 2 seasonal channels
- 7 validity masks
- 14-day temporal window

The output is:

- `[B, 15, H, W]`

where each pixel contains the reconstructed temperature at the 15 standard depths.

## Official implementation status

### Completed

- Data acquisition and preprocessing verified
- Input/output shape verified on CUDA
- Full composite-loss training completed for three seeds
- Official ensemble performance evaluated
- Independent ARGO validation completed
- Final plots generated

### Metrics

- Ensemble GLORYS RMSE: 0.8746°C
- Climatology RMSE: 1.0926°C
- Overall skill score: 0.1995
- Ensemble ARGO RMSE: 1.1859°C

### Remaining work

- Final presentation polish for the release
- Optional uncertainty quantification
- Optional OMNI validation
- Optional deeper-depth correction and deployment improvements

## How this connects to daily satellite-driven output

The live inference pipeline is:

1. Download or ingest recent daily satellite fields over the region.
2. Reproject and align them to the project grid.
3. Build the 19-channel daily input tensor with a 14-day contextual window.
4. Load the trained checkpoint from `checkpoints/`.
5. Run model inference to generate a daily 3D temperature field.
6. Save the gridded output for each standard depth.

This makes the model usable as a daily operational reconstruction service for subsurface temperature.

## Repository cleanliness notes

This repository is focused on the main OceanEmbed pipeline. Legacy exploration files and ablation-only experiments were removed from the release-facing version so the codebase stays easy to follow and open-source ready.

## Main files

- `Model/training/model.py` — core model
- `Model/training/losses.py` — training loss
- `Model/training/config.py` — configuration values
- `Model/training/train_stage1.py` — self-supervised pretraining
- `Model/training/train_stage2.py` — supervised fine-tuning
- `Model/training/09_train_stage2_ensemble.py` — official 3-seed ensemble training
- `Model/training/14_evaluate_official_ensemble.py` — official validation
- `Model/evaluation/10_evaluate_ensemble_against_argo.py` — ARGO validation
- `Model/training/15_generate_final_plots.py` — final plots and summary tables
