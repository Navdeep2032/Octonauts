# OceanEmbed

OceanEmbed is a deep-learning pipeline for reconstructing daily subsurface ocean temperature from surface satellite observations over the North Indian Ocean.

The model ingests a multi-source, multi-day surface signal and predicts temperature at 15 standard ocean depths from 0 to 1000 m. It is trained and validated using GLORYS reanalysis and checked against independent ARGO float profiles.

## Ocean temperature explorer

The React dashboard explores the daily ensemble outputs stored in the Cloudflare D1 `temperatures` table. Its date selector is populated from the dates actually present in D1. Pan and zoom the map to inspect a region, select a model depth to color its grid cells, and click a cell to display its 15-level temperature profile and frontend-calculated temperature gradient.

The browser calls a Cloudflare Worker proxy with a native D1 binding; account credentials and API tokens are never bundled into the frontend or stored as Worker secrets.

### Run locally

1. Install Node.js 20.19+ and dependencies:

   ```bash
   npm install
   ```

2. Enable a `workers.dev` subdomain for the Cloudflare account if it does not have one yet. Wrangler requires this to open a remote-binding preview.

3. In one terminal, run the Worker. Its D1 binding is configured to read the remote database:

   ```bash
   npm run worker:dev -- --port 8787
   ```

4. In another terminal, start the frontend. Vite proxies `/api/ocean-query` to the local Worker:

   ```bash
   npm run dev
   ```

### Deploy the data proxy

Build the Vite website before deploying the Worker that serves both the UI and API:

```bash
npm ci
npm run deploy
```

For a Cloudflare Workers Git deployment, use `npm ci` as the install command and `npm run build` as the build command. Set the deploy command to `npx wrangler deploy` and the build output directory to `dist`. Wrangler serves the built single-page app through the `ASSETS` binding and routes `/api/ocean-query` to the D1-backed Worker.

For a separately hosted frontend, set `VITE_OCEAN_API_URL` to the deployed Worker URL before building. When the frontend and Worker share an origin, the default `/api/ocean-query` URL can be used.

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
