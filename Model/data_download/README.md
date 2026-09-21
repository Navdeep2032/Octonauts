# OceanEmbed — Data Download, From Scratch

This folder contains six scripts, run in numbered order, that download every
data source OceanEmbed needs, cropped to the North Indian Ocean box
(5°N–30°N, 45°E–105°E), one year at a time, to keep disk usage manageable.

## Date range covered

**2015–2023** (9 years total), split as:
- 2015–2018 → Stage 2 training (GLORYS-labeled)
- 2019 → Stage 2 validation
- 2020 → Stage 2 test (held out, touched once)
- 2021–2023 → Stage 1 self-supervised pretraining (surface-only, unlabeled)

Scripts 01–05 (SST, SSS, SSH, currents, winds) download the full 2015–2023 span.
Script 06 (GLORYS target) only downloads 2015–2020, since the pretraining years
(2021–2023) are deliberately surface-only with no subsurface label.

## Step 0 — One-time setup (do this before running anything)

### 0.1 Install required packages
```bash
pip install copernicusmarine earthaccess xarray dask tqdm netCDF4
```

### 0.2 Copernicus Marine account (needed for scripts 01, 02, 03, 06)
1. Sign up free at https://data.marine.copernicus.eu
2. Run once in your terminal:
```bash
copernicusmarine login
```
This will prompt for your username/password and store credentials locally —
scripts 01, 02, 03, and 06 will then authenticate automatically.

### 0.3 NASA Earthdata account (needed for scripts 04, 05)
1. Sign up free at https://urs.earthdata.nasa.gov
2. Create a `.netrc` file in your home directory:

**Linux/macOS:**
```bash
nano ~/.netrc
```
```
machine urs.earthdata.nasa.gov
login YOUR_EARTHDATA_USERNAME
password YOUR_EARTHDATA_PASSWORD
```
```bash
chmod 600 ~/.netrc
```

**Windows (PowerShell):**
```powershell
notepad $HOME\_netrc
```
Paste the same three lines. Confirm in File Explorer (with extensions shown)
that it saved as `_netrc`, not `_netrc.txt`.

### 0.4 Verify both logins work
```python
import earthaccess
auth = earthaccess.login(strategy="netrc")
print(auth.authenticated)  # should print True
```
```bash
copernicusmarine describe --product-id SST_GLO_SST_L4_REP_OBSERVATIONS_010_011
```
If this returns dataset info without an error, Copernicus login is working.

## Step 1 — Run the scripts, in this exact order

```bash
python 01_download_sst.py          # OSTIA SST, 0.05deg -> cropped, ~2015-2023
python 02_download_sss.py          # Multi-obs SSS, 0.125deg -> cropped, ~2015-2023
python 03_download_ssh.py          # DUACS SSH/SLA + ugos/vgos, 0.25deg -> cropped, ~2015-2023
python 04_download_currents.py     # OSCAR u/v currents (only if not using ugos/vgos from 03)
python 05_download_winds.py        # CCMP u/v winds, ~2015-2023
python 06_download_glorys_target.py  # GLORYS12V1 subsurface temperature target, 2015-2020 only
```

**Run one script fully before starting the next** — each one downloads ~9 years
(or 6, for GLORYS) of data and can take a while depending on your connection.

## Notes specific to each script

- **01/02/03/06 (Copernicus-hosted):** use server-side subsetting — you get one
  already-cropped NetCDF per year directly, no manual cropping needed.
- **04/05 (NASA-hosted, OSCAR + CCMP):** Harmony server-side subsetting was found
  unreliable for these collections (unsupported concatenation, high job failure
  rate), so these scripts instead download small daily global granules, crop
  each one locally with `xarray`, delete the raw global file immediately, and
  merge the year's cropped days into one file — keeping peak disk usage low
  throughout. This is slower (many small network calls) but far more reliable.
- **03 also fetches `ugos`/`vgos`** (geostrophic currents) bundled with SSH —
  check if these are sufficient for your needs before running 04, since OSCAR
  (04) may then be skippable, saving a slow download.
- **06 stops at 2020** — do not extend its year range to 2021-2023; that would
  break the deliberate non-overlap between Stage 1 pretraining data and Stage 2
  labeled data described in the technical approach document.
- **07 (verification)** should be run after each individual download script
  finishes, not just once at the very end — catching a bad year right after
  its download completes is much faster to fix than discovering it after all
  six sources have finished.

## After all six finish

You should have, under `./data/raw/`:
```
sst/SST_NIO_2015.nc ... SST_NIO_2023.nc
sss/SSS_NIO_2015.nc ... SSS_NIO_2023.nc
ssh/SSH_NIO_2015.nc ... SSH_NIO_2023.nc
currents/CURRENTS_NIO_2015.nc ... CURRENTS_NIO_2023.nc   (if you ran 04)
wind/WIND_NIO_2015.nc ... WIND_NIO_2023.nc
glorys/GLORYS_NIO_2015.nc ... GLORYS_NIO_2020.nc
```

## Step 2 — Verify everything downloaded correctly

**Don't skip this.** A partial or corrupted download can look fine in a file
listing (the file exists, has *some* size) but be missing days, missing
variables, or have a huge chunk of unexpected NaNs — any of which will
silently produce bad training data later, and be much harder to debug once
mixed into a 19-channel stacked tensor than to catch right now.

```bash
python 07_verify_downloads.py
```

This checks, per file, per source:
1. The file exists and isn't suspiciously small (a common sign of a failed download).
2. It opens correctly as NetCDF (catches truncated/corrupted downloads).
3. The expected variable is actually present.
4. The time dimension has roughly 365 (or 366, leap years) days — catches
   partial-year downloads.
5. The spatial extent roughly matches the 5–30°N, 45–105°E box.
6. The fraction of NaN values isn't unexpectedly high.
7. (GLORYS only) all 15 standard depth levels are present.

**Sample output:**
```
--- SST ---
  [2015] OK
  [2016] OK
  [2017] ISSUES FOUND:
      - only 210 time steps found, expected ~365 — likely a partial download
  ...
Checked 51 files, 1 had issues.
Re-run the relevant download script(s) for any year/source flagged above
before proceeding.
```

If a year/source shows issues, delete just that file and re-run the
corresponding download script — every script already skips years whose files
exist, so re-running is safe and won't re-download everything.

## Manual spot-checks worth doing once, beyond the automated script

- **Open one file in Panoply or QGIS** (or just `xr.open_dataset(...).plot()` in
  a notebook) and visually confirm the map looks like a real ocean region —
  not all zeros, not all NaNs, land clearly distinguishable from ocean.
- **Check the longitude convention matches across sources** — some products use
  0–360°, others use -180 to 180°. The verification script flags an obvious
  mismatch, but a quick manual look at one SST file and one currents file
  side-by-side is worth doing once to be sure they'll actually overlay
  correctly once regridded.
- **Check file sizes are roughly consistent year-to-year within the same
  source** — SST files should all be similar size to each other; if one year
  is 10x smaller than its neighbors, that's worth a second look even if it
  technically passed the automated checks.
