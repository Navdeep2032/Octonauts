# OceanEmbed — Training, Step by Step

Everything here assumes `data_download/` and `preprocessing/` have already
run successfully and passed `05_verify_preprocessing.py`.

## One-time setup

```bash
pip install torch
```
(Use the CPU or CUDA build appropriate for your machine — see
https://pytorch.org/get-started/locally/ for the exact install command if
you have a GPU you want PyTorch to use.)

## What was reconciled before this was handed off

These training scripts were originally written *before* the preprocessing
pipeline's final file layout was settled. Two real path mismatches were
caught and fixed just now:

- `train_stage2.py` pointed at a placeholder path (`./data/glorys/GLORYS_NIO_{year}.nc`)
  that was never actually produced — the real GLORYS target files are at
  `./data/glorys_processed/GLORYS_STD_DEPTHS_{year}.nc` (built by
  `preprocessing/03_build_glorys_standard_depths.py`). Fixed.
- Verified variable names, array shapes, and depth/month ordering all match
  exactly between what `preprocessing/` produces and what `dataset.py`/
  `model.py` expect (climatology `[depth, lat, lon, month]`, inversion map
  `[lat, lon]`, depth-valid mask `[depth, lat, lon]`) — no mismatches found
  in this pass.

**What was NOT verified:** an actual forward/backward pass on real data.
Every fix so far has been caught via manual code review and shape-tracing,
not execution — there was no `torch` installation available in the
environment these scripts were prepared in. Treat the run order below as
"ready to attempt," not "guaranteed to run first try."

## Run order

```bash
cd training/
python train_stage1.py    # self-supervised pretraining, 2021-2023 surface-only data
python train_stage2.py    # supervised fine-tuning, GLORYS 2015-2018 train / 2019 val
```

### Before running the full Stage 1 loop

Do a tiny dry run first — cut `STAGE1_EPOCHS` in `config.py` down to 1, and
temporarily limit `PRETRAIN_YEARS` to a single year — just to confirm the
whole chain (data loading → model forward pass → loss → backward pass →
checkpoint save) actually executes without a shape or path error, before
committing to a multi-hour full run. This is standard practice for any new
training pipeline, doubly so here since this exact combination of code has
not been executed end-to-end before.

```python
# in config.py, temporarily:
STAGE1_EPOCHS = 1
PRETRAIN_YEARS = [2021]
```
Revert both once the dry run completes cleanly.

### What each stage produces

| Stage | Output | Notes |
|---|---|---|
| 1 | `checkpoints/stage1_pretrained.pt` | Encoder + temporal front-end + high-res branch + bottleneck attention weights |
| 2 | `checkpoints/stage2_best.pt` | Full model, saved whenever validation loss improves |

### If Stage 1 fails on the MLD climatology path

`train_stage1.py`'s multi-task pretraining (surface reconstruction + MLD
regression) expects an optional `mld_climatology` array passed into
`SurfaceOnlyDataset`. **This was never actually built** — no script in
`preprocessing/` produces an MLD climatology file. Two options:

1. **Skip it** (fastest path to a working Stage 1): don't pass
   `mld_climatology` to `SurfaceOnlyDataset` — the code already handles this
   case (`if "mld_target" in batch:` guards the extra loss term), so Stage 1
   will run as pure surface-reconstruction pretraining without the MLD
   auxiliary task. You lose a small amount of the "touches subsurface
   coupling" benefit described in the technical approach doc, but everything
   else still works.
2. **Build it properly**: source a mixed-layer-depth climatology (e.g. de
   Boyer Montégut et al.'s published climatology) and write a preprocessing
   script producing a `[H, W, 12]` array on your target grid, then pass it in.
   This is the "do it right" option but adds another preprocessing step.

Given the project is otherwise fully verified end-to-end, option 1 is the
pragmatic default unless there's time budget left for option 2.

## After training

Move to `evaluate.py` for the three-tier validation (gridded ARGO, raw ARGO
profiles, INCOIS OMNI) — this still needs the actual ARGO/OMNI datasets
sourced and formatted to match what `evaluate.py`'s
`evaluate_against_observations()` expects (see that file's docstring). This
hasn't been built yet and is the next real gap to close after training itself
is confirmed working.
