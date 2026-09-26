# OceanEmbed — Simple Summary

OceanEmbed is a project that turns everyday satellite data into a daily picture of subsurface ocean temperature.

Instead of only seeing the sea surface, the model tries to estimate what the water looks like below the surface, down to 1000 m.

## What it does

It uses satellite information like:

- sea surface temperature
- salinity
- sea level
- ocean currents
- wind

and turns this into a daily 3D temperature map of the North Indian Ocean.

## What has already been done

- The data pipeline is working
- The preprocessing steps are complete
- The main model architecture has been verified
- The official model has been trained and evaluated
- The results are better than climatology baseline
- ARGO validation has also been completed

## Current status

The main model is working and validated. The project is mostly complete from a technical standpoint.

What is left is mostly polishing and optional enhancements:

- final report and presentation cleanup
- uncertainty estimation
- OMNI validation
- future operational improvements for deeper layers

## Example of the final idea

A user can provide recent satellite data for a region, and the model can produce a daily subsurface temperature estimate that helps with ocean monitoring, fisheries planning, and climate-related analysis.

## Key result

The model reduced GLORYS RMSE from 1.0926°C climatology baseline to 0.8746°C, which is a 20% improvement in skill.

## Why this matters

Most ocean monitoring tools see only the surface. OceanEmbed adds a practical way to estimate deeper water temperatures from the information already available from satellites.
