# PFM vs observations at the moorings

Compares the PFM day-1 forecast with observed currents, temperature and salinity at four groups of moorings. All times are UTC.

| Folder | Moorings | Model grid |
|---|---|---|
| `SBOO/` | South Bay Ocean Outfall mooring | LV4 |
| `IB/` | Imperial Beach mini moorings IB-N, IB-C, IB-W, IB-S | LV4 |
| `PLOO/` | Point Loma Ocean Outfall mooring | LV3 |
| `DelMar/` | Del Mar mooring (delmar1) | LV3 |

Each mooring folder holds only its own two scripts and its plots:

| File | What it does |
|---|---|
| `plot_<name>_3weeks.py` | Plots of the past 3 weeks for the QC website, saved to `plots_web/`. Updates the data first. |
| `plot_<name>_range.py` | The same plots for any start/end dates, saved to `plots/`. Not for the website. |

The code they share, and the data cache, are in `common/`:

| File | What it does |
|---|---|
| `common/runner.py` | Command-line options and output folders for the two scripts. |
| `common/plot_model_vs_obs.py` | The figures. |
| `common/extract_model.py` | Extracts model profiles from `/dataSIO/PFM_Simulations/Archive/LV4_His` (SBOO, IB) and `LV3_His` (PLOO, Del Mar) into `common/cache/model/`. Python port of `PFM_Master_Code_For_TimeSeriesExtraction.m`. |
| `common/download_obs.py` | Downloads mooring data from https://mooring.ucsd.edu (Jeff Sevadjian, SIO) into `common/cache/obs/`. |

## Running

```bash
python SBOO/plot_sboo_3weeks.py                    # update data, plot the past 3 weeks up to today
python SBOO/plot_sboo_3weeks.py --no-update        # plot from cached data only
python SBOO/plot_sboo_3weeks.py --date 2026-09-30  # end on another day (older plots are kept)

python PLOO/plot_ploo_range.py --start 2026-06-01 --end 2026-08-31
python PLOO/plot_ploo_range.py --end today --duration 45   # the last 45 days, today included
python PLOO/plot_ploo_range.py --start 2025-01-01 --update
```

The same options work for `IB/plot_ib_*.py` and `DelMar/plot_delmar_*.py`.

- 3-week output: `plots_web/<figure>_<YYYYMMDD>_3weeks.png`. A normal run (without `--date`) deletes that folder's older files, so `plots_web/` only holds the newest set.
- Range output: `plots/<figure>_<start>_<end>.png`, both dates included. `--end` takes a date or `today`; `--duration N` can't be combined with `--start`. The range scripts use the cached data unless you add `--update`.
- An update extracts new model days for every mooring on the same grid, so after `SBOO/plot_sboo_3weeks.py` the IB run has nothing left to extract (and the same for PLOO and Del Mar).
- The day's history files are usually written by about 05:30 Pacific; if one isn't there yet, the next run picks it up.

## Figures

| Folder | Figure | Panels |
|---|---|---|
| SBOO | `SBOO_currents` | Northward top 5 m, northward depth-averaged, eastward top 5 m, eastward depth-averaged |
| SBOO | `SBOO_temperature` | All four sensor depths (1, 10, 18 and 26 m) in one panel, one colour per depth |
| SBOO | `SBOO_temperature_panels`, `SBOO_salinity` | One panel per sensor depth |
| SBOO | `SBOO_TS_contours` | Depth-time contours: observed temperature, model temperature, observed salinity, model salinity |
| IB | `IB-C_currents` | As `SBOO_currents`, with the top 3 m |
| IB | `IB_temperature_salinity` | Temperature at 1 m and at the deep sensor, salinity at 1 m and at the deep sensor, all four IB moorings in each panel. The deep sensor is at 8 m at IB-N and IB-C, 10 m at IB-W and IB-S. |
| PLOO | `PLOO_currents` | As `SBOO_currents` (top 5 m) |
| PLOO | `PLOO_temperature_panels`, `PLOO_salinity` | One panel per sensor depth (1, 10, 20, 30, 45, 60, 75, 87 and 89 m) |
| PLOO | `PLOO_TS_contours` | As `SBOO_TS_contours` |
| DelMar | `DelMar_currents` | As `SBOO_currents`, with the top 10 m (the first ADCP bin is at 5.42 m) |
| DelMar | `DelMar_temperature_panels`, `DelMar_salinity` | One panel per sensor depth (1, 7, 15, 24, 35, 47, 60, 74 and 90 m) |
| DelMar | `DelMar_TS_contours` | As `SBOO_TS_contours` |

- PLOO and Del Mar sensor depths vary a little between deployments (for example 4 m instead of 1 m in delmar1_21), so their panels and contours use the depths that have data in the window.
- Panels with both model and observations show r, RMSE and bias (model minus observed). In the overlay figures (`SBOO_temperature`, `IB_temperature_salinity`) they sit to the right of each panel, one entry per mooring or depth.
- All panels in a figure share one y range. In `IB_temperature_salinity`, the two temperature panels share one range and the two salinity panels share another.
- In `IB_temperature_salinity` each mooring has its own colour (IB-N blue, IB-C orange, IB-W green, IB-S yellow); the model is solid and the observations dashed.
- Model temperature and salinity are linearly interpolated between model levels to each sensor's fixed depth below mean sea level, as in `Model_vs_Observation_comparison_temp_sal.m`. The sensors hang on mooring lines and do not move with the tide.
- **Contours:** see [Temperature and salinity contours](#temperature-and-salinity-contours) below.
- **Model forecast:** after the last day-1 hour, the model continues with days 2–5 of the latest history file (about 4 days ahead), on a light grey background labelled "model forecast". Model lines are dashed there (dotted in the overlay figures, where the observations are already dashed); in the contours a white dashed line marks where the forecast starts. A window that ends before the forecast starts has no forecast. The statistics only use hours with observations, so the forecast never enters them.
- Every window is drawn hourly, however long (no daily means), and the statistics are computed from the same hourly values.

### Currents: which depths

The model is averaged over the depths the ADCP measures, each model layer weighted by how much of it lies in the range.

| Mooring | ADCP | Top layer | Depth average |
|---|---|---|---|
| SBOO | Downward looking, 1 m bins from 3.3 m below the mean surface | 3.3, 4.3 and 5.3 m bins (model 2.83–5.83 m) | 3.43 m to the side-lobe cutoff at 92% of the 31.5 m water depth |
| IB-C | Upward looking, 0.5 m bins below the moving surface | 1.75, 2.25 and 2.75 m bins (model 1.5–3 m) | 1.5 to 9 m below the surface |
| PLOO | Downward looking, 1 m bins from 3.43 m | 3.43, 4.43 and 5.43 m bins (model 2.93–5.93 m) | 3.43 to 88.0 m, obs and model |
| Del Mar | Downward looking, 2 m bins from 5.42 m; SIO QC flags, only "good" kept | 5.42, 7.42 and 9.42 m bins (model 4.42–10.42 m) | 5.42 to 93.8 m, obs and model |

At PLOO and Del Mar the LV3 model is shallower than the moorings (model depth 95.6 m at PLOO and 102 m at Del Mar; the ADCPs have good data to about 123 and 131 m). So both observations and model are averaged from the first bin down to 92% of the model depth, as `DelMar_Model_DataExtraction.m` cuts the model at 0.92 h.

Only IB-C has an ADCP among the IB moorings; IB-N, IB-W and IB-S only appear in `IB_temperature_salinity`.

### Temperature and salinity contours

`<site>_TS_contours` (SBOO, PLOO, Del Mar) has four depth-time panels: observed temperature, model temperature, observed salinity, model salinity. The observed and model panels are built differently, because the mooring only measures T and S at a few fixed depths while the model has the whole water column. Code: `obs_ts_grid`, `model_ts_profiles` and `ts_contours` in `common/plot_model_vs_obs.py`.

**Observed panels: interpolated between the sensors**

1. The sensor depths are those in the deployments' `temp.csv` / `sal.csv` that have data in the window (SBOO 1, 10, 18, 26 m; PLOO and Del Mar about 9 depths, which change slightly between deployments). Each sensor hangs at a fixed depth below mean sea level.
2. Each sensor is averaged to hourly values (values outside 5–30 °C and 20–36 are dropped first).
3. A regular depth grid is made every 0.25 m from the shallowest to the deepest sensor.
4. At every hour, each grid depth is linearly interpolated between the sensor just above and the sensor just below it. A point gets a value only when both of those sensors have data at that hour.
5. Nothing is extrapolated: there is no colour above the top sensor or below the bottom sensor.

So everything between two dotted lines (the sensor depths) is a straight line between two measurements; a thermocline between two sensors is smoothed out.

What shows up as grey (no data) in the observed panels:
- **Above the top sensor and below the bottom sensor**, always.
- **Both layers next to a sensor that has no data**: interpolation does not bridge a missing sensor. Example: when the Del Mar 35 m sensor is out (Feb–Mar 2025, Apr–Jul 2026), 24–47 m is grey.
- **Where one sensor changed depth between deployments.** Del Mar's near-surface sensor is at 1 m in some deployments and 4 m in others, and its deepest at 89 m or 90 m. They are treated as different sensors that never have data at the same time, so for example the 1–7 m layer is grey even when the 4 m sensor has data.

**Model panels: all model levels, up to the free surface**

1. The model profiles are hourly day-1 output at the grid point nearest the mooring (cached by `extract_model.py`), with T and S at every vertical level. After the white dashed line, days 2–5 of the latest forecast follow.
2. Each level is plotted at its real depth below mean sea level at that hour, from the ROMS s-coordinate transform (Vtransform 2) with the hourly free-surface height zeta. The levels therefore move up and down with the tide.
3. One extra row at the free surface (depth = −zeta) repeats the top level's value, so the colour reaches the moving sea surface. The tide shows as a wavy top edge, with grey above it.
4. The model is not interpolated to the sensor depths here: it shows its full vertical resolution from the surface down to its deepest level (model depth h: about 31 m at SBOO, 95.6 m at PLOO and 102 m at Del Mar).

**Shared by both**

- The y-axis is depth below mean sea level. It runs from the model depth h up to the highest model free surface in the window, so the top is slightly above mean sea level (negative depths). A grey line marks mean sea level (0 m), and dotted lines mark the sensor depths.
- The observed and model panels of one variable use the same cmocean colormap (`thermal` for temperature, `haline` for salinity) and the same 14 colour levels. The levels span the 0.5–99.5 percentiles of the observed and model values together, so a few spikes don't set the scale; values beyond it use the end colours.
- Grey always means no data.
- The IB moorings get no contours: they have at most two T/S sensors.

**Comparing the panels:** the model panel has much more vertical detail than the observed one, including the top few metres and the deepest layers where there are often no sensors. Structure that appears only in the model may simply fall between or outside the sensors. For a like-for-like comparison at the measured depths, use `<site>_temperature_panels` and `<site>_salinity`, where the model is interpolated to each sensor's depth.

## Data

All in `common/cache/`: model files in `model/`, observations in `obs/`.

- **Model:** `extract_model.py` reads each history file once for all the moorings on that grid (LV4: about 35 s per file for SBOO and the four IB sites; LV3: about 5 s per file for PLOO and Del Mar, with 6 files in parallel) and saves `model/<site>_<LV>_day1_profiles.nc` and `<site>_<LV>_day1_hourly.csv`. Later runs only read new days. To rebuild from scratch, delete the `*_profiles.nc` files. Only files named `LV<n>_ocean_his_<YYYYMMDDHHMM>.nc` are read, so failed runs such as `LV4_ocean_his_202610110000_dt2_BLOWUP.nc` are skipped.
- **Model forecast:** each update also reads days 2–5 of the latest history file into `model/<site>_<LV>_forecast_profiles.nc` and `_hourly.csv`, replaced on every run.
- **Model positions** are those of the MATLAB registry: SBOO ID 2, IB IDs 10/101/102/103, Del Mar LV3 ID 14 (-117.31661, 32.92947), Point Loma ID 15 (-117.32421, 32.67005).
- **Observations:** `download_obs.py` only downloads files that have changed on the server, and deletes local ADCP files the server no longer lists (the server renames a deployment's file as it grows).
  - PLOO and Del Mar: only deployments from ploo_05 and delmar1_20 on (earlier ones end before the model starts on 5 Dec 2024). Each ADCP file is about 400 MB.
  - SBOO and PLOO currents come from the 10-minute ADCP netCDF files, Del Mar from the 15-minute ones, IB-C from its ADCP netCDF files with surface-contaminated bins removed.
  - Temperature and salinity come from each deployment's `temp.csv` and `sal.csv`. They are calibrated but not yet QC'd for outliers, so values outside 5–30 °C and 20–36 are dropped.
  - Each hour is the mean of samples from 30 minutes before to 30 minutes after it, as in the MATLAB observation code.

## Requirements

Python with `pandas`, `numpy`, `scipy`, `matplotlib`, `netCDF4` and `cmocean` (installed in `/home/akg004/anaconda3`). It also needs read access to `/dataSIO/PFM_Simulations`.
