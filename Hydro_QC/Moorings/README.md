# PFM vs observations at the moorings

Compares the PFM day-1 forecast with observed currents, temperature and salinity at four groups of moorings. Times on the plots, and the dates of the plot windows, are Pacific (PDT/PST, America/Los_Angeles); the data and caches stay UTC.

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

The data cache is on the shared disk, `/dataSIO/Hydro_QC/Moorings/cache/`, so runs on swell and backwash use and update the same files (group `feddersen-lab`, group-writable). Set the environment variable `HYDRO_QC_DATA` to use another root (the cache is then `$HYDRO_QC_DATA/Moorings/cache/`). Plots stay next to the scripts.

The code they share is in `common/`:

| File | What it does |
|---|---|
| `common/runner.py` | Command-line options and output folders for the two scripts. |
| `common/plot_model_vs_obs.py` | The figures. |
| `common/extract_model.py` | Extracts model profiles from `/dataSIO/PFM_Simulations/Archive/LV4_His` (SBOO, IB) and `LV3_His` (PLOO, Del Mar) into `/dataSIO/Hydro_QC/Moorings/cache/model/`. Python port of `PFM_Master_Code_For_TimeSeriesExtraction.m`. |
| `common/download_obs.py` | Downloads mooring data from https://mooring.ucsd.edu (Jeff Sevadjian, SIO) into `/dataSIO/Hydro_QC/Moorings/cache/obs/`. |
| `common/export_model_timeseries.m` | Old MATLAB export, replaced by `extract_model.py`. |

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
| SBOO | `SBOO_density_panels` | Density σ0, one panel per sensor depth |
| SBOO | `SBOO_density_N2_contours` | Depth-time contours: observed σ0, model σ0, observed N² (one value per layer between sensors), model N² (between neighbouring s-levels); each observed/model pair on one colour scale |
| SBOO | `SBOO_N2_panels` | Observed buoyancy frequency N², one panel per pair of neighbouring sensors (1–10, 10–18, 18–26 m); no model |
| IB | `IB-C_currents` | As `SBOO_currents`, with the top 3 m |
| IB | `IB_temperature_salinity` | Temperature at 1 m and at the deep sensor, salinity at 1 m and at the deep sensor, all four IB moorings in each panel. The deep sensor is at 8 m at IB-N and IB-C, 10 m at IB-W and IB-S. |
| IB | `IB_density` | Density σ0 at 1 m and at the deep sensor, all four IB moorings in each panel |
| IB | `IB_N2` | Observed N² between the 1 m and the deep sensor, all four IB moorings in one panel, no model (a mooring without a 1 m sensor, e.g. IB-C now, has no observed line) |
| PLOO | `PLOO_currents` | As `SBOO_currents` (top 5 m) |
| PLOO | `PLOO_temperature_panels`, `PLOO_salinity` | One panel per sensor depth (1, 10, 20, 30, 45, 60, 75, 87 and 89 m) |
| PLOO | `PLOO_TS_contours` | As `SBOO_TS_contours` |
| PLOO | `PLOO_density_panels`, `PLOO_density_N2_contours`, `PLOO_N2_panels` | As for SBOO |
| DelMar | `DelMar_currents` | As `SBOO_currents`, with the top 10 m (the first ADCP bin is at 5.42 m) |
| DelMar | `DelMar_temperature_panels`, `DelMar_salinity` | One panel per sensor depth (1, 7, 15, 24, 35, 47, 60, 74 and 90 m) |
| DelMar | `DelMar_TS_contours` | As `SBOO_TS_contours` |
| DelMar | `DelMar_density_panels`, `DelMar_density_N2_contours`, `DelMar_N2_panels` | As for SBOO |

- PLOO and Del Mar sensor depths vary a little between deployments (for example 4 m instead of 1 m in delmar1_21), so their panels and contours use the depths that have data in the window.
- Panels with both model and observations show r, RMSE and bias (model minus observed). In the overlay figures (`SBOO_temperature`, `IB_temperature_salinity`) they sit to the right of each panel, one entry per mooring or depth.
- All panels in a figure share one y range. In `IB_temperature_salinity`, the two temperature panels share one range and the two salinity panels share another.
- In `IB_temperature_salinity` each mooring has its own colour (IB-N blue, IB-C orange, IB-W green, IB-S yellow); the model is solid and the observations dashed.
- **Density:** σ0 = ρ(SA, CT, 0) − 1000 kg/m³, potential density anomaly from TEOS-10 (Gibbs SeaWater, `gsw`; `sigma0()` in `plot_model_vs_obs.py`). Practical salinity and in-situ temperature are converted to absolute salinity (`gsw.SA_from_SP`, at the site's lon/lat) and conservative temperature (`gsw.CT_from_t`), with pressure from the depth below mean sea level (`gsw.p_from_z`). Computed from the hourly T and S of each sensor and of each model level with the same function. In the contours the observed σ0 is computed at each sensor and then interpolated between sensors like T and S; the model σ0 is computed at every model level.
- **N²:** `gsw.Nsquared` (TEOS-10, local gravity) from SA, CT and pressure at two neighbouring depths, plotted in 10⁻⁴ s⁻². Observed (`*_N2_panels`, `IB_N2`, the observed N² panel of `*_density_N2_contours`): between each pair of neighbouring sensors, hour by hour. Model (the model N² panel of `*_density_N2_contours`; no model in the line panels): between each pair of neighbouring s-levels, placed halfway between them, with pressure from each level's depth below mean sea level. The s-levels are much closer together than the sensors, so the model resolves thin layers (e.g. the near-surface pycnocline) the moorings cannot. The observed and model N² panels share one colour scale, to which observations and model contribute equally (see [Colour and axis limits](#colour-and-axis-limits)). A sensor whose depth changed a little between deployments (Del Mar 1 m / 4 m, 89 m / 90 m: within 3 m and never both with data) is treated as one sensor at its hourly depth. The N² contours use a diverging colormap (cmocean `balance`) centred on 0: white at 0, blue for negative (unstable), red for positive (stable), with separate scales either side of 0. Negative observed N² over long periods usually means a salinity offset or drift between two sensors rather than real instability.
- Model temperature and salinity are linearly interpolated between model levels to each sensor's fixed depth below mean sea level, as in `Model_vs_Observation_comparison_temp_sal.m`. The sensors hang on mooring lines and do not move with the tide.
- **Contours:** see [Temperature and salinity contours](#temperature-and-salinity-contours) below.
- **Model forecast:** after the last day-1 hour, the model continues with days 2–5 of the latest history file (about 4 days ahead), on a light grey background labelled "model forecast". Model lines are dashed there (dotted in the overlay figures, where the observations are already dashed); in the contours a white dashed line marks where the forecast starts. The forecast is shown whenever the window reaches the end of the day-1 data, so the 3-week plots (and range plots ending today) always run on about 4 days past the window; a window that ends earlier has no forecast. The statistics only use hours with observations, so the forecast never enters them.
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
- The observed and model panels of one variable use the same cmocean colormap (`thermal` for temperature, `haline` for salinity) and the same 14 colour levels (see [Colour and axis limits](#colour-and-axis-limits)).
- Grey always means no data.
- The IB moorings get no contours: they have at most two T/S sensors.

**Comparing the panels:** the model panel has much more vertical detail than the observed one, including the top few metres and the deepest layers where there are often no sensors. Structure that appears only in the model may simply fall between or outside the sensors. For a like-for-like comparison at the measured depths, use `<site>_temperature_panels` and `<site>_salinity`, where the model is interpolated to each sensor's depth.

### Colour and axis limits

All limits are computed from the data in the plotted window, so they change from one plot to the next: the same colour can mean a different value in last week's website plot. Percentiles are used instead of min–max because T and S only have the gross range check (5–30 °C, 20–36), and a single bad hour inside that range would otherwise set the whole scale. Code: `ts_contours`, `panel`, `overlay_panel` and `finish` in `common/plot_model_vs_obs.py`.

**Contour plots** (14 colour levels; values beyond the limits get the end colours, shown by the arrow tips on the colour bar)

| Figure | Values the limits come from | Limits |
|---|---|---|
| `*_TS_contours`, density panels of `*_density_N2_contours` | Every hourly value of every sensor, plus every model value (all levels, every hour, including forecast days 2–5), pooled | 0.5–99.5 percentiles, evenly spaced levels |
| N² panels of `*_density_N2_contours` | Every hourly observed sensor-pair value, plus every model level-pair value, with **equal total weight** for observations and model | 0.5–99.5 weighted percentiles, then the lower limit is pushed to at least −5 % of the upper one |

- The observed and model panels of one variable share one colour bar, so the same colour means the same value in both.
- **The limits come from the measured values, not from the 0.25 m grid the observed panels are drawn on.** That grid only copies or interpolates the sensor values between sensors; counting its points would weight each sensor layer by its thickness and make the scale depend on the grid spacing (with a 1 m grid instead of 0.25 m the N² limit at Del Mar moved from 11 to 15 with the same data). Before 19 Oct 2026 the limits were taken from the grid.
- **T, S, σ0:** plain pooling. The model has more values per hour (~40 levels against 4–9 sensors), but observed and model ranges are similar, so the scale fits both: in the 3 weeks to 19 Oct 2026 at most about 1 % of either panel fell beyond the limits at SBOO, PLOO and Del Mar.
- **N²:** model N² between thin s-levels reaches much higher values than N² across a 6–10 m sensor layer, and the model has ~5 times as many values per hour. Plain pooling would let the model set the scale and leave the observed panel pale; observations alone would saturate the model's near-surface layers (15 % of the SBOO model panel). With equal weight each dataset counts half: in the 3 weeks to 19 Oct 2026 the limits were about −0.6 to 21 (SBOO), −1.2 to 15 (PLOO) and −1.3 to 17 (Del Mar) × 10⁻⁴ s⁻², with about 1 % of each panel beyond them.
- N² colours (cmocean `balance`) are centred on 0 (white) with `TwoSlopeNorm`. The levels are split between the negative and positive sides in proportion to their ranges, with at least 2 below 0, so the colour steps can differ in size on the two sides.

**Line panels** (T, S, density and observed N² at each depth; the IB overlay figures)

- The y-axis spans the 0.2–99.8 percentiles of the values in the panel (observed and model together where both are drawn), plus 5 % padding at each end. Clipped spikes run off the top or bottom of the panel.
- All panels of one figure are then given one shared y-range (the lowest bottom and highest top of all panels), so depths can be compared directly; panels with little variation look flat. Exception: in `IB_temperature_salinity` the temperature panels and the salinity panels each share their own range.
- Currents panels are not clipped: matplotlib's automatic range, then shared across the figure in the same way.

**Other choices considered (Oct 2026), kept as above for now:** 2–98 or 5–95 percentiles give more contrast in the middle of the range but saturate the extremes (with 5–95 the model's cold bottom water and the warm surface layer at Del Mar become flat end colours, and negative observed N² gets only about 2 colour levels). Min–max saturates nothing but lets one spike set the scale, and for N² a single thin layer (model N² up to 47 × 10⁻⁴ s⁻²) would squash the rest of the water column into 2–3 colours. Fixed limits per site would make website plots comparable from week to week.

## Data

All in `/dataSIO/Hydro_QC/Moorings/cache/`: model files in `model/`, observations in `obs/`.

- **Model:** `extract_model.py` reads each history file once for all the moorings on that grid (LV4: about 35 s per file for SBOO and the four IB sites; LV3: about 5 s per file for PLOO and Del Mar, with 6 files in parallel) and saves `model/<site>_<LV>_day1_profiles.nc` and `<site>_<LV>_day1_hourly.csv`. Later runs only read new days. To rebuild from scratch, delete the `*_profiles.nc` files. Only files named `LV<n>_ocean_his_<YYYYMMDDHHMM>.nc` are read, so failed runs such as `LV4_ocean_his_202610110000_dt2_BLOWUP.nc` are skipped.
- **Model forecast:** each update also reads days 2–5 of the latest history file into `model/<site>_<LV>_forecast_profiles.nc` and `_hourly.csv`, replaced on every run.
- **Model positions** are those of the MATLAB registry: SBOO ID 2, IB IDs 10/101/102/103, Del Mar LV3 ID 14 (-117.31661, 32.92947), Point Loma ID 15 (-117.32421, 32.67005).
- **Observations:** `download_obs.py` only downloads files that have changed on the server, and deletes local ADCP files the server no longer lists (the server renames a deployment's file as it grows).
  - PLOO and Del Mar: only deployments from ploo_05 and delmar1_20 on (earlier ones end before the model starts on 5 Dec 2024). Each ADCP file is about 400 MB.
  - SBOO and PLOO currents come from the 10-minute ADCP netCDF files, Del Mar from the 15-minute ones, IB-C from its ADCP netCDF files with surface-contaminated bins removed.
  - Temperature and salinity come from each deployment's `temp.csv` and `sal.csv`. They are calibrated but not yet QC'd for outliers, so values outside 5–30 °C and 20–36 are dropped.
  - Each hour is the mean of samples from 30 minutes before to 30 minutes after it, as in the MATLAB observation code.

## Requirements

Python with `pandas`, `numpy`, `scipy`, `matplotlib`, `netCDF4`, `cmocean` (installed in `/home/akg004/anaconda3`) and `gsw` (Gibbs SeaWater / TEOS-10, for density and N²; `pip install gsw`, currently in the user site-packages `~/.local`). It also needs read access to `/dataSIO/PFM_Simulations`.
