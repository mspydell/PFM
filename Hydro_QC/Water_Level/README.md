# PFM vs observed sea surface height: San Diego Bay and La Jolla

Compares the PFM day-1 forecast sea surface height (`zeta`) with the NOAA tide gauges at San Diego Bay and La Jolla. All times are UTC.

| Station key | Gauge | PFM grid | Model grid point |
|---|---|---|---|
| `SDBay` | NOAA 9410170, San Diego Bay | LV4 | −117.17663, 32.71484, depth 11.1 m |
| `LaJolla` | NOAA 9410230, Scripps Pier | LV3 (LV4 doesn't reach La Jolla) | −117.25647, 32.86733, depth 10.7 m |

Each station has its own scripts (`plot_sdbay_*.py`, `plot_lajolla_*.py`). They share the plotting code, and the station settings are in `stations.py`.

This is a Python version of MATLAB location IDs 1 (SD Bay) and 11 (La Jolla) in `PFM_Master_Code_For_TimeSeriesExtraction.m`, `Observed_Master_Code_For_TimeSeriesExtraction.m` and `PFM_Master_Code_For_Plots.m` (which calls `Plot_SSH.m`).

## Files

| File | What it does | Website |
|---|---|---|
| `plot_sdbay_3weeks.py`, `plot_lajolla_3weeks.py` | Plot of the past 3 weeks for that station, saved to `plots_web/`. Updates the data first. | yes |
| `plot_ssh.py` | Shared code for the hourly figure and the 3-week run. | yes |
| `stations.py` | Station settings: NOAA ID, position, grid. | yes |
| `download_obs.py` | Downloads water level, air pressure and tide predictions from the NOAA CO-OPS API into `obs_cache/<station>/`. | yes |
| `extract_model.py` | Extracts `zeta` at each gauge from `/dataSIO/PFM_Simulations/Archive/<LV>_His` into `model_cache/`. | yes |
| `plot_sdbay_range.py`, `plot_lajolla_range.py` | Hourly and tides-removed figures for any dates, saved to `plots/`. | no |
| `plot_ssh_lowpass.py` | Shared code: the PL64 low-pass filter, the tides-removed figure and the date-range run. | no |
| `plot_sdbay_noaa_tide.py`, `plot_lajolla_noaa_tide.py` | NOAA measured vs predicted water level and their difference; no model. | no |
| `noaa_tide.py` | Shared code for the NOAA plots. | no |

The website plot of one station needs only its 3-week script plus the four shared files marked "yes". None of them contains any filter code. `download_obs.py` and `extract_model.py` can also be run on their own; they take `--station SDBay` or `--station LaJolla` (default both).

## Website plots: plot_sdbay_3weeks.py and plot_lajolla_3weeks.py

```bash
python plot_sdbay_3weeks.py                    # update data, plot the past 3 weeks up to today
python plot_lajolla_3weeks.py
python plot_sdbay_3weeks.py --no-update        # plot from the cached data only
python plot_sdbay_3weeks.py --date 2026-09-30  # end on another day (older plots are kept)
```

Writes `plots_web/SDBay_ssh_<YYYYMMDD>_3weeks.png` or `plots_web/LaJolla_ssh_<YYYYMMDD>_3weeks.png`, and deletes that station's older plot. It must run on a machine that can read `/dataSIO`.

The figure has two panels, laid out like `Plot_SSH.m`:

| Panel | Contents |
|---|---|
| 1 | Hourly observed water level (pressure-corrected, MSL datum) and model `zeta`, with r, RMSE, bias and the number of hours compared |
| 2 | ΔSSH = observed − model, with its mean and standard deviation; y axis fixed at −0.5 to 0.5 m |

## Date-range plots: plot_sdbay_range.py and plot_lajolla_range.py

```bash
python plot_sdbay_range.py                                     # START in the script to today
python plot_sdbay_range.py --start 2026-06-01 --end 2026-08-31
python plot_lajolla_range.py --end today --duration 45         # the last 45 days, today included
python plot_lajolla_range.py --start 2024-12-05 --update
```

- `--end` takes a date or `today` (the default).
- `--duration N` plots the N days ending on `--end`, both days included; it can't be combined with `--start`.
- By default the script uses the cached data; `--update` downloads new data first.

Each writes two figures for its station to `plots/`:

| File | Contents |
|---|---|
| `<station>_ssh_<start>_<end>.png` | The website figure for this window |
| `<station>_ssh_lowpass_<start>_<end>.png` | The same with the tides removed |

**How the tides are removed:** with the PL64 low-pass filter, a Python port of `pl64ta.m` (Rosenfeld 1983) that matches MATLAB to 1e-10. It has its half amplitude at 33 h and is applied separately to the hourly observed and model series.
- Each gap-free stretch is filtered on its own (`pl64ta.m` would join the pieces on either side of a gap). The first and last 66 h of each stretch, the filter half-width, are blanked next to gaps.
- The last 66 h before the end of the data are shown but shaded grey. The filter needs data up to 66 h later, so these values change as new data arrive.
- 10 extra days before the window are loaded, so the start of the window isn't a filter edge.
- PL64 removes the semidiurnal tides (M2, S2) completely but passes about 3 % of K1 and 9 % of O1. A wiggle of about 2–3 cm with a period near 25 h therefore remains in both series; it mostly cancels in ΔSSH.

## NOAA measured vs predicted: plot_sdbay_noaa_tide.py and plot_lajolla_noaa_tide.py

```bash
python plot_sdbay_noaa_tide.py                              # 2025-01-01 to today
python plot_lajolla_noaa_tide.py --end today --duration 90
python plot_sdbay_noaa_tide.py --start 2026-06-01 --end 2026-09-30 --update
```

They take the same `--start`/`--end`/`--duration`/`--update` options. Writes `plots/<station>_noaa_tide_<start>_<end>.png`:

| Panel | Contents |
|---|---|
| 1 | NOAA measured water level (verified in blue, preliminary in green) and NOAA's harmonic tide prediction, hourly, MSL datum, no pressure correction |
| 2 | Residual = measured − predicted, hourly and PL64 low-passed; y axis fixed at −0.5 to 0.5 m |

The residual is the sea level the tide prediction leaves out:
- weather (pressure, wind, storms),
- coastal-trapped and Kelvin waves,
- seasonal and year-to-year anomalies beyond NOAA's average annual cycle (the Sa constituent, about 8 cm),
- the sea-level rise since NOAA's MSL datum period (1983–2001), which makes its mean positive.

## Method

**Model**
- `zeta` at the grid point nearest each gauge (table at the top).
- Forecast day 1: hours 1–24 of each history file, rounded to the hour. Where history files overlap, the later file wins.

**Observations**
- 6-minute water level relative to mean sea level, in GMT. Each hour uses the sample on the hour, as in `NOAA_SeaSurfaceHeight_Extraction.m`.
- **Inverse-barometer correction**, as in `correction_WaterLevel_InverseBarometer.m`:
  - `wl_corr = wl − 0.01 m/hPa × (P − 1013.25 hPa)`
  - P is the air pressure on the hour; missing hours take the nearest available value.
  - This removes the pressure effect from the gauge level so it compares with the model.
- Pressure comes from the NOAA CO-OPS API for the same station. NDBC `sdbc1` and `ljac1`, which the MATLAB used, carry the same station data (identical values at San Diego Bay).

**Difference from the MATLAB:** `correction_WaterLevel_InverseBarometer.m` only reads the NDBC pressure files for 2024 and 2025. From January 2026 its pressure stays at the last 2025 value. At San Diego Bay that puts the corrected level off by up to about 14 cm (the real pressure ranged 1008–1028 hPa in Jan–Feb 2026, while the MATLAB used 1014.8 hPa). The Python uses the real pressure. For 2024–2025 it matches the saved MATLAB output.

## Data

- **Model:** `model_cache/SDBay_LV4_day1_zeta.csv` and `model_cache/LaJolla_LV3_day1_zeta.csv`. Later runs only read new days. A full rebuild (delete the file) reads every history file since 5 Dec 2024 and takes about 6–8 minutes per station. To start from a later date on a new machine, run `python extract_model.py --start YYYY-MM-DD` once first.
- **Observations:** `obs_cache/<station>/<product>_<YYYYMM>.csv`, one file per month for `water_level`, `air_pressure` and `predictions`. Each run downloads missing months plus the current and previous month again, so preliminary data are replaced by verified data within about a month.

## Requirements

- Python 3 with `numpy`, `pandas`, `matplotlib` and `netCDF4`. `scipy` is also needed for the date-range and NOAA tide plots (the filter), but not for the website plot.
- Read access to `/dataSIO/PFM_Simulations` (model archive and grids).
- Internet access to the NOAA CO-OPS API.
- On the SIO server these are all available in `/home/akg004/anaconda3`.
