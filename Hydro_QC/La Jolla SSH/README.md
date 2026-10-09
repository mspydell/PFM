# PFM vs observed sea surface height at La Jolla

Compares the PFM LV3 day-1 forecast sea surface height (`zeta`) with the NOAA tide gauge at La Jolla (station 9410230 (Scripps Pier)). All times are UTC.

The LV4 grid doesn't reach La Jolla, so this uses LV3. The code is the same as in `../SanDiegoBay`; only the station, grid and position differ.

This is a Python version of MATLAB location ID 11 (La Jolla) in `PFM_Master_Code_For_TimeSeriesExtraction.m`, `Observed_Master_Code_For_TimeSeriesExtraction.m` and `PFM_Master_Code_For_Plots.m` (which calls `Plot_SSH.m`).

## Files

| File | What it does | Website |
|---|---|---|
| `plot_lajolla_3weeks.py` | Daily plot of the past 3 weeks, saved to `plots_web/`. Updates the data first. | yes |
| `plot_ssh.py` | Shared code for the hourly figure, and the station settings (`NAME`, `TITLE`, `GAUGE`). | yes |
| `download_obs.py` | Downloads water level, air pressure and tide predictions from the NOAA CO-OPS API into `obs_cache/`. | yes |
| `extract_model.py` | Extracts `zeta` at the gauge from `/dataSIO/PFM_Simulations/Archive/LV3_His` into `model_cache/`. | yes |
| `plot_lajolla_range.py` | Hourly and tides-removed figures for any dates, saved to `plots/`. | no |
| `plot_ssh_lowpass.py` | The PL64 low-pass filter and the tides-removed figure. | no |
| `plot_noaa_tide.py` | NOAA measured vs predicted water level and their difference; no model. | no |

The website plot needs only the four files marked "yes". None of them contains any filter code.

## Website plot: plot_lajolla_3weeks.py

```bash
python plot_lajolla_3weeks.py                    # update data, plot the past 3 weeks up to today
python plot_lajolla_3weeks.py --no-update        # plot from the cached data only
python plot_lajolla_3weeks.py --date 2026-09-30  # end on another day (older plots are kept)
```

Writes `plots_web/LaJolla_ssh_<YYYYMMDD>_3weeks.png` and deletes the previous day's plot, so `plots_web/` holds only the newest one. It runs daily from cron on the SIO server, because it needs `/dataSIO`.

The figure has two panels, laid out like `Plot_SSH.m`:

| Panel | Contents |
|---|---|
| 1 | Hourly observed water level (pressure-corrected, MSL datum) and model `zeta`, with r, RMSE, bias and the number of hours compared |
| 2 | ΔSSH = observed − model, with its mean and standard deviation; y axis fixed at −0.5 to 0.5 m |

## Date-range plots: plot_lajolla_range.py

```bash
python plot_lajolla_range.py                                       # default start (START in the script) to today
python plot_lajolla_range.py --start 2026-06-01 --end 2026-08-31
python plot_lajolla_range.py --end today --duration 45             # the last 45 days, today included
python plot_lajolla_range.py --start 2024-12-05 --end 2026-10-09 --update
```

- `--end` takes a date or `today` (the default).
- `--duration N` plots the N days ending on `--end`, both days included; it can't be combined with `--start`.
- By default the script uses the cached data; `--update` downloads new data first.

Writes two figures to `plots/`:

| File | Contents |
|---|---|
| `LaJolla_ssh_<start>_<end>.png` | The website figure for this window |
| `LaJolla_ssh_lowpass_<start>_<end>.png` | The same with the tides removed |

**How the tides are removed:** with the PL64 low-pass filter, a Python port of `pl64ta.m` (Rosenfeld 1983) that matches MATLAB to 1e-10. It has its half amplitude at 33 h and is applied separately to the hourly observed and model series.
- Each gap-free stretch is filtered on its own (`pl64ta.m` would join the pieces on either side of a gap). The first and last 66 h of each stretch, the filter half-width, are blanked next to gaps.
- The last 66 h before the end of the data are shown but shaded grey. The filter needs data up to 66 h later, so these values change as new data arrive.
- 10 extra days before the window are loaded, so the start of the window isn't a filter edge.
- PL64 removes the semidiurnal tides (M2, S2) completely but passes about 3 % of K1 and 9 % of O1. A wiggle of about 2–3 cm with a period near 25 h therefore remains in both series; it mostly cancels in ΔSSH.

## NOAA measured vs predicted: plot_noaa_tide.py

```bash
python plot_noaa_tide.py                             # 2025-01-01 to today
python plot_noaa_tide.py --end today --duration 90
python plot_noaa_tide.py --start 2026-06-01 --end 2026-09-30 --update
```

Takes the same `--start`/`--end`/`--duration`/`--update` options. Writes `plots/LaJolla_noaa_tide_<start>_<end>.png`:

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
- `zeta` at the LV3 grid point nearest NOAA's gauge position (−117.25714, 32.86689): −117.25647, 32.86733 (79 m away), depth 10.7 m.
- The MATLAB registry entry for ID 11 (`locs(25)`) has the SD Bay position (−117.1767, 32.715) by mistake, so it samples San Diego Bay instead of La Jolla.
- Forecast day 1: hours 1–24 of each history file, rounded to the hour. Where history files overlap, the later file wins.

**Observations**
- 6-minute water level relative to mean sea level, in GMT. Each hour uses the sample on the hour, as in `NOAA_SeaSurfaceHeight_Extraction.m`.
- **Inverse-barometer correction**, as in `correction_WaterLevel_InverseBarometer.m`:
  - `wl_corr = wl − 0.01 m/hPa × (P − 1013.25 hPa)`
  - P is the air pressure on the hour; missing hours take the nearest available value.
  - This removes the pressure effect from the gauge level so it compares with the model.
- Pressure comes from the NOAA CO-OPS API for station 9410230 (Scripps Pier). NDBC `ljac1`, which the MATLAB used, republishes the same station's data.

**Difference from the MATLAB:** `correction_WaterLevel_InverseBarometer.m` only reads the NDBC pressure files for 2024 and 2025 (for La Jolla, `ljac1`). From January 2026 its pressure therefore stays at the last 2025 value, as at San Diego Bay, where this put the corrected level off by up to about 14 cm. The Python uses the real pressure.

## Data

- **Model:** `model_cache/LaJolla_LV3_day1_zeta.csv`. Later runs only read new days. A full rebuild (delete the file) reads every history file since 5 Dec 2024 and takes about 6 minutes. To start from a later date on a new machine, run `python extract_model.py --start YYYY-MM-DD` once first.
- **Observations:** `obs_cache/<product>_<YYYYMM>.csv`, one file per month for `water_level`, `air_pressure` and `predictions`. Each run downloads missing months plus the current and previous month again, so preliminary data are replaced by verified data within about a month.

## Requirements

- Python 3 with `numpy`, `pandas`, `matplotlib` and `netCDF4`. `scipy` is also needed for the date-range and NOAA tide plots (the filter), but not for the website plot.
- Read access to `/dataSIO/PFM_Simulations` (model archive and grid).
- Internet access to the NOAA CO-OPS API.
- On the SIO server these are all available in `/home/akg004/anaconda3`.
