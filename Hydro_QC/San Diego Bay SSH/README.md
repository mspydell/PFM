# PFM vs observed sea surface height in San Diego Bay

Compares the PFM LV4 day-1 forecast sea surface height (`zeta`) with the NOAA tide gauge in San Diego Bay (station 9410170). All times are UTC.

This is a Python version of the MATLAB location ID 1 (SD Bay):

- `PFM_Master_Code_For_TimeSeriesExtraction.m`
- `Observed_Master_Code_For_TimeSeriesExtraction.m`
- `PFM_Master_Code_For_Plots.m`, which calls `Plot_SSH.m`

## Files

| File | What it does |
|---|---|
| `plot_sdbay_3weeks.py` | Daily plot of the past 3 weeks for the QC website, saved to `plots_web/`. Updates the data first. |
| `plot_sdbay_range.py` | The same plot for any start/end dates, saved to `plots/`. Not for the website. |
| `plot_ssh.py` | Shared plotting code. |
| `extract_model.py` | Extracts `zeta` at the gauge from `/dataSIO/PFM_Simulations/Archive/LV4_His` into `model_cache/`. |
| `download_obs.py` | Downloads water level, air pressure and NOAA tide predictions from the NOAA CO-OPS API into `obs_cache/`. |
| `plot_noaa_tide.py` | NOAA measured vs predicted water level and the residual, no model. |

## The figures

Each run draws two figures, each with two panels laid out like `Plot_SSH.m`:

| Figure | Panel 1 | Panel 2 |
|---|---|---|
| `SDBay_ssh` | Hourly observed water level and model `zeta`, with r, RMSE, bias and hours compared | ΔSSH = observed − model, fixed at −0.5 to 0.5 m |
| `SDBay_ssh_lowpass` | The same after removing the tides | ΔSSH of the low-passed series, fixed at −0.5 to 0.5 m |

**How the tides are removed:** with the PL64 low-pass filter from your `pl64ta.m` (Rosenfeld 1983), half amplitude at 33 h, applied separately to the hourly observed and model series. The Python port matches MATLAB `pl64ta` to 1e-10.

- Each gap-free stretch is filtered on its own. `pl64ta.m` would join the pieces on either side of a gap.
- The first and last 66 h of each stretch (the filter half-width) are blanked next to gaps.
- The last 66 h before the end of the data are shown but shaded grey: the filter needs data from up to 66 h later, so these values change as new data arrive.
- PL64 removes the semidiurnal tides (M2, S2) completely but passes about 3 % of K1 and 9 % of O1. A wiggle of about 2–3 cm with a period near 25 h therefore remains in both series; it mostly cancels in ΔSSH.

## Method

**Model**
- `zeta` at the LV4 grid point nearest the gauge (−117.1767, 32.715). The grid point is at −117.17663, 32.71484, depth 11.1 m.
- Forecast day 1: hours 1–24 of each history file. Where history files overlap, the later file wins.

**Observations**
- 6-minute water level relative to mean sea level, in GMT. Each hour uses the sample on the hour, as in `NOAA_SeaSurfaceHeight_Extraction.m`.
- **Inverse-barometer correction**, as in `correction_WaterLevel_InverseBarometer.m`:
  - `wl_corr = wl − 0.01 m/hPa × (P − 1013.25 hPa)`
  - P is the air pressure on the hour. Missing hours take the nearest available value.
  - This removes the pressure effect on the gauge level so it compares with the model.
- Pressure comes from the NOAA CO-OPS API for 9410170. NDBC `sdbc1`, which the MATLAB used, is the same sensor (identical values).

**Difference from the MATLAB:** `correction_WaterLevel_InverseBarometer.m` only reads the NDBC files for 2024 and 2025. In 2026 its pressure was stuck at the last 2025 value (1014.8 hPa), when the actual range was 1008–1028 hPa. The corrected level was therefore off by up to about 14 cm. For 2024–2025 the Python matches the saved MATLAB output.

## Usage

```bash
python plot_sdbay_3weeks.py                    # update data, plot the past 3 weeks up to today
python plot_sdbay_3weeks.py --no-update        # plot from cached data only
python plot_sdbay_range.py --start 2026-06-01 --end 2026-08-31
python plot_sdbay_range.py --start 2024-12-05 --end 2026-10-09 --update
```

- `plot_sdbay_3weeks.py` writes `plots_web/SDBay_ssh_<YYYYMMDD>_3weeks.png` and `plots_web/SDBay_ssh_lowpass_<YYYYMMDD>_3weeks.png`, and deletes the previous day's plots.
- `plot_sdbay_range.py` writes `plots/SDBay_ssh_<start>_<end>.png` and `plots/SDBay_ssh_lowpass_<start>_<end>.png`. By default it uses the cached data; add `--update` to fetch new data first.

## plot_noaa_tide.py: NOAA measured vs predicted (no model)

```bash
python plot_noaa_tide.py                                    # 2025-01-01 to today
python plot_noaa_tide.py --start 2026-06-01 --end 2026-09-30 --update
```

Writes `plots/SDBay_noaa_tide_<start>_<end>.png` with two panels:

| Panel | Contents |
|---|---|
| 1 | NOAA measured water level (verified in blue, preliminary in green) and NOAA's harmonic tide prediction, hourly, MSL datum. No pressure correction. |
| 2 | Residual = measured − predicted, hourly and PL64 low-passed, fixed at −0.5 to 0.5 m |

The residual is the sea level that the tide prediction leaves out: weather, coastal-trapped and Kelvin waves, and seasonal or year-to-year anomalies beyond NOAA's average annual cycle (Sa, about 8 cm). It also includes the sea-level rise since NOAA's MSL datum period (1983–2001). The predictions are downloaded by `download_obs.py` with the other NOAA data.

## Data

- **Model:** `model_cache/SDBay_LV4_day1_zeta.csv`. A full rebuild (delete the file) takes about 8 minutes. Later runs only read new days.
- **Observations:** `obs_cache/<product>_<YYYYMM>.csv`, one file per month. Only the current and previous months are downloaded again.

## Requirements

Python with `pandas`, `numpy`, `matplotlib` and `netCDF4` (installed in `/home/akg004/anaconda3`). It also needs read access to `/dataSIO/PFM_Simulations` and internet access to the NOAA API.
