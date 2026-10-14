# Flow rate: observed vs model

Plots of the flow rates, observed IBWC flow against the PFM model forcing: Punta Bandera (PTB) and SAB PreTreatment, and the Tijuana River.

| File | What it does |
|---|---|
| `plot_flowrate.py` | QC-website plot: the past 3 weeks, into `plots_web/`. Also holds the shared plotting code. |
| `plot_flowrate_range.py` | The same plot for any start and end dates, into `plots/` (not for the website). |

## What is plotted

| Line | Source |
|---|---|
| Observed Punta Bandera, hourly (grey) and daily mean (black) | IBWC AQWebportal `Discharge.Telemetry-ADS-mgd@11-PUNTA-BANDERA`, MGD converted to m³/s. Each hourly value is the mean of the five samples centred on the first sample of the hour, as in `PTB_data.m`. |
| Observed SAB PreTreatment, hourly (light purple) and daily mean (purple) | IBWC `Discharge.Telemetry-ADS-mgd@11-SAB-Pre` (as `PTB_TJRE_DyeFactor/IBWC_Flow.py`), MGD converted to m³/s, hourly values made the same way. |
| Model Q<sub>ptb</sub> (blue) | Total discharge of the river sources carrying dye_01 in `/dataSIO/PFM_Simulations/Archive/Forcing/river_LV4_*.nc`. |
| Model Q<sub>ptb,ww</sub> (green) | The raw-sewage part: the sum of \|river_transport\| × river_dye_01, weighted over the layers by river_Vshape (as `Dye_Loading.py`). |

A second panel (log axis) shows the Tijuana River, as `Hourly_Flow_rate_only_SDTJRE.png` of `Model_vs_Observation_river_flow.m` (`PFM_Master_Code_For_Plots.m`, `Obs_Lo = 6`):

| Line | Source |
|---|---|
| Observed Tijuana River, hourly (grey) and daily mean (black) | IBWC `Discharge.Best Available@11013300` (as `IBWC_data.m`), already in m³/s, hourly values made the same way. |
| Model Q<sub>tj</sub> (orange) | Total discharge of the river sources carrying dye_02 (rivers 1–5, as `data_river.m`) in the same forcing files. |

The log axis runs from 0.01 to 300 m³/s as in the MATLAB plot, and goes higher when a storm peak needs it.

- Left axis: m³/s. Right axis: the same scale in MGD.
- Times are Pacific (PDT/PST), as on the dye vs ddPCR QC plots. The data are read in UTC and converted; the plot window, dates and daily means are Pacific days.
- Each model hour comes from the latest forecast that covers it, so days without a forcing file are filled from the earlier forecasts, which run 5 days ahead.
- The model lines continue past the plot window to the end of the latest forecast (the next ~5 days; the forcing files hold 111–123 hours), **dashed** after now, with a dotted vertical line at now. For an older window (`--date` or `--end` in the past) the dashed part is the forecast issued on the last day, starting at the end of that day.
- Longer gaps keep the last forecast value. Hours before the first forcing file (2025-02-20) take its first value, as `Dye_Loading.py` does.

## plot_flowrate.py: plot for the QC website

```bash
python plot_flowrate.py                     # past 3 weeks to the end of today (Pacific) -> plots_web/
python plot_flowrate.py --date 2026-09-30   # 3 weeks ending on another day (older plots are kept)
```

A normal run (without `--date`) writes `plots_web/FlowRate_<YYYYMMDD>_3weeks.png` and deletes older `*_3weeks.png`, so `plots_web/` only holds the newest plot. This is the same 3-week window as the SBOO / IB QC plots.

## plot_flowrate_range.py: any start and end dates

```bash
python plot_flowrate_range.py --start YYYY-MM-DD [--end YYYY-MM-DD|today] [--out PATH]
python plot_flowrate_range.py --duration N [--end YYYY-MM-DD|today] [--out PATH]
```

| Option | Meaning | Default |
|---|---|---|
| `--start` | First day to plot (Pacific). Give this or `--duration`. | |
| `--end` | Last day to plot (Pacific), included in the plot; a date or `today` | today |
| `--duration` | Number of days to plot, ending on `--end` (both days included). Can't be combined with `--start`. | |
| `--out` | Where to save the PNG | `plots/FlowRate_<start>_<end>.png` |

Both dates are included: `--end 2026-07-31` plots through the end of July 31.

Examples:

```bash
python plot_flowrate_range.py --start 2025-01-01 --end 2026-10-08   # whole record -> plots/FlowRate_20250101_20261008.png
python plot_flowrate_range.py --start 2026-01-01                    # 1 January 2026 to today
python plot_flowrate_range.py --end today --duration 45            # the last 45 days, today included
python plot_flowrate_range.py --start 2026-07-01 --end 2026-07-31 --out july.png
```

For long ranges the plot draws thinner hourly lines and puts minor ticks at months instead of days.
The first long range downloads the IBWC data (about a minute per year of data); later runs read it from the cache.

## Cache

Both scripts keep their data in `cache/`, next to the scripts, so a run only downloads or reads what is new:

| Folder | Contents | Refreshed |
|---|---|---|
| `cache/obs/` | Raw IBWC samples, one CSV per data set and UTC month, e.g. `Discharge_Telemetry_ADS_mgd_11_PUNTA_BANDERA_202610.csv` (time UTC, flow in the data set's unit: MGD for Punta Bandera and SAB, m³/s for the Tijuana River). | Months not yet in the cache are downloaded; the current and previous month are downloaded again on every run, so IBWC revisions are picked up. If IBWC is unreachable, the cached month is used. |
| `cache/model/` | Hourly model flows of each river forcing file, `river_LV4_<YYYYMMDDHH>.csv` (time UTC, `q_ptb`, `q_ptb_ww`, `q_tj` in m³/s). | Computed once per forcing file; recomputed if the forcing file is newer than its CSV. |

The hourly averaging and the "latest forecast wins" merge are done after reading the cache, so the plots are the same as without it. Delete a file (or the whole `cache/`) to rebuild it; nothing else is needed.

## Requirements

Base anaconda Python (`/home/akg004/anaconda3/bin/python`), with `pandas`, `matplotlib`, `numpy` and `netCDF4`.
It needs read access to `/dataSIO/PFM_Simulations/Archive/Forcing` and internet access to `waterdata.ibwc.gov`.
