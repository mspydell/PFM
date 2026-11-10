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
| Model Q<sub>ptb</sub> (blue) | Total discharge of the river sources carrying dye_01 in the LV4 river forcing (see [Sources of model data](#sources-of-model-data)). |
| Model Q<sub>ptb,ww</sub> (green) | The raw-sewage part: the sum of \|river_transport\| × river_dye_01, weighted over the layers by river_Vshape (as `Dye_Loading.py`). |

A second panel (log axis) shows the Tijuana River, as `Hourly_Flow_rate_only_SDTJRE.png` of `Model_vs_Observation_river_flow.m` (`PFM_Master_Code_For_Plots.m`, `Obs_Lo = 6`):

| Line | Source |
|---|---|
| Observed Tijuana River, hourly (grey) and daily mean (black) | IBWC `Discharge.Best Available@11013300` (as `IBWC_data.m`), already in m³/s, hourly values made the same way. |
| Model Q<sub>tj</sub> (orange) | Total discharge of the TJR sources (rivers 1–5, as `data_river.m`) in the same forcing. |

The log axis runs from 0.01 to 300 m³/s as in the MATLAB plot, and goes higher when a storm peak needs it.

- Left axis: m³/s. Right axis: the same scale in MGD.
- Times are Pacific (PDT/PST), as on the dye vs ddPCR QC plots. The data are read in UTC and converted; the plot window, dates and daily means are Pacific days.
- The model lines continue past the plot window to the end of the latest forecast (the next ~5 days; the forcing files hold 111–123 hours), **dashed** after now, with a dotted vertical line at now. For an older window (`--date` or `--end` in the past) the dashed part is the forecast issued on the last day, starting at the end of that day.
- The record starts on 2024-12-05, for both the model and the observations. Nothing earlier is plotted, read or cached.

## Sources of model data

The model flows are the LV4 river forcing the model ran with. They come from two sources:

| Source | Files | Contents | Available |
|---|---|---|---|
| LV4 river forcing archive | `/dataSIO/PFM_Simulations/Archive/Forcing/river_LV4_<YYYYMMDDHH>.nc` | The ROMS river file of each forecast: `river_transport`, `river_Vshape`, `river_dye_01`, `river_dye_02` for 9 sources, about 5 days ahead. Gives Q<sub>ptb</sub>, Q<sub>ptb,ww</sub> and Q<sub>tj</sub>. | From 2025-02-20, with some days missing |
| Matt's river discharge files | `/dataSIO/PFM_Simulations/Archive/riverQ_from_matt/LV4_riverQ_<YYYYMMDD>.nc` | Hourly discharge for Sweetwater, Otay and TJRE (5 days, from 01:00 UTC of the day). No Punta Bandera. | 2024-10-01 to 2025-10-16; daily to 2025-02-28, then mostly only on days missing from the forcing archive |

### Which source is used when

Each model hour takes the **latest forecast that covers it**. Matt's file for a day counts as the forecast issued that day. A forcing file issued the same day wins, and where both exist they agree.

| Period | Q<sub>tj</sub> | Q<sub>ptb</sub>, Q<sub>ptb,ww</sub> |
|---|---|---|
| Before 2024-12-05 | Not used. The October–November 2024 LV4 runs had no rivers ("No Waves No Rivers" in the logs). | Not used |
| 2024-12-05 to 2025-02-19 (no forcing archive) | Matt's files: Q<sub>tj</sub> = TJRE | Hardcoded in `get_pb_flow_and_dye`: 2.1906 m³/s with dye 0.7, so Q<sub>ptb,ww</sub> = 1.533 m³/s |
| 2025-02-20 to 2025-07-31 | Forcing archive. 36 days are missing from it. On 35 of them (2025-03-06, 03-08, 03-23, 03-24, 04-04 to 04-06, 06-08 to 06-29, and 07-01 to 07-07 except 07-03), Matt's file for that day. 2025-03-27 has no Matt file and takes the earlier forecasts. | Forcing archive. On missing days before 2025-04-01, the hardcoded values above; after that, the last forcing value. |
| From 2025-08-01 | Forcing archive only. Missing days take the earlier forecasts, which run 5 days ahead. | Forcing archive only, the same way |

Other rules:

- **Q<sub>tj</sub>** is the sum of rivers 1–5, selected by index, since `mk_lv4_river_nc` puts 0.2 × Q on each. Forcing files up to mid-2025 tag only 4 of the 5 with dye_02, so selecting them by the tag gave only 0.8 × Q<sub>tj</sub>.
- **Q<sub>ptb</sub>** comes from the sources carrying dye_01.
- **Gaps:** hours that no source covers keep the last value. Hours before the first file take its first value, as `Dye_Loading.py` does.

### How Matt's files were checked

The question was whether Matt's files are the forcing the model actually used.

| Period | Check | Result |
|---|---|---|
| 2024-12-06 to 2025-02-21 | ROMS prints the min/max of `river_transport` for every record it reads. Compared with `Archive/Log/LV4_forecast<date>.log`. | Each TJ source = 0.2 × Matt's TJRE exactly. PTB = 2.1906 m³/s. On 2024-12-05 alone the run used a placeholder of 0.01 m³/s per TJ source; the plot shows Matt's value for that day. |
| 2025-02-20 to 2025-07-07 | Compared with forcing files issued within 2 days, over the same hours | Ratio 1.00. The exception is 2025-03-06 and 03-08: during a storm the forecasts differ (ratio 0.15–0.89), and there's no log with river records to check. |
| 2025-08-15 to 2025-10-16 | The same comparison | The forcing is only 0.01–0.5 × Matt's TJRE. The TJ input changed (IBWC observed flow plus the NWM forecast change from 2025-08-22), so these 12 files are **not** used (`MATT_END = 2025-08-01`). |

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

The data cache is on the shared disk, `/dataSIO/PFM_Simulations/cache/FlowRate/`, so runs on swell and backwash use and update the same files (group `feddersen-lab`, group-writable). Set the environment variable `HYDRO_QC_DATA` to use another root (the cache is then `$HYDRO_QC_DATA/FlowRate/cache/`). Plots stay next to the scripts. A run only downloads or reads what is new:

| Folder | Contents | Refreshed |
|---|---|---|
| `cache/obs/` | Raw IBWC samples, one CSV per data set and UTC month, e.g. `Discharge_Telemetry_ADS_mgd_11_PUNTA_BANDERA_202610.csv` (time UTC, flow in the data set's unit: MGD for Punta Bandera and SAB, m³/s for the Tijuana River). | Months not yet in the cache are downloaded; the current and previous month are downloaded again on every run, so IBWC revisions are picked up. If IBWC is unreachable, the cached month is used. |
| `cache/model/` | Hourly model flows of each river forcing file, `river_LV4_<YYYYMMDDHH>.csv` (time UTC, `q_ptb`, `q_ptb_ww`, `q_tj` in m³/s). | Computed once per forcing file; recomputed if the forcing file is newer than its CSV. |
| `cache/model/` | Hourly TJRE discharge from each of Matt's river files, `river_LV4_<YYYYMMDD>.csv` (time UTC, `q_tj` in m³/s). These have 8 digits in the name; the forcing files' caches have 10 (`YYYYMMDDHH`). | The same: computed once per file and recomputed if the file is newer. |

The hourly averaging and the "latest forecast wins" merge are done after reading the cache, so the plots are the same as without it. Delete a file (or the whole `cache/`) to rebuild it; nothing else is needed.

## Requirements

Base anaconda Python (`/home/akg004/anaconda3/bin/python`), with `pandas`, `matplotlib`, `numpy` and `netCDF4`.
It needs read access to `/dataSIO/PFM_Simulations/Archive/Forcing` and internet access to `waterdata.ibwc.gov`.
