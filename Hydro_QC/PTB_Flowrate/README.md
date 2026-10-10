# PTB discharge: observed vs model

Plots of the Punta Bandera (PTB) discharge: the observed IBWC flow against the PFM model forcing.

| File | What it does |
|---|---|
| `plot_ptb_q.py` | Daily QC-website plot: the past 3 weeks, into `plots_web/`. Also holds the shared plotting code. |
| `plot_ptb_q_range.py` | The same plot for any start and end dates, into `plots/` (not for the website). |

## What is plotted

| Line | Source |
|---|---|
| Observed Punta Bandera, hourly (grey) and daily mean (black) | IBWC AQWebportal `Discharge.Telemetry-ADS-mgd@11-PUNTA-BANDERA`, MGD converted to m³/s. Each hourly value is the mean of the five samples centred on the first sample of the hour, as in `PTB_data.m`. |
| Model Q<sub>ptb</sub> (blue) | Total discharge of the river sources carrying dye_01 in `/dataSIO/PFM_Simulations/Archive/Forcing/river_LV4_*.nc`. |
| Model Q<sub>ptb,ww</sub> (green) | The raw-sewage part: the sum of \|river_transport\| × river_dye_01, weighted over the layers by river_Vshape (as `Dye_Loading.py`). |

- Left axis: m³/s. Right axis: the same scale in MGD.
- Times are Pacific (PDT/PST), as on the dye vs ddPCR QC plots. The data are read in UTC and converted; the plot window, dates and daily means are Pacific days.
- Each model hour comes from the latest forecast that covers it, so days without a forcing file are filled from the earlier forecasts, which run 5 days ahead.
- Longer gaps keep the last forecast value. Hours before the first forcing file (2025-02-20) take its first value, as `Dye_Loading.py` does.

## plot_ptb_q.py: daily plot for the QC website

```bash
python plot_ptb_q.py                     # past 3 weeks to the end of today (Pacific) -> plots_web/
python plot_ptb_q.py --date 2026-09-30   # 3 weeks ending on another day (older plots are kept)
```

The daily run writes `plots_web/PTB_discharge_<YYYYMMDD>_3weeks.png` and deletes older `*_3weeks.png`, so `plots_web/` only holds the newest plot. This is the same 3-week window as the SBOO / IB QC plots.

Cron runs it every morning at 06:00 (server time), with the log in `cron.log`:

```
0 6 * * * cd /home/akg004/modelQC/PTB_Q && /home/akg004/anaconda3/bin/python plot_ptb_q.py >> cron.log 2>&1
```

## plot_ptb_q_range.py: any start and end dates

```bash
python plot_ptb_q_range.py --start YYYY-MM-DD [--end YYYY-MM-DD|today] [--out PATH]
python plot_ptb_q_range.py --duration N [--end YYYY-MM-DD|today] [--out PATH]
```

| Option | Meaning | Default |
|---|---|---|
| `--start` | First day to plot (Pacific). Give this or `--duration`. | |
| `--end` | Last day to plot (Pacific), included in the plot; a date or `today` | today |
| `--duration` | Number of days to plot, ending on `--end` (both days included). Can't be combined with `--start`. | |
| `--out` | Where to save the PNG | `plots/PTB_discharge_<start>_<end>.png` |

Both dates are included: `--end 2026-07-31` plots through the end of July 31.

Examples:

```bash
python plot_ptb_q_range.py --start 2025-01-01 --end 2026-10-08   # whole record -> plots/PTB_discharge_20250101_20261008.png
python plot_ptb_q_range.py --start 2026-01-01                    # 1 January 2026 to today
python plot_ptb_q_range.py --end today --duration 45            # the last 45 days, today included
python plot_ptb_q_range.py --start 2026-07-01 --end 2026-07-31 --out july.png
```

For long ranges the plot draws thinner hourly lines and puts minor ticks at months instead of days.
The IBWC download takes about a minute per year of data.

## Requirements

Base anaconda Python (`/home/akg004/anaconda3/bin/python`), with `pandas`, `matplotlib`, `numpy` and `netCDF4`.
It needs read access to `/dataSIO/PFM_Simulations/Archive/Forcing` and internet access to `waterdata.ibwc.gov`.
