# PTB and TJRE dye volume

Total raw-sewage (dye) volume in the LV4 domain: PTB (dye_01), TJRE (dye_02), and the two combined.

| File | What it does |
|---|---|
| `plot_dye_volume.py` | QC-website plot of the past 3 weeks, saved to `plots_web/`. Also holds the shared plotting code. |
| `plot_dye_volume_range.py` | The same plot for any start and end dates, saved to `plots/` (not for the website). |
| `volume_cache.csv` | Hourly volumes already computed, one row per forecast file and record. |

## What is plotted

- Volume = sum over wet cells of dye × Hz × dx × dy (m³). This is the same integral as `PTB_TJRE_DyeFactor/Dye_Volume.py`. The script imports `file_volume`, `forecast_files` and `gap_plan` from it.
- Each hour comes from the latest LV4 forecast that covers it (`/dataSIO/PFM_Simulations/Archive/LV4_His/LV4_ocean_his_*.nc`). Past hours are day-1 values. Days without a forecast are filled from earlier forecasts, as in `fill_Dye_Volume.py`. These values match `Volf_ptb` / `Volf_tjre` in `Dye_Volume_01-Jan-2025_30-Sep-2026.nc`.
- The dotted line marks the start of the newest forecast. Everything to its right is forecast.
- Times are Pacific (PDT/PST), as on the other QC plots.

## plot_dye_volume.py: plot for the QC website

```bash
python plot_dye_volume.py                     # past 3 weeks to the end of today (Pacific) -> plots_web/
python plot_dye_volume.py --date 2026-09-30   # 3 weeks ending on another day (older plots are kept)
```

A normal run (without `--date`) writes `plots_web/Dye_volume_<YYYYMMDD>_3weeks.png` and deletes older `*_3weeks.png`.

## plot_dye_volume_range.py: any start and end dates

```bash
python plot_dye_volume_range.py --start YYYY-MM-DD [--end YYYY-MM-DD|today] [--out PATH]
python plot_dye_volume_range.py --duration N [--end YYYY-MM-DD|today] [--out PATH]
```

| Option | Meaning | Default |
|---|---|---|
| `--start` | First day to plot (Pacific). Give this or `--duration`. | |
| `--end` | Last day to plot (Pacific), included in the plot; a date or `today` | today |
| `--duration` | Number of days to plot, ending on `--end` (both days included). Can't be combined with `--start`. | |
| `--out` | Where to save the PNG | `plots/Dye_volume_<start>_<end>.png` |

Examples:

```bash
python plot_dye_volume_range.py --start 2026-08-01 --end 2026-10-08   # -> plots/Dye_volume_20260801_20261008.png
python plot_dye_volume_range.py --start 2026-01-01                    # 1 January 2026 to today
python plot_dye_volume_range.py --end today --duration 45            # the last 45 days, today included
python plot_dye_volume_range.py --start 2026-07-01 --end 2026-07-31 --out july.png
```

For long ranges the plot draws thinner lines and puts minor ticks at months instead of days.

## Run time

One hour takes about 2.5 s to compute. Hours are computed in parallel (16 processes) and cached in `volume_cache.csv`, which both scripts share. A run only computes hours that aren't cached yet. A range that is not cached yet takes about 3 minutes per month of data (Aug 1 to Oct 8 took 6 minutes with 3 weeks already cached). Rerunning a range, or any part of it, takes about a second.

## Requirements

Base anaconda Python (`/home/akg004/anaconda3/bin/python`) with `pandas`, `matplotlib`, `numpy` and `netCDF4`. It needs `/home/akg004/Python_files/PTB_TJRE_DyeFactor/Dye_Volume.py` and read access to the LV4 his archive.
