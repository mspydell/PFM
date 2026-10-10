# PFM dye forecast vs ddPCR Enterococcus

Plots the PFM dye forecast (day 1 and day 5) against ddPCR Enterococcus measurements for three sites, one panel per site:

| Panel | Dye site | ddPCR station |
|---|---|---|
| Imperial Beach Pier | `IB_pier` | EH-030 |
| Silver Strand | `SilverStrand` | IB-068 |
| Coronado (Avenida Lunar) | `Coronado_AvLunar` | IB-079 |

- Left axis: total dye (log scale), with green, yellow and red risk bands.
- Right axis: ddPCR Enterococcus (copies/100 ml). Red markers are values above 1413; the rest are small black markers.
- X axis: Pacific time (PDT/PST).

## Files

| File | What it does |
|---|---|
| `plot_dye_ddpcr_range.py` | Plots any start/end date range (described below). |
| `plot_dye_ddpcr.py` | Plots the past 2 weeks, month or year up to today, plus the next 5 days. Also holds the shared plotting code. |
| `make_forecast_csv.py` | Builds `sites_dye_tot_day1_day5_<YYYYMMDDHH>.csv` from the PFM netCDF archive. |
| `download_water_quality.py` | Downloads ddPCR data from the San Diego County portal or State BeachWatch. |

## Requirements

Python 3.9 or newer, with `pandas`, `matplotlib`, `netCDF4` and `numpy`.

```bash
pip install pandas matplotlib netCDF4 numpy
```

## plot_dye_ddpcr_range.py: plot any date range

### Usage

```bash
python plot_dye_ddpcr_range.py [--start YYYY-MM-DD] [--end YYYY-MM-DD|today] [--duration DAYS] [--out PATH]
```

| Option | Meaning | Default |
|---|---|---|
| `--start` | First day to plot (Pacific date or `today`) | `2026-07-01` |
| `--end` | Last day to plot, included in the plot (Pacific date or `today`) | `2026-07-15` (or `today` if `--duration` is given) |
| `--duration` | Number of days back from `--end` (cannot be combined with `--start`) | None |
| `--out` | Where to save the PNG | `plots/dye_ddpcr_<start>_<end>.png` |

Both dates are included: `--end 2026-07-15` plots through the end of July 15. When using `--end today`, today's local Pacific date is used.

### Examples

Plot the past 14 days up to today:

```bash
python plot_dye_ddpcr_range.py --end today --duration 14
# -> plots/dye_ddpcr_<14_days_ago>_<today>.png
```

Plot 1–15 July 2026:

```bash
python plot_dye_ddpcr_range.py --start 2026-07-01 --end 2026-07-15
# -> plots/dye_ddpcr_20260701_20260715.png
```

Plot the whole summer:

```bash
python plot_dye_ddpcr_range.py --start 2026-06-01 --end 2026-08-31
# -> plots/dye_ddpcr_20260601_20260831.png
```

Save to a file name you choose:

```bash
python plot_dye_ddpcr_range.py --start 2026-09-01 --end 2026-09-30 --out sept2026.png
```

Use the default dates (set by `START` and `END` near the top of the script):

```bash
python plot_dye_ddpcr_range.py
```

You can also call it from Python:

```python
import pandas as pd
from plot_dye_ddpcr import TZ, plot_window

start = pd.Timestamp("2026-07-01", tz=TZ)
end = pd.Timestamp("2026-07-15", tz=TZ) + pd.Timedelta(days=1)  # +1 day so July 15 is included
plot_window(start, end, "my_plot.png", "PFM dye vs ddPCR, 1-15 July 2026")
```

### Where the data comes from

- **Dye:** the newest `sites_dye_tot_day1_day5_*.csv` in this folder. To include the latest forecast, run `python make_forecast_csv.py` first. That script reads the PFM archive at `/dataSIO/PFM_Simulations/Archive/web`, so you need access to that path.
- **ddPCR:** downloaded automatically for the date range and saved to `ddpcr_ent_recent.csv`. The County portal only goes back a few months, so for start dates more than 60 days ago the script uses State BeachWatch instead.

The tick spacing on the x axis is chosen from the length of the range. A gray dotted vertical line marks the current time; dye after that line is forecast.

## plot_dye_ddpcr.py: plot up to today

```bash
python plot_dye_ddpcr.py                    # past 2 weeks up to today, plus 5 forecast days
python plot_dye_ddpcr.py --period month     # 2weeks | month | year
python plot_dye_ddpcr.py --date 2026-10-04  # end the plot on a different "today"
```
