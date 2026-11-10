# PFM dye vs ddPCR Enterococcus

QC plots of the PFM dye forecast against ddPCR Enterococcus at three beaches, and the shared cache of dye and ddPCR data from 1 January 2025 to today.

| Panel | Dye site (column names in both dye files) | ddPCR station |
|---|---|---|
| Imperial Beach Pier | `IB_pier` | EH-030 |
| Silver Strand (Guard Shack) | `SilverStrand` | IB-068 |
| Coronado (Avenida Lunar) | `Coronado_AvLunar` | IB-079 |

The code used to be in `~/pfm_ddPCRvsDye_forecast_csv` (and its copy in `~/PFM/ddPCR_ENT`). The data it made are now in the shared cache, so runs on swell and backwash use the same files.

## Files

| File | What it does |
|---|---|
| `plot_dye_ddpcr.py` | QC-website plot: the past 2 weeks, month or year up to today, plus the next 5 days. Refreshes the ddPCR in the cache first. Also holds the shared plotting code. |
| `plot_dye_ddpcr_range.py` | The same plot for any start and end dates, into `plots/`, and the dye vs ddPCR scatter of the same window next to it (`<name>_scatter.png`). The QC website's 40-day pair: `--end today --duration 40 --out plots/dye_ddpcr_40day.png` writes `dye_ddpcr_40day.png` and `dye_ddpcr_40day_scatter.png`. |
| `make_forecast_csv.py` | Builds the website-dye CSV in the cache from `/dataSIO/PFM_Simulations/Archive/web` (reads every web file each time, about 10 s). |
| `update_cache.py` | Updates the ddPCR and the shoreline dye in the cache. The ddPCR dashboard (`~/Python_files/PTB_TJRE_DyeFactor`) runs it every morning. |
| `download_water_quality.py` | Downloads ddPCR from the San Diego County portal or State BeachWatch. |
| `station_dye.py` | Shoreline dye at a station from an LV4 history file (Python port of the MATLAB `Shoreline_Dye_5days.m`). |
| `plot_water_quality.py` | ddPCR summary figures and table for the QC website (`make_ddPCR_ENT_Plots.sh`). |

## Running

```bash
python make_forecast_csv.py                       # add new website forecasts to the cache
python plot_dye_ddpcr.py                          # past 2 weeks + 5 forecast days -> plots/dye_ddpcr_<YYYYMMDD>_2weeks.png
python plot_dye_ddpcr.py --period month           # 2weeks | month | year
python plot_dye_ddpcr.py --date 2026-10-04        # end on another day (older plots are kept)
python plot_dye_ddpcr.py --no-update              # use the cached ddPCR, no download
python plot_dye_ddpcr_range.py --start 2026-06-01 --end 2026-08-31
python plot_dye_ddpcr_range.py --end today --duration 40 --out plots/dye_ddpcr_40day.png
python plot_dye_ddpcr_range.py --start 2025-01-01 --end today --update   # download new ddPCR first
python update_cache.py                            # update ddPCR and shoreline dye (--only ddpcr | dye)
```

- Dates are Pacific days, and both the start and end day are included.
- A normal `plot_dye_ddpcr.py` run deletes the older plot of the same period, so `plots/` holds only the newest one.
- The range script uses the cached data unless you add `--update`.

## Plot

- Left axis: total dye (log scale, 10⁻⁶ to 10⁻²), with green, yellow and red risk bands.
- Right axis: ddPCR Enterococcus (copies/100 ml), the left axis multiplied by 1413/10⁻⁵, so the 1413 threshold lines up with dye 10⁻⁵ (the top of the green band) and both axes span the same decades. Red markers are above 1413; the rest are small black markers.
- X axis: Pacific time (PDT/PST). A grey dotted line marks now; dye after it is forecast.
- Blue: day-1 dye. Dashed purple: day-5 dye.
- Each panel title shows the Pearson correlation r of log10(day-1 dye) and log10(ddPCR ENT): each past sample is paired with the mean dye within ±3 h (`PAIR_HOURS`, as `dt` in the MATLAB scripts).

## Scatter plot

- Two panels, dye day 1 (blue) and dye day 5 (purple), as in the time series, over the same green/yellow/red dye bands. ddPCR ENT on x, dye on y (both log).
- The three beaches in each panel: circles IB Pier, squares Silver Strand, triangles Avenida Lunar. Each past sample with its mean dye within ±3 h.
- Legend (below each panel): per beach, Pearson r of log10(dye) and log10(ddPCR ENT) and its least-squares line `dye = 10^a(±sa) ENT^b(±sb)` with the standard errors of a and b in brackets, as on the ddPCR dashboard; title: r of all three; dashed: least-squares line of all three in log-log space (as `polyfit` in the MATLAB scripts), its equation in the lower left (`dye = 10^a ENT^b`). Dotted: ENT = 1413. Dye below 10⁻⁶ is drawn at 10⁻⁶.

## Cache

`/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT/`, group `feddersen-lab`, group-writable. Set `HYDRO_QC_DATA` to use another parent folder. All times are UTC unless a column says local.

| File | Contents | Written by |
|---|---|---|
| `sites_dye_tot_day1_day5_<YYYYMMDDHH>.csv` | Hourly website forecast dye (`sites_dye_tot`) at the website sites, day 1 and day 5. Named after the newest run; only the newest file is kept. | `make_forecast_csv.py` |
| `ddpcr_ent_recent.csv` | Every ddPCR Enterococcus sample at all South County stations: station, description, beach, position, local and UTC time, ENT, qualifier, method, source. BeachWatch for the whole history, the County portal (last 60 days) on top. | `update_cache.py` |
| `station_dye01_dye02_day1.csv`, `station_dye01_dye02_day5.csv` | Hourly shoreline dye_01 (`ptb_<site>`, Punta Bandera) and dye_02 (`tjre_<site>`, Tijuana River estuary) at the website sites IB_pier (ddPCR EH-030), SilverStrand (IB-068) and Coronado_AvLunar (IB-079), from the LV4 history files. Day 1 is records 0–23 of each day's 00 UTC run (on days without one, its 06 or 12 UTC run), day 5 records 96–119. | `update_cache.py` |

- **Website dye:** day 1 is lead hours 1–24 and day 5 lead hours 97–120. If two runs cover the same hour, the newer one wins. The web files start on 12 April 2025.
- **Shoreline dye:** each site is the shoreline cell in one LV4 eta row (MATLAB 1-based): `IB_pier` j = 748 (the cell nearest the website's `IB_pier` point), `SilverStrand` j = 910, `Coronado_AvLunar` j = 1034. Columns are `ptb_<site>` (dye_01) and `tjre_<site>` (dye_02); the website file has `<site>_day1` and `<site>_day5` (total dye). The two dye sources agree closely (Day 1, log r 0.995 at Silver Strand and 0.975 at Avenida Lunar).
- **ddPCR:** BeachWatch and the County portal agree on every sample in the County's 60 days. Samples are matched by station and local time, and the newest download wins.

## Requirements

Python with `pandas`, `numpy`, `matplotlib` and `netCDF4` (the `roms-env` environment or `/home/akg004/anaconda3`). Read access to `/dataSIO/PFM_Simulations/Archive`, write access to the cache, and internet access for the ddPCR downloads.
