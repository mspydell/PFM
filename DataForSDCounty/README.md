# DataForSDCounty

Weekly CSVs for the San Diego County. Each one has the forecast total dye plus the ddPCR Enterococcus at three South County beaches: Avenida Lunar (IB-079), Silver Strand (IB-068) and Imperial Beach Pier (EH-030).

## Files

| File | What it does |
|---|---|
| `make_county_csv.py` | Writes one CSV per beach. Each row is one forecast date and prediction day (Day 1–5). Columns `dye_0600`..`dye_0900` are in local Pacific time, followed by `ddPCR_ENT` and `ddPCR_time`. |
| `run_weekly.sh` | Cron wrapper. It waits up to 90 min for today's 00Z web file, then writes the weekly files and the full history (`--all`). |
| `extract_mat_dye.py` | One-off script that writes `mat_dye_hourly.csv` from the MATLAB shoreline-dye .mat file. It needs scipy. |
| `mat_dye_hourly.csv` | Fills dye that is missing from the web files: before 2025-04-12, and Avenida Lunar before 2025-06-23. |

## Data sources

- **Dye:** `sites_dye_tot` in `/dataSIO/PFM_Simulations/Archive/web/web_data_YYYYMMDDHH.nc`. The 00Z run is used, else 06Z, else 12Z. This is the same source as the QC website.
- **ddPCR:** the shared cache `/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT`, which is refreshed through `../ddPCR_ENT/update_cache.py`.

## Usage

```bash
python make_county_csv.py                         # the 7 forecasts ending today
python make_county_csv.py 2026-10-20 2026-10-26   # a date range
python make_county_csv.py --all                   # 2025-01-01 to today; replaces the previous full-history file
```

The output goes into this folder as `<beach>_dye_forecast_<first>_<last>.csv`. It is ignored by git.

The weekly run happens from cron on swell every Monday at 06:00:

```
0 6 * * 1 /path/to/PFM/DataForSDCounty/run_weekly.sh >> /path/to/PFM/DataForSDCounty/run_weekly.log 2>&1
```

Set `PY=/path/to/python` if `python3` on PATH lacks netCDF4/pandas.

Notes: about 1 in 6 runs stop at hour 108, which leaves Day 5 blank. Days with no run are skipped.
