#!/bin/bash
# Weekly County CSVs, run from cron Monday 06:00: the 7 forecasts Tue..Mon ending today, and the
# full history since 2025-01-01. ddPCR comes from the shared cache /dataSIO/PFM_Simulations/cache/Dye_ddPCRENT,
# refreshed by make_county_csv.py (ddPCR_ENT/update_cache.py).
# Today's web file usually lands 05:20-06:30, so wait up to 90 min for it first.
# Log: run_weekly.log
cd "$(dirname "$0")"
umask 002
echo "=== $(date '+%F %T %Z')"
WEB=/dataSIO/PFM_Simulations/Archive/web/web_data_$(date -u +%Y%m%d)00.nc
for i in $(seq 18); do
    if [ -f "$WEB" ]; then
        s1=$(stat -c %s "$WEB"); sleep 60; s2=$(stat -c %s "$WEB")
        [ "$s1" = "$s2" ] && break                      # size steady: file complete
    else
        sleep 300
    fi
done
[ -f "$WEB" ] && echo "web file: $(basename "$WEB")" || echo "no $(basename "$WEB") after 90 min; using the files there are"
PY=${PY:-python3}    # set PY to use another python; needs netCDF4, numpy, pandas
$PY make_county_csv.py                 # the week: <beach>_dye_forecast_<first>_<last>.csv
$PY make_county_csv.py --all           # since 2025-01-01: <beach>_dye_forecast_20250101_<today>.csv
