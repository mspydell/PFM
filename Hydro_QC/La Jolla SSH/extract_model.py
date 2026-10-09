#!/usr/bin/env python3
"""Extract PFM LV3 sea surface height (zeta) at the La Jolla tide gauge (Scripps Pier).

Python port of PFM_Master_Code_For_TimeSeriesExtraction.m for location ID 11 ('La Jolla', LV3,
vars {'zeta'}) with LV1to4_model_data_serial_PHM_PFM_V2.m: nearest rho point to the gauge
(knnsearch on lon/lat), the forecast-day window of each history file (hours 1-24 for day 1),
times rounded to the hour, later files win for duplicate hours.
Output: model_cache/LaJolla_LV3_day<N>_zeta.csv (time UTC, zeta m). Reruns only read files from
the last saved day on.

  python extract_model.py                         # 2024-12-05 (first run) or last saved day to today
  python extract_model.py --start 2024-12-05 --end 2026-10-07
"""
import argparse
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import netCDF4
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(OUT_DIR, "model_cache")
BASE_DATA = "/dataSIO/PFM_Simulations/Archive"
GRID = "/dataSIO/PFM_Simulations/Grid/GRID_SDTJRE_LV3_rx020.nc"  # LV4 does not reach La Jolla
TIME0 = pd.Timestamp("1999-01-01", tz="UTC")  # ocean_time is seconds since this
LON, LAT = -117.25714, 32.86689  # NOAA 9410230 (Scripps Pier)
LEVEL = "LV3"


def nearest_rho():
    with netCDF4.Dataset(GRID) as g:
        lon, lat = g["lon_rho"][:], g["lat_rho"][:]
        j, i = np.unravel_index(np.argmin((lon - LON) ** 2 + (lat - LAT) ** 2), lon.shape)
        return int(j), int(i), float(lon[j, i]), float(lat[j, i]), float(g["h"][j, i])


def file_date(name):
    return pd.Timestamp(re.findall(r"\d{8}", name)[-1], tz="UTC")


def read_file(path, j, i, forecast_day):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        nt = len(ds["ocean_time"])
        t1, t2 = 24 * (forecast_day - 1), min(24 * forecast_day, nt)
        if t1 >= nt:
            return None
        t = (TIME0 + pd.to_timedelta(ds["ocean_time"][t1:t2], unit="s")).round("h")
        z = np.asarray(ds["zeta"][t1:t2, j, i], float)
    z[np.abs(z) > 1e30] = np.nan
    return pd.Series(z, index=t)


def cache_path(forecast_day):
    return os.path.join(MODEL_DIR, f"LaJolla_{LEVEL}_day{forecast_day}_zeta.csv")


def load_model(forecast_day=1):
    path = cache_path(forecast_day)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    return pd.Series(df["zeta"].values, index=df["time"].dt.tz_localize("UTC"))


def extract(start="2024-12-05", end=None, forecast_day=1, workers=6):
    end = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC").normalize()
    old = load_model(forecast_day)
    first = old.index.max().floor("D") if old is not None and len(old) else pd.Timestamp(start, tz="UTC")
    files = sorted(glob.glob(os.path.join(BASE_DATA, f"{LEVEL}_His", f"{LEVEL}_ocean_his_*.nc")))
    todo = [f for f in files if first <= file_date(f) <= end]
    print(f"{len(todo)} history files to read ({first:%Y-%m-%d} to {end:%Y-%m-%d})")
    if not todo:
        return
    j, i, lon, lat, h = nearest_rho()
    print(f"grid point (eta {j}, xi {i}) at {lon:.5f}, {lat:.5f}, depth {h:.1f} m")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = [p for p in pool.map(read_file, todo, [j] * len(todo), [i] * len(todo),
                                     [forecast_day] * len(todo)) if p is not None]
    new = pd.concat(parts)
    s = pd.concat([old, new]) if old is not None else new  # later files win
    s = s[~s.index.duplicated(keep="last")].sort_index()
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = pd.DataFrame({"time": s.index.tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"), "zeta": s.values})
    out.to_csv(cache_path(forecast_day) + ".part", index=False, float_format="%.4f")
    os.replace(cache_path(forecast_day) + ".part", cache_path(forecast_day))
    print(f"{len(todo)} files in {time.time() - t0:.0f} s; model {s.index.min()} to {s.index.max()} "
          f"-> {cache_path(forecast_day)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", default="2024-12-05", help="UTC start date (first run only; later runs resume)")
    p.add_argument("--end", help="UTC end date, inclusive (default today)")
    p.add_argument("--forecast-day", type=int, default=1, choices=range(1, 6))
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    extract(a.start, a.end, a.forecast_day, a.workers)


if __name__ == "__main__":
    main()
