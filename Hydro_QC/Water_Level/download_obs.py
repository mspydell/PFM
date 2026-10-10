#!/usr/bin/env python3
"""Download and cache NOAA tide-gauge water level, air pressure and tide predictions (NOAA CO-OPS API).

Python port of NOAATide_WaterLevel.m + NOAA_SeaSurfaceHeight_Extraction.m +
correction_WaterLevel_InverseBarometer.m (Observed_Master_Code_For_TimeSeriesExtraction.m):
  water level: 6-min, datum MSL, GMT, metric; hourly value = the sample on the hour
  air pressure: same station (NDBC sdbc1 / ljac1, which the MATLAB code read, carry the same
    station's data), the sample on the hour; missing hours filled with the nearest value
  inverse-barometer correction: wl_corr = wl - 0.01 m/hPa * (P - 1013.25), so the gauge can be
    compared with the model (which has no atmospheric pressure forcing)
  predictions: NOAA's hourly harmonic tide prediction (MSL), used by plot_noaa_tide.py
Each month is cached in obs_cache/<station>/<product>_<YYYYMM>.csv; only the current and
previous month are downloaded again on later runs.

  python download_obs.py                      # refresh both stations from 2024-12 to now
  python download_obs.py --station LaJolla
From other scripts:
  from download_obs import load_obs
  obs = load_obs("SDBay", "2026-09-01", "2026-10-07")  # hourly: wl, pres, wl_corr (m, hPa, m), UTC index
"""
import argparse
import os
import urllib.request

import pandas as pd

from stations import STATIONS

API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(OUT_DIR, "obs_cache")
FIRST_MONTH = "2024-12"
P_REF = 1013.25      # hPa, reference pressure
IB_M_PER_HPA = 0.01  # 1 hPa ~ 1 cm of sea level (rho g ~ 1e4 Pa/m)
PRODUCTS = {"water_level": ("Water Level", "&datum=MSL"), "air_pressure": ("Pressure", ""),
            "predictions": ("Prediction", "&datum=MSL&interval=h")}  # NOAA harmonic tide prediction, hourly


def cache_dir(station):
    return os.path.join(CACHE_DIR, station)


def _fetch_month(station, product, month):
    """One month of data as CSV text."""
    start = pd.Timestamp(month)
    end = start + pd.offsets.MonthEnd(0)
    url = (f"{API}?product={product}{PRODUCTS[product][1]}&station={STATIONS[station]['noaa']}&time_zone=GMT"
           f"&units=metric&format=csv&application=PFM_QC&begin_date={start:%Y%m%d}&end_date={end:%Y%m%d}%2023:59")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def update(station, end=None):
    """Download months not yet cached, and always the current and previous month."""
    d = cache_dir(station)
    os.makedirs(d, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    end = pd.Timestamp(end) if end else now
    recent = {f"{now:%Y%m}", f"{now - pd.offsets.MonthBegin(1):%Y%m}"}
    for month in pd.period_range(FIRST_MONTH, f"{end:%Y-%m}", freq="M"):
        for product in PRODUCTS:
            path = os.path.join(d, f"{product}_{month.strftime('%Y%m')}.csv")
            if os.path.exists(path) and month.strftime("%Y%m") not in recent:
                continue
            text = _fetch_month(station, product, month.start_time)
            if not text.startswith("Date Time"):
                print(f"{station} {product} {month}: no data ({text.strip()[:80]})")
                continue
            with open(path + ".part", "w") as f:
                f.write(text)
            os.replace(path + ".part", path)
            print(f"{station} {product} {month}: {text.count(chr(10)) - 1} samples")


def _read_csvs(station, product):
    d = cache_dir(station)
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.startswith(product + "_") and name.endswith(".csv"):
            df = pd.read_csv(os.path.join(d, name), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            yield df


def _read(station, product):
    frames = [pd.Series(pd.to_numeric(df[PRODUCTS[product][0]], errors="coerce").values,
                        index=pd.to_datetime(df["Date Time"], utc=True)) for df in _read_csvs(station, product)]
    if not frames:
        return pd.Series(dtype=float)
    s = pd.concat(frames)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _window(df, start, end):
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return df


def load_obs(station, start=None, end=None):
    """Hourly observed water level (m, MSL), air pressure (hPa) and inverse-barometer-corrected level (m)."""
    wl, pres = _read(station, "water_level"), _read(station, "air_pressure")
    on_hour = lambda s: s[s.index.minute == 0]  # MATLAB: unique(dateshift(t, 'start', 'hour')) -> the :00 sample
    df = pd.DataFrame({"wl": on_hour(wl)})
    df["pres"] = on_hour(pres).reindex(df.index)
    df["pres"] = df["pres"].interpolate(method="nearest", limit_direction="both")  # fillmissing(..., 'nearest')
    df["wl_corr"] = df["wl"] - IB_M_PER_HPA * (df["pres"] - P_REF)
    return _window(df, start, end)


def load_noaa(station, start=None, end=None):
    """NOAA's own hourly series, without the pressure correction: measured water level (the sample on
    the hour; verified where NOAA has verified it, preliminary otherwise), its quality flag
    ('v'/'p'), and NOAA's harmonic tide prediction (m, MSL)."""
    frames = [pd.DataFrame({"measured": pd.to_numeric(df["Water Level"], errors="coerce").values,
                            "quality": df["Quality"].astype(str).str.strip().values},
                           index=pd.to_datetime(df["Date Time"], utc=True))
              for df in _read_csvs(station, "water_level")]
    wl = pd.concat(frames)
    wl = wl[~wl.index.duplicated(keep="last")].sort_index()
    wl = wl[wl.index.minute == 0]
    pred = _read(station, "predictions")
    df = pd.DataFrame({"predicted": pred[pred.index.minute == 0]}).join(wl, how="outer")
    return _window(df, start, end)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--station", nargs="+", choices=STATIONS, default=list(STATIONS))
    for s in p.parse_args().station:
        update(s)
        obs = load_obs(s)
        print(f"{s}: hourly obs {obs.index.min()} to {obs.index.max()}, {obs['wl'].notna().sum()} h water level, "
              f"{obs['pres'].notna().sum()} h pressure")


if __name__ == "__main__":
    main()
