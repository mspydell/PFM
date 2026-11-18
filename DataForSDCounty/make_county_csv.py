"""CSVs of forecast total dye and ddPCR for the San Diego County, at the 3 South County beaches.

Dye: sites_dye_tot in the PFM website files /dataSIO/PFM_Simulations/Archive/web/web_data_YYYYMMDDHH.nc
(same as the QC website, ~/PFM/ddPCR_ENT/make_forecast_csv.py), the day's 00 UTC run (else 06, else 12).
Where the website files lack the site (before 2025-04-12; Avenida Lunar before 2025-06-23) it comes
from mat_dye_hourly.csv (extract_mat_dye.py, the MATLAB Shoreline_Dye_5days output). For each forecast writes prediction Day 1 - Day 5 at 06, 07, 08 and
09 local Pacific time (PDT/PST): Day N is the local date forecast_date + N - 1. Days with no run are
skipped; runs that stop at hour 108 leave Day 5 blank.

ddPCR Enterococcus (copies/100 ml) and sample time (local) at each beach from the shared cache
/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT/ddpcr_ent_recent.csv (State BeachWatch history, the County
portal's past ~2 months on top), refreshed first with ddPCR_ENT/update_cache.py. Put on
the row whose date is the sample date (the County samples about once a day, 06:30-08:30); the first sample
of the day. Blank where not sampled (yet).

Output: one file per beach, one row per forecast / prediction day, columns dye_0600 .. dye_0900:
  <beach>_dye_forecast_<first>_<last>.csv   the week
  <beach>_dye_forecast_20250101_<last>.csv  with --all: every forecast from 2025-01-01 to today (replaces
                                            the previous one)
Usage: python make_county_csv.py                    # the 7 forecasts ending today
       python make_county_csv.py 2026-10-20 2026-10-26
       python make_county_csv.py --all
"""

import glob
import os
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ddPCR_ENT'))
from update_cache import CACHE, update_ddpcr  # noqa: E402

WEB = '/dataSIO/PFM_Simulations/Archive/web/web_data_{:%Y%m%d}{}.nc'
RUNS = ['00', '06', '12']
MAT_CSV = os.path.join(HERE, 'mat_dye_hourly.csv')
VAR = 'sites_dye_tot'
EPOCH = pd.Timestamp('1999-01-01')
TZ = 'America/Los_Angeles'
ALL_FIRST = pd.Timestamp('2025-01-01')
# sites matched by coordinates, as in make_forecast_csv.py; County ddPCR station of each
BEACHES = {
    'Avenida_Lunar':       (32.67415, -117.17386),
    'Silver_Strand':       (32.62492, -117.13978),
    'Imperial_Beach_Pier': (32.58008, -117.13338),
}
STATIONS = {'Avenida_Lunar': 'IB-079', 'Silver_Strand': 'IB-068', 'Imperial_Beach_Pier': 'EH-030'}
TOL = 1e-3
HOURS = [6, 7, 8, 9]
DAYS = range(1, 6)


def load_ddpcr(start, end):
    """{(beach, local date): (ENT, 'YYYY-MM-DD HH:MM')}, first sample of the day, from the shared cache."""
    try:
        update_ddpcr()
    except Exception as e:
        print(f'ddPCR update failed, using the cached samples: {e}')
    df = pd.read_csv(os.path.join(CACHE, 'ddpcr_ent_recent.csv'), parse_dates=['time_local'])
    beach = {s: b for b, s in STATIONS.items()}
    df['beach'] = df['station'].map(beach)
    df = df.dropna(subset=['beach', 'time_local', 'ent']).sort_values('time_local')
    df = df[(df['time_local'] >= start) & (df['time_local'] < end + pd.Timedelta(days=1))]
    out = {}
    for b, t, v in zip(df['beach'], df['time_local'], df['ent']):
        out.setdefault((b, t.date()), (v, f'{t:%Y-%m-%d %H:%M}'))
    return out


def forecast(fdate, mat):
    """{beach: (UTC hours, dye)} of the day's first run, from the MATLAB extract where the website lacks
    the site; None if there is none."""
    out = {}
    for run in RUNS:
        path = WEB.format(fdate, run)
        if os.path.exists(path):
            with nc.Dataset(path) as d:
                t = pd.DatetimeIndex(EPOCH + pd.to_timedelta(np.ma.filled(d['time'][:].astype(float), np.nan), 'D'))
                data = np.ma.filled(d[VAR][:].astype(float), np.nan)
                lat, lon = np.asarray(d['sites_lat'][:]), np.asarray(d['sites_lon'][:])
            for name, (la, lo) in BEACHES.items():
                i = np.flatnonzero((np.abs(lat - la) < TOL) & (np.abs(lon - lo) < TOL))
                if len(i):
                    out[name] = (t.round('h'), data[:, i[0]])
            break
    if mat is not None and len(out) < len(BEACHES):
        runs = mat[mat.run.dt.normalize() == fdate]
        if len(runs):
            one = mat[mat.run == runs.run.min()]
            for name in BEACHES:
                out.setdefault(name, (pd.DatetimeIndex(one.valid_utc), one[name].to_numpy()))
    return out or None


def make(first, last, full=False):
    now = pd.Timestamp.now(TZ).tz_localize(None)
    mat = (pd.read_csv(MAT_CSV, parse_dates=['run', 'valid_utc'])
           if first < pd.Timestamp('2025-06-23') and os.path.exists(MAT_CSV) else None)
    ddpcr = load_ddpcr(first, min(last + pd.Timedelta(days=max(DAYS) - 1), now))
    rows = []
    for fdate in pd.date_range(first, last):
        fc = forecast(fdate, mat)
        if fc is None:
            print(f'no forecast for {fdate:%Y-%m-%d}')
            continue
        for day in DAYS:
            date = fdate + pd.Timedelta(days=day - 1)
            row = {'forecast_date': f'{fdate:%Y-%m-%d}', 'prediction_day': day, 'date': f'{date:%Y-%m-%d}'}
            for hr in HOURS:
                local = (date + pd.Timedelta(hours=hr)).tz_localize(TZ)
                utc = local.tz_convert('UTC').tz_localize(None)
                for name in BEACHES:
                    t, dye = fc.get(name, ([], []))
                    k = np.flatnonzero(t == utc)
                    row[f'{name}_dye_{hr:02d}00'] = dye[k[0]] if len(k) else np.nan
            for name in BEACHES:
                ent, tsample = ddpcr.get((name, date.date()), (np.nan, ''))
                row[f'{name}_ddPCR_ENT'] = ent
                row[f'{name}_ddPCR_time'] = tsample
            rows.append(row)
    if not rows:
        sys.exit('no forecasts found')
    out = pd.DataFrame(rows)
    span = f'{out.forecast_date.iloc[0].replace("-", "")}_{out.forecast_date.iloc[-1].replace("-", "")}'
    # one file per beach: forecast / day / date, dye at each hour, ddPCR value and sample time
    cols = [f'dye_{hr:02d}00' for hr in HOURS] + ['ddPCR_ENT', 'ddPCR_time']
    for b in BEACHES:
        one = out[list(out.columns[:3]) + [f'{b}_{c}' for c in cols]]
        one.columns = list(out.columns[:3]) + cols
        one['ddPCR_ENT'] = one['ddPCR_ENT'].round().astype('Int64')
        out_csv = os.path.join(HERE, f'{b}_dye_forecast_{span}.csv')
        if full:   # drop the previous full-history file
            for old in glob.glob(os.path.join(HERE, f'{b}_dye_forecast_{ALL_FIRST:%Y%m%d}_*.csv')):
                if old != out_csv:
                    os.remove(old)
        one.to_csv(out_csv, index=False, float_format='%.3e')
        print(f'{out_csv}: {len(one)} rows')


def main():
    today = pd.Timestamp.now(TZ).normalize().tz_localize(None)
    if sys.argv[1:] == ['--all']:
        make(ALL_FIRST, today, full=True)
    elif len(sys.argv) == 3:
        make(pd.Timestamp(sys.argv[1]), pd.Timestamp(sys.argv[2]))
    else:
        make(today - pd.Timedelta(days=6), today)


if __name__ == '__main__':
    main()
