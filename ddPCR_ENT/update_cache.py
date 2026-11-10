#!/usr/bin/env python3
"""Build and update the shared dye + ddPCR Enterococcus cache, from 2025-01-01 to today.

Dye at three sites, named as on the website: IB_pier (ddPCR EH-030), SilverStrand (IB-068), Coronado_AvLunar (IB-079).
ddPCR at every station the County portal and State BeachWatch report south of Point Loma.
Cache folder: $HYDRO_QC_DATA/Dye_ddPCRENT (default /dataSIO/PFM_Simulations/cache/Dye_ddPCRENT).

  ddpcr_ent_recent.csv          Every ddPCR Enterococcus sample at every South County station: State
                                BeachWatch for the full history, the County portal (past ~60 days) on top.
  station_dye01_dye02_day1.csv  Hourly shoreline dye_01 (ptb_<site>) and dye_02 (tjre_<site>), kept
  station_dye01_dye02_day5.csv  separate, from the LV4 his files (station_dye.py).
                                Day 1 = records 0-23 of each day's 00 UTC run (else its 06 or 12 UTC run),
                                day 5 = records 96-119.

Each run only reads what is new; rows already in the cache are kept.

  python update_cache.py              # update everything
  python update_cache.py --only ddpcr # ddpcr | dye
"""
import argparse
import glob
import os
import re
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd

from download_water_quality import download_sd_water_quality
import station_dye

CACHE = os.path.join(os.environ.get('HYDRO_QC_DATA', '/dataSIO/PFM_Simulations/cache'), 'Dye_ddPCRENT')
FIRST = pd.Timestamp('2025-01-01')
TZ = 'America/Los_Angeles'

# site (named as in sites_dye_tot_day1_day5_<run>.csv) -> MATLAB 1-based LV4 eta row of its shoreline cell.
# ddPCR stations: IB_pier EH-030 (j 748 = the cell nearest the website's IB_pier site),
# SilverStrand IB-068, Coronado_AvLunar IB-079.
STATION_J = {'IB_pier': 748, 'SilverStrand': 910, 'Coronado_AvLunar': 1034}
DAYS = {1: range(0, 24), 5: range(96, 120)}
DYE_COLS = [f'{p}_{s}' for s in STATION_J for p in ('ptb', 'tjre')]

PORTAL_DAYS = 60   # the County portal reaches back about this far
DDPCR_COLS = ['station', 'description', 'beach_name', 'latitude', 'longitude', 'time_local', 'time_utc',
              'ent', 'qualifier', 'result', 'method', 'source']


def path(name):
    return os.path.join(CACHE, name)


def write_csv(df, name, **kw):
    tmp = path(name) + '.tmp'
    df.to_csv(tmp, **kw)
    os.replace(tmp, path(name))


def save_dye(cache, new, name):
    """Put the new hours into the cache, write it and return it."""
    if new:
        # reindex: from_dict drops hours with no values (the NaN hours of a cut-short run)
        add = pd.DataFrame.from_dict(new, orient='index').reindex(index=list(new), columns=DYE_COLS)
        cache = pd.concat([cache[~cache.index.isin(add.index)], add])
        new.clear()
    cache = cache[cache.index >= FIRST].sort_index()
    cache.index.name = 'time'
    write_csv(cache, name, date_format='%Y-%m-%dT%H:%M:%S', float_format='%.6e')
    return cache


def update_dye():
    runs = {}   # date -> {'HHMM': path}
    for p in sorted(glob.glob(station_dye.HIS)):
        m = re.search(r'his_(\d{8})(\d{4})\.nc$', p)  # skips e.g. ..._dt2_BLOWUP.nc
        if m and pd.Timestamp(m.group(1)) >= FIRST - pd.Timedelta(days=5):
            runs.setdefault(pd.Timestamp(m.group(1)), {})[m.group(2)] = p
    # each date's run: the 00 UTC run, else that day's 06 or 12 UTC run
    files = {}
    for date, v in runs.items():
        hhmm = next((r for r in ('0000', '0600', '1200') if r in v), None)
        if hhmm:
            files[date + pd.Timedelta(hours=int(hhmm[:2]))] = v[hhmm]
    has00 = {date for date, v in runs.items() if '0000' in v}
    with nc.Dataset(next(iter(files.values()))) as d:
        shorei = station_dye.shore_index(d['mask_rho'][:].filled(0))
    today = pd.Timestamp.now('UTC').tz_localize(None).normalize()

    for day, recs in DAYS.items():
        name = f'station_dye01_dye02_day{day}.csv'
        cache = (pd.read_csv(path(name), parse_dates=['time'], index_col='time') if os.path.exists(path(name))
                 else pd.DataFrame(columns=DYE_COLS, index=pd.DatetimeIndex([], name='time')))
        have = set(cache.index)
        new, n_read = {}, 0
        for t0, p in files.items():
            hours = [t0 + pd.Timedelta(hours=k) for k in recs]
            if t0.hour:
                # a 06/12 UTC run reaches into the next day; those hours come from that day's 00 UTC run if any
                hours = [h for h in hours if (h - pd.Timedelta(days=day - 1)).normalize() not in has00]
            if not hours or hours[-1] < FIRST or all(h in have for h in hours):
                continue
            keep = set(hours)
            new |= {h: row for h, row in station_dye.read_file(p, recs, shorei, STATION_J).items() if h in keep}
            if t0.normalize() < today:
                # a finished run cut short (e.g. at hour 108): keep its missing hours as NaN so it is not reread
                new |= {h: {} for h in hours if h not in new}
            n_read += 1
            if n_read % 20 == 0:  # save as we go, so an interrupted first build keeps its progress
                print(f'day {day}: read {n_read} files, to {os.path.basename(p)}', flush=True)
                cache = save_dye(cache, new, name)
        cache = save_dye(cache, new, name)
        print(f'dye day {day}: read {n_read} his files; {cache.index.min()} to {cache.index.max()}; '
              + ', '.join(f'{s} {cache[f"ptb_{s}"].notna().sum()} h' for s in STATION_J))


def fetch_ddpcr(start, end, beachwatch):
    tmp = path('ddpcr_download.tmp.csv')
    df = download_sd_water_quality(output_csv=tmp, start_date=f'{start:%Y-%m-%d}', end_date=f'{end:%Y-%m-%d}',
                                   parameter='Enterococcus', ddpcr_only=True, force_beachwatch=beachwatch)
    if os.path.exists(tmp):
        os.remove(tmp)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=DDPCR_COLS)
    local = pd.to_datetime(df['SampleDate'].astype(str).str[:10] + ' ' + df['SampleTime'].fillna('12:00:00'))
    utc = local.dt.tz_localize(TZ, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').dt.tz_localize(None)
    return pd.DataFrame({
        'station': df['Station Name'].str.strip(),  # BeachWatch puts a numeric id in Station_ID
        'description': df['Description'],
        'beach_name': df['Beach Name'], 'latitude': df['Latitude'], 'longitude': df['Longitude'],
        'time_local': local, 'time_utc': utc,
        'ent': pd.to_numeric(df['Result'], errors='coerce'), 'qualifier': df['qualifier'],
        'result': df['Result'], 'method': df['method'],
        'source': 'beachwatch' if beachwatch else 'county'})


def update_ddpcr():
    out = path('ddpcr_ent_recent.csv')
    old = pd.read_csv(out, parse_dates=['time_local', 'time_utc']) if os.path.exists(out) else None
    today = pd.Timestamp.now(TZ).tz_localize(None).normalize()
    parts = [old] if old is not None else []
    # BeachWatch: everything the first time, then the past 90 days (it can lag the County portal)
    bw_start = FIRST if old is None else today - pd.Timedelta(days=90)
    for start, bw in ((bw_start, True), (today - pd.Timedelta(days=PORTAL_DAYS), False)):
        try:
            parts.append(fetch_ddpcr(start, today, bw))
        except Exception as e:
            print(f'ddPCR download ({"BeachWatch" if bw else "County"}) failed: {e}', file=sys.stderr)
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df[df['time_local'] >= FIRST]
    # same sample = same station and local time; the newest download wins
    df = df.drop_duplicates(['station', 'time_local'], keep='last').sort_values(['time_local', 'station'])
    write_csv(df[DDPCR_COLS], 'ddpcr_ent_recent.csv', index=False, date_format='%Y-%m-%d %H:%M')
    print(f'ddPCR: {len(df)} samples at {df["station"].nunique()} stations, '
          f'{df["time_local"].min():%Y-%m-%d} to {df["time_local"].max():%Y-%m-%d} (local)')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--only', choices=['ddpcr', 'dye'], help='update only this part')
    args = p.parse_args()
    os.umask(0o002)  # shared with the group, like the other ModelQC caches
    os.makedirs(CACHE, exist_ok=True)
    for part, fn in (('ddpcr', update_ddpcr), ('dye', update_dye)):
        if args.only in (None, part):
            fn()


if __name__ == '__main__':
    main()
