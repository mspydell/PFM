"""Hourly shoreline dye_01 / dye_02 at IB-079, IB-068 and Imperial Beach Pier (EH-030), from the PFM DA
test LV4 his files. The operational Day 1 with the same method is in the shared cache
/dataSIO/PFM_Simulations/cache/Dye_ddPCRENT/station_dye01_dye02_day1.csv (~/modelQC/Dye_vs_ddPCRENT).

Shoreline-cell method of station_dye.py (dye vs ddPCR code, the same as update_station_dye.py in
~/Python_files/PTB_TJRE_DyeFactor):
surface dye at the most onshore water cell with depth <= 1.2 m in the station's eta row.
Eta rows: IB-079 1034 and IB-068 910 as in the dashboard; Imperial Beach Pier 748 = nearest shoreline cell to
the QC-website IB_pier point (32.58008 N, -117.13338 E).
Records 0-23 of each daily file (the runs start at 00 UTC), as operational Day 1.

Cache (only new files are read): da_station_dye.csv
Usage: python extract_da_dye.py [start, default 2026-10-19]
"""

import glob
import os
import re
import sys

import netCDF4 as nc
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DA_CACHE = os.path.join(os.environ.get('HYDRO_QC_DATA', '/dataSIO/PFM_Simulations/cache'), 'DA')  # CSV caches
from qc_paths import DYE_DDPCR  # noqa: E402
sys.path.insert(0, DYE_DDPCR)
import station_dye as usd  # noqa: E402

STATION_J = {'IB-079': 1034, 'IB-068': 910, 'EH-030': 748}
SOURCES = {
    'da': ('/dataSIO/PFM_Simulations/Archive/DA/PFM_testing/LV4_His/LV4_ocean_his_*0000_testing.nc',
           r'his_(\d{8})0000_testing\.nc$', 'da_station_dye.csv'),
}
RECS = range(0, 24)
COLS = [f'{p}_{s.replace("-", "")}' for s in STATION_J for p in ('ptb', 'tjre')]


def update(src, start):
    pattern, regex, name = SOURCES[src]
    out = os.path.join(DA_CACHE, name)
    files = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(regex, p)
        if m and pd.Timestamp(m.group(1)) >= start:
            files[pd.Timestamp(m.group(1))] = p
    cache = pd.read_csv(out, parse_dates=['time'], index_col='time') if os.path.exists(out) else None
    if cache is None or list(cache.columns) != COLS:     # new station set: rebuild
        cache = pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name='time'))
    with nc.Dataset(next(iter(files.values()))) as d:
        shorei = usd.shore_index(d['mask_rho'][:].filled(0))
    new = {}
    for t0, path in files.items():
        if all(t0 + pd.Timedelta(hours=k) in cache.index for k in RECS):
            continue
        new |= usd.read_file(path, RECS, shorei, STATION_J)
        print(f'{src}: read {os.path.basename(path)}', flush=True)
    if new:
        add = pd.DataFrame.from_dict(new, orient='index')[COLS]
        cache = pd.concat([cache[~cache.index.isin(add.index)], add]).sort_index()
        cache.index.name = 'time'
        cache.to_csv(out, date_format='%Y-%m-%dT%H:%M:%S')
    print(f'{out}: {len(cache)} hours, {cache.index.min()} to {cache.index.max()}')


if __name__ == '__main__':
    start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else '2026-10-19')
    for src in SOURCES:
        update(src, start)
