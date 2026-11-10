"""Hourly SBOO currents from the PFM DA test LV4 his files, computed exactly as the QC mooring model series.

Uses Moorings/common/extract_model.py (qc_paths.MOORINGS_COMMON) (same grid point, grid rotation and depth ranges):
  u_top, v_top    east/north, layer-weighted mean 2.83-5.83 m below mean sea level (ADCP top bins)
  u_davg, v_davg  east/north, layer-weighted mean 3.43 m to 0.92 * 31.48 m (ADCP depth average)
Records 0-23 of each daily DA file (runs from 00 UTC; 1-day runs to 2026-11-02, 5-day runs from 11-03).

Cache: da_sboo_currents.csv; only DA files not in it yet are read.
Usage: ~/anaconda3/bin/python extract_da_sboo.py
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DA_CACHE = os.path.join(os.environ.get('HYDRO_QC_DATA', '/dataSIO/PFM_Simulations/cache'), 'DA')  # CSV caches
from qc_paths import MOORINGS_COMMON  # noqa: E402
sys.path.insert(0, MOORINGS_COMMON)
from extract_model import SITES, layer_mean, nearest_points, read_file, static_fields, z_levels  # noqa: E402

DA_HIS = '/dataSIO/PFM_Simulations/Archive/DA/PFM_testing/LV4_His/LV4_ocean_his_*0000_testing.nc'
OUT = os.path.join(DA_CACHE, 'da_sboo_currents.csv')
SITE = 'SBOO'


def currents(path, point, static):
    """Hourly u_top, v_top, u_davg, v_davg (east/north) for day 1 of one DA file."""
    t, out = read_file(path, {SITE: point}, 1)
    d = {k: np.asarray(v, float) for k, v in out[SITE].items()}
    for k in d:
        d[k][np.abs(d[k]) > 1e30] = np.nan
    _, zw = z_levels(static, d['zeta'])
    ang = point['angle']
    ue = d['u'] * np.cos(ang) - d['v'] * np.sin(ang)
    vn = d['u'] * np.sin(ang) + d['v'] * np.cos(ang)
    ref = np.zeros(len(t))                       # SBOO depths are below mean sea level
    cfg = SITES[SITE]
    df = pd.DataFrame(index=pd.DatetimeIndex(t, name='time'))
    for name in ('top', 'davg'):
        df[f'u_{name}'] = layer_mean(ue, zw, ref, *cfg[name])
        df[f'v_{name}'] = layer_mean(vn, zw, ref, *cfg[name])
    return df


def main():
    files = sorted(p for p in glob.glob(DA_HIS) if re.search(r'his_\d{8}0000_testing\.nc$', p))
    cache = pd.read_csv(OUT, parse_dates=['time'], index_col='time') if os.path.exists(OUT) else None
    if cache is not None and cache.index.tz is None:      # CSV stores UTC without tz; read_file times are UTC-aware
        cache.index = cache.index.tz_localize('UTC')
    point = nearest_points('LV4', [SITE])[SITE]
    static = static_fields(files[0], {SITE: point})[SITE]
    new = []
    for p in files:
        day = pd.Timestamp(re.search(r'his_(\d{8})', p).group(1), tz='UTC')
        if cache is not None and (cache.index.floor('D') == day).sum() >= 24:
            continue
        new.append(currents(p, point, static))
        print('read', os.path.basename(p))
    if new:
        add = pd.concat(new)
        cache = add if cache is None else pd.concat([cache[~cache.index.isin(add.index)], add])
        cache = cache.sort_index()
        cache.to_csv(OUT, date_format='%Y-%m-%dT%H:%M:%S', float_format='%.5g')
    print(f'{OUT}: {len(cache)} hours, {cache.index.min()} to {cache.index.max()}')


if __name__ == '__main__':
    main()
